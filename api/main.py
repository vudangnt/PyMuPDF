import urllib.request
import urllib.parse
import asyncio
import subprocess
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Literal
from pathlib import PurePosixPath

import pymupdf
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status

from api.auth import verify_api_key
from api.config import settings

app = FastAPI(
    title="PDF to Text API",
    description="Extract text from PDF and other documents using PyMuPDF (with OCR support)",
    version="1.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

_pool = ThreadPoolExecutor(max_workers=4)

SUPPORTED_EXTENSIONS = {
    ".pdf", ".epub", ".xps", ".fb2", ".cbz", ".svg",
    ".txt", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
    ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif",
}

OCR_LANGUAGES = "eng+vie"
TEXT_THRESHOLD = 20
OCR_DPI = 200  # 200 đủ cho resume/doc, giảm từ 300 → nhanh ~2x


def _parse_page_range(pages_str: str, total_pages: int) -> list[int]:
    result = set()
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            result.update(range(int(start) - 1, int(end)))
        else:
            result.add(int(part) - 1)
    return sorted(i for i in result if 0 <= i < total_pages)


def _ext_from_filename(filename: str) -> str:
    if "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()


def _fetch_url(url: str) -> tuple[bytes, str]:
    """Download file from URL. Returns (bytes, filename)."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="URL must use http or https scheme",
        )
    filename = PurePosixPath(parsed.path).name or "document.pdf"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "pdf-to-text-api/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to fetch URL: {exc}",
        )
    if len(content) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds {settings.MAX_FILE_SIZE_MB}MB limit",
        )
    return content, filename


def _ocr_page_subprocess(img_bytes: bytes, language: str) -> str:
    """OCR via temp file + tesseract subprocess."""
    fd, img_path = tempfile.mkstemp(suffix=".png")
    try:
        os.write(fd, img_bytes)
        os.close(fd)
        result = subprocess.run(
            ["tesseract", img_path, "stdout", "-l", language, "--psm", "3"],
            capture_output=True, text=True, timeout=60,
        )
        return result.stdout
    except Exception:
        return ""
    finally:
        try:
            os.unlink(img_path)
        except OSError:
            pass


def _extract_page_text(page: pymupdf.Page, mode: str, ocr: str) -> str:
    """Extract text from a single page. Falls back to OCR if text is sparse."""
    if mode == "blocks":
        blocks = page.get_text("blocks")
        text = "\n".join(b[4] for b in blocks if b[6] == 0)
    elif mode == "words":
        words = page.get_text("words")
        text = " ".join(w[4] for w in words)
    else:
        text = page.get_text("text")

    if ocr != "off" and len(text.strip()) < TEXT_THRESHOLD:
        pix = page.get_pixmap(dpi=OCR_DPI)
        img_bytes = pix.tobytes("png")
        ocr_text = _ocr_page_subprocess(img_bytes, ocr)
        if len(ocr_text.strip()) > len(text.strip()):
            return ocr_text
    return text


def _process_document(file_bytes: bytes, filename: str, mode: str, ocr_lang: str,
                      page_indices: list[int] | None = None) -> dict:
    """Process document - runs in thread pool."""
    ext = _ext_from_filename(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext or 'unknown'}")
    try:
        doc = pymupdf.open(stream=file_bytes, filetype=ext.lstrip("."))
    except Exception as exc:
        raise ValueError(f"Failed to open document: {exc}")

    try:
        total = len(doc)
        indices = page_indices if page_indices is not None else list(range(total))

        # OCR pages in parallel when needed
        ocr_needed = ocr_lang != "off"
        if ocr_needed:
            # Pre-check which pages need OCR
            page_data = []
            for i in indices:
                page = doc[i]
                text = page.get_text("text") if mode == "text" else ""
                needs_ocr = len(text.strip()) < TEXT_THRESHOLD
                if needs_ocr:
                    pix = page.get_pixmap(dpi=OCR_DPI)
                    page_data.append((i, text, pix.tobytes("png")))
                else:
                    page_data.append((i, None, None))

            # Run OCR in parallel for pages that need it
            ocr_results = {}
            imgs_to_ocr = [(i, img) for i, _, img in page_data if img is not None]
            if imgs_to_ocr:
                with ThreadPoolExecutor(max_workers=min(4, len(imgs_to_ocr))) as ocr_pool:
                    futures = {
                        ocr_pool.submit(_ocr_page_subprocess, img, ocr_lang): i
                        for i, img in imgs_to_ocr
                    }
                    for future in futures:
                        ocr_results[futures[future]] = future.result()

            # Assemble results
            results = []
            for i, orig_text, img in page_data:
                if img is not None and i in ocr_results:
                    ocr_text = ocr_results[i]
                    text = ocr_text if len(ocr_text.strip()) > len((orig_text or "").strip()) else (orig_text or "")
                else:
                    text = _extract_page_text(doc[i], mode, "off")
                results.append((i, text))
        else:
            results = [(i, _extract_page_text(doc[i], mode, "off")) for i in indices]

        if page_indices is not None:
            return {
                "total_pages": total,
                "pages": [{"page": i + 1, "text": t} for i, t in results],
            }
        full_text = "\n\n".join(t for _, t in results)
        return {"total_pages": total, "text": full_text, "char_count": len(full_text)}
    finally:
        doc.close()


@app.get("/health")
def health():
    return {"status": "ok", "pymupdf_version": pymupdf.__version__}


async def _resolve_input(file: UploadFile | None, url: str | None) -> tuple[bytes, str]:
    """Resolve file upload or URL to (bytes, filename). Handles Swagger empty file."""
    has_file = file is not None and file.filename
    if not has_file and not url:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Provide either 'file' or 'url'")
    if has_file and url:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Provide only one of 'file' or 'url', not both")
    if url:
        return _fetch_url(url)
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Uploaded file is empty")
    if len(content) > settings.max_file_size_bytes:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"File size exceeds {settings.MAX_FILE_SIZE_MB}MB limit")
    return content, file.filename or "document"


@app.post("/extract")
async def extract(
    file: UploadFile = File(None),
    url: str = Form(None),
    mode: Literal["text", "blocks", "words"] = Form("text"),
    ocr: str = Form("auto", description="OCR language: 'auto' (eng+vie fallback), 'off', or Tesseract lang code e.g. 'eng+vie'"),
    _key: str = Depends(verify_api_key),
):
    """Extract text from entire document. Provide either `file` (upload) or `url`.
    OCR auto-activates when page has little/no text (image-based PDF)."""
    content, filename = await _resolve_input(file, url)
    ocr_lang = OCR_LANGUAGES if ocr == "auto" else ocr

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _pool, _process_document, content, filename, mode, ocr_lang, None
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    return {
        "source": url or filename,
        "pages": result["total_pages"],
        "text": result["text"],
        "char_count": result["char_count"],
        "ocr": ocr_lang != "off",
    }


@app.post("/extract/pages")
async def extract_pages(
    file: UploadFile = File(None),
    url: str = Form(None),
    pages: str = Form(None, description="Page range e.g. '1-3,5' (default: all)"),
    mode: Literal["text", "blocks", "words"] = Form("text"),
    ocr: str = Form("auto", description="OCR language: 'auto' (eng+vie fallback), 'off', or Tesseract lang code"),
    _key: str = Depends(verify_api_key),
):
    """Extract text per page. Provide either `file` (upload) or `url`.
    OCR auto-activates when page has little/no text (image-based PDF)."""
    content, filename = await _resolve_input(file, url)
    ocr_lang = OCR_LANGUAGES if ocr == "auto" else ocr

    loop = asyncio.get_event_loop()
    try:
        # Quick open to get total pages for range parsing
        ext = _ext_from_filename(filename)
        doc = pymupdf.open(stream=content, filetype=ext.lstrip("."))
        total = len(doc)
        doc.close()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=f"Failed to open document: {exc}")

    indices = _parse_page_range(pages, total) if pages else list(range(total))
    try:
        result = await loop.run_in_executor(
            _pool, _process_document, content, filename, mode, ocr_lang, indices
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    return {
        "source": url or filename,
        "total_pages": result["total_pages"],
        "extracted_pages": len(result["pages"]),
        "pages": result["pages"],
    }
