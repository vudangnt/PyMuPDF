import urllib.request
import urllib.parse
from typing import Literal
from pathlib import PurePosixPath

import pymupdf
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status

from api.auth import verify_api_key
from api.config import settings

app = FastAPI(
    title="PDF to Text API",
    description="Extract text from PDF and other documents using PyMuPDF",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

SUPPORTED_EXTENSIONS = {
    ".pdf", ".epub", ".xps", ".fb2", ".cbz", ".svg",
    ".txt", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
}


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


def _open_document(file_bytes: bytes, filename: str) -> pymupdf.Document:
    ext = _ext_from_filename(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type: {ext or 'unknown'}",
        )
    try:
        return pymupdf.open(stream=file_bytes, filetype=ext.lstrip("."))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to open document: {exc}",
        )


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


def _extract_text(doc: pymupdf.Document, mode: str) -> list[str]:
    pages_text = []
    for page in doc:
        if mode == "text":
            pages_text.append(page.get_text("text"))
        elif mode == "blocks":
            blocks = page.get_text("blocks")
            pages_text.append("\n".join(b[4] for b in blocks if b[6] == 0))
        else:  # words
            words = page.get_text("words")
            pages_text.append(" ".join(w[4] for w in words))
    return pages_text


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
    _key: str = Depends(verify_api_key),
):
    """Extract text from entire document. Provide either `file` (upload) or `url`."""
    content, filename = await _resolve_input(file, url)

    doc = _open_document(content, filename)
    try:
        pages_text = _extract_text(doc, mode)
        full_text = "\n\n".join(pages_text)
        return {
            "source": url or filename,
            "pages": len(doc),
            "text": full_text,
            "char_count": len(full_text),
        }
    finally:
        doc.close()


@app.post("/extract/pages")
async def extract_pages(
    file: UploadFile = File(None),
    url: str = Form(None),
    pages: str = Form(None, description="Page range e.g. '1-3,5' (default: all)"),
    mode: Literal["text", "blocks", "words"] = Form("text"),
    _key: str = Depends(verify_api_key),
):
    """Extract text per page. Provide either `file` (upload) or `url`."""
    content, filename = await _resolve_input(file, url)

    doc = _open_document(content, filename)
    try:
        total = len(doc)
        indices = _parse_page_range(pages, total) if pages else list(range(total))

        result_pages = []
        for i in indices:
            page = doc[i]
            if mode == "text":
                text = page.get_text("text")
            elif mode == "blocks":
                blocks = page.get_text("blocks")
                text = "\n".join(b[4] for b in blocks if b[6] == 0)
            else:
                words = page.get_text("words")
                text = " ".join(w[4] for w in words)
            result_pages.append({"page": i + 1, "text": text})

        return {
            "source": url or filename,
            "total_pages": total,
            "extracted_pages": len(result_pages),
            "pages": result_pages,
        }
    finally:
        doc.close()
