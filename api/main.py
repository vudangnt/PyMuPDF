import urllib.request
import urllib.parse
import asyncio
import subprocess
import tempfile
import os
import re
import io
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Literal
from pathlib import PurePosixPath

import pymupdf
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response

from api.auth import verify_api_key
from api.config import settings

app = FastAPI(
    title="PDF to Text API",
    description="Extract text from PDF and other documents using PyMuPDF (with OCR support)",
    version="1.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

_pool = ThreadPoolExecutor(max_workers=1)

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


# ── Redaction ────────────────────────────────────────────────────────────────
#
# True PDF redaction using PyMuPDF's add_redact_annot + apply_redactions.
# Text is permanently removed from the content stream, not just visually
# covered. For image-only pages, OCR via Tesseract locates text positions,
# then black rectangles are drawn directly on the rendered pixmap.
#
# Strategies (applied per page in order):
#   1. Image-only pages → OCR-based pixel redaction
#   2. search_for()     → direct text-layer match (handles simple spans)
#   3. Word-level       → get_text("words") grouped by line (handles fragmented spans)
#   4. Link annotations → mailto:, tel:, social URLs in PDF link objects
#   5. QR code images   → small square images in the contact area
# ─────────────────────────────────────────────────────────────────────────────

# -- Patterns ------------------------------------------------------------------

_REDACT_PATTERNS: dict[str, re.Pattern] = {
    # Email
    "email": re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", re.I,
    ),
    "email_fuzzy": re.compile(
        r"[A-Za-z0-9._%+\-]+[\s(]*@[\s)]*[A-Za-z0-9.\-\u00C0-\u024F]+"
        r"(?:\.(?:com|net|org|vn|io|co|edu|gov)\b"
        r"|gm[a\u00E0\u00E1\u1EA3\u00E3\u1EA1]il\.?\w*"
        r"|yahoo\.?\w*|outlook\.?\w*|hotmail\.?\w*)",
        re.I,
    ),
    "email_at_line": re.compile(r"\S+@\S+", re.I),
    # Phone – Vietnamese
    "phone": re.compile(r"(?<!\d)(?:\+84|0084|84|0)[235789]\d{8}(?!\d)"),
    "phone_spaced": re.compile(r"(?<!\d)0[235789]\d{1,2}[\s.\-]\d{3,4}[\s.\-]\d{3,4}(?!\d)"),
    "phone_vn_paren": re.compile(r"\(\+?84\)[\s.\-]?\d[\d\s.\-]{7,12}\d(?!\d)"),
    "phone_dot": re.compile(r"(?<!\d)0[235789]\d{1,2}\.\d{3}\.\d{3,4}(?!\d)"),
    # Phone – international
    "phone_intl": re.compile(r"(?<!\d)\+\d{1,3}[\s.\-]?\(?\d{2,4}\)?[\s.\-]?\d{3,4}[\s.\-]?\d{3,4}(?!\d)"),
    # Social / professional URLs
    "linkedin": re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/(?:in|pub|company)/[\w\-%]+/?", re.I),
    "facebook": re.compile(r"(?:https?://)?(?:www\.)?(?:fb|facebook)\.com/[\w.\-/]+", re.I),
    "github": re.compile(r"(?:https?://)?(?:www\.)?github\.com/[\w\-]+(?:/[\w\-.]+)*/?", re.I),
    "twitter": re.compile(r"(?:https?://)?(?:www\.)?(?:twitter|x)\.com/\w{1,50}/?", re.I),
    "instagram": re.compile(r"(?:https?://)?(?:www\.)?instagram\.com/[\w.\-]+/?", re.I),
    "telegram": re.compile(r"(?:https?://)?t\.me/\w{3,50}/?", re.I),
    "zalo": re.compile(r"zalo[:\s./]?\s*[\d\s\-]{9,15}", re.I),
    "behance": re.compile(r"(?:https?://)?(?:www\.)?behance\.net/[\w\-]+/?", re.I),
    "url_personal": re.compile(
        r"(?:https?://)?(?:www\.)?(?:about\.me|portfolio|dribbble\.com|medium\.com)/[\w\-]+/?", re.I,
    ),
    # Portfolio-style subdomain (e.g. portfolio-shindo806.vercel.app, my-portfolio.netlify.app)
    "portfolio_subdomain": re.compile(
        r"(?:https?://)?(?:www\.)?portfolio[\w\-]*\.[\w\-.]+(?:/\S*)?", re.I,
    ),
    # Free hosting / no-code portfolio domains
    "portfolio_hosting": re.compile(
        r"(?:https?://)?(?:[\w\-]+\.)+(?:vercel\.app|netlify\.app|github\.io|my\.canva\.site|canva\.site|notion\.site|webflow\.io|pages\.dev|framer\.website|glitch\.me|replit\.app|surge\.sh)(?:/\S*)?",
        re.I,
    ),
    # Canva design links
    "canva_design": re.compile(
        r"(?:https?://)?(?:www\.)?canva\.com/design/[\w\-/]+", re.I,
    ),
}

_TARGET_GROUPS: dict[str, list[str]] = {
    "email": ["email", "email_fuzzy", "email_at_line"],
    "phone": ["phone", "phone_spaced", "phone_vn_paren", "phone_intl", "phone_dot"],
    "linkedin": ["linkedin"],
    "social": ["facebook", "github", "twitter", "instagram", "telegram", "zalo", "behance", "url_personal", "portfolio_subdomain", "portfolio_hosting", "canva_design"],
    "all": list(_REDACT_PATTERNS.keys()),
}

# -- Helpers -------------------------------------------------------------------

_RECT_PAD = pymupdf.Rect(-2, -1, 2, 1)
_RECT_PAD_IMG = 4
_QR_MAX_SIZE = 200
_QR_MIN_SIZE = 15
_QR_ASPECT_THRESHOLD = 0.75
_OCR_REDACT_DPI = 150


def _resolve_targets(targets: list[str]) -> dict[str, re.Pattern]:
    """Map user-facing target names (email, phone, …) to compiled patterns."""
    keys: set[str] = set()
    for t in targets:
        name = t.strip().lower()
        if name in _TARGET_GROUPS:
            keys.update(_TARGET_GROUPS[name])
        elif name in _REDACT_PATTERNS:
            keys.add(name)
    return {k: _REDACT_PATTERNS[k] for k in keys}


def _rect_key(rect: pymupdf.Rect) -> tuple[int, int, int, int]:
    """Rounded rect coordinates as hashable key for dedup."""
    return (round(rect.x0), round(rect.y0), round(rect.x1), round(rect.y1))


# -- Core redaction ------------------------------------------------------------


class _PageRedactor:
    """Handles redaction for a single PDF page."""

    def __init__(self, page: pymupdf.Page, page_num: int,
                 patterns: dict[str, re.Pattern], fill: tuple,
                 seen: set[tuple]):
        self.page = page
        self.page_num = page_num
        self.patterns = patterns
        self.fill = fill
        self.seen = seen
        self.stats: dict[str, int] = {}
        self.count = 0

    def _mark(self, rect: pymupdf.Rect, label: str, pad: pymupdf.Rect = _RECT_PAD) -> bool:
        """Add a redaction annotation if not already seen. Returns True if added."""
        rk = _rect_key(rect)
        if (self.page_num, label, rk) in self.seen:
            return False
        self.seen.add((self.page_num, label, rk))
        self.page.add_redact_annot(rect + pad, fill=self.fill)
        self.count += 1
        self.stats[label] = self.stats.get(label, 0) + 1
        return True

    # ── Strategy 1: search_for() on full regex matches ───────────────────────

    def redact_via_search(self, page_text: str) -> None:
        for label, pattern in self.patterns.items():
            for match in pattern.finditer(page_text):
                hit = match.group(0).strip()
                if len(hit) < 4:
                    continue
                for rect in self.page.search_for(hit):
                    self._mark(rect, label)

    # ── Strategy 2: word-level fallback for fragmented text ──────────────────

    def redact_via_words(self) -> None:
        words = self.page.get_text("words")
        if not words:
            return
        # Group words into lines by (block_no, line_no)
        lines: dict[tuple, list] = defaultdict(list)
        for w in words:
            lines[(w[5], w[6])].append(w)

        for key in sorted(lines):
            line_words = sorted(lines[key], key=lambda w: w[0])
            line_text = " ".join(w[4] for w in line_words)

            for label, pattern in self.patterns.items():
                for match in pattern.finditer(line_text):
                    m_start, m_end = match.start(), match.end()
                    pos = 0
                    for w in line_words:
                        w_start, w_end = pos, pos + len(w[4])
                        if w_end > m_start and w_start < m_end:
                            self._mark(pymupdf.Rect(w[0], w[1], w[2], w[3]), label)
                        pos = w_end + 1  # +1 for the join space

    # ── Strategy 3: link annotations ─────────────────────────────────────────

    def redact_links(self) -> None:
        link = self.page.first_link
        while link:
            uri = link.uri or ""
            matched_label = self._match_uri(uri)
            if matched_label and link.rect.is_valid and not link.rect.is_empty:
                self._mark(link.rect, matched_label, pad=pymupdf.Rect(0, 0, 0, 0))
            link = link.next

    def _match_uri(self, uri: str) -> str | None:
        if not uri:
            return None
        for label, pattern in self.patterns.items():
            if pattern.search(uri):
                return label
        if uri.startswith("mailto:"):
            return "email"
        if uri.startswith("tel:"):
            return "phone"
        return None

    # ── Strategy 4: QR code images ───────────────────────────────────────────

    def redact_qr_images(self) -> None:
        try:
            images = self.page.get_images(full=True)
        except Exception:
            return

        page_height = self.page.rect.height
        for img_info in images:
            xref = img_info[0]
            try:
                rects = self.page.get_image_rects(xref)
            except Exception:
                continue
            for rect in rects:
                if not self._looks_like_qr(rect, page_height):
                    continue
                # Only redact if pyzbar actually decodes a QR payload.
                # Previously a fallback heuristic (small square in top 40%)
                # also marked as QR, but it matched candidate avatar/photo
                # on CVs and wiped the portrait. No heuristic fallback:
                # if pyzbar can't decode, treat it as a normal image.
                self._try_decode_qr(xref, rect)

    def _looks_like_qr(self, rect: pymupdf.Rect, page_height: float) -> bool:
        if rect.is_empty or not rect.is_valid:
            return False
        w, h = rect.width, rect.height
        if w < _QR_MIN_SIZE or h < _QR_MIN_SIZE or max(w, h) > _QR_MAX_SIZE:
            return False
        return min(w, h) / max(w, h) >= _QR_ASPECT_THRESHOLD

    def _try_decode_qr(self, xref: int, rect: pymupdf.Rect) -> bool:
        """Try pyzbar decode. Returns True if QR was found and redacted."""
        try:
            from pyzbar.pyzbar import decode as qr_decode
            from PIL import Image
            pix = pymupdf.Pixmap(self.page.parent, xref)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            if qr_decode(img):
                self._mark(rect, "qr_code", pad=pymupdf.Rect(-3, -3, 3, 3))
                return True
        except ImportError:
            pass
        except Exception:
            pass
        return False

    # ── Strategy 5: OCR-based redaction for image-only pages ─────────────────

    def redact_image_page(self) -> bool:
        """OCR the page, find contact info, draw black rects on the pixmap,
        then replace the page content with the redacted image.
        Returns True if any redactions were applied."""
        # Cap DPI so pixmap stays under ~8MP to avoid OOM with large scans.
        page_rect = self.page.rect
        max_pixels = 8_000_000
        natural_w = page_rect.width * _OCR_REDACT_DPI / 72
        natural_h = page_rect.height * _OCR_REDACT_DPI / 72
        if natural_w * natural_h > max_pixels:
            scale = (max_pixels / (natural_w * natural_h)) ** 0.5
            dpi = max(72, int(_OCR_REDACT_DPI * scale))
        else:
            dpi = _OCR_REDACT_DPI
        pix = self.page.get_pixmap(dpi=dpi)

        ocr_words = self._ocr_to_words(pix.tobytes("png"))
        if not ocr_words:
            return False

        rects = self._match_ocr_words(ocr_words)
        if not rects:
            return False

        # Draw black/white blocks directly on the pixmap
        fill_val = (0, 0, 0) if self.fill == (0, 0, 0) else (255, 255, 255)
        for x0, y0, x1, y1, label in rects:
            irect = pymupdf.IRect(
                max(0, x0 - _RECT_PAD_IMG),
                max(0, y0 - _RECT_PAD_IMG),
                min(pix.width, x1 + _RECT_PAD_IMG),
                min(pix.height, y1 + _RECT_PAD_IMG),
            )
            pix.set_rect(irect, fill_val)
            self.count += 1
            self.stats[label] = self.stats.get(label, 0) + 1

        # Replace page content: draw a white background then overlay the redacted pixmap.
        # Avoids update_stream(xref, b"") which can corrupt PDF docs converted from images.
        page_rect = self.page.rect
        shape = self.page.new_shape()
        shape.draw_rect(page_rect)
        shape.finish(fill=(1, 1, 1), color=None, width=0)
        shape.commit()
        self.page.insert_image(page_rect, stream=pix.tobytes("png"), overlay=True)
        return True

    def _ocr_to_words(self, img_bytes: bytes) -> list[dict]:
        """Run Tesseract TSV and parse into word dicts with bounding boxes."""
        fd, img_path = tempfile.mkstemp(suffix=".png")
        tsv_base = img_path.replace(".png", "_ocr")
        tsv_file = tsv_base + ".tsv"
        try:
            os.write(fd, img_bytes)
            os.close(fd)
            result = subprocess.run(
                ["tesseract", img_path, tsv_base, "-l", OCR_LANGUAGES, "--psm", "3", "tsv"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0 or not os.path.exists(tsv_file):
                return []

            words = []
            with open(tsv_file) as f:
                f.readline()  # skip header
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 12:
                        continue
                    text = parts[11].strip()
                    try:
                        conf = float(parts[10])
                    except (ValueError, IndexError):
                        continue
                    if not text or conf < 0:
                        continue
                    try:
                        words.append({
                            "text": text,
                            "left": int(parts[6]), "top": int(parts[7]),
                            "width": int(parts[8]), "height": int(parts[9]),
                            "line": int(parts[4]), "block": int(parts[2]),
                        })
                    except (ValueError, IndexError):
                        continue
            return words
        except Exception:
            return []
        finally:
            for p in (img_path, tsv_file):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    def _match_ocr_words(self, words: list[dict]) -> list[tuple]:
        """Group OCR words into lines, match patterns, return redact rects."""
        lines: dict[tuple, list] = defaultdict(list)
        for w in words:
            lines[(w["block"], w["line"])].append(w)

        rects = []
        for key in sorted(lines):
            line_words = sorted(lines[key], key=lambda w: w["left"])
            line_text = " ".join(w["text"] for w in line_words)

            for label, pattern in self.patterns.items():
                for match in pattern.finditer(line_text):
                    m_start, m_end = match.start(), match.end()
                    pos = 0
                    for w in line_words:
                        w_start, w_end = pos, pos + len(w["text"])
                        if w_end > m_start and w_start < m_end:
                            rects.append((
                                w["left"], w["top"],
                                w["left"] + w["width"], w["top"] + w["height"],
                                label,
                            ))
                        pos = w_end + 1
        return rects

    # ── Apply ────────────────────────────────────────────────────────────────

    def apply(self) -> None:
        """Commit all pending redaction annotations on this page.
        images=PDF_REDACT_IMAGE_NONE to preserve personal photos/avatars."""
        self.page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE, graphics=True)

    # ── Strategy 6: OCR text-in-image spans (e.g. email rendered as image) ───

    def redact_text_images(self) -> None:
        """Find small images in the contact/header area that contain text
        (email, phone, etc.) rendered as images instead of text layer.
        OCR each candidate image and redact if PII is found."""
        try:
            images = self.page.get_images(full=True)
        except Exception:
            return

        page_height = self.page.rect.height
        for img_info in images:
            xref = img_info[0]
            try:
                rects = self.page.get_image_rects(xref)
            except Exception:
                continue
            for rect in rects:
                if not rect.is_valid or rect.is_empty:
                    continue
                w, h = rect.width, rect.height
                # Skip large images (photos/backgrounds) and tiny icons
                if w < 30 or h < 8 or h > 60 or w > 500:
                    continue
                # Only top 30% of page (contact/header area)
                if rect.y0 > page_height * 0.3:
                    continue
                # Aspect ratio: text images are wide, not square (skip QR/icons)
                if w / h < 2.0:
                    continue
                # OCR this image
                try:
                    pix = pymupdf.Pixmap(self.page.parent, xref)
                    if pix.width < 20 or pix.height < 8:
                        continue
                    img_bytes = pix.tobytes("png")
                    ocr_text = _ocr_page_subprocess(img_bytes, OCR_LANGUAGES)
                    if not ocr_text.strip():
                        continue
                    # Check if OCR text matches any pattern
                    for label, pattern in self.patterns.items():
                        if pattern.search(ocr_text):
                            self._mark(rect, label, pad=pymupdf.Rect(-2, -2, 2, 2))
                            break
                except Exception:
                    continue


# -- Document-level orchestration ----------------------------------------------


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}


_MAGIC_BYTES: list[tuple[bytes, str]] = [
    (b"\x89PNG", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"GIF8", ".gif"),
    (b"BM", ".bmp"),
    (b"II\x2a\x00", ".tiff"),
    (b"MM\x00\x2a", ".tiff"),
    (b"%PDF", ".pdf"),
]


def _detect_ext(file_bytes: bytes) -> str | None:
    for magic, ext in _MAGIC_BYTES:
        if file_bytes[:len(magic)] == magic:
            return ext
    return None


def _redact_document(file_bytes: bytes, filename: str,
                     targets: list[str], fill_color_name: str) -> tuple[bytes, dict]:
    """Redact contact information from a PDF document. Runs in thread pool."""
    ext = _ext_from_filename(filename)
    # Override extension when magic bytes disagree — prevents sending PNG with .pdf filename
    detected = _detect_ext(file_bytes)
    if detected and detected != ext:
        ext = detected
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext or 'unknown'}")

    fill = (0, 0, 0) if fill_color_name == "black" else (1, 1, 1)
    patterns = _resolve_targets(targets)
    if not patterns:
        raise ValueError(f"No valid redaction targets. Available: {list(_TARGET_GROUPS.keys())}")

    # Images must be converted to PDF first — opening as image doc causes segfault
    # in clean_contents() when redact_image_page tries to replace page content.
    if ext in _IMAGE_EXTENSIONS:
        try:
            img_doc = pymupdf.open(stream=file_bytes, filetype=ext.lstrip("."))
            pdf_bytes = img_doc.convert_to_pdf()
            img_doc.close()
            file_bytes = pdf_bytes
            filename = filename.rsplit(".", 1)[0] + ".pdf"
            ext = ".pdf"
        except Exception as exc:
            raise ValueError(f"Failed to convert image to PDF: {exc}")

    try:
        doc = pymupdf.open(stream=file_bytes, filetype=ext.lstrip("."))
    except Exception as exc:
        raise ValueError(f"Failed to open document: {exc}")

    total_stats: dict[str, int] = {}
    total_count = 0
    seen: set[tuple] = set()

    try:
        for page_num, page in enumerate(doc):
            r = _PageRedactor(page, page_num, patterns, fill, seen)
            page_text = page.get_text("text")

            if len(page_text.strip()) < TEXT_THRESHOLD:
                # Image-only page → OCR path
                r.redact_image_page()
            else:
                # Text-based page → layered strategies
                r.redact_via_search(page_text)
                r.redact_via_words()
                r.redact_links()
                r.redact_qr_images()
                r.redact_text_images()
                r.apply()

            total_count += r.count
            for k, v in r.stats.items():
                total_stats[k] = total_stats.get(k, 0) + v

        buf = io.BytesIO()
        doc.save(buf, deflate=True, garbage=2)
        return buf.getvalue(), {"total_redactions": total_count, "details": total_stats}
    finally:
        doc.close()


def _verify_redaction(redacted_bytes: bytes, patterns: dict[str, re.Pattern]) -> list[str]:
    """Re-extract text from the redacted PDF and check for remaining PII."""
    warnings = []
    try:
        doc = pymupdf.open(stream=redacted_bytes, filetype="pdf")
        full_text = "\n".join(page.get_text("text") for page in doc)
        doc.close()
        for label, pattern in patterns.items():
            n = len(pattern.findall(full_text))
            if n:
                warnings.append(f"{label}: {n} match(es) still found")
    except Exception:
        pass
    return warnings


# -- Endpoint ------------------------------------------------------------------


@app.post("/redact")
async def redact(
    file: UploadFile = File(None),
    url: str = Form(None),
    targets: str = Form(
        "email,phone,linkedin,social",
        description="Comma-separated: email, phone, linkedin, social, all",
    ),
    fill_color: str = Form("black", description="Redaction fill: 'black' or 'white'"),
    verify: bool = Form(True, description="Verify no PII remains after redaction"),
    _key: str = Depends(verify_api_key),
):
    """Redact contact information from a PDF document.

    Permanently removes email, phone, LinkedIn, and social-media URLs from
    both the text layer and image layer (for scanned/image-only PDFs).

    Returns the redacted PDF as a binary download.
    """
    content, filename = await _resolve_input(file, url)
    target_list = [t.strip() for t in targets.split(",") if t.strip()]

    loop = asyncio.get_event_loop()
    try:
        redacted_bytes, stats = await loop.run_in_executor(
            _pool, _redact_document, content, filename, target_list, fill_color,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    warnings: list[str] = []
    if verify and stats["total_redactions"] > 0:
        active = _resolve_targets(target_list)
        warnings = await loop.run_in_executor(_pool, _verify_redaction, redacted_bytes, active)

    return Response(
        content=redacted_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="redacted_{filename}"',
            "X-Redaction-Count": str(stats["total_redactions"]),
            "X-Redaction-Details": str(stats["details"]),
            "X-Redaction-Warnings": str(warnings) if warnings else "none",
        },
    )
