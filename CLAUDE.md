# CLAUDE.md — extract-text

## Project Overview
Python FastAPI service for PDF text extraction and contact info redaction. Used by recruiter.digisource.vn (via digi-lib `PdfToText` / `PdfRedact` APIs).

## Tech Stack
- **Language:** Python 3.10+
- **Framework:** FastAPI + Uvicorn
- **PDF:** PyMuPDF (pymupdf) for text extraction + redaction
- **OCR:** pyzbar (QR), Tesseract (image-only pages)
- **Deploy:** Supervisor on production server

## Key Files
| File | Purpose |
|------|---------|
| `api/main.py` | Core API: `/extract` (text extraction), `/redact` (contact redaction) |
| `api/auth.py` | API key authentication middleware |
| `api/config.py` | Pydantic settings (env-based) |
| `deploy.sh` | Production deployment script |
| `supervisor/` | Supervisor config for process management |

## Redaction System (`api/main.py`)
5 redaction strategies run in order per page:
1. **search** — regex match on extracted text → `add_redact_annot`
2. **words** — per-word bounding box scan for partial matches
3. **links** — strip hyperlinks (mailto, tel, linkedin, social)
4. **QR codes** — detect via pyzbar → black rect overlay
5. **OCR (image-only pages)** — Tesseract → regex on OCR text → pixel overlay

### Pattern Groups (`_TARGET_GROUPS`)
- `email` — email, email_fuzzy, email_at_line
- `phone` — phone, phone_spaced, phone_vn_paren, phone_intl, phone_dot, phone_84_spaced
- `linkedin` — linkedin URLs
- `social` — facebook, github, twitter, instagram, telegram, zalo, behance, portfolio URLs
- `all` — all patterns

### Phone Patterns
| Pattern | Example |
|---------|---------|
| `phone` | `0935887255`, `+84935887255` |
| `phone_spaced` | `093 588 7255` |
| `phone_vn_paren` | `(+84) 935 887 255` |
| `phone_84_spaced` | `+84 3558 72 558` |
| `phone_dot` | `093.588.7255` |
| `phone_intl` | `+1 415 555 2671` |

## API Endpoints
- `POST /extract` — extract text from PDF/DOC/image, returns `{text, pages, char_count}`
- `POST /redact` — redact contact info, returns redacted PDF bytes + metadata
- Auth: `X-API-Key` header

## Local Development
```bash
pip install -r requirements.txt  # or: pip install pymupdf fastapi pydantic-settings pyzbar
uvicorn api.main:app --reload --port 8002
```

## Production
- Server: same as recruiter (103.133.224.137)
- Deploy: `bash deploy.sh`
- Supervisor manages the process

## Common Tasks
- **Add new phone format:** Edit `_REDACT_PATTERNS` dict in `api/main.py`, add to `_TARGET_GROUPS["phone"]`
- **Add new social URL:** Add regex to `_REDACT_PATTERNS`, add to `_TARGET_GROUPS["social"]`
- **Test redaction locally:** `python3.11 -c "from api.main import _redact_document; ..."`
