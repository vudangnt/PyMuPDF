"""force_ocr cho /extract: CV thiet ke (chu ve thanh anh/outline) chi co lop chu mong — vd 47 ky tu
"Photography Cat, cat and cat" — vuot nguong TEXT_THRESHOLD=20 nen khong bao gio duoc OCR.
Client (Laravel) goi lai voi force_ocr=true khi text qua ngan.
"""
import pymupdf

import api.main as m


def _thin_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Photography Cat, cat and cat")  # 28 ky tu > nguong 20
    data = doc.tobytes()
    doc.close()
    return data


OCR_TEXT = "Nguyen Van A\nSenior Designer\n" + "Kinh nghiem 5 nam Figma, Photoshop. " * 5


def test_default_keeps_thin_text_layer(monkeypatch):
    called = []
    monkeypatch.setattr(m, "_ocr_page_subprocess", lambda img, lang: called.append(1) or OCR_TEXT)
    out = m._process_document(_thin_pdf(), "cv.pdf", "text", m.OCR_LANGUAGES)
    assert "Photography" in out["text"]
    assert called == []
    assert out["ocr_pages"] == 0


def test_force_ocr_uses_longer_ocr_text(monkeypatch):
    monkeypatch.setattr(m, "_ocr_page_subprocess", lambda img, lang: OCR_TEXT)
    out = m._process_document(_thin_pdf(), "cv.pdf", "text", m.OCR_LANGUAGES, None, True)
    assert "Senior Designer" in out["text"]
    assert out["ocr_pages"] == 1


def test_force_ocr_keeps_text_layer_when_ocr_is_worse(monkeypatch):
    monkeypatch.setattr(m, "_ocr_page_subprocess", lambda img, lang: "x")
    out = m._process_document(_thin_pdf(), "cv.pdf", "text", m.OCR_LANGUAGES, None, True)
    assert "Photography" in out["text"]
    assert out["ocr_pages"] == 0


def test_force_ocr_ignored_when_ocr_off(monkeypatch):
    called = []
    monkeypatch.setattr(m, "_ocr_page_subprocess", lambda img, lang: called.append(1) or OCR_TEXT)
    out = m._process_document(_thin_pdf(), "cv.pdf", "text", "off", None, True)
    assert "Photography" in out["text"]
    assert called == []
