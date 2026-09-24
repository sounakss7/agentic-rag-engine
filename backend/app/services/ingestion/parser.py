"""
NexusRAG Layout-Aware Document Ingestion & Multimodal Vision Parser.
Extracts clean text, preserves markdown tables, and handles scanned documents via Vision OCR.
"""

import io
import re
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from pdf2image import convert_from_bytes
except ImportError:
    convert_from_bytes = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

from backend.app.core.config import settings, register_degraded
from backend.app.core.logging import logger

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


class LayoutAwareParser:
    """
    Parses unstructured and structured documents, extracting section headers,
    tables formatted as clean Markdown, and scanned text via Multimodal Vision OCR.
    """

    def __init__(self):
        self._genai_client = None

    def _get_genai_client(self):
        if self._genai_client is None and genai is not None and settings.active_gemini_key:
            self._genai_client = genai.Client(api_key=settings.active_gemini_key)
        return self._genai_client

    def parse_document(self, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Parses document into a list of structured page/section records:
        [
            {
                "page": int,
                "text": str,
                "tables": List[str],  # Markdown representations of tables
                "ocr_used": bool,
                "source": str
            }, ...
        ]
        """
        ext = filename.lower().split(".")[-1]

        if ext == "pdf":
            return self._parse_pdf(file_bytes, filename)
        elif ext in ["png", "jpg", "jpeg", "webp", "bmp", "tiff"]:
            return self._parse_image(file_bytes, filename)
        elif ext in ["csv", "tsv"]:
            return self._parse_tabular_text(file_bytes, filename, is_csv=True)
        elif ext in ["txt", "md", "json", "yaml", "xml", "log"]:
            return self._parse_text(file_bytes, filename)
        else:
            return self._parse_text(file_bytes, filename)

    def _parse_pdf(self, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """Extracts text and structured tables from PDF, with scanned OCR fallback."""
        pages_data = []
        needs_ocr_pages = []

        # 1. Primary extraction with pdfplumber (table & layout aware)
        if pdfplumber is not None:
            try:
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    for i, page in enumerate(pdf.pages):
                        page_num = i + 1
                        page_tables = []
                        
                        # Extract structured tables and format as Markdown
                        tables = page.extract_tables() or []
                        for tbl in tables:
                            md_table = self._table_to_markdown(tbl)
                            if md_table:
                                page_tables.append(md_table)

                        raw_text = page.extract_text(layout=False) or ""
                        
                        # Check if page is an image/scanned (fewer than 40 chars of text)
                        if len(raw_text.strip()) < 40 and not page_tables:
                            needs_ocr_pages.append(page_num)
                        else:
                            full_page_content = raw_text.strip()
                            if page_tables:
                                full_page_content += "\n\n" + "\n\n".join(page_tables)

                            pages_data.append({
                                "page": page_num,
                                "text": full_page_content,
                                "tables": page_tables,
                                "ocr_used": False,
                                "source": filename
                            })
            except Exception as e:
                logger.warning(f"pdfplumber extraction error on {filename}: {e}")

        # Fallback to pypdf if pdfplumber failed or is unavailable
        if not pages_data and pypdf is not None:
            try:
                reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                for i, page in enumerate(reader.pages):
                    page_num = i + 1
                    t = page.extract_text() or ""
                    if len(t.strip()) < 40:
                        needs_ocr_pages.append(page_num)
                    else:
                        pages_data.append({
                            "page": page_num,
                            "text": t.strip(),
                            "tables": [],
                            "ocr_used": False,
                            "source": filename
                        })
            except Exception as e:
                logger.warning(f"pypdf extraction error on {filename}: {e}")

        # 2. Vision OCR for Scanned Pages
        if needs_ocr_pages:
            resolved_ocr = self._ocr_scanned_pdf(file_bytes, needs_ocr_pages, filename)
            pages_data.extend(resolved_ocr)

        # Ensure pages are sorted
        pages_data.sort(key=lambda x: x["page"])
        if not pages_data:
            pages_data.append({
                "page": 1,
                "text": f"[Empty or unreadable document: {filename}]",
                "tables": [],
                "ocr_used": False,
                "source": filename
            })

        return pages_data

    def _ocr_scanned_pdf(self, file_bytes: bytes, pages: List[int], filename: str) -> List[Dict[str, Any]]:
        """Multi-tier OCR fallback: pdf2image+tesseract or direct Gemini Multimodal Vision."""
        ocr_results = []
        unresolved = set(pages)

        # Tier 1: Local Tesseract OCR
        if convert_from_bytes is not None and pytesseract is not None:
            try:
                images = convert_from_bytes(file_bytes)
                for p_num in list(unresolved):
                    idx = p_num - 1
                    if idx < len(images):
                        text = pytesseract.image_to_string(images[idx]).strip()
                        if text:
                            ocr_results.append({
                                "page": p_num,
                                "text": text,
                                "tables": [],
                                "ocr_used": True,
                                "source": filename
                            })
                            unresolved.remove(p_num)
            except Exception as e:
                logger.info(f"Local Tesseract OCR skipped: {e}")

        # Tier 2: Gemini Multimodal Vision OCR
        if unresolved:
            client = self._get_genai_client()
            if client:
                try:
                    prompt = (
                        "Extract all text, headers, and tabular data from this document accurately. "
                        "Format any tables cleanly as Markdown tables (| Col 1 | Col 2 |). "
                        "Preserve exact numbers, dates, and currency symbols."
                    )
                    res = client.models.generate_content(
                        model=settings.PRIMARY_LLM,
                        contents=[
                            types.Part.from_bytes(data=file_bytes, mime_type="application/pdf"),
                            prompt
                        ]
                    )
                    if res and res.text:
                        extracted = res.text.strip()
                        for p_num in unresolved:
                            ocr_results.append({
                                "page": p_num,
                                "text": extracted,
                                "tables": [],
                                "ocr_used": True,
                                "source": filename
                            })
                        unresolved.clear()
                except Exception as e:
                    logger.warning(f"Gemini Multimodal Vision PDF OCR error: {e}")

        return ocr_results

    def _parse_image(self, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """Parses standalone image, handling RGBA flattening and vision OCR."""
        text = ""
        ocr_used = True

        try:
            img = Image.open(io.BytesIO(file_bytes))
            # Flatten transparency to white RGB
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                bg = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                bg.paste(img, mask=img.split()[-1] if len(img.split()) == 4 else None)
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # Local OCR
            if pytesseract is not None:
                try:
                    text = pytesseract.image_to_string(img).strip()
                except Exception:
                    pass

            # Gemini Vision OCR fallback
            if not text:
                client = self._get_genai_client()
                if client:
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=95)
                    res = client.models.generate_content(
                        model=settings.PRIMARY_LLM,
                        contents=[
                            types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"),
                            "Thoroughly extract all text, data, and tables from this image as clean markdown."
                        ]
                    )
                    if res and res.text:
                        text = res.text.strip()
        except Exception as e:
            text = f"[Image OCR parse failed: {e}]"

        return [{
            "page": 1,
            "text": text or "[No text detected in image]",
            "tables": [],
            "ocr_used": ocr_used,
            "source": filename
        }]

    def _parse_tabular_text(self, file_bytes: bytes, filename: str, is_csv: bool = True) -> List[Dict[str, Any]]:
        """Parses CSV/TSV into clean Markdown tables."""
        try:
            content = file_bytes.decode("utf-8", errors="replace")
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            if not lines:
                return [{"page": 1, "text": "", "tables": [], "ocr_used": False, "source": filename}]

            sep = "," if is_csv else "\t"
            rows = [line.split(sep) for line in lines[:200]]  # limit preview lines for table
            md_table = self._table_to_markdown(rows)
            return [{
                "page": 1,
                "text": f"Structured Tabular Dataset ({filename}):\n\n{md_table}",
                "tables": [md_table],
                "ocr_used": False,
                "source": filename
            }]
        except Exception as e:
            return self._parse_text(file_bytes, filename)

    def _parse_text(self, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """Parses UTF-8 or Latin-1 text files."""
        try:
            content = file_bytes.decode("utf-8", errors="replace")
        except Exception:
            content = file_bytes.decode("latin-1", errors="replace")

        return [{
            "page": 1,
            "text": content,
            "tables": [],
            "ocr_used": False,
            "source": filename
        }]

    @staticmethod
    def _table_to_markdown(rows: List[List[Any]]) -> str:
        """Converts raw 2D list of cells into clean GitHub-flavored Markdown table."""
        if not rows or len(rows) < 1:
            return ""

        clean_rows = []
        max_cols = max(len(r) for r in rows)
        if max_cols == 0:
            return ""

        for r in rows:
            cells = [str(c or "").replace("\n", " ").replace("|", "\\|").strip() for c in r]
            while len(cells) < max_cols:
                cells.append("")
            clean_rows.append(cells)

        header = "| " + " | ".join(clean_rows[0]) + " |"
        sep = "| " + " | ".join(["---"] * max_cols) + " |"
        body = "\n".join(["| " + " | ".join(row) + " |" for row in clean_rows[1:]])

        return f"{header}\n{sep}\n{body}" if body else f"{header}\n{sep}"
