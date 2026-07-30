import io
import uuid
import logging
from typing import List, Dict, Any, Tuple
from PIL import Image

# Import document splitters with fallback
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=800, chunk_overlap=120, **kwargs):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_text(self, text: str) -> List[str]:
            if not text:
                return []
            chunks = []
            start = 0
            text_len = len(text)
            while start < text_len:
                end = min(start + self.chunk_size, text_len)
                chunks.append(text[start:end])
                start += self.chunk_size - self.chunk_overlap
                if start >= end:
                    break
            return chunks


from config import register_degraded_component

# Setup Logging
logger = logging.getLogger("OCRParser")
logger.setLevel(logging.INFO)

# Optional Imports with Graceful Fallbacks
try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import pytesseract
except ImportError:
    register_degraded_component("PyTesseract (stub)")
    pytesseract = None

try:
    from pdf2image import convert_from_bytes
except ImportError:
    register_degraded_component("pdf2image (stub)")
    convert_from_bytes = None



class DocumentIngestor:
    """
    Unified document parsing and OCR ingestion pipeline.
    Supports PDF (native text & scanned OCR), images (PNG, JPG, JPEG), and plain text (TXT, MD).
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_file(self, file_bytes: bytes, filename: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Parses a file and splits it into structured chunks with rich metadata.
        Returns:
            Tuple of (list_of_chunks, processing_summary)
        """
        ext = filename.lower().split(".")[-1]
        pages_data = []

        if ext == "pdf":
            pages_data = self._parse_pdf(file_bytes, filename)
        elif ext in ["png", "jpg", "jpeg", "bmp", "tiff", "webp"]:
            pages_data = self._parse_image(file_bytes, filename)
        elif ext in ["txt", "md", "csv", "json"]:
            pages_data = self._parse_text(file_bytes, filename)
        else:
            # Fallback text decoding
            pages_data = self._parse_text(file_bytes, filename)

        chunks = self._chunk_pages(pages_data, filename)
        
        summary = {
            "filename": filename,
            "total_pages": len(pages_data),
            "total_chunks": len(chunks),
            "ocr_pages_count": sum(1 for p in pages_data if p.get("ocr_used", False))
        }

        return chunks, summary

    def _parse_pdf(self, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """Extracts text from PDF, falling back to Tesseract OCR for scanned pages."""
        pages_data = []
        
        # 1. Primary Attempt: pdfplumber or pypdf
        text_by_page = []
        
        if pdfplumber is not None:
            try:
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    for i, page in enumerate(pdf.pages):
                        text = page.extract_text() or ""
                        text_by_page.append((i + 1, text.strip()))
            except Exception as e:
                logger.warning(f"pdfplumber extraction failed for {filename}: {e}")

        if not text_by_page and pypdf is not None:
            try:
                reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    text_by_page.append((i + 1, text.strip()))
            except Exception as e:
                logger.warning(f"pypdf extraction failed for {filename}: {e}")

        # Check if native extraction yielded sufficient text per page
        needs_ocr_pages = []
        if text_by_page:
            for page_num, text in text_by_page:
                # If page has fewer than 30 characters, treat it as scanned/image page
                if len(text) < 30:
                    needs_ocr_pages.append(page_num)
                else:
                    pages_data.append({
                        "page": page_num,
                        "text": text,
                        "ocr_used": False
                    })
        else:
            needs_ocr_pages = [1]  # Fallback to OCR if native failed entirely

        # 2. OCR Fallback for scanned pages using pdf2image + pytesseract
        if needs_ocr_pages and convert_from_bytes is not None and pytesseract is not None:
            try:
                images = convert_from_bytes(file_bytes)
                for page_num in needs_ocr_pages:
                    idx = page_num - 1
                    if idx < len(images):
                        ocr_text = self._ocr_image(images[idx])
                        if ocr_text.strip():
                            pages_data.append({
                                "page": page_num,
                                "text": ocr_text.strip(),
                                "ocr_used": True
                            })
                        else:
                            # Retain existing minimal text if OCR returned empty
                            existing_text = next((t for p, t in text_by_page if p == page_num), "")
                            pages_data.append({
                                "page": page_num,
                                "text": existing_text or f"[Page {page_num}: No readable text found]",
                                "ocr_used": False
                            })
            except Exception as e:
                logger.warning(f"PDF OCR conversion failed: {e}")
                # If OCR fails (e.g., poppler/tesseract not installed), fallback to whatever text was found
                for page_num in needs_ocr_pages:
                    existing_text = next((t for p, t in text_by_page if p == page_num), "")
                    pages_data.append({
                        "page": page_num,
                        "text": existing_text or f"[Page {page_num}: Unreadable content]",
                        "ocr_used": False
                    })

        # Ensure pages are sorted by page number
        pages_data.sort(key=lambda x: x["page"])
        return pages_data

    def _parse_image(self, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """Extracts text from images using PyTesseract OCR."""
        try:
            image = Image.open(io.BytesIO(file_bytes))
            ocr_text = self._ocr_image(image)
            return [{
                "page": 1,
                "text": ocr_text.strip() if ocr_text.strip() else "[Image containing no readable text]",
                "ocr_used": True
            }]
        except Exception as e:
            logger.error(f"Image parsing error for {filename}: {e}")
            return [{
                "page": 1,
                "text": f"[Error processing image {filename}: {str(e)}]",
                "ocr_used": False
            }]

    def _ocr_image(self, image: Image.Image) -> str:
        """Helper to run PyTesseract with fallback to Gemini 2.5 Flash Vision OCR."""
        if pytesseract is not None:
            try:
                text = pytesseract.image_to_string(image)
                if text.strip():
                    return text
            except Exception as e:
                logger.warning(f"Tesseract OCR execution warning: {e}")

        # Gemini 2.5 Flash Multimodal Vision OCR Fallback
        try:
            from google import genai
            from google.genai import types
            from config import config
            if genai is not None and config.GEMINI_API_KEY:
                client = genai.Client(api_key=config.GEMINI_API_KEY)
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG')
                img_bytes = img_byte_arr.getvalue()

                res = client.models.generate_content(
                    model=config.LLM_MODEL,
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                        "Extract all text from this document image completely and accurately. Do not summarize."
                    ]
                )
                if res and res.text:
                    return res.text.strip()
        except Exception as e:
            logger.warning(f"Gemini Vision OCR fallback warning: {e}")

        return ""


    def _parse_text(self, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """Parses plain text files."""
        try:
            content = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            content = file_bytes.decode("latin-1", errors="ignore")

        return [{
            "page": 1,
            "text": content.strip(),
            "ocr_used": False
        }]

    def _chunk_pages(self, pages_data: List[Dict[str, Any]], filename: str) -> List[Dict[str, Any]]:
        """Splits page data into chunks with metadata."""
        chunks = []
        chunk_idx = 0

        for p_info in pages_data:
            page_num = p_info["page"]
            text = p_info["text"]
            ocr_used = p_info.get("ocr_used", False)

            if not text.strip():
                continue

            sub_chunks = self.splitter.split_text(text)
            for sub_text in sub_chunks:
                if not sub_text.strip():
                    continue
                
                chunk_id = f"{filename}_p{page_num}_c{chunk_idx}_{uuid.uuid4().hex[:6]}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "content": sub_text.strip(),
                    "metadata": {
                        "source": filename,
                        "page": page_num,
                        "chunk_index": chunk_idx,
                        "ocr_used": ocr_used,
                        "chunk_id": chunk_id
                    }
                })
                chunk_idx += 1

        return chunks
