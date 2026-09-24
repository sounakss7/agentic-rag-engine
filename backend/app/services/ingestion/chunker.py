"""
NexusRAG Parent-Child Hierarchical Document Chunker.
Produces high-precision atomic child chunks linked to rich parent context sections.
Preserves Markdown tables without fragmentation.
"""

import uuid
from typing import List, Dict, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.app.core.config import settings


class HierarchicalChunker:
    """
    Implements Parent-Child Hierarchical Chunking.
    - Parent Sections: Large context blocks (~1200 chars) retaining thematic coherence and complete tables.
    - Child Chunks: Compact atomic passages (~250 chars) for precise vector & BM25 needle matching.
    """

    def __init__(
        self,
        parent_chunk_size: int = settings.PARENT_CHUNK_SIZE,
        child_chunk_size: int = settings.CHILD_CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ):
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.chunk_overlap = chunk_overlap

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.chunk_overlap * 2,
            separators=["\n## ", "\n### ", "\n\n", "\n", " "]
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " "]
        )

    def process_document(
        self, pages_data: List[Dict[str, Any]], filename: str
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        Splits parsed page records into child chunks and a parent section lookup dictionary.
        Returns:
            Tuple of (child_chunks_list, parent_sections_map)
        """
        child_chunks: List[Dict[str, Any]] = []
        parent_sections_map: Dict[str, Dict[str, Any]] = {}

        for p_idx, page in enumerate(pages_data):
            page_num = page.get("page", p_idx + 1)
            raw_text = page.get("text", "")
            ocr_used = page.get("ocr_used", False)
            source = page.get("source", filename)
            tables = page.get("tables", [])

            # Identify if text contains markdown tables
            has_table = bool(tables or ("|" in raw_text and "---" in raw_text))

            # 1. Create Parent Blocks
            parent_texts = self.parent_splitter.split_text(raw_text)
            if not parent_texts and raw_text:
                parent_texts = [raw_text]

            for p_sub_idx, p_text in enumerate(parent_texts):
                parent_id = f"{filename}_p{page_num}_sec{p_sub_idx}_{uuid.uuid4().hex[:6]}"
                is_table_section = "|" in p_text and "---" in p_text

                parent_record = {
                    "parent_id": parent_id,
                    "content": p_text,
                    "metadata": {
                        "source": source,
                        "page": page_num,
                        "is_table": is_table_section or has_table,
                        "ocr_used": ocr_used,
                    }
                }
                parent_sections_map[parent_id] = parent_record

                # 2. Create Atomic Child Chunks under this Parent
                child_texts = self.child_splitter.split_text(p_text)
                if not child_texts and p_text:
                    child_texts = [p_text]

                for c_sub_idx, c_text in enumerate(child_texts):
                    child_id = f"{parent_id}_c{c_sub_idx}"
                    child_chunks.append({
                        "chunk_id": child_id,
                        "content": c_text,
                        "parent_id": parent_id,
                        "parent_content": p_text,
                        "metadata": {
                            "source": source,
                            "page": page_num,
                            "chunk_index": len(child_chunks) + 1,
                            "is_table": is_table_section,
                            "ocr_used": ocr_used,
                            "parent_id": parent_id
                        }
                    })

        return child_chunks, parent_sections_map
