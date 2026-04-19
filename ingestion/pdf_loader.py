import fitz

from ingestion.pdf_text_extract import extract_page_text
from ingestion.text_cleaner import clean_text


def extract_pages(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []
    try:
        for i, page in enumerate(doc):
            page_num = i + 1
            raw = extract_page_text(page, pdf_path, i)
            pages.append({"page": page_num, "text": clean_text(raw)})
    finally:
        doc.close()
    return pages
