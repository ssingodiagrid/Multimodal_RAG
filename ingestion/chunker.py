"""Page-level text chunking using LangChain RecursiveCharacterTextSplitter."""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(
    doc_id: str,
    pages: list[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks: list[dict] = []
    for page_obj in pages:
        text = page_obj["text"]
        page = page_obj["page"]
        parts = splitter.split_text(text)
        chunk_num = 0
        for piece in parts:
            chunk_text = piece.strip()
            if not chunk_text:
                continue
            chunks.append(
                {
                    "chunk_id": f"{doc_id}_p{page}_c{chunk_num}",
                    "doc_id": doc_id,
                    "page": page,
                    "text": chunk_text,
                }
            )
            chunk_num += 1
    return chunks
