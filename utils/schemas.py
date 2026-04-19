from typing import TypedDict


class TextChunk(TypedDict):
    chunk_id: str
    doc_id: str
    page: int
    text: str


class ChunkMetadata(TypedDict):
    chunk_id: str
    text: str
    page: int
    doc_id: str


class ContextItem(TypedDict):
    chunk_id: str
    text: str
    page: int
    score: float
