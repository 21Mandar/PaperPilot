"""PaperPilot — ingestion pipeline.

Load a PDF, split it into overlapping chunks (keeping page numbers for
citations), embed each chunk with sentence-transformers, and build a FAISS
index for dense retrieval.

Usage:
    python -m src.ingest                      # uses defaults below
    python -m src.ingest --pdf data/sample.pdf --out index
    python -m src.ingest --chunk-size 800 --overlap 150
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Defaults chosen to match the project stack / one-day plan.
DEFAULT_PDF = "data/sample.pdf"
DEFAULT_OUT = "index"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 800   # characters
DEFAULT_OVERLAP = 150      # characters


@dataclass
class Chunk:
    """A retrievable unit of text plus the metadata we need to cite it."""

    id: int
    page: int        # 1-based page number in the source PDF
    text: str


def load_pdf_pages(pdf_path: Path) -> list[str]:
    """Return the extracted text of each page (index 0 == page 1)."""
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        # extract_text() can return None for image-only pages.
        pages.append(page.extract_text() or "")
    return pages


def chunk_pages(
    pages: list[str],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Split each page's text into overlapping character windows.

    Chunking per page (rather than across the whole document) keeps an exact
    page number on every chunk, which the UI uses to cite sources. Overlap
    avoids cutting a relevant sentence across a chunk boundary.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    next_id = 0
    step = chunk_size - overlap

    for page_idx, text in enumerate(pages):
        text = " ".join(text.split())  # collapse whitespace/newlines
        if not text:
            continue
        for start in range(0, len(text), step):
            window = text[start : start + chunk_size].strip()
            if not window:
                continue
            chunks.append(Chunk(id=next_id, page=page_idx + 1, text=window))
            next_id += 1
            if start + chunk_size >= len(text):
                break  # last window already covered the tail
    return chunks


def embed_chunks(
    chunks: list[Chunk],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
) -> np.ndarray:
    """Embed chunk texts into a normalized float32 matrix (n_chunks, dim)."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        [c.text for c in chunks],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so inner product == cosine similarity
    )
    return embeddings.astype("float32")


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a flat inner-product FAISS index over normalized vectors."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # exact search; fine for a single-machine demo
    index.add(embeddings)
    return index


def save_artifacts(
    out_dir: Path,
    index: faiss.Index,
    chunks: list[Chunk],
    model_name: str,
) -> None:
    """Persist the FAISS index, chunk metadata, and a small manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(out_dir / "faiss.index"))

    with open(out_dir / "chunks.json", "w") as f:
        json.dump([asdict(c) for c in chunks], f, ensure_ascii=False)

    manifest = {
        "model": model_name,
        "num_chunks": len(chunks),
        "dim": index.d,
        "metric": "inner_product (cosine on normalized vectors)",
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def ingest(
    pdf_path: Path,
    out_dir: Path,
    model_name: str = DEFAULT_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> None:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    t0 = time.perf_counter()
    pages = load_pdf_pages(pdf_path)
    print(f"Loaded {len(pages)} pages from {pdf_path}")

    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    print(f"Created {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={overlap})")
    if not chunks:
        raise RuntimeError(
            "No text extracted — the PDF may be scanned/image-only (needs OCR)."
        )

    embeddings = embed_chunks(chunks, model_name=model_name)
    print(f"Embedded {embeddings.shape[0]} chunks -> dim {embeddings.shape[1]}")

    index = build_index(embeddings)
    save_artifacts(out_dir, index, chunks, model_name)

    dt = time.perf_counter() - t0
    print(f"Saved index + metadata to {out_dir}/  ({dt:.1f}s total)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest a PDF into a FAISS index.")
    p.add_argument("--pdf", default=DEFAULT_PDF, help="path to source PDF")
    p.add_argument("--out", default=DEFAULT_OUT, help="output directory")
    p.add_argument("--model", default=DEFAULT_MODEL, help="embedding model")
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ingest(
        pdf_path=Path(args.pdf),
        out_dir=Path(args.out),
        model_name=args.model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
