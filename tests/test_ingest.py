"""Unit tests for src.ingest — pure chunking logic, no models/network."""

import pytest

from src.ingest import chunk_pages


def test_chunk_pages_respects_chunk_size():
    text = "a" * 1000
    chunks = chunk_pages([text], chunk_size=200, overlap=50)
    assert chunks
    assert all(len(c.text) <= 200 for c in chunks)
    assert all(c.page == 1 for c in chunks)


def test_chunk_pages_ids_are_sequential():
    text = "word " * 500
    chunks = chunk_pages([text], chunk_size=200, overlap=50)
    assert [c.id for c in chunks] == list(range(len(chunks)))


def test_chunk_pages_skips_empty_pages():
    chunks = chunk_pages(["", "   ", "real text here"], chunk_size=800, overlap=150)
    assert len(chunks) == 1
    assert chunks[0].page == 3


def test_chunk_pages_page_numbers_are_one_based():
    chunks = chunk_pages(
        ["page one text", "page two text"], chunk_size=800, overlap=150
    )
    pages = {c.page for c in chunks}
    assert pages == {1, 2}


def test_chunk_pages_rejects_overlap_ge_chunk_size():
    with pytest.raises(ValueError):
        chunk_pages(["some text"], chunk_size=100, overlap=100)
