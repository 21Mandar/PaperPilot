"""Unit tests for src.agent's tool functions.

Tests call the plain (undecorated) implementations directly against the
FAISS index already committed under index/ — no live Claude API call, no
ANTHROPIC_API_KEY required, keeps CI free.
"""

from src.agent import (
    _get_page_impl,
    _search_document_impl,
    build_tools,
    is_missing_credential_error,
)
from src.rag import Retriever


def test_search_document_returns_formatted_context():
    retriever = Retriever("index")
    result = _search_document_impl(
        retriever, "How are radioactive materials transported?", k=3
    )
    assert "[p." in result


def test_search_document_handles_no_results(monkeypatch):
    retriever = Retriever("index")
    monkeypatch.setattr(retriever, "search", lambda query, k=4: ([], 0.0))
    result = _search_document_impl(retriever, "irrelevant query")
    assert result == "No relevant passages found."


def test_get_page_returns_text_for_known_page():
    retriever = Retriever("index")
    result = _get_page_impl(retriever, 9)
    assert result
    assert result != "Page 9 not found in the document."


def test_get_page_handles_missing_page():
    retriever = Retriever("index")
    result = _get_page_impl(retriever, 9999)
    assert result == "Page 9999 not found in the document."


def test_build_tools_returns_two_tools():
    retriever = Retriever("index")
    tools = build_tools(retriever)
    assert len(tools) == 2


def test_is_missing_credential_error_matches_sdk_message():
    # Regression test: this is the exact TypeError anthropic.Anthropic() raises
    # client-side (no dedicated exception class) when no api_key/auth_token/
    # OAuth profile resolves at all — distinct from anthropic.AuthenticationError,
    # which only fires when a key IS sent but rejected by the server.
    exc = TypeError(
        "Could not resolve authentication method. Expected one of api_key, "
        "auth_token, or credentials to be set."
    )
    assert is_missing_credential_error(exc)


def test_is_missing_credential_error_ignores_unrelated_type_errors():
    assert not is_missing_credential_error(TypeError("unrelated failure"))
    assert not is_missing_credential_error(ValueError("wrong exception type"))
