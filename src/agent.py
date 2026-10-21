"""PaperPilot — agentic retrieval mode (Claude API, optional).

An alternate backend to ``src.rag``: instead of a single fixed
retrieve-then-generate pass, Claude drives a tool-use loop over the same
FAISS index — issuing further searches for multi-part questions and
double-checking a page's full text before citing it.

This mode calls the Anthropic API and requires ``ANTHROPIC_API_KEY`` to be
set. It is intentionally NOT wired into the Docker/Hugging Face Spaces
deployment (see README) — it's a local-dev/demo mode so an API key is never
exposed to public, unauthenticated traffic.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m src.agent "How are radioactive materials transported?"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import anthropic
from anthropic import beta_tool

from src.rag import DEFAULT_INDEX_DIR, Retriever, format_context

DEFAULT_AGENT_MODEL = "claude-opus-4-8"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_ITERATIONS = 6

AGENT_SYSTEM_PROMPT = (
    "You are a precise assistant that answers questions about a document "
    "using ONLY the tools provided — you have no other source of truth. "
    "Use search_document to find relevant passages; call it again with a "
    "different query if the question has multiple parts or the first search "
    "doesn't fully cover it. Use get_page to double-check a page's full text "
    "before citing it, especially for numbers, names, or specific claims. "
    "Cite the page number(s) you used like (p. 12). If the tools don't turn "
    "up the answer after a genuine search, say you don't know rather than "
    "guessing. Be concise."
)


@dataclass
class AgentResult:
    """The agent's final answer plus a trace of the tools it called."""

    answer: str
    tool_calls: list[dict]
    model: str
    truncated: bool = False


TRUNCATED_NOTICE = (
    "The agent stopped before finishing (it hit the "
    f"{DEFAULT_MAX_ITERATIONS}-step search limit or the token limit), so there "
    "is no complete answer. Try a narrower question."
)

NO_CREDENTIAL_HELP = (
    "No Claude API credentials found. This mode calls the Anthropic API — set "
    "an API key in your environment and try again:\n"
    "  export ANTHROPIC_API_KEY=sk-ant-...\n"
)

# The SDK raises two different exceptions depending on *how* auth fails:
#   - anthropic.AuthenticationError: a key was sent but the server rejected it
#     (bad/revoked key) — a normal HTTP 401 response.
#   - a plain TypeError, raised client-side before any request is sent, when
#     no credential (api_key / auth_token / OAuth profile) resolves at all.
#     This has no dedicated exception class in the SDK, so it's matched by
#     message — the far more common case for a first-time run with no key
#     configured yet, and easy to miss if you only test with a bad key.
_NO_CREDENTIAL_MARKER = "Could not resolve authentication method"


def is_missing_credential_error(exc: Exception) -> bool:
    """True if `exc` is the SDK's client-side 'no credential configured at
    all' error, as opposed to some unrelated TypeError."""
    return isinstance(exc, TypeError) and _NO_CREDENTIAL_MARKER in str(exc)


def _search_document_impl(retriever: Retriever, query: str, k: int = 4) -> str:
    """Plain implementation behind the search_document tool — kept undecorated
    so it can be unit-tested directly, without going through the tool-call
    machinery."""
    results, _ = retriever.search(query, k=k)
    if not results:
        return "No relevant passages found."
    return format_context(results)


def _get_page_impl(retriever: Retriever, page: int) -> str:
    """Plain implementation behind the get_page tool — see _search_document_impl."""
    chunks_on_page = [c for c in retriever.chunks if c["page"] == page]
    if not chunks_on_page:
        return f"Page {page} not found in the document."
    return "\n".join(c["text"] for c in chunks_on_page)


def build_tools(retriever: Retriever) -> list:
    """Build the two document tools, bound to one Retriever instance."""

    @beta_tool
    def search_document(query: str, k: int = 4) -> str:
        """Search the ingested document for passages relevant to a query.

        Args:
            query: A natural-language question or sub-question to search for.
            k: Number of top passages to retrieve.
        """
        return _search_document_impl(retriever, query, k=k)

    @beta_tool
    def get_page(page: int) -> str:
        """Retrieve the full text of one page, to verify a citation before answering.

        Args:
            page: The 1-based page number to retrieve.
        """
        return _get_page_impl(retriever, page)

    return [search_document, get_page]


def _final_answer(last_message) -> tuple[str, bool]:
    """Extract the agent's answer text and whether the run was cut off.

    The loop ends cleanly when Claude replies with text and no further tool
    call. If the iteration cap (or max_tokens) stops it mid-search, the last
    message is a tool call (or clipped) and there is no usable answer — report
    that instead of returning an empty string.
    """
    if last_message is None:
        return TRUNCATED_NOTICE, True
    text = "".join(
        block.text for block in last_message.content if block.type == "text"
    ).strip()
    cut_off = getattr(last_message, "stop_reason", None) in {"tool_use", "max_tokens"}
    if not text or cut_off:
        return text or TRUNCATED_NOTICE, True
    return text, False


def answer(
    query: str,
    retriever: Retriever,
    model: str = DEFAULT_AGENT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> AgentResult:
    """Run the tool-use loop for a single question and return the final answer."""
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    tools = build_tools(retriever)

    runner = client.beta.messages.tool_runner(
        model=model,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        system=AGENT_SYSTEM_PROMPT,
        tools=tools,
        messages=[{"role": "user", "content": query}],
        max_iterations=DEFAULT_MAX_ITERATIONS,
    )

    tool_calls: list[dict] = []
    last_message = None
    for message in runner:
        last_message = message
        for block in message.content:
            if block.type == "tool_use":
                tool_calls.append({"name": block.name, "input": block.input})

    final_text, truncated = _final_answer(last_message)
    return AgentResult(
        answer=final_text, tool_calls=tool_calls, model=model, truncated=truncated
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Query PaperPilot via a Claude tool-use agent."
    )
    p.add_argument("query", help="the question to ask")
    p.add_argument("--index", default=DEFAULT_INDEX_DIR, help="index directory")
    p.add_argument("--model", default=DEFAULT_AGENT_MODEL, help="Claude model id")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    retriever = Retriever(args.index)

    try:
        result = answer(args.query, retriever, model=args.model)
    except anthropic.AuthenticationError:
        raise SystemExit(f"Invalid ANTHROPIC_API_KEY (rejected by the API).\n{NO_CREDENTIAL_HELP}")
    except TypeError as e:
        if not is_missing_credential_error(e):
            raise
        raise SystemExit(NO_CREDENTIAL_HELP)

    print(f"\nAnswer ({result.model}):\n{result.answer}\n")
    if result.tool_calls:
        print("Tool calls:")
        for call in result.tool_calls:
            print(f"  {call['name']}({call['input']})")
    else:
        print("Tool calls: none")


if __name__ == "__main__":
    main()
