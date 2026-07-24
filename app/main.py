"""PaperPilot — Streamlit UI.

Ask a question about the ingested PDF and get a grounded, page-cited answer
from the local LLM. The RAG pipeline (embedder + FAISS index + LLM) is loaded
once and cached, so the model-load cost is paid a single time per session and
every question is just retrieval + generation.

Run from the project root:
    streamlit run app/main.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the project root importable so `src` resolves no matter where Streamlit
# is launched from, and resolve the default index path relative to the repo.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import anthropic
import streamlit as st

from src.agent import (
    DEFAULT_AGENT_MODEL,
    answer as agent_answer,
    is_missing_credential_error,
)
from src.rag import DEFAULT_K, DEFAULT_LLM, RAGPipeline, Retriever  # noqa: E402

# Allow the deployment (Docker/HF Spaces) to pick a lighter model than the local
# default — a 1.5B model is slow on a free CPU Space, so we ship 0.5B there.
DEFAULT_LLM = os.environ.get("PAPERPILOT_LLM", DEFAULT_LLM)

# Agentic mode is local-dev only — it is not set in the Docker/HF Spaces
# deployment, so this override only ever takes effect when you export it
# yourself. See README for why.
DEFAULT_AGENT_MODEL = os.environ.get("PAPERPILOT_AGENT_MODEL", DEFAULT_AGENT_MODEL)

st.set_page_config(page_title="PaperPilot", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading embedder, FAISS index, and LLM…")
def load_pipeline(index_dir: str, model_name: str) -> RAGPipeline:
    """Load (and cache) the full RAG pipeline. Cached on its arguments."""
    return RAGPipeline(index_dir=index_dir, model_name=model_name)


@st.cache_resource(show_spinner="Loading embedder and FAISS index…")
def load_retriever(index_dir: str) -> Retriever:
    """Load (and cache) just the retriever — no local LLM, used by agentic mode."""
    return Retriever(index_dir)


def sidebar() -> dict:
    st.sidebar.header("Settings")
    backend = st.sidebar.radio(
        "Backend",
        options=["Local (offline)", "Agentic (Claude API)"],
        help=(
            "Local runs entirely on this machine, no API key needed. Agentic "
            "has Claude drive a multi-step tool-use search loop over the same "
            "index — requires ANTHROPIC_API_KEY and calls the Anthropic API."
        ),
    )
    cfg: dict = {"backend": backend}

    if backend == "Local (offline)":
        cfg["model_name"] = st.sidebar.text_input("LLM model", value=DEFAULT_LLM)
        cfg["mode"] = st.sidebar.radio(
            "Prompt mode",
            options=["few_shot", "zero_shot"],
            help="few_shot adds a worked example — measurably improves citation rate.",
        )
        cfg["k"] = st.sidebar.slider("Chunks retrieved (k)", 1, 10, DEFAULT_K)
        cfg["max_new_tokens"] = st.sidebar.slider(
            "Max answer tokens", 64, 512, 256, step=64
        )
    else:
        cfg["model_name"] = st.sidebar.text_input(
            "Claude model", value=DEFAULT_AGENT_MODEL
        )

    return cfg


def main() -> None:
    st.title("📄 PaperPilot")
    st.caption(
        "RAG over your PDF — dense retrieval + a local open-weights LLM, "
        "with grounded, page-cited answers."
    )

    cfg = sidebar()
    index_dir = str(ROOT / "index")

    query = st.text_input(
        "Ask a question about the document",
        placeholder="e.g. How are radioactive materials transported?",
    )
    ask = st.button("Ask", type="primary")

    if not (ask and query.strip()):
        if ask:
            st.warning("Enter a question first.")
        return

    if cfg["backend"] == "Local (offline)":
        try:
            pipeline = load_pipeline(index_dir, cfg["model_name"])
        except FileNotFoundError:
            st.error(
                "No index found. Build it first:\n\n"
                "```\npython -m src.ingest\n```"
            )
            st.stop()

        with st.spinner("Retrieving and generating…"):
            result = pipeline.answer(
                query,
                k=cfg["k"],
                mode=cfg["mode"],
                max_new_tokens=cfg["max_new_tokens"],
            )

        st.subheader("Answer")
        st.write(result.answer)

        # Latency metrics — the same numbers the eval harness reports.
        c1, c2, c3 = st.columns(3)
        c1.metric("Retrieval", f"{result.retrieval_latency * 1000:.0f} ms")
        c2.metric("Generation", f"{result.generation_latency:.1f} s")
        c3.metric("Prompt mode", result.mode)

        st.subheader("Sources")
        for c in result.contexts:
            with st.expander(f"p. {c.page}  ·  similarity {c.score:.3f}"):
                st.write(c.text)
        return

    # Agentic (Claude API) backend
    try:
        retriever = load_retriever(index_dir)
    except FileNotFoundError:
        st.error(
            "No index found. Build it first:\n\n"
            "```\npython -m src.ingest\n```"
        )
        st.stop()

    with st.spinner("Agent is searching the document…"):
        try:
            result = agent_answer(query, retriever, model=cfg["model_name"])
        except anthropic.AuthenticationError:
            st.info(
                "`ANTHROPIC_API_KEY` was rejected by the API (invalid or "
                "revoked). Set a valid key in your environment and restart:\n\n"
                "```\nexport ANTHROPIC_API_KEY=sk-ant-...\n```"
            )
            st.stop()
        except TypeError as e:
            if not is_missing_credential_error(e):
                raise
            st.info(
                "No `ANTHROPIC_API_KEY` found. This mode calls the Claude "
                "API — set a key in your environment and restart:\n\n"
                "```\nexport ANTHROPIC_API_KEY=sk-ant-...\n```"
            )
            st.stop()

    st.subheader("Answer")
    st.write(result.answer)
    st.caption(f"Model: {result.model}")

    st.subheader("Tool calls")
    if result.tool_calls:
        for call in result.tool_calls:
            st.write(f"`{call['name']}`  →  `{call['input']}`")
    else:
        st.write("None — the agent answered directly.")


if __name__ == "__main__":
    main()
