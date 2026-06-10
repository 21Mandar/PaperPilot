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

import streamlit as st

from src.rag import DEFAULT_K, DEFAULT_LLM, RAGPipeline  # noqa: E402

# Allow the deployment (Docker/HF Spaces) to pick a lighter model than the local
# default — a 1.5B model is slow on a free CPU Space, so we ship 0.5B there.
DEFAULT_LLM = os.environ.get("PAPERPILOT_LLM", DEFAULT_LLM)

st.set_page_config(page_title="PaperPilot", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading embedder, FAISS index, and LLM…")
def load_pipeline(index_dir: str, model_name: str) -> RAGPipeline:
    """Load (and cache) the full RAG pipeline. Cached on its arguments."""
    return RAGPipeline(index_dir=index_dir, model_name=model_name)


def sidebar() -> dict:
    st.sidebar.header("Settings")
    model_name = st.sidebar.text_input("LLM model", value=DEFAULT_LLM)
    mode = st.sidebar.radio(
        "Prompt mode",
        options=["few_shot", "zero_shot"],
        help="few_shot adds a worked example — measurably improves citation rate.",
    )
    k = st.sidebar.slider("Chunks retrieved (k)", 1, 10, DEFAULT_K)
    max_new_tokens = st.sidebar.slider("Max answer tokens", 64, 512, 256, step=64)
    return {
        "model_name": model_name,
        "mode": mode,
        "k": k,
        "max_new_tokens": max_new_tokens,
    }


def main() -> None:
    st.title("📄 PaperPilot")
    st.caption(
        "RAG over your PDF — dense retrieval + a local open-weights LLM, "
        "with grounded, page-cited answers."
    )

    cfg = sidebar()
    index_dir = str(ROOT / "index")

    try:
        pipeline = load_pipeline(index_dir, cfg["model_name"])
    except FileNotFoundError:
        st.error(
            "No index found. Build it first:\n\n"
            "```\npython -m src.ingest\n```"
        )
        st.stop()

    query = st.text_input(
        "Ask a question about the document",
        placeholder="e.g. How are radioactive materials transported?",
    )
    ask = st.button("Ask", type="primary")

    if ask and query.strip():
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
    elif ask:
        st.warning("Enter a question first.")


if __name__ == "__main__":
    main()
