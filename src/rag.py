"""PaperPilot — retrieval + generation pipeline.

Loads the FAISS index built by ``src.ingest``, retrieves the most relevant
chunks for a query via dense vector search, and generates a grounded answer
with a local open-weights LLM (Hugging Face Transformers).

The pieces are deliberately separable so ``src.eval`` can reuse them:
    - Retriever.search(query, k)      -> ranked chunks + scores + latency
    - build_messages(query, ctx, mode)-> chat messages (zero-shot vs few-shot)
    - Generator.generate(messages)    -> answer text

Usage:
    python -m src.rag "How does merge sort work?"
    python -m src.rag "..." --k 4 --mode few_shot
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

# IMPORT ORDER MATTERS — do not reorder/sort these.
# faiss, torch, and sklearn (via sentence-transformers) each bundle their own
# OpenMP runtime. If faiss initializes OpenMP first, a later model move to MPS
# segfaults. Importing torch/transformers first makes torch claim OpenMP and
# avoids the crash. Keep faiss last.  (isort: off)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss  # noqa: E402  (must stay after torch — see note above)

DEFAULT_INDEX_DIR = "index"
# Small open-weights instruct model — runs on an M1 (MPS) or CPU for the demo.
DEFAULT_LLM = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_K = 4
DEFAULT_MAX_NEW_TOKENS = 256

SYSTEM_PROMPT = (
    "You are a precise assistant that answers questions using ONLY the "
    "provided context passages. If the answer is not contained in the "
    "context, say you don't know. Cite the page number(s) you used like "
    "(p. 12). Be concise."
)

# One worked exemplar for the few-shot variant. It demonstrates the desired
# behaviour: ground the answer in context, cite the page, and abstain when the
# context doesn't cover the question.
FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": (
            "Context:\n"
            "[p. 3] A stack is a last-in, first-out (LIFO) structure. The "
            "most recently inserted element is removed first.\n\n"
            "Question: What ordering does a stack use?"
        ),
    },
    {
        "role": "assistant",
        "content": "A stack uses last-in, first-out (LIFO) ordering (p. 3).",
    },
]


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Retrieved:
    """A chunk returned from search, with its similarity score."""

    id: int
    page: int
    text: str
    score: float


class Retriever:
    """Dense retriever over the FAISS index produced by ``src.ingest``."""

    def __init__(self, index_dir: str | Path = DEFAULT_INDEX_DIR):
        index_dir = Path(index_dir)
        manifest_path = index_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No index found in {index_dir}/. Run `python -m src.ingest` first."
            )

        self.manifest = json.loads(manifest_path.read_text())
        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        self.chunks = json.loads((index_dir / "chunks.json").read_text())
        # Same embedding model as ingestion — required for comparable vectors.
        # Pin to the GPU (MPS/CUDA) when available, else CPU.
        self.embedder = SentenceTransformer(self.manifest["model"], device=_select_device())

    def search(self, query: str, k: int = DEFAULT_K) -> tuple[list[Retrieved], float]:
        """Return the top-k chunks for a query and the retrieval latency (s)."""
        t0 = time.perf_counter()
        q = self.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, idxs = self.index.search(q, k)
        latency = time.perf_counter() - t0

        results: list[Retrieved] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:  # FAISS pads with -1 when fewer than k results
                continue
            c = self.chunks[idx]
            results.append(
                Retrieved(id=c["id"], page=c["page"], text=c["text"], score=float(score))
            )
        return results, latency


def format_context(contexts: list[Retrieved]) -> str:
    """Render retrieved chunks as a numbered, page-tagged context block."""
    return "\n\n".join(f"[p. {c.page}] {c.text}" for c in contexts)


def build_messages(
    query: str, contexts: list[Retrieved], mode: str = "zero_shot"
) -> list[dict]:
    """Build chat messages for the LLM.

    ``mode`` is "zero_shot" (instruction only) or "few_shot" (instruction plus
    a worked example). The two modes are what ``src.eval`` compares.
    """
    if mode not in {"zero_shot", "few_shot"}:
        raise ValueError(f"unknown mode: {mode!r}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if mode == "few_shot":
        messages.extend(FEW_SHOT_EXAMPLES)

    user = f"Context:\n{format_context(contexts)}\n\nQuestion: {query}"
    messages.append({"role": "user", "content": user})
    return messages


class Generator:
    """Wraps a local causal-LM for grounded answer generation."""

    def __init__(self, model_name: str = DEFAULT_LLM, device: str | None = None):
        self.device = device or _select_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32 if self.device == "cpu" else torch.float16,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self, messages: list[dict], max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    ) -> tuple[str, float]:
        """Generate an answer from chat messages; return (text, latency_s)."""
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        t0 = time.perf_counter()
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy -> deterministic, fair prompt comparison
            pad_token_id=self.tokenizer.eos_token_id,
        )
        latency = time.perf_counter() - t0

        # Decode only the newly generated tokens, not the prompt.
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return text, latency


@dataclass
class RAGResult:
    answer: str
    contexts: list[Retrieved]
    retrieval_latency: float
    generation_latency: float
    mode: str


class RAGPipeline:
    """End-to-end retrieve-then-generate pipeline."""

    def __init__(
        self,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
        model_name: str = DEFAULT_LLM,
    ):
        self.retriever = Retriever(index_dir)
        self.generator = Generator(model_name)

    def answer(
        self,
        query: str,
        k: int = DEFAULT_K,
        mode: str = "zero_shot",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> RAGResult:
        contexts, r_lat = self.retriever.search(query, k=k)
        messages = build_messages(query, contexts, mode=mode)
        text, g_lat = self.generator.generate(messages, max_new_tokens=max_new_tokens)
        return RAGResult(
            answer=text,
            contexts=contexts,
            retrieval_latency=r_lat,
            generation_latency=g_lat,
            mode=mode,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Query the PaperPilot RAG pipeline.")
    p.add_argument("query", help="the question to ask")
    p.add_argument("--index", default=DEFAULT_INDEX_DIR, help="index directory")
    p.add_argument("--model", default=DEFAULT_LLM, help="local LLM model id")
    p.add_argument("--k", type=int, default=DEFAULT_K, help="chunks to retrieve")
    p.add_argument(
        "--mode", choices=["zero_shot", "few_shot"], default="zero_shot"
    )
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pipe = RAGPipeline(index_dir=args.index, model_name=args.model)
    result = pipe.answer(
        args.query, k=args.k, mode=args.mode, max_new_tokens=args.max_new_tokens
    )

    print(f"\nAnswer ({result.mode}):\n{result.answer}\n")
    print("Sources:")
    for c in result.contexts:
        print(f"  p. {c.page}  (score {c.score:.3f})")
    print(
        f"\nretrieval: {result.retrieval_latency*1000:.0f} ms  |  "
        f"generation: {result.generation_latency:.1f} s"
    )


if __name__ == "__main__":
    main()
