"""
eval/runners/baseline_runner.py

Global-view baseline (control group).

Simulates a "flat agent" that:
  1. Uses a web search tool to retrieve ALL relevant documents for a question
  2. Concatenates the full retrieved context into ONE prompt
  3. Asks the LLM to answer—no decomposition, no isolation, no escalation

This mirrors "traditional flat RAG / ACN" described in the ROMA paper.

For SEAL-0, the misleading sources are included verbatim in the global context
to demonstrate the anti-interference disadvantage of the global view.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from eval.config import (
    BASELINE_API_KEY,
    BASELINE_BASE_URL,
    BASELINE_MODEL,
)

_SYSTEM_FLAT = """\
You are a knowledgeable assistant. You will be given a question and a set of \
retrieved web passages that may help you answer it. Some passages may be \
misleading, outdated, or irrelevant. Read ALL passages and answer as accurately \
as you can. Reply with ONLY the final answer—no preamble.\
"""

_SYSTEM_MULTIHOP = """\
You are a research assistant. You will be given a multi-hop factual question \
and a collection of retrieved documents. Synthesise the information across all \
documents to produce the correct answer. Reply with ONLY the final answer.\
"""

_SYSTEM_WRITING_GLOBAL = """\
You are a creative writing assistant. You will receive the full prior text and \
a continuation prompt. Write the next ~1,000-word section in a consistent style \
and voice. Output the continuation text only.\
"""

_SYSTEM_ABLATION_GLOBAL = """\
You are an expert ML researcher. You will receive a description of a system, \
a hypothesis about it, and extensive background context. Design a rigorous \
ablation study. List the ablation dimensions with clear variable-control rules.\
"""


class BaselineRunner:
    """
    Flat single-LLM-call baseline — the global-view control group.
    Accepts the same source snippets that EVABot sees but dumps them
    all into a single context window.
    """

    def __init__(
        self,
        model: str = BASELINE_MODEL,
        api_key: str = BASELINE_API_KEY,
        base_url: str = BASELINE_BASE_URL,
    ):
        self.model  = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    # ──────────────────────────────────────────────
    # Dataset-specific entry points
    # ──────────────────────────────────────────────

    def run_seal0(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        SEAL-0 global view: all sources (incl. misleading ones) concatenated.
        """
        sources_block = "\n\n".join(
            f"[Source {i+1}] {s}" for i, s in enumerate(sample["sources"])
        )
        user_msg = (
            f"Question: {sample['question']}\n\n"
            f"Retrieved passages:\n{sources_block}"
        )
        return self._call(_SYSTEM_FLAT, user_msg)

    def run_frames(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        FRAMES global view: Wikipedia article titles used as retrieval hints.
        We prompt the model to answer as if all wiki pages were retrieved.
        """
        wiki_hint = ", ".join(sample.get("wiki_titles", [])) or "general knowledge"
        user_msg = (
            f"Question: {sample['question']}\n\n"
            f"Relevant Wikipedia articles (all retrieved): {wiki_hint}\n\n"
            "Using information from all the above articles, answer the question."
        )
        return self._call(_SYSTEM_MULTIHOP, user_msg)

    def run_simpleqa(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        SimpleQA global view: retrieve the question with no context isolation.
        """
        user_msg = (
            f"Question: {sample['question']}\n\n"
            "Search the web and return the single correct factual answer."
        )
        return self._call(_SYSTEM_FLAT, user_msg)

    def run_eqbench(self, sample: Dict[str, Any], turn_idx: int, history: str) -> Dict[str, Any]:
        """
        EQ-Bench global view: the FULL prior text is fed as context each turn.
        This simulates context bloat and potential attention drift.
        """
        user_msg = (
            f"Title: {sample['title']}\n"
            f"Genre: {sample['genre']}\n"
            f"Characters: {sample['character_brief']}\n\n"
            f"--- FULL PRIOR TEXT ({len(history.split())} words) ---\n{history}\n\n"
            f"--- CONTINUATION PROMPT (Turn {turn_idx+1}/8) ---\n{sample['turns'][turn_idx]}"
        )
        return self._call(_SYSTEM_WRITING_GLOBAL, user_msg)

    def run_abgen(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        AbGen global view: full noisy context including irrelevant sentences.
        """
        components = "\n".join(f"  - {c}" for c in sample["components"])
        user_msg = (
            f"System: {sample['system']}\n"
            f"Hypothesis: {sample['hypothesis']}\n\n"
            f"Context (full, may contain irrelevant information):\n{sample['context']}\n\n"
            f"Known system components:\n{components}\n\n"
            "Design a rigorous ablation study testing the hypothesis."
        )
        return self._call(_SYSTEM_ABLATION_GLOBAL, user_msg)

    # ──────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────

    def _call(self, system: str, user: str) -> Dict[str, Any]:
        t0 = time.time()
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            content = resp.choices[0].message.content or ""
            usage   = resp.usage
            cost = self._calc_cost(
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0,
            )
        except Exception as e:
            content = f"[BASELINE ERROR] {e}"
            cost = 0.0

        return {
            "final_answer": content.strip(),
            "cost":         cost,
            "latency_s":    round(time.time() - t0, 2),
            # Baseline never escalates or evolves
            "escalate_count":       0,
            "escalate_resolved":    0,
            "evolution_triggered":  False,
        }

    def _calc_cost(self, input_tokens: int, output_tokens: int) -> float:
        # GPT-4o-mini pricing as of 2025
        PRICES = {
            "gpt-4o-mini": (0.15, 0.60),     # $/1M tokens
            "gpt-4o":      (5.00, 15.00),
            "gpt-4":       (30.00, 60.00),
        }
        inp_price, out_price = PRICES.get(self.model, (1.0, 3.0))
        return round(
            (input_tokens / 1_000_000) * inp_price +
            (output_tokens / 1_000_000) * out_price,
            6,
        )
