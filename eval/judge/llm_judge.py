"""
eval/judge/llm_judge.py

LLM-as-judge evaluation — mirrors ROMA's use of GPT-4o-mini as judge.

Provides four judge modes:
  1. factual_accuracy   – binary correct/incorrect for SEAL-0, FRAMES, SimpleQA
  2. writing_quality    – 0-100 Likert score for EQ-Bench (8-turn stories)
  3. ablation_quality   – Likert score for AbGen (variable control, completeness)
  4. noise_resistance   – fraction of misleading sources cited in the answer
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from eval.config import JUDGE_API_KEY, JUDGE_BASE_URL, JUDGE_MODEL

# ─────────────────────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────────────────────

_SYS_FACTUAL = """\
You are an expert evaluator assessing whether an AI system's answer is \
factually correct. You will be given the question, the gold-standard \
reference answer, and the system's answer.

Return ONLY a JSON object with two fields:
  "correct": true | false
  "reason":  one-sentence explanation (max 40 words)

Rules:
- Minor phrasing differences are fine if the core fact matches.
- Partial credit: if the answer is partially correct (e.g. right entity, \
  wrong year), set "correct": false but note this in "reason".
- Do NOT penalise for extra information if the key fact is present.\
"""

_SYS_WRITING = """\
You are a literary editor scoring a section of a multi-turn creative writing \
task. Score the writing on the following dimensions (each 0-10):

  narrative_quality     – flow, prose style, scene construction
  character_depth       – voice, motivation, authenticity
  structural_consistency– coherence with prior turns, no plot holes
  emotional_resonance   – impact, subtext, reader engagement

Return ONLY a JSON object:
{
  "narrative_quality":      <int 0-10>,
  "character_depth":        <int 0-10>,
  "structural_consistency": <int 0-10>,
  "emotional_resonance":    <int 0-10>,
  "total":                  <int 0-40>,
  "notes":                  "<one sentence>"
}\
"""

_SYS_ABLATION = """\
You are a senior ML researcher evaluating an ablation study design.
Score on four Likert dimensions (each 1-5):

  variable_control    – does each ablation change exactly ONE variable?
  completeness        – are all critical components covered?
  logical_soundness   – is the hypothesis–experiment link defensible?
  noise_exclusion     – did the designer correctly ignore irrelevant context?

Return ONLY a JSON object:
{
  "variable_control":  <int 1-5>,
  "completeness":      <int 1-5>,
  "logical_soundness": <int 1-5>,
  "noise_exclusion":   <int 1-5>,
  "total":             <int 4-20>,
  "notes":             "<one sentence>"
}\
"""

_SYS_NOISE_RESISTANCE = """\
You are evaluating whether an AI answer was influenced by planted misleading \
evidence. You will receive:
  - The question
  - A list of ALL sources (some marked as misleading)
  - The system's answer

Determine how many of the misleading sources (if any) influenced the answer.

Return ONLY a JSON object:
{
  "misleading_sources_cited": <int>,  // 0 = fully resistant
  "total_misleading":         <int>,
  "resistance_score":         <float>, // (1 - cited/total), 0.0-1.0
  "explanation":              "<one sentence>"
}\
"""


# ─────────────────────────────────────────────────────────────
# Judge class
# ─────────────────────────────────────────────────────────────

class LLMJudge:
    def __init__(
        self,
        model: str = JUDGE_MODEL,
        api_key: str = JUDGE_API_KEY,
        base_url: str = JUDGE_BASE_URL,
        max_retries: int = 3,
    ):
        self.model       = model
        self.client      = OpenAI(api_key=api_key, base_url=base_url)
        self.max_retries = max_retries

    # ──────────────────────────────────────────────
    # 1. Factual accuracy (SEAL-0 / FRAMES / SimpleQA)
    # ──────────────────────────────────────────────

    def factual_accuracy(
        self,
        question: str,
        reference_answer: str,
        system_answer: str,
    ) -> Dict[str, Any]:
        """Returns {"correct": bool, "reason": str, "raw": str}"""
        prompt = (
            f"Question: {question}\n\n"
            f"Reference answer: {reference_answer}\n\n"
            f"System answer: {system_answer}"
        )
        raw = self._call(_SYS_FACTUAL, prompt)
        parsed = _parse_json(raw)
        return {
            "correct": bool(parsed.get("correct", False)),
            "reason":  parsed.get("reason", ""),
            "raw":     raw,
        }

    # ──────────────────────────────────────────────
    # 2. Writing quality (EQ-Bench)
    # ──────────────────────────────────────────────

    def writing_quality(
        self,
        turn_prompt: str,
        prior_summary: str,
        generated_text: str,
        turn_idx: int,
    ) -> Dict[str, Any]:
        """Returns Likert dimension scores + total (0-40)."""
        prompt = (
            f"Turn {turn_idx+1}/8\n\n"
            f"Prior story summary: {prior_summary or '(First turn, no prior)'}\n\n"
            f"This turn's writing prompt: {turn_prompt}\n\n"
            f"Generated text:\n{generated_text}"
        )
        raw = self._call(_SYS_WRITING, prompt)
        parsed = _parse_json(raw)
        return {
            "narrative_quality":      int(parsed.get("narrative_quality", 0)),
            "character_depth":        int(parsed.get("character_depth", 0)),
            "structural_consistency": int(parsed.get("structural_consistency", 0)),
            "emotional_resonance":    int(parsed.get("emotional_resonance", 0)),
            "total":                  int(parsed.get("total", 0)),
            "notes":                  parsed.get("notes", ""),
            "raw":                    raw,
        }

    # ──────────────────────────────────────────────
    # 3. Ablation quality (AbGen)
    # ──────────────────────────────────────────────

    def ablation_quality(
        self,
        system: str,
        hypothesis: str,
        context: str,
        generated_ablation: str,
        expected_ablations: List[str],
    ) -> Dict[str, Any]:
        """Returns Likert scores (1-5 each) + total (4-20)."""
        expected_block = "\n".join(f"  - {a}" for a in expected_ablations)
        prompt = (
            f"System: {system}\n"
            f"Hypothesis: {hypothesis}\n\n"
            f"Context given to the system (may contain noise):\n{context}\n\n"
            f"Gold-standard ablations (for reference):\n{expected_block}\n\n"
            f"Generated ablation study:\n{generated_ablation}"
        )
        raw = self._call(_SYS_ABLATION, prompt)
        parsed = _parse_json(raw)
        return {
            "variable_control":  int(parsed.get("variable_control",  0)),
            "completeness":      int(parsed.get("completeness",      0)),
            "logical_soundness": int(parsed.get("logical_soundness", 0)),
            "noise_exclusion":   int(parsed.get("noise_exclusion",   0)),
            "total":             int(parsed.get("total", 0)),
            "notes":             parsed.get("notes", ""),
            "raw":               raw,
        }

    # ──────────────────────────────────────────────
    # 4. Noise resistance (SEAL-0)
    # ──────────────────────────────────────────────

    def noise_resistance(
        self,
        question: str,
        all_sources: List[str],
        misleading_indices: List[int],
        system_answer: str,
    ) -> Dict[str, Any]:
        """
        Measures how much misleading evidence influenced the answer.
        Returns resistance_score ∈ [0, 1]  (1 = fully resistant).
        """
        sources_block = "\n".join(
            f"[{'MISLEADING' if i in misleading_indices else 'REAL'} Source {i+1}] {s}"
            for i, s in enumerate(all_sources)
        )
        prompt = (
            f"Question: {question}\n\n"
            f"All sources:\n{sources_block}\n\n"
            f"System answer: {system_answer}\n\n"
            f"Total misleading sources: {len(misleading_indices)}"
        )
        raw = self._call(_SYS_NOISE_RESISTANCE, prompt)
        parsed = _parse_json(raw)
        return {
            "misleading_sources_cited": int(parsed.get("misleading_sources_cited", 0)),
            "total_misleading":         int(parsed.get("total_misleading", len(misleading_indices))),
            "resistance_score":         float(parsed.get("resistance_score", 1.0)),
            "explanation":              parsed.get("explanation", ""),
            "raw":                      raw,
        }

    # ──────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────

    def _call(self, system: str, user: str) -> str:
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    response_format={"type": "json_object"},
                )
                return resp.choices[0].message.content or "{}"
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return json.dumps({"error": str(e)})
                time.sleep(2 ** attempt)
        return "{}"


def _parse_json(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except Exception:
        # Try extracting JSON from markdown code block
        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {}
