"""
eval/metrics/collector.py

Aggregates per-question results into benchmark-level statistics.

Three core capability axes (from the EVABot paper):
  1. Anti-interference  : local_view_accuracy vs. global_view_accuracy
                         + noise_resistance_score (SEAL-0 only)
  2. Escalate efficacy  : escalate_trigger_rate + win_rate_after_escalate
  3. Evolution delta    : pre_evolution_accuracy vs. post_evolution_accuracy

All metrics are persisted to JSON and summarised as a flat dict for the
HTML report generator.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional


class MetricsCollector:
    """
    Stateful collector.  Call .record() for each evaluated question,
    then .summary() / .save() at the end.
    """

    def __init__(self, benchmark: str, results_dir: str):
        self.benchmark   = benchmark
        self.results_dir = results_dir
        self.records: List[Dict[str, Any]] = []

    # ──────────────────────────────────────────────
    # Recording
    # ──────────────────────────────────────────────

    def record(
        self,
        sample_id: str,
        *,
        # EVABot (local-view) metrics
        evabot_answer:          str,
        evabot_correct:         bool,
        evabot_cost:            float,
        evabot_latency_s:       float,
        evabot_escalate_count:  int,
        evabot_escalate_resolved: int,
        evabot_evolution_triggered: bool,
        # Baseline (global-view) metrics
        baseline_answer:        str,
        baseline_correct:       bool,
        baseline_cost:          float,
        baseline_latency_s:     float,
        # Noise resistance (SEAL-0 specific, optional)
        evabot_noise_resistance:    float = 1.0,
        baseline_noise_resistance:  float = 1.0,
        # Writing quality (EQ-Bench specific, optional)
        evabot_writing_score:   Optional[float] = None,
        baseline_writing_score: Optional[float] = None,
        # Ablation quality (AbGen specific, optional)
        evabot_ablation_score:  Optional[float] = None,
        baseline_ablation_score: Optional[float] = None,
        # Evolution experiment (second pass, optional)
        post_evolution_correct: Optional[bool] = None,
        # Extra metadata
        extra: Optional[Dict] = None,
    ) -> None:
        self.records.append({
            "sample_id":   sample_id,
            "timestamp":   time.time(),
            # ── EVABot ──
            "evabot": {
                "answer":             evabot_answer,
                "correct":            evabot_correct,
                "cost":               evabot_cost,
                "latency_s":          evabot_latency_s,
                "escalate_count":     evabot_escalate_count,
                "escalate_resolved":  evabot_escalate_resolved,
                "evolution_triggered": evabot_evolution_triggered,
                "noise_resistance":   evabot_noise_resistance,
                "writing_score":      evabot_writing_score,
                "ablation_score":     evabot_ablation_score,
            },
            # ── Baseline ──
            "baseline": {
                "answer":             baseline_answer,
                "correct":            baseline_correct,
                "cost":               baseline_cost,
                "latency_s":          baseline_latency_s,
                "noise_resistance":   baseline_noise_resistance,
                "writing_score":      baseline_writing_score,
                "ablation_score":     baseline_ablation_score,
            },
            # ── Evolution ──
            "post_evolution_correct": post_evolution_correct,
            "extra": extra or {},
        })

    # ──────────────────────────────────────────────
    # Aggregation
    # ──────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Compute benchmark-level aggregate statistics."""
        n = len(self.records)
        if n == 0:
            return {"benchmark": self.benchmark, "n": 0}

        eva = [r["evabot"]    for r in self.records]
        bas = [r["baseline"]  for r in self.records]

        # ── Accuracy ──
        eva_acc  = _mean([e["correct"]  for e in eva])
        bas_acc  = _mean([b["correct"]  for b in bas])
        acc_delta = round(eva_acc - bas_acc, 4)

        # ── Noise resistance (SEAL-0) ──
        eva_nr  = _mean([e["noise_resistance"] for e in eva])
        bas_nr  = _mean([b["noise_resistance"] for b in bas])

        # ── Escalate metrics ──
        total_escalations = sum(e["escalate_count"]    for e in eva)
        total_resolved    = sum(e["escalate_resolved"] for e in eva)
        escalate_rate     = total_escalations / n                           # avg per question
        win_rate_after    = (
            total_resolved / total_escalations if total_escalations > 0 else 0.0
        )

        # ── Evolution ──
        evolution_triggered = sum(1 for e in eva if e["evolution_triggered"])
        post_evo_records = [
            r for r in self.records if r["post_evolution_correct"] is not None
        ]
        if post_evo_records:
            pre_acc  = _mean([r["evabot"]["correct"]        for r in post_evo_records])
            post_acc = _mean([r["post_evolution_correct"]   for r in post_evo_records])
            evo_delta = round(post_acc - pre_acc, 4)
        else:
            pre_acc = post_acc = evo_delta = None

        # ── Cost & Latency ──
        eva_cost  = sum(e["cost"]      for e in eva)
        bas_cost  = sum(b["cost"]      for b in bas)
        eva_lat   = _mean([e["latency_s"]  for e in eva])
        bas_lat   = _mean([b["latency_s"]  for b in bas])

        # ── Writing / Ablation scores (optional benchmarks) ──
        eva_writing  = _mean([e["writing_score"]  for e in eva  if e["writing_score"]  is not None]) or None
        bas_writing  = _mean([b["writing_score"]  for b in bas  if b["writing_score"]  is not None]) or None
        eva_ablation = _mean([e["ablation_score"] for e in eva  if e["ablation_score"] is not None]) or None
        bas_ablation = _mean([b["ablation_score"] for b in bas  if b["ablation_score"] is not None]) or None

        return {
            "benchmark":            self.benchmark,
            "n":                    n,
            # Capability 1: Anti-interference
            "evabot_accuracy":      round(eva_acc,  4),
            "baseline_accuracy":    round(bas_acc,  4),
            "accuracy_delta":       acc_delta,   # positive = EVABot wins
            "evabot_noise_resistance":   round(eva_nr, 4),
            "baseline_noise_resistance": round(bas_nr, 4),
            # Capability 2: Escalate
            "total_escalations":    total_escalations,
            "escalate_rate_per_q":  round(escalate_rate, 4),
            "escalate_win_rate":    round(win_rate_after, 4),
            "evolution_triggered_count": evolution_triggered,
            # Capability 3: Evolution
            "pre_evolution_accuracy":  pre_acc,
            "post_evolution_accuracy": post_acc,
            "evolution_delta":         evo_delta,
            # Cost / Latency
            "evabot_total_cost_usd":  round(eva_cost, 6),
            "baseline_total_cost_usd": round(bas_cost, 6),
            "evabot_avg_latency_s":   round(eva_lat, 2),
            "baseline_avg_latency_s": round(bas_lat, 2),
            # Optional benchmark-specific scores
            "evabot_writing_score":   eva_writing,
            "baseline_writing_score": bas_writing,
            "evabot_ablation_score":  eva_ablation,
            "baseline_ablation_score": bas_ablation,
        }

    # ──────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────

    def save(self) -> str:
        """Saves raw records + summary to JSON. Returns file path."""
        fname = f"{self.benchmark}_{int(time.time())}.json"
        path  = os.path.join(self.results_dir, fname)
        data  = {
            "summary": self.summary(),
            "records": self.records,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "MetricsCollector":
        """Restore a collector from a saved JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        benchmark = data["summary"]["benchmark"]
        results_dir = os.path.dirname(path)
        collector = cls(benchmark, results_dir)
        collector.records = data.get("records", [])
        return collector


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _mean(values) -> float:
    vals = [v for v in values if v is not None]
    if not vals:
        return 0.0
    return round(sum(vals) / len(vals), 4)
