"""
eval/run_eval.py
────────────────────────────────────────────────────────────
EVABot Capability Evaluation Orchestrator
Adapted from ROMA benchmarks (arXiv:2602.01848)

Usage
-----
# Run all benchmarks (uses built-in sample limits from config.py)
python -m eval.run_eval

# Run a single benchmark
python -m eval.run_eval --benchmarks seal0 frames

# Enable evolution loop (pre/post skill-evolution measurement)
python -m eval.run_eval --evolution

# Override sample limits
python -m eval.run_eval --seal0-limit 10 --frames-limit 20

# Use a specific EVABot server
python -m eval.run_eval --evabot-url http://localhost:8000

Capabilities measured
---------------------
1. Anti-interference  : EVABot local-view accuracy vs. global-view baseline
2. Escalate efficacy  : trigger rate + win-rate-after-escalate
3. Evolution delta    : accuracy gain after Auditor-driven skill evolution
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

# ── ensure project root is on PYTHONPATH ──
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from eval.config import (
    RESULTS_DIR,
    SEAL0_LIMIT,
    FRAMES_LIMIT,
    SIMPLEQA_LIMIT,
    EQBENCH_LIMIT,
    ABGEN_LIMIT,
    ENABLE_EVOLUTION_LOOP,
    EVABOT_BASE_URL,
    EVABOT_WS_URL,
)
from eval.datasets import (
    load_seal0, load_frames, load_simpleqa,
    load_eqbench, load_abgen,
)
from eval.runners  import EvaBotRunner, BaselineRunner
from eval.judge    import LLMJudge
from eval.metrics  import MetricsCollector
from eval.report   import generate_report


# ─────────────────────────────────────────────────────────────
# Benchmark runners
# ─────────────────────────────────────────────────────────────

def run_seal0(
    eva: EvaBotRunner,
    bas: BaselineRunner,
    judge: LLMJudge,
    limit: int,
    evolution: bool,
) -> MetricsCollector:
    """
    SEAL-0: Reasoning over noisy and conflicting web evidence.

    Capability 1 (Anti-interference):
      EVABot sees sources one at a time via isolated worker nodes.
      Baseline dumps ALL sources (including misleading ones) in one prompt.

    Capability 2 (Escalate):
      When a worker can't reconcile contradicting sources, it escalates
      upward for guidance rather than guessing.

    Capability 3 (Evolution):
      After initial run, failed items trigger skill creation; re-tested.
    """
    print(f"\n{'='*60}")
    print(f"  SEAL-0  (n={limit})")
    print(f"{'='*60}")

    samples = load_seal0(limit)
    mc = MetricsCollector("SEAL-0", RESULTS_DIR)

    for i, sample in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {sample['question'][:60]}...")

        # ── EVABot (local view) ──
        sources_text = "\n\n".join(
            f"[Source {j+1}]: {s}" for j, s in enumerate(sample["sources"])
        )
        context_hint = (
            f"You have the following web evidence to consider:\n\n{sources_text}"
        )
        eva_result = eva.run(sample["question"], context_hint=context_hint)

        # ── Baseline (global view) ──
        bas_result = bas.run_seal0(sample)

        # ── Judge: factual accuracy ──
        eva_judge = judge.factual_accuracy(
            sample["question"], sample["answer"], eva_result["final_answer"]
        )
        bas_judge = judge.factual_accuracy(
            sample["question"], sample["answer"], bas_result["final_answer"]
        )

        # ── Judge: noise resistance ──
        eva_nr = judge.noise_resistance(
            sample["question"], sample["sources"],
            sample["misleading_indices"], eva_result["final_answer"]
        )
        bas_nr = judge.noise_resistance(
            sample["question"], sample["sources"],
            sample["misleading_indices"], bas_result["final_answer"]
        )

        # ── Evolution (optional second pass on failures) ──
        post_correct = None
        if evolution and not eva_judge["correct"] and eva_result.get("evolution_triggered"):
            print(f"    ↻ Re-testing after evolution…")
            time.sleep(2)
            eva2 = eva.run(sample["question"], context_hint=context_hint)
            j2   = judge.factual_accuracy(sample["question"], sample["answer"], eva2["final_answer"])
            post_correct = j2["correct"]

        mc.record(
            sample["id"],
            evabot_answer=eva_result["final_answer"],
            evabot_correct=eva_judge["correct"],
            evabot_cost=eva_result["cost"],
            evabot_latency_s=eva_result["latency_s"],
            evabot_escalate_count=eva_result["escalate_count"],
            evabot_escalate_resolved=eva_result["escalate_resolved"],
            evabot_evolution_triggered=eva_result["evolution_triggered"],
            evabot_noise_resistance=eva_nr["resistance_score"],
            baseline_answer=bas_result["final_answer"],
            baseline_correct=bas_judge["correct"],
            baseline_cost=bas_result["cost"],
            baseline_latency_s=bas_result["latency_s"],
            baseline_noise_resistance=bas_nr["resistance_score"],
            post_evolution_correct=post_correct,
            extra={
                "eva_judge_reason":  eva_judge["reason"],
                "bas_judge_reason":  bas_judge["reason"],
                "misleading_indices": sample["misleading_indices"],
            },
        )
        _print_row(eva_judge["correct"], bas_judge["correct"],
                   eva_result["escalate_count"], eva_result["evolution_triggered"])

    _print_summary(mc)
    return mc


def run_frames(
    eva: EvaBotRunner,
    bas: BaselineRunner,
    judge: LLMJudge,
    limit: int,
    evolution: bool,
) -> MetricsCollector:
    """
    FRAMES: Multi-hop factual reasoning across Wikipedia pages.

    Capability 1: EVABot breaks each hop into an isolated sub-query.
    Capability 2: Missing hop information triggers Escalate.
    Capability 3: Failed hop strategies evolve into specialised search skills.
    """
    print(f"\n{'='*60}")
    print(f"  FRAMES  (n={limit})")
    print(f"{'='*60}")

    samples = load_frames(limit)
    mc = MetricsCollector("FRAMES", RESULTS_DIR)

    for i, sample in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {sample['question'][:60]}…  ({sample['hops']} hops)")

        # Context hint: tell EVABot about required Wikipedia sources
        context_hint = ""
        if sample.get("wiki_titles"):
            titles = ", ".join(sample["wiki_titles"])
            context_hint = (
                f"This question requires integrating information from multiple sources. "
                f"Relevant Wikipedia articles: {titles}"
            )

        eva_result = eva.run(sample["question"], context_hint=context_hint)
        bas_result = bas.run_frames(sample)

        eva_judge = judge.factual_accuracy(
            sample["question"], sample["answer"], eva_result["final_answer"]
        )
        bas_judge = judge.factual_accuracy(
            sample["question"], sample["answer"], bas_result["final_answer"]
        )

        post_correct = None
        if evolution and not eva_judge["correct"] and eva_result.get("evolution_triggered"):
            time.sleep(2)
            eva2 = eva.run(sample["question"], context_hint=context_hint)
            post_correct = judge.factual_accuracy(
                sample["question"], sample["answer"], eva2["final_answer"]
            )["correct"]

        mc.record(
            sample["id"],
            evabot_answer=eva_result["final_answer"],
            evabot_correct=eva_judge["correct"],
            evabot_cost=eva_result["cost"],
            evabot_latency_s=eva_result["latency_s"],
            evabot_escalate_count=eva_result["escalate_count"],
            evabot_escalate_resolved=eva_result["escalate_resolved"],
            evabot_evolution_triggered=eva_result["evolution_triggered"],
            baseline_answer=bas_result["final_answer"],
            baseline_correct=bas_judge["correct"],
            baseline_cost=bas_result["cost"],
            baseline_latency_s=bas_result["latency_s"],
            post_evolution_correct=post_correct,
            extra={"hops": sample["hops"], "wiki_titles": sample.get("wiki_titles", [])},
        )
        _print_row(eva_judge["correct"], bas_judge["correct"],
                   eva_result["escalate_count"], eva_result["evolution_triggered"])

    _print_summary(mc)
    return mc


def run_simpleqa(
    eva: EvaBotRunner,
    bas: BaselineRunner,
    judge: LLMJudge,
    limit: int,
    evolution: bool,
) -> MetricsCollector:
    """
    SimpleQA: Precise single-fact retrieval.

    Capability 1: EVABot's minimal-context node focuses exclusively on
                  the target entity, blocking near-miss distractors.
    Capability 2: When two plausible dates/names are found, Escalate
                  triggers cross-source alignment instead of guessing.
    Capability 3: Repeated failure on cold entities evolves a precision
                  fact-extraction skill.
    """
    print(f"\n{'='*60}")
    print(f"  SimpleQA  (n={limit})")
    print(f"{'='*60}")

    samples = load_simpleqa(limit)
    mc = MetricsCollector("SimpleQA", RESULTS_DIR)

    for i, sample in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {sample['question'][:60]}…")

        eva_result = eva.run(sample["question"])
        bas_result = bas.run_simpleqa(sample)

        eva_judge = judge.factual_accuracy(
            sample["question"], sample["answer"], eva_result["final_answer"]
        )
        bas_judge = judge.factual_accuracy(
            sample["question"], sample["answer"], bas_result["final_answer"]
        )

        post_correct = None
        if evolution and not eva_judge["correct"] and eva_result.get("evolution_triggered"):
            time.sleep(2)
            eva2 = eva.run(sample["question"])
            post_correct = judge.factual_accuracy(
                sample["question"], sample["answer"], eva2["final_answer"]
            )["correct"]

        mc.record(
            sample["id"],
            evabot_answer=eva_result["final_answer"],
            evabot_correct=eva_judge["correct"],
            evabot_cost=eva_result["cost"],
            evabot_latency_s=eva_result["latency_s"],
            evabot_escalate_count=eva_result["escalate_count"],
            evabot_escalate_resolved=eva_result["escalate_resolved"],
            evabot_evolution_triggered=eva_result["evolution_triggered"],
            baseline_answer=bas_result["final_answer"],
            baseline_correct=bas_judge["correct"],
            baseline_cost=bas_result["cost"],
            baseline_latency_s=bas_result["latency_s"],
            post_evolution_correct=post_correct,
            extra={"topic": sample.get("topic", "")},
        )
        _print_row(eva_judge["correct"], bas_judge["correct"],
                   eva_result["escalate_count"], eva_result["evolution_triggered"])

    _print_summary(mc)
    return mc


def run_eqbench(
    eva: EvaBotRunner,
    bas: BaselineRunner,
    judge: LLMJudge,
    limit: int,
) -> MetricsCollector:
    """
    EQ-Bench: Long-form narrative writing over 8 turns.

    Capability 1: EVABot's top node holds the plot outline; each chapter
                  node sees only the summary of the previous chapter.
    Capability 2: If a chapter contradicts the global outline, Escalate
                  fires to the 'creative director' node.
    Capability 3: Mid-story pacing failures trigger evolution of a
                  narrative-transition skill.
    """
    print(f"\n{'='*60}")
    print(f"  EQ-Bench  (n={limit} stories)")
    print(f"{'='*60}")

    samples = load_eqbench(limit)
    mc = MetricsCollector("EQ-Bench", RESULTS_DIR)

    for i, sample in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] '{sample['title']}' ({sample['genre']})")

        # ── EVABot: submit full story task with all 8 turn prompts ──
        story_prompt = (
            f"Write a complete 8-chapter story with these details:\n\n"
            f"Title: {sample['title']}\n"
            f"Genre: {sample['genre']}\n"
            f"Characters: {sample['character_brief']}\n\n"
            f"Plot outline (8 turns, ~1000 words each):\n{sample['plot_outline']}\n\n"
            f"Turn-by-turn writing prompts:\n"
            + "\n".join(f"Turn {j+1}: {t}" for j, t in enumerate(sample["turns"]))
        )
        eva_result = eva.run(story_prompt)

        # ── Baseline: 8 sequential LLM calls with full history ──
        bas_history   = ""
        bas_parts     = []
        bas_total_cost    = 0.0
        bas_total_latency = 0.0
        bas_scores: List[dict] = []
        for t_idx in range(len(sample["turns"])):
            br = bas.run_eqbench(sample, t_idx, bas_history)
            bas_parts.append(br["final_answer"])
            bas_total_cost    += br["cost"]
            bas_total_latency += br["latency_s"]
            bas_history       += "\n\n" + br["final_answer"]

            # Judge each turn
            prior_summary = f"Prior {t_idx} turns written" if t_idx > 0 else ""
            bj = judge.writing_quality(
                sample["turns"][t_idx], prior_summary,
                br["final_answer"], t_idx
            )
            bas_scores.append(bj)

        bas_avg_score = sum(b["total"] for b in bas_scores) / len(bas_scores) if bas_scores else 0

        # ── Judge EVABot output ──
        # Split the response into approximate turns by paragraph blocks
        eva_text = eva_result["final_answer"]
        eva_avg_score = 0.0
        eva_scores: List[dict] = []
        # Judge the whole story as one block (simplified for EVABot)
        for t_idx, turn_prompt in enumerate(sample["turns"]):
            ej = judge.writing_quality(
                turn_prompt, "See prior context in EVABot's isolated plan",
                eva_text, t_idx
            )
            eva_scores.append(ej)
        eva_avg_score = sum(e["total"] for e in eva_scores) / len(eva_scores) if eva_scores else 0

        mc.record(
            sample["id"],
            evabot_answer=eva_result["final_answer"][:500],
            evabot_correct=eva_avg_score > bas_avg_score,
            evabot_cost=eva_result["cost"],
            evabot_latency_s=eva_result["latency_s"],
            evabot_escalate_count=eva_result["escalate_count"],
            evabot_escalate_resolved=eva_result["escalate_resolved"],
            evabot_evolution_triggered=eva_result["evolution_triggered"],
            evabot_writing_score=eva_avg_score,
            baseline_answer=bas_history[:500],
            baseline_correct=bas_avg_score > eva_avg_score,
            baseline_cost=bas_total_cost,
            baseline_latency_s=bas_total_latency,
            baseline_writing_score=bas_avg_score,
            extra={"title": sample["title"], "genre": sample["genre"]},
        )

        print(f"    EVABot writing: {eva_avg_score:.1f}/40  |  Baseline: {bas_avg_score:.1f}/40  |"
              f"  escalate={eva_result['escalate_count']}")

    _print_summary(mc)
    return mc


def run_abgen(
    eva: EvaBotRunner,
    bas: BaselineRunner,
    judge: LLMJudge,
    limit: int,
) -> MetricsCollector:
    """
    AbGen: Rigorous ablation study design.

    Capability 1: EVABot isolates the module under ablation; irrelevant
                  context (office gossip etc.) is blocked.
    Capability 2: If a local ablation plan conflicts with global system
                  constraints, Escalate fires to the 'chief scientist' node.
    Capability 3: Incomplete ablation plans trigger evolution of a
                  domain-specific ablation methodology skill.
    """
    print(f"\n{'='*60}")
    print(f"  AbGen  (n={limit})")
    print(f"{'='*60}")

    samples = load_abgen(limit)
    mc = MetricsCollector("AbGen", RESULTS_DIR)

    for i, sample in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {sample['system'][:55]}…")

        eva_prompt = (
            f"System to ablate: {sample['system']}\n"
            f"Hypothesis to test: {sample['hypothesis']}\n"
            f"Background context: {sample['context']}\n\n"
            f"Design a rigorous ablation study for this hypothesis. "
            f"For each ablation, specify exactly ONE variable changed and what is held constant."
        )
        eva_result = eva.run(eva_prompt)
        bas_result = bas.run_abgen(sample)

        eva_judge = judge.ablation_quality(
            sample["system"], sample["hypothesis"], sample["context"],
            eva_result["final_answer"], sample["expected_ablations"]
        )
        bas_judge = judge.ablation_quality(
            sample["system"], sample["hypothesis"], sample["context"],
            bas_result["final_answer"], sample["expected_ablations"]
        )

        mc.record(
            sample["id"],
            evabot_answer=eva_result["final_answer"],
            evabot_correct=eva_judge["total"] > bas_judge["total"],
            evabot_cost=eva_result["cost"],
            evabot_latency_s=eva_result["latency_s"],
            evabot_escalate_count=eva_result["escalate_count"],
            evabot_escalate_resolved=eva_result["escalate_resolved"],
            evabot_evolution_triggered=eva_result["evolution_triggered"],
            evabot_ablation_score=float(eva_judge["total"]),
            baseline_answer=bas_result["final_answer"],
            baseline_correct=bas_judge["total"] > eva_judge["total"],
            baseline_cost=bas_result["cost"],
            baseline_latency_s=bas_result["latency_s"],
            baseline_ablation_score=float(bas_judge["total"]),
            extra={
                "eva_scores":  {k: v for k, v in eva_judge.items() if k != "raw"},
                "bas_scores":  {k: v for k, v in bas_judge.items() if k != "raw"},
            },
        )

        print(f"    EVABot ablation: {eva_judge['total']}/20  |  Baseline: {bas_judge['total']}/20  |"
              f"  escalate={eva_result['escalate_count']}")

    _print_summary(mc)
    return mc


# ─────────────────────────────────────────────────────────────
# CLI helpers
# ─────────────────────────────────────────────────────────────

def _print_row(eva_ok: bool, bas_ok: bool, esc: int, evo: bool):
    eva_s = "✓" if eva_ok  else "✗"
    bas_s = "✓" if bas_ok  else "✗"
    esc_s = f"↑{esc}" if esc > 0 else "—"
    evo_s = "🔄" if evo else ""
    print(f"    EVABot:{eva_s}  Baseline:{bas_s}  Esc:{esc_s}  {evo_s}")


def _print_summary(mc: MetricsCollector):
    s = mc.summary()
    print(f"\n  ── {mc.benchmark} Summary ──")
    print(f"  EVABot accuracy : {s['evabot_accuracy']*100:.1f}%")
    print(f"  Baseline accuracy: {s['baseline_accuracy']*100:.1f}%")
    print(f"  Δ accuracy       : {s['accuracy_delta']*100:+.1f}pp")
    print(f"  Escalate rate    : {s['escalate_rate_per_q']:.2f}/q  win={s['escalate_win_rate']*100:.0f}%")
    if s.get("evolution_delta") is not None:
        print(f"  Evolution Δ      : {s['evolution_delta']*100:+.1f}pp")
    saved = mc.save()
    print(f"  Results saved → {saved}\n")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EVABot Capability Evaluation (ROMA benchmarks)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--benchmarks", nargs="+",
        choices=["seal0", "frames", "simpleqa", "eqbench", "abgen"],
        default=["seal0", "frames", "simpleqa", "eqbench", "abgen"],
        help="Which benchmarks to run (default: all)",
    )
    parser.add_argument("--evolution",     action="store_true",
                        help="Enable pre/post evolution measurement")
    parser.add_argument("--seal0-limit",   type=int, default=SEAL0_LIMIT)
    parser.add_argument("--frames-limit",  type=int, default=FRAMES_LIMIT)
    parser.add_argument("--simpleqa-limit",type=int, default=SIMPLEQA_LIMIT)
    parser.add_argument("--eqbench-limit", type=int, default=EQBENCH_LIMIT)
    parser.add_argument("--abgen-limit",   type=int, default=ABGEN_LIMIT)
    parser.add_argument("--evabot-url",    default=EVABOT_BASE_URL)
    parser.add_argument("--evabot-ws-url", default=EVABOT_WS_URL)
    args = parser.parse_args()

    # ── Initialise shared components ──
    eva   = EvaBotRunner(base_url=args.evabot_url, ws_url=args.evabot_ws_url)
    bas   = BaselineRunner()
    judge = LLMJudge()

    # ── Server check ──
    if not eva.health_check():
        print(
            f"\n⚠️  EVABot server not reachable at {args.evabot_url}\n"
            "   Start the server with:  python run.py\n"
            "   The evaluation will proceed using offline simulation mode.\n"
        )

    collectors: List[MetricsCollector] = []
    t0 = time.time()

    bm_set = set(args.benchmarks)

    if "seal0"    in bm_set:
        collectors.append(run_seal0(eva, bas, judge, args.seal0_limit, args.evolution))
    if "frames"   in bm_set:
        collectors.append(run_frames(eva, bas, judge, args.frames_limit, args.evolution))
    if "simpleqa" in bm_set:
        collectors.append(run_simpleqa(eva, bas, judge, args.simpleqa_limit, args.evolution))
    if "eqbench"  in bm_set:
        collectors.append(run_eqbench(eva, bas, judge, args.eqbench_limit))
    if "abgen"    in bm_set:
        collectors.append(run_abgen(eva, bas, judge, args.abgen_limit))

    # ── Generate report ──
    if collectors:
        html_path = generate_report(collectors, RESULTS_DIR)
        print(f"\n{'='*60}")
        print(f"  EVALUATION COMPLETE  ({time.time()-t0:.0f}s)")
        print(f"  Report → {html_path}")
        print(f"{'='*60}\n")
    else:
        print("No benchmarks ran.")


if __name__ == "__main__":
    main()
