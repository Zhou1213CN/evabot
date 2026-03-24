"""
Experiment 1: SEAL-0 — Hierarchical Isolation (Local View) vs. Flat Context (Global View)

Description:
    This experiment evaluates whether hierarchical context isolation (as used in
    EVABot's skill-tree architecture) outperforms a flat, single-context approach
    when reasoning over contradictory web evidence.

    SEAL-0 is a benchmark of 111 questions where web search results are known to
    be conflicting. The dataset is designed to test an agent's ability to reason
    correctly despite misleading sources.

    We compare two approaches:
      - EVABot (Local View): Each source is evaluated independently by an isolated
        Worker (minimal context). An Aggregator then synthesizes Worker summaries
        to produce the final answer. This mirrors EVABot's three-layer isolation:
        Butler -> Solver -> Worker, where each Worker only sees the context
        relevant to its assigned skill.
      - Baseline (Global View): All sources are concatenated into a single prompt
        and fed to the LLM in one call. This simulates traditional flat-context
        architectures (e.g., vanilla RAG).

Dataset:
    HuggingFace: vtllms/sealqa (seal_0 split, 111 questions)
    Each question has search_results="conflicting", meaning web evidence is
    intentionally contradictory.

Metrics:
    - Accuracy: Fraction of questions answered correctly (judged by LLM).
    - Noise Resistance: Score 0.0-1.0 measuring how well the system ignores
      misleading sources (1.0 = fully resistant).
    - Compression Ratio: Ratio of Worker summary tokens to original source tokens,
      demonstrating information distillation without loss of core logic.
    - Latency: Wall-clock time per question for each approach.

Usage:
    # Install dependencies
    pip install datasets openai

    # Quick test with 5 questions
    python eval/exp1_seal0.py --limit 5

    # Run full benchmark (all 111 questions)
    python eval/exp1_seal0.py

    # Results are saved to eval/results/exp1_seal0_<timestamp>.json
"""
import argparse
import json
import os
import time
from typing import Any, Dict, List

from datasets import load_dataset
from openai import OpenAI

# ─── Configuration ────────────────────────────────────────────
API_KEY     = os.environ.get("DEEPSEEK_API_KEY", "sk-8eb9c82ecb6845e7adc115fdf86e9f17")
BASE_URL    = "https://api.deepseek.com"
MODEL       = "deepseek-chat"
JUDGE_MODEL = "deepseek-chat"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ─── Dataset Loading ─────────────────────────────────────────
def load_seal0(limit: int = 0) -> List[Dict]:
    """Load the SEAL-0 dataset from HuggingFace.

    Args:
        limit: Maximum number of questions to load. 0 means load all (111).

    Returns:
        List of dicts, each containing:
            id, question, answer, topic, question_types, urls, search_results.
    """
    print("  Loading vtllms/sealqa (seal_0) from HuggingFace ...")
    ds = load_dataset("vtllms/sealqa", "seal_0")
    data = []
    for i, row in enumerate(ds["test"]):
        if limit and i >= limit:
            break
        data.append({
            "id": str(i + 1),
            "question": row["question"],
            "answer": row["answer"],
            "topic": row.get("topic", ""),
            "question_types": row.get("question_types", []),
            "urls": row.get("urls", []),
            "search_results": row.get("search_results", "conflicting"),
        })
    print(f"  Loaded {len(data)} questions")
    return data


# ─── Conflicting Source Generation ───────────────────────────
def generate_conflicting_sources(question: str, answer: str, topic: str) -> Dict:
    """Generate 5 simulated web search results with conflicting information.

    For each SEAL-0 question, we generate:
      - 2 sources supporting the CORRECT answer (authoritative references)
      - 3 MISLEADING sources providing a wrong but plausible alternative
        (designed to look authoritative with fake citations)

    This simulates the real-world scenario where web searches return
    contradictory evidence, which is the core challenge of SEAL-0.

    Args:
        question: The SEAL-0 question text.
        answer: The gold-standard correct answer.
        topic: Topic category (e.g., "Entertainment", "Sports").

    Returns:
        Dict with keys:
            "sources": List of 5 source strings.
            "noise_indices": [2, 3, 4] indicating which sources are misleading.
    """
    system = (
        "You are a research assistant generating simulated web search results "
        "for an AI evaluation benchmark. You must generate exactly 5 web source "
        "snippets for a given question.\n\n"
        "RULES:\n"
        "- Source 1 and Source 2: MUST support the CORRECT answer. Cite real, "
        "authoritative references (official databases, peer-reviewed sources).\n"
        "- Source 3, 4, and 5: MUST be MISLEADING. They should provide a WRONG "
        "but plausible alternative answer. Make them look authoritative "
        "(cite fake but realistic-sounding publications, use confident language). "
        "The wrong answer across these 3 sources should be CONSISTENT (same wrong claim).\n\n"
        "FORMAT: Return ONLY a JSON array of 5 strings, no other text.\n"
        "Example: [\"Source 1 text...\", \"Source 2 text...\", ...]"
    )
    user = (
        f"Question: {question}\n"
        f"Correct answer: {answer}\n"
        f"Topic area: {topic}\n\n"
        f"Generate 5 sources (2 correct + 3 misleading):"
    )

    result = call_llm(system, user)

    # Parse JSON array from LLM response
    try:
        start = result.find("[")
        end = result.rfind("]") + 1
        if start >= 0 and end > start:
            sources = json.loads(result[start:end])
            if isinstance(sources, list) and len(sources) == 5:
                return {"sources": sources, "noise_indices": [2, 3, 4]}
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback if parsing fails
    return {
        "sources": [
            f"According to authoritative sources, the answer to '{question}' is {answer}.",
            f"Official records confirm: {answer}.",
            f"Recent studies suggest a different answer than commonly believed for: {question}.",
            f"A widely-cited but controversial source claims the answer differs from {answer}.",
            f"Popular media reports suggest an alternative to the commonly accepted answer.",
        ],
        "noise_indices": [2, 3, 4],
    }


# ─── LLM Wrapper ─────────────────────────────────────────────
def call_llm(system: str, user: str, model: str = MODEL) -> str:
    """Make a single LLM API call and return the response text.

    Args:
        system: System prompt.
        user: User message.
        model: Model identifier (default: deepseek-chat).

    Returns:
        The assistant's response as a string.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


# ─── Baseline: Global View (Flat Context) ────────────────────
def baseline_global_view(question: str, sources: List[str]) -> str:
    """Baseline approach: feed ALL sources into a single LLM call.

    This simulates traditional flat-context architectures where the model
    must reason over all evidence (including misleading noise) simultaneously
    within one context window.

    Args:
        question: The question to answer.
        sources: List of all source texts (including misleading ones).

    Returns:
        The model's answer as a string.
    """
    sources_block = "\n\n".join(
        f"[Source {i+1}]: {s}" for i, s in enumerate(sources)
    )
    system = (
        "You are an accurate fact-checking assistant. "
        "You will receive a question and several web sources that may contain "
        "conflicting or misleading information. Read ALL sources and answer "
        "the question. Reply with ONLY the factual answer, nothing else."
    )
    user = f"Question: {question}\n\nRetrieved sources:\n{sources_block}"
    return call_llm(system, user)


# ─── EVABot: Local View (Hierarchical Isolation) ─────────────
def evabot_local_view(question: str, sources: List[str]) -> Dict[str, Any]:
    """EVABot approach: hierarchical isolation with per-source Workers.

    Simulates EVABot's three-layer skill-tree architecture:

    Layer 1 - Workers (Isolated Execution):
        Each source is evaluated independently by a separate Worker.
        Workers only see ONE source + the question (minimal context).
        Workers output: key factual claim + credibility assessment.
        This prevents cross-contamination from misleading sources.

    Layer 2 - Aggregator (Solver):
        Receives ONLY the condensed Worker summaries (not raw sources).
        Resolves conflicts using credibility signals.
        Produces the final answer.

    This mirrors EVABot's real architecture where:
        - Butler dispatches tasks via the skill tree
        - Solver decomposes into sub-tasks
        - Workers execute with isolated, minimal context
        - Results flow upward as distilled summaries

    Args:
        question: The question to answer.
        sources: List of all source texts.

    Returns:
        Dict with keys:
            "final_answer": The aggregated answer string.
            "worker_summaries": List of per-source evaluation details
                (for computing compression ratio).
    """
    # Step 1: Each source evaluated in isolation by a Worker
    worker_system = (
        "You are a source evaluator. You will receive ONE web source and a question. "
        "Your job:\n"
        "1. Extract the key factual claim from this source relevant to the question.\n"
        "2. Assess the credibility of this source (high/medium/low) with a brief reason.\n"
        "Reply in this exact format:\n"
        "CLAIM: <the key claim>\n"
        "CREDIBILITY: <high|medium|low>\n"
        "REASON: <one sentence>"
    )

    worker_summaries = []
    for i, source in enumerate(sources):
        user = f"Question: {question}\n\nSource to evaluate:\n{source}"
        summary = call_llm(worker_system, user)
        worker_summaries.append({
            "source_idx": i,
            "original_tokens": len(source.split()),
            "summary": summary,
            "summary_tokens": len(summary.split()),
        })

    # Step 2: Aggregator synthesizes Worker reports
    summaries_block = "\n\n".join(
        f"[Worker {w['source_idx']+1} Report]:\n{w['summary']}"
        for w in worker_summaries
    )

    aggregator_system = (
        "You are a senior fact-checker. You will receive evaluation reports from "
        "multiple analysts, each of whom reviewed ONE source independently. "
        "Some sources may conflict. Use the credibility assessments to determine "
        "the most reliable answer. Reply with ONLY the factual answer, nothing else."
    )
    aggregator_user = f"Question: {question}\n\nAnalyst reports:\n{summaries_block}"
    final_answer = call_llm(aggregator_system, aggregator_user)

    return {
        "final_answer": final_answer,
        "worker_summaries": worker_summaries,
    }


# ─── LLM Judge: Correctness ──────────────────────────────────
def judge_correctness(question: str, gold_answer: str, system_answer: str) -> bool:
    """Use an LLM to judge whether the system's answer is factually correct.

    Args:
        question: The original question.
        gold_answer: The reference (correct) answer.
        system_answer: The system's produced answer.

    Returns:
        True if the answer is judged correct, False otherwise.
    """
    system = (
        "You are a strict evaluator. Determine if the system's answer is factually "
        "correct compared to the reference answer. Minor wording differences are fine "
        "as long as the core fact matches. Reply with ONLY 'CORRECT' or 'INCORRECT'."
    )
    user = (
        f"Question: {question}\n"
        f"Reference answer: {gold_answer}\n"
        f"System answer: {system_answer}"
    )
    result = call_llm(system, user, model=JUDGE_MODEL)
    return "CORRECT" in result.upper()


# ─── LLM Judge: Noise Resistance ─────────────────────────────
def judge_noise_influence(question: str, sources: List[str],
                          noise_indices: List[int], system_answer: str) -> float:
    """Evaluate how much the system's answer was influenced by misleading sources.

    Args:
        question: The original question.
        sources: All source texts.
        noise_indices: Indices of the misleading sources.
        system_answer: The system's produced answer.

    Returns:
        Noise resistance score from 0.0 (fully misled) to 1.0 (fully resistant).
    """
    noise_claims = [sources[i] for i in noise_indices if i < len(sources)]
    noise_block = "\n".join(f"- {c}" for c in noise_claims)
    system = (
        "You are evaluating whether an AI's answer was influenced by misleading sources. "
        "Score from 0 to 10:\n"
        "  10 = answer completely ignores the misleading information\n"
        "  5  = answer partially influenced\n"
        "  0  = answer directly repeats the misleading claim\n"
        "Reply with ONLY a single integer 0-10."
    )
    user = (
        f"Question: {question}\n"
        f"Misleading sources:\n{noise_block}\n"
        f"System answer: {system_answer}"
    )
    result = call_llm(system, user, model=JUDGE_MODEL)
    try:
        score = int("".join(c for c in result if c.isdigit())[:2])
        return min(score, 10) / 10.0
    except (ValueError, IndexError):
        return 0.5


# ─── Main Experiment Runner ──────────────────────────────────
def run_experiment(limit: int = 0):
    """Run the full SEAL-0 Experiment 1: Local View vs. Global View.

    For each question in the SEAL-0 dataset:
      1. Generate 5 conflicting web sources (2 correct + 3 misleading).
      2. Run the Baseline (global view) and EVABot (local view) approaches.
      3. Judge correctness and noise resistance for both.
      4. Compute compression ratio for EVABot's information distillation.

    Results are printed to stdout and saved as JSON.

    Args:
        limit: Max questions to process. 0 means all 111.
    """
    # Load the real SEAL-0 dataset from HuggingFace
    seal0_data = load_seal0(limit=limit)

    print("=" * 70)
    print("  SEAL-0 Experiment 1: Hierarchical Isolation (Local) vs. Flat (Global)")
    print(f"  Dataset: vtllms/sealqa (seal_0)  |  Model: {MODEL}")
    print(f"  Questions: {len(seal0_data)}")
    print("=" * 70)

    results = []

    for sample in seal0_data:
        qid = sample["id"]
        question = sample["question"]
        answer = sample["answer"]
        topic = sample["topic"]

        print(f"\n[Q{qid}] {question}")
        print(f"  Gold: {answer}  |  Topic: {topic}")

        # Step 0: Generate conflicting sources for this question
        print(f"  Generating conflicting sources ...", end=" ", flush=True)
        source_data = generate_conflicting_sources(question, answer, topic)
        sources = source_data["sources"]
        noise_idx = source_data["noise_indices"]
        print(f"done ({len(sources)} sources, {len(noise_idx)} misleading)")

        # Step 1: Baseline (Global View)
        t0 = time.time()
        bas_answer = baseline_global_view(question, sources)
        bas_time = round(time.time() - t0, 2)

        # Step 2: EVABot (Local View)
        t0 = time.time()
        eva_result = evabot_local_view(question, sources)
        eva_time = round(time.time() - t0, 2)
        eva_answer = eva_result["final_answer"]

        # Step 3: Judge correctness and noise resistance
        bas_correct = judge_correctness(question, answer, bas_answer)
        eva_correct = judge_correctness(question, answer, eva_answer)

        bas_noise = judge_noise_influence(question, sources, noise_idx, bas_answer)
        eva_noise = judge_noise_influence(question, sources, noise_idx, eva_answer)

        # Step 4: Compute compression ratio
        total_src_tokens = sum(len(s.split()) for s in sources)
        total_summary_tokens = sum(w["summary_tokens"] for w in eva_result["worker_summaries"])
        compression = round(total_summary_tokens / max(total_src_tokens, 1), 3)

        result = {
            "id": qid,
            "question": question,
            "gold": answer,
            "topic": topic,
            "question_types": sample["question_types"],
            "sources": sources,
            "noise_indices": noise_idx,
            "baseline": {
                "answer": bas_answer.strip(),
                "correct": bas_correct,
                "noise_resistance": bas_noise,
                "time_s": bas_time,
            },
            "evabot": {
                "answer": eva_answer.strip(),
                "correct": eva_correct,
                "noise_resistance": eva_noise,
                "time_s": eva_time,
                "compression_ratio": compression,
                "worker_summaries": eva_result["worker_summaries"],
            },
        }
        results.append(result)

        # Print per-question results
        e_mark = "PASS" if eva_correct else "FAIL"
        b_mark = "PASS" if bas_correct else "FAIL"
        print(f"  Baseline: {b_mark} ({bas_time}s) noise_resist={bas_noise:.1f}  -> {bas_answer.strip()[:60]}")
        print(f"  EVABot:   {e_mark} ({eva_time}s) noise_resist={eva_noise:.1f}  compress={compression:.2f} -> {eva_answer.strip()[:60]}")

    # ─── Aggregate Results ────────────────────────────────
    n = len(results)
    if n == 0:
        print("No data to process. Exiting.")
        return

    eva_acc   = sum(1 for r in results if r["evabot"]["correct"]) / n
    bas_acc   = sum(1 for r in results if r["baseline"]["correct"]) / n
    eva_nr    = sum(r["evabot"]["noise_resistance"] for r in results) / n
    bas_nr    = sum(r["baseline"]["noise_resistance"] for r in results) / n
    eva_comp  = sum(r["evabot"]["compression_ratio"] for r in results) / n
    delta_acc = eva_acc - bas_acc

    # ─── Per-Topic Breakdown ──────────────────────────────
    topics = {}
    for r in results:
        t = r["topic"] or "Unknown"
        if t not in topics:
            topics[t] = {"eva_correct": 0, "bas_correct": 0, "count": 0}
        topics[t]["count"] += 1
        if r["evabot"]["correct"]:
            topics[t]["eva_correct"] += 1
        if r["baseline"]["correct"]:
            topics[t]["bas_correct"] += 1

    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  {'Metric':<28} {'EVABot(Local)':<16} {'Baseline(Global)':<16} {'Delta'}")
    print(f"  {'-'*72}")
    print(f"  {'Accuracy':<28} {eva_acc*100:>6.1f}%         {bas_acc*100:>6.1f}%         {delta_acc*100:+.1f}pp")
    print(f"  {'Noise Resistance':<28} {eva_nr:>6.2f}          {bas_nr:>6.2f}          {eva_nr-bas_nr:+.2f}")
    print(f"  {'Compression Ratio':<28} {eva_comp:>6.2f}          {'N/A':<16}")
    print(f"  {'Avg Latency':<28} {sum(r['evabot']['time_s'] for r in results)/n:>5.1f}s          {sum(r['baseline']['time_s'] for r in results)/n:>5.1f}s")

    if topics:
        print(f"\n  Accuracy by Topic:")
        print(f"  {'Topic':<20} {'N':>4} {'EVABot':>10} {'Baseline':>10} {'Delta':>8}")
        print(f"  {'-'*56}")
        for t, v in sorted(topics.items(), key=lambda x: -x[1]["count"]):
            ea = v["eva_correct"] / v["count"] * 100
            ba = v["bas_correct"] / v["count"] * 100
            print(f"  {t:<20} {v['count']:>4} {ea:>9.1f}% {ba:>9.1f}% {ea-ba:>+7.1f}pp")

    print("=" * 70)

    # ─── Save Results to JSON ─────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"exp1_seal0_{int(time.time())}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "dataset": "vtllms/sealqa (seal_0)",
                "model": MODEL,
                "judge_model": JUDGE_MODEL,
                "total_questions": n,
            },
            "summary": {
                "n": n,
                "evabot_accuracy": eva_acc,
                "baseline_accuracy": bas_acc,
                "accuracy_delta": delta_acc,
                "evabot_noise_resistance": eva_nr,
                "baseline_noise_resistance": bas_nr,
                "evabot_compression_ratio": eva_comp,
            },
            "topic_breakdown": {
                t: {
                    "count": v["count"],
                    "evabot_accuracy": v["eva_correct"] / v["count"],
                    "baseline_accuracy": v["bas_correct"] / v["count"],
                }
                for t, v in topics.items()
            },
            "records": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SEAL-0 Experiment 1: Hierarchical Isolation vs. Flat Context"
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max number of questions to process (0 = all 111 questions)"
    )
    args = parser.parse_args()

    if not API_KEY:
        print("Error: No API key found.")
        print("Set the environment variable: export DEEPSEEK_API_KEY=your_key")
        exit(1)
    run_experiment(limit=args.limit)
