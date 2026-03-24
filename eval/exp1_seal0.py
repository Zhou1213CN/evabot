"""
实验 1：SEAL-0 — 层级隔离 vs 全局视图

数据来源：HuggingFace vtllms/sealqa (seal_0 split, 111 道题)
SEAL-0 特点：每道题的网络搜索结果存在矛盾 (search_results=conflicting)

核心逻辑：
  1. 从 HuggingFace 加载真实 SEAL-0 数据集（111题）
  2. 为每道题用 LLM 生成 5 条模拟网络检索结果（2条正确 + 3条误导）
  3. 局部视图（模拟 EVABot）：每条 source 单独送给 Worker 做摘要判断 → Aggregator 汇总裁决
  4. 全局视图（Baseline）：全部 source 拼在一起，单次 LLM 调用

运行：
  pip install datasets openai
  python eval/exp1_seal0.py [--limit N]
"""
import argparse
import json
import os
import time
from typing import Any, Dict, List

from datasets import load_dataset
from openai import OpenAI

# ─── 配置 ────────────────────────────────────────────────────
API_KEY     = os.environ.get("DEEPSEEK_API_KEY", "sk-8eb9c82ecb6845e7adc115fdf86e9f17")
BASE_URL    = "https://api.deepseek.com"
MODEL       = "deepseek-chat"
JUDGE_MODEL = "deepseek-chat"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ─── 加载 SEAL-0 数据集 ─────────────────────────────────────
def load_seal0(limit: int = 0) -> List[Dict]:
    """
    从 HuggingFace 加载 SEAL-0 数据集。
    返回 [{id, question, answer, topic, question_types, urls}, ...]
    """
    print("  从 HuggingFace 加载 vtllms/sealqa (seal_0) ...")
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
    print(f"  已加载 {len(data)} 道题")
    return data


# ─── 为每道题生成矛盾性 sources ─────────────────────────────
def generate_conflicting_sources(question: str, answer: str, topic: str) -> Dict:
    """
    用 LLM 为一道 SEAL-0 题目生成 5 条模拟网络检索结果。
    - 2 条支持正确答案（来自权威来源）
    - 3 条误导性（看起来权威但答案错误）

    返回 {"sources": [...], "noise_indices": [2,3,4]}
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

    # 解析 JSON 数组
    try:
        # 尝试提取 JSON 数组
        start = result.find("[")
        end = result.rfind("]") + 1
        if start >= 0 and end > start:
            sources = json.loads(result[start:end])
            if isinstance(sources, list) and len(sources) == 5:
                return {"sources": sources, "noise_indices": [2, 3, 4]}
    except (json.JSONDecodeError, ValueError):
        pass

    # 解析失败时的 fallback：手工构造
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


# ─── LLM 调用封装 ─────────────────────────────────────────────
def call_llm(system: str, user: str, model: str = MODEL) -> str:
    """单次 LLM 调用，返回文本."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


# ─── 全局视图（Baseline）──────────────────────────────────────
def baseline_global_view(question: str, sources: List[str]) -> str:
    """
    对照组：把所有 source 一次性喂给 LLM。
    模型需要在一个 context 里同时处理矛盾信息。
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


# ─── 局部视图（模拟 EVABot 隔离架构）──────────────────────────
def evabot_local_view(question: str, sources: List[str]) -> Dict[str, Any]:
    """
    实验组：模拟 EVABot 的三层隔离。

    Step 1 (Worker 层): 每条 source 单独交给一个 Worker。
            Worker 只看到这一条 source + 问题。
            Worker 输出：该 source 的关键声明 + 可信度判断。

    Step 2 (Solver/Aggregator 层): 收集所有 Worker 摘要。
            Aggregator 只看到精炼摘要（非原始全文）。
            Aggregator 进行冲突裁决 → 输出最终答案。

    返回 final_answer + worker_summaries（用于分析信息压缩率）。
    """
    # ── Step 1: 每条 source 独立评估 ──
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

    # ── Step 2: Solver 聚合裁决 ──
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


# ─── 裁判 ─────────────────────────────────────────────────────
def judge_correctness(question: str, gold_answer: str, system_answer: str) -> bool:
    """用 LLM 判断 system_answer 是否正确。"""
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


def judge_noise_influence(question: str, sources: List[str],
                          noise_indices: List[int], system_answer: str) -> float:
    """
    判断误导性 source 对答案的影响程度。
    返回 resistance_score: 0.0（完全被误导）到 1.0（完全抗干扰）。
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


# ─── 主实验 ───────────────────────────────────────────────────
def run_experiment(limit: int = 0):
    # 加载真实 SEAL-0 数据集
    seal0_data = load_seal0(limit=limit)

    print("=" * 70)
    print("  SEAL-0 实验 1：层级隔离（局部视图）vs 全局视图")
    print(f"  数据集: vtllms/sealqa (seal_0)  |  Model: {MODEL}")
    print(f"  题目数: {len(seal0_data)}")
    print("=" * 70)

    results = []

    for sample in seal0_data:
        qid = sample["id"]
        question = sample["question"]
        answer = sample["answer"]
        topic = sample["topic"]

        print(f"\n[Q{qid}] {question}")
        print(f"  Gold: {answer}  |  Topic: {topic}")

        # ── Step 0: 生成矛盾性 sources ──
        print(f"  生成矛盾证据 ...", end=" ", flush=True)
        source_data = generate_conflicting_sources(question, answer, topic)
        sources = source_data["sources"]
        noise_idx = source_data["noise_indices"]
        print(f"done ({len(sources)} sources, {len(noise_idx)} misleading)")

        # ── Baseline (全局视图) ──
        t0 = time.time()
        bas_answer = baseline_global_view(question, sources)
        bas_time = round(time.time() - t0, 2)

        # ── EVABot (局部视图) ──
        t0 = time.time()
        eva_result = evabot_local_view(question, sources)
        eva_time = round(time.time() - t0, 2)
        eva_answer = eva_result["final_answer"]

        # ── 裁判评估 ──
        bas_correct = judge_correctness(question, answer, bas_answer)
        eva_correct = judge_correctness(question, answer, eva_answer)

        bas_noise = judge_noise_influence(question, sources, noise_idx, bas_answer)
        eva_noise = judge_noise_influence(question, sources, noise_idx, eva_answer)

        # ── 信息压缩率 ──
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

        # ── 打印单题结果 ──
        e_mark = "✓" if eva_correct else "✗"
        b_mark = "✓" if bas_correct else "✗"
        print(f"  Baseline: {b_mark} ({bas_time}s) noise_resist={bas_noise:.1f}  → {bas_answer.strip()[:60]}")
        print(f"  EVABot:   {e_mark} ({eva_time}s) noise_resist={eva_noise:.1f}  compress={compression:.2f} → {eva_answer.strip()[:60]}")

    # ─── 汇总 ─────────────────────────────────────────────
    n = len(results)
    if n == 0:
        print("没有数据，退出。")
        return

    eva_acc   = sum(1 for r in results if r["evabot"]["correct"]) / n
    bas_acc   = sum(1 for r in results if r["baseline"]["correct"]) / n
    eva_nr    = sum(r["evabot"]["noise_resistance"] for r in results) / n
    bas_nr    = sum(r["baseline"]["noise_resistance"] for r in results) / n
    eva_comp  = sum(r["evabot"]["compression_ratio"] for r in results) / n
    delta_acc = eva_acc - bas_acc

    # ─── 按 topic 分组统计 ──────────────────────────────────
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
    print("  汇总结果")
    print("=" * 70)
    print(f"  {'指标':<28} {'EVABot(局部)':<16} {'Baseline(全局)':<16} {'Δ'}")
    print(f"  {'-'*72}")
    print(f"  {'准确率 (Accuracy)':<28} {eva_acc*100:>6.1f}%         {bas_acc*100:>6.1f}%         {delta_acc*100:+.1f}pp")
    print(f"  {'噪声抵抗 (Noise Resist)':<28} {eva_nr:>6.2f}          {bas_nr:>6.2f}          {eva_nr-bas_nr:+.2f}")
    print(f"  {'信息压缩率 (Compression)':<28} {eva_comp:>6.2f}          {'N/A':<16}")
    print(f"  {'平均延迟':<28} {sum(r['evabot']['time_s'] for r in results)/n:>5.1f}s          {sum(r['baseline']['time_s'] for r in results)/n:>5.1f}s")

    if topics:
        print(f"\n  按 Topic 分类准确率:")
        print(f"  {'Topic':<20} {'N':>4} {'EVABot':>10} {'Baseline':>10} {'Δ':>8}")
        print(f"  {'-'*56}")
        for t, v in sorted(topics.items(), key=lambda x: -x[1]["count"]):
            ea = v["eva_correct"] / v["count"] * 100
            ba = v["bas_correct"] / v["count"] * 100
            print(f"  {t:<20} {v['count']:>4} {ea:>9.1f}% {ba:>9.1f}% {ea-ba:>+7.1f}pp")

    print("=" * 70)

    # ─── 保存 JSON ────────────────────────────────────────
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
    print(f"\n  结果已保存 → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEAL-0 Experiment 1")
    parser.add_argument("--limit", type=int, default=0,
                        help="限制题目数量 (0=全部111题)")
    args = parser.parse_args()

    if not API_KEY:
        print("请先设置环境变量: export DEEPSEEK_API_KEY=你的key")
        exit(1)
    run_experiment(limit=args.limit)
