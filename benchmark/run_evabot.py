"""
benchmark/run_evabot.py

阶段二：Evabot 基准测试运行器
- 读取 cached_sources.json（由 fetch_sources.py 生成）
- 用 Evabot 的两阶段多智能体架构处理每道题：
    Stage 1 (Worker)  ：分析每个来源，提取与问题相关的关键事实
    Stage 2 (Solver)  ：综合所有事实，输出最终答案
- 评分：严格关键词匹配（忽略大小写）
- 输出：准确率、平均耗时、逐题详情

用法：
    qwen_key=<YOUR_KEY> python benchmark/run_evabot.py --limit 5
    qwen_key=<YOUR_KEY> python benchmark/run_evabot.py           # 全部 111 题
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# 把项目根目录加入 sys.path，确保 backend 包可以 import
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI

# ─────────────────────────────────────────────
# 配置：直接使用 Aliyun DashScope / qwen3-max
# ─────────────────────────────────────────────
ALIYUN_KEY  = os.environ.get("deepseek_key", "sk-b6cd43bbc301481fa51120a56e53a39d")
BASE_URL    = "https://api.deepseek.com"
MODEL       = "deepseek-chat"

INPUT_PATH  = Path(__file__).parent / "cached_sources.json"
OUTPUT_PATH = Path(__file__).parent / "evabot_results.json"

# ─────────────────────────────────────────────
# 底层 LLM 调用（OpenAI-compatible）
# ─────────────────────────────────────────────

def llm_call(messages: list, temperature: float = 0.1, retries: int = 3) -> tuple:
    """
    调用 LLM，返回 (content_text, latency_seconds)。带重试逻辑。
    """
    client = OpenAI(base_url=BASE_URL, api_key=ALIYUN_KEY, timeout=60.0)
    for attempt in range(retries):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
            )
            latency = round(time.time() - t0, 2)
            content = resp.choices[0].message.content or ""
            return content, latency
        except Exception as e:
            print(f"    [重试 {attempt+1}/{retries}] 请求失败: {type(e).__name__}")
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
    return "[LLM调用失败]", 0.0


# ─────────────────────────────────────────────
# 两阶段推理（Evabot 架构核心）
# ─────────────────────────────────────────────

WORKER_SYSTEM = """You are a precise information extraction agent.
You will be given a question and a web page snippet.
Your job: extract EVERY fact in the snippet that could answer or contradict the question.
Be comprehensive and verbatim where possible. If the snippet is irrelevant, say so.
Reply in English."""

SOLVER_SYSTEM = """You are a careful synthesis agent tasked with resolving conflicting web information.
You will receive a question and a set of extracted facts from multiple sources, which may contradict each other.
Rules:
1. Carefully evaluate which fact is most likely correct based on recency, source authority, and internal consistency.
2. Provide ONE definitive short answer (a number, a name, a title, etc.).
3. Do NOT hedge or give multiple answers.
4. End your response with: FINAL ANSWER: <your answer>"""


def worker_stage(question: str, sources: list) -> tuple:
    """
    Stage 1 – Worker：逐一分析来源，提取关键事实。
    返回 (aggregated_facts_str, total_latency)
    """
    all_facts = []
    total_latency = 0.0

    for i, src in enumerate(sources):
        snippet = src.get("snippet", "").strip()
        if not snippet:
            continue

        messages = [
            {"role": "system", "content": WORKER_SYSTEM},
            {"role": "user",   "content": (
                f"Question: {question}\n\n"
                f"--- Source {i+1} ({src.get('url', 'unknown')}) ---\n{snippet[:7000]}"
            )},
        ]
        facts, lat = llm_call(messages)
        all_facts.append(f"[Source {i+1}] {facts}")
        total_latency += lat

    return "\n\n".join(all_facts), total_latency


def solver_stage(question: str, aggregated_facts: str) -> tuple:
    """
    Stage 2 – Solver：综合所有事实，输出最终答案。
    返回 (final_answer_text, latency)
    """
    messages = [
        {"role": "system", "content": SOLVER_SYSTEM},
        {"role": "user",   "content": (
            f"Question: {question}\n\n"
            f"Extracted facts from sources:\n{aggregated_facts}"
        )},
    ]
    answer_text, lat = llm_call(messages)
    return answer_text, lat


def extract_final_answer(response_text: str) -> str:
    """
    从 Solver 的回复中提取 'FINAL ANSWER: ...' 后面的部分。
    如果没有找到标签，返回整个回复文本。
    """
    marker = "FINAL ANSWER:"
    idx = response_text.upper().find(marker)
    if idx != -1:
        return response_text[idx + len(marker):].strip().split("\n")[0].strip()
    return response_text.strip()


# ─────────────────────────────────────────────
# 评分
# ─────────────────────────────────────────────

def score_answer(predicted: str, ground_truth: str) -> bool:
    """
    严格关键词匹配（忽略大小写）：
    ground_truth 的每个词都必须出现在 predicted 中。
    """
    pred_lower = predicted.lower()
    # 将标准答案中的关键词逐一检查
    for keyword in ground_truth.lower().split():
        # 跳过连接词
        if keyword in {"the", "a", "an", "of", "in", "at", "to", "and", "or", "is", "are"}:
            continue
        if keyword not in pred_lower:
            return False
    return True


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evabot 基准测试")
    parser.add_argument("--limit", type=int, default=None, help="仅测试前 N 道题")
    parser.add_argument("--input",  type=str, default=str(INPUT_PATH),  help="cached_sources.json 路径")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH), help="结果输出路径")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if args.limit:
        dataset = dataset[:args.limit]

    print(f"\n{'='*60}")
    print(f"Evabot Benchmark  |  模型: {MODEL}  |  共 {len(dataset)} 道题")
    print(f"{'='*60}\n")

    results = []
    correct = 0

    for i, item in enumerate(dataset):
        qid      = item["id"]
        question = item["question"]
        answer   = item["answer"]
        sources  = item["sources"]

        print(f"[{i+1}/{len(dataset)}] Q: {question[:80]}")
        print(f"  标准答案: {answer}")

        t_start = time.time()

        # Stage 1: Worker 分析来源
        facts, worker_lat = worker_stage(question, sources)

        # Stage 2: Solver 综合答案
        raw_response, solver_lat = solver_stage(question, facts)
        predicted = extract_final_answer(raw_response)

        total_lat = round(time.time() - t_start, 2)
        is_correct = score_answer(predicted, answer)
        if is_correct:
            correct += 1

        status = "✅" if is_correct else "❌"
        print(f"  预测答案: {predicted[:120]}")
        print(f"  {status}  耗时: {total_lat}s  (Worker: {worker_lat}s + Solver: {solver_lat}s)\n")

        results.append({
            "id": qid,
            "question": question,
            "ground_truth": answer,
            "predicted": predicted,
            "raw_response": raw_response,
            "correct": is_correct,
            "latency_s": total_lat,
            "worker_latency_s": worker_lat,
            "solver_latency_s": solver_lat,
        })

    # ── 汇总统计 ──
    total = len(results)
    accuracy = round(correct / total * 100, 1) if total else 0
    avg_lat  = round(sum(r["latency_s"] for r in results) / total, 2) if total else 0

    summary = {
        "model": MODEL,
        "total_questions": total,
        "correct": correct,
        "accuracy_pct": accuracy,
        "avg_latency_s": avg_lat,
        "results": results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Evabot 测试结果汇总")
    print(f"{'='*60}")
    print(f"  模型        : {MODEL}")
    print(f"  测试题数    : {total}")
    print(f"  正确数      : {correct}")
    print(f"  准确率      : {accuracy}%")
    print(f"  平均耗时    : {avg_lat}s / 题")
    print(f"  结果已保存  : {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
