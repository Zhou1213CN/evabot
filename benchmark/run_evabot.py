"""
benchmark/run_evabot.py

阶段二：Evabot 基准测试运行器（忠实架构版）

真实 Evabot 架构：
  Butler  →  Solver（拿到问题后自主决策）
                ├── 判断是否需要派 Worker
                ├── 决定派哪些 Worker、每个做什么
                └── Worker 汇报后综合答案

本脚本用 tool_call 模拟 Solver 的自主派发：
  - Solver 可调用 analyze_source(source_id, aspect) 派 Worker 分析指定来源
  - Solver 可调用 final_answer(answer) 直接给出答案
  - Worker 执行 Solver 指定的任务并返回结果
  - Solver 基于所有 Worker 反馈综合最终答案

用法：
    python benchmark/run_evabot.py --limit 5
    python benchmark/run_evabot.py          # 全部 111 题
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

DEEPSEEK_KEY = os.environ.get("deepseek_key", "sk-b6cd43bbc301481fa51120a56e53a39d")
BASE_URL     = "https://api.deepseek.com"
MODEL        = "deepseek-chat"
MAX_SOLVER_ROUNDS = 8   # Solver 最多与 Worker 交互几轮

INPUT_PATH   = Path(__file__).parent / "cached_sources.json"
OUTPUT_PATH  = Path(__file__).parent / "evabot_results.json"

client = OpenAI(base_url=BASE_URL, api_key=DEEPSEEK_KEY, timeout=90.0)

# ─────────────────────────────────────────────
# Solver 的 tool schema（它能调用的两个工具）
# ─────────────────────────────────────────────
SOLVER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_source",
            "description": (
                "Dispatch a Worker to carefully analyze one specific web source. "
                "The Worker will read the source and extract facts relevant to your specified aspect."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_id": {
                        "type": "integer",
                        "description": "1-based index of the source to analyze"
                    },
                    "aspect": {
                        "type": "string",
                        "description": "Specific question or aspect you want the Worker to focus on"
                    }
                },
                "required": ["source_id", "aspect"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Submit your definitive answer. Call this once you have enough information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Short, definitive answer (name / number / title)"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this is correct despite conflicting sources"
                    }
                },
                "required": ["answer"]
            }
        }
    }
]

SOLVER_SYSTEM = """\
You are Solver, the strategic orchestration agent in the Evabot multi-agent system.

Your job: answer the user's question by reasoning over multiple web sources that may contain conflicting information.

You have two tools:
1. analyze_source(source_id, aspect) — dispatch a Worker to extract facts from a specific source
2. final_answer(answer, reasoning)   — submit your definitive answer

Strategy:
- Start by reviewing the available sources (listed in the user message).
- Decide which sources are worth analyzing and what specific aspect each Worker should focus on.
- You may call analyze_source multiple times in parallel or sequentially.
- Once you have enough Worker feedback, call final_answer with a SHORT answer (a number, name, or title).
- If sources conflict, reason about which is more authoritative or recent before deciding.
"""

WORKER_SYSTEM = """\
You are a Worker in the Evabot system. Solver has dispatched you to analyze one web source.
Extract every fact that is relevant to the given aspect/question.
Be precise and verbatim. If the source is irrelevant, say so.
"""

# ─────────────────────────────────────────────
# LLM 调用
# ─────────────────────────────────────────────

def llm_call(messages: list, tools=None, retries: int = 3) -> tuple:
    """返回 (response_message_dict, latency_s)"""
    for attempt in range(retries):
        try:
            t0 = time.time()
            kwargs = dict(model=MODEL, messages=messages, temperature=0.1)
            if tools:
                kwargs["tools"] = tools
            resp = client.chat.completions.create(**kwargs)
            latency = round(time.time() - t0, 2)
            msg = resp.choices[0].message
            return msg, latency
        except Exception as e:
            print(f"    [重试 {attempt+1}/{retries}] {type(e).__name__}: {e}")
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
    return None, 0.0


# ─────────────────────────────────────────────
# Worker 执行（由 Solver 的 tool call 触发）
# ─────────────────────────────────────────────

def run_worker(question: str, source: dict, aspect: str) -> tuple:
    """Worker 分析指定来源，返回 (facts_text, latency_s)"""
    snippet = source.get("snippet", "").strip()
    url     = source.get("url", "unknown")
    if not snippet:
        return "Source is empty.", 0.0

    messages = [
        {"role": "system", "content": WORKER_SYSTEM},
        {"role": "user", "content": (
            f"Original question: {question}\n"
            f"Aspect to focus on: {aspect}\n\n"
            f"Source URL: {url}\n\n"
            f"{snippet}"
        )},
    ]
    msg, lat = llm_call(messages)
    return (msg.content or "[No output]") if msg else "[Worker failed]", lat


# ─────────────────────────────────────────────
# Solver 主循环（自主决策 + 派发 Worker）
# ─────────────────────────────────────────────

def run_solver(question: str, sources: list) -> tuple:
    """
    Solver 自主决定策略：可以不派 Worker 直接回答，也可以派多个 Worker。
    返回 (final_answer_str, total_latency_s, worker_calls_count)
    """
    # 构建来源索引表（告知 Solver 有哪些来源可用）
    sources_index = "\n".join(
        f"  Source {i+1}: {s.get('url', 'unknown')}"
        for i, s in enumerate(sources)
    )

    messages = [
        {"role": "system", "content": SOLVER_SYSTEM},
        {"role": "user", "content": (
            f"Question: {question}\n\n"
            f"Available sources ({len(sources)} total):\n{sources_index}\n\n"
            f"Use analyze_source to dispatch Workers as needed, then call final_answer."
        )},
    ]

    total_latency   = 0.0
    worker_calls    = 0

    for round_idx in range(MAX_SOLVER_ROUNDS):
        msg, lat = llm_call(messages, tools=SOLVER_TOOLS)
        total_latency += lat

        if msg is None:
            return "[Solver failed]", total_latency, worker_calls

        # 把 Solver 的回复加入历史
        msg_dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                }
                for tc in msg.tool_calls
            ]
        messages.append(msg_dict)

        # 没有 tool call → Solver 直接用文字给出了答案
        if not msg.tool_calls:
            return msg.content or "[No answer]", total_latency, worker_calls

        # 处理 tool calls
        all_done = False
        for tc in msg.tool_calls:
            func_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}

            if func_name == "final_answer":
                all_done = True
                answer = args.get("answer", "[No answer]")
                # 把 tool result 加回去（保持对话完整性）
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "Answer submitted."
                })
                return answer, total_latency, worker_calls

            elif func_name == "analyze_source":
                source_id = args.get("source_id", 1)
                aspect    = args.get("aspect", question)
                src_idx   = source_id - 1
                if 0 <= src_idx < len(sources):
                    facts, w_lat = run_worker(question, sources[src_idx], aspect)
                    total_latency += w_lat
                    worker_calls  += 1
                    tool_result = facts
                else:
                    tool_result = f"Source {source_id} does not exist."

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result
                })

        if all_done:
            break

    return "[Max rounds reached]", total_latency, worker_calls


# ─────────────────────────────────────────────
# 评分
# ─────────────────────────────────────────────

def score_answer(predicted: str, ground_truth: str) -> bool:
    """
    关键实体匹配：
    1. 若标准答案含数字 → 只需数字匹配
    2. 否则 → 取最长非停用词匹配
    """
    pred_lower = predicted.lower()
    gt_lower   = ground_truth.lower()

    gt_numbers = re.findall(r'\d+', gt_lower)
    if gt_numbers:
        pred_numbers = re.findall(r'\d+', pred_lower)
        return any(n in pred_numbers for n in gt_numbers)

    stopwords = {"the","a","an","of","in","at","to","and","or",
                 "is","are","was","were","years","old","players","films"}
    tokens = [t for t in re.findall(r"[a-z']+", gt_lower) if t not in stopwords]
    if not tokens:
        return gt_lower in pred_lower
    return max(tokens, key=len) in pred_lower


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",  type=int, default=None)
    parser.add_argument("--input",  type=str, default=str(INPUT_PATH))
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if args.limit:
        dataset = dataset[:args.limit]

    print(f"\n{'='*60}")
    print(f"Evabot Benchmark  |  模型: {MODEL}  |  共 {len(dataset)} 道题")
    print(f"架构: Butler → Solver（自主决策）→ Worker（按需派发）")
    print(f"{'='*60}\n")

    results = []
    correct = 0

    for i, item in enumerate(dataset):
        question = item["question"]
        answer   = item["answer"]
        sources  = item["sources"]

        print(f"[{i+1}/{len(dataset)}] Q: {question[:80]}")
        print(f"  标准答案: {answer}")

        predicted, latency, n_workers = run_solver(question, sources)
        is_correct = score_answer(predicted, answer)
        if is_correct:
            correct += 1

        status = "✅" if is_correct else "❌"
        print(f"  预测答案: {predicted[:120]}")
        print(f"  {status}  耗时: {latency}s  Worker调用: {n_workers}次\n")

        results.append({
            "id": item["id"],
            "question": question,
            "ground_truth": answer,
            "predicted": predicted,
            "correct": is_correct,
            "latency_s": latency,
            "worker_calls": n_workers,
        })

    total    = len(results)
    accuracy = round(correct / total * 100, 1) if total else 0
    avg_lat  = round(sum(r["latency_s"] for r in results) / total, 2) if total else 0
    avg_w    = round(sum(r["worker_calls"] for r in results) / total, 1) if total else 0

    summary = {
        "model": MODEL,
        "architecture": "Butler→Solver(autonomous)→Worker(on-demand)",
        "total_questions": total,
        "correct": correct,
        "accuracy_pct": accuracy,
        "avg_latency_s": avg_lat,
        "avg_worker_calls": avg_w,
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Evabot 测试结果汇总")
    print(f"{'='*60}")
    print(f"  模型              : {MODEL}")
    print(f"  测试题数          : {total}")
    print(f"  正确数            : {correct}")
    print(f"  准确率            : {accuracy}%")
    print(f"  平均耗时          : {avg_lat}s / 题")
    print(f"  平均 Worker 调用  : {avg_w}次 / 题")
    print(f"  结果已保存        : {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
