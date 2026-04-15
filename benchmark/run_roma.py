"""
benchmark/run_roma.py

阶段三：ROMA 基准测试运行器（DSPy 实现）
- 读取同一份 cached_sources.json
- 用 DSPy ChainOfThought 单阶段推理处理每道题
- 相同模型（deepseek-chat）、相同数据、不同架构 → 公平对比

用法：
    python benchmark/run_roma.py --limit 5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import dspy

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
DEEPSEEK_KEY = os.environ.get("deepseek_key", "sk-b6cd43bbc301481fa51120a56e53a39d")
MODEL        = "deepseek/deepseek-chat"   # litellm 格式

INPUT_PATH   = Path(__file__).parent / "cached_sources.json"
OUTPUT_PATH  = Path(__file__).parent / "roma_results.json"


# ─────────────────────────────────────────────
# DSPy 模块定义（ROMA 风格）
# ─────────────────────────────────────────────

class ConflictAwareQA(dspy.Signature):
    """
    You are given a question and multiple web page snippets that may contain conflicting information.
    Carefully reason through all sources, identify contradictions, and determine the single most
    accurate answer. Provide a short, definitive answer (a name, number, or title).
    """
    question: str = dspy.InputField(desc="The question to answer")
    sources:  str = dspy.InputField(desc="Web page snippets, possibly with conflicting information")
    answer:   str = dspy.OutputField(desc="A short, definitive answer (name / number / title only)")


class ROMAModule(dspy.Module):
    """
    ROMA 核心模块：Chain-of-Thought 单阶段推理。
    将所有来源拼接后一次性传给 LLM，利用 CoT 自动处理冲突。
    """
    def __init__(self):
        self.qa = dspy.ChainOfThought(ConflictAwareQA)

    def forward(self, question: str, sources: str) -> dspy.Prediction:
        return self.qa(question=question, sources=sources)


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def format_sources(sources: list) -> str:
    """把 sources 列表拼成单个字符串传给 DSPy"""
    parts = []
    for i, s in enumerate(sources):
        url = s.get("url", "unknown")
        snippet = s.get("snippet", "").strip()[:7000]
        if snippet:
            parts.append(f"[Source {i+1}] ({url})\n{snippet}")
    return "\n\n---\n\n".join(parts) if parts else "No sources available."


def score_answer(predicted: str, ground_truth: str) -> bool:
    """严格关键词匹配（忽略大小写）"""
    pred_lower = predicted.lower()
    skip = {"the", "a", "an", "of", "in", "at", "to", "and", "or", "is", "are"}
    for kw in ground_truth.lower().split():
        if kw in skip:
            continue
        if kw not in pred_lower:
            return False
    return True


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ROMA 基准测试（DSPy）")
    parser.add_argument("--limit",  type=int, default=None, help="仅测试前 N 道题")
    parser.add_argument("--input",  type=str, default=str(INPUT_PATH))
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    # 配置 DSPy LLM（通过 litellm 路由到 DeepSeek）
    lm = dspy.LM(
        model=MODEL,
        api_key=DEEPSEEK_KEY,
        api_base="https://api.deepseek.com/v1",
        temperature=0.1,
        max_tokens=2048,
    )
    dspy.configure(lm=lm)

    roma = ROMAModule()

    with open(args.input, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if args.limit:
        dataset = dataset[:args.limit]

    print(f"\n{'='*60}")
    print(f"ROMA Benchmark  |  模型: deepseek-chat  |  共 {len(dataset)} 道题")
    print(f"{'='*60}\n")

    results = []
    correct = 0

    for i, item in enumerate(dataset):
        question = item["question"]
        answer   = item["answer"]
        sources  = item["sources"]

        print(f"[{i+1}/{len(dataset)}] Q: {question[:80]}")
        print(f"  标准答案: {answer}")

        sources_str = format_sources(sources)

        for attempt in range(3):
            try:
                t0 = time.time()
                pred = roma(question=question, sources=sources_str)
                latency = round(time.time() - t0, 2)
                predicted = pred.answer.strip()
                break
            except Exception as e:
                print(f"    [重试 {attempt+1}/3] {type(e).__name__}")
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))
                else:
                    predicted, latency = "[调用失败]", 0.0

        is_correct = score_answer(predicted, answer)
        if is_correct:
            correct += 1

        status = "✅" if is_correct else "❌"
        print(f"  预测答案: {predicted[:120]}")
        print(f"  {status}  耗时: {latency}s\n")

        results.append({
            "id": item["id"],
            "question": question,
            "ground_truth": answer,
            "predicted": predicted,
            "correct": is_correct,
            "latency_s": latency,
        })

    total    = len(results)
    accuracy = round(correct / total * 100, 1) if total else 0
    avg_lat  = round(sum(r["latency_s"] for r in results) / total, 2) if total else 0

    summary = {
        "model": "deepseek-chat",
        "framework": "DSPy ChainOfThought (ROMA)",
        "total_questions": total,
        "correct": correct,
        "accuracy_pct": accuracy,
        "avg_latency_s": avg_lat,
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"ROMA 测试结果汇总")
    print(f"{'='*60}")
    print(f"  模型        : deepseek-chat")
    print(f"  框架        : DSPy ChainOfThought")
    print(f"  测试题数    : {total}")
    print(f"  正确数      : {correct}")
    print(f"  准确率      : {accuracy}%")
    print(f"  平均耗时    : {avg_lat}s / 题")
    print(f"  结果已保存  : {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
