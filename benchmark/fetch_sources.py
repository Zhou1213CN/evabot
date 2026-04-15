"""
benchmark/fetch_sources.py

阶段一：数据准备脚本
- 从 Hugging Face 加载 vtllms/sealqa (seal_0) 数据集，共 111 道题
- 直接抓取数据集自带 urls 字段中的网页内容（这些是 SealQA 基准设计时使用的冲突来源页面）
- 将结果保存为 cached_sources.json，供 Evabot 和 ROMA 共享作为同卷输入

用法：
    python benchmark/fetch_sources.py              # 抓全部 111 题
    python benchmark/fetch_sources.py --limit 5    # 仅抓前 5 题（测试用）
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import requests
from markdownify import markdownify as md

OUTPUT_PATH = Path(__file__).parent / "cached_sources.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_url_content(url: str, timeout: int = 15) -> dict:
    """
    抓取单个 URL 的网页内容，转换为 Markdown 格式。
    返回 {"url": ..., "snippet": ...}
    """
    # 处理 Wikipedia fragment URL（截断 #:~:text= 部分）
    clean_url = url.split("#:~:text=")[0].split("#")[0]

    try:
        resp = requests.get(clean_url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        # 将 HTML 转为 Markdown，截取前 3000 字符避免 token 爆炸
        text = md(resp.text, strip=["script", "style", "nav", "footer"])
        # 清理多余空行
        lines = [l for l in text.splitlines() if l.strip()]
        snippet = "\n".join(lines)[:8000]
        return {"url": url, "snippet": snippet, "error": None}
    except Exception as e:
        return {"url": url, "snippet": "", "error": str(e)}


def load_sealqa_dataset(limit: Optional[int] = None) -> list:
    """
    加载 vtllms/sealqa (seal_0) 数据集。
    """
    try:
        from datasets import load_dataset
        print("正在从 Hugging Face 加载 vtllms/sealqa (seal_0) ...")
        ds = load_dataset("vtllms/sealqa", "seal_0", split="test")
        questions = []
        for i, row in enumerate(ds):
            if limit is not None and i >= limit:
                break
            questions.append({
                "id": i,
                "question": row["question"],
                "answer": row["answer"],
                "urls": row.get("urls", []),
                "freshness": row.get("freshness", ""),
                "question_types": row.get("question_types", ""),
            })
        print(f"成功加载 {len(questions)} 道题。")
        return questions
    except Exception as e:
        print(f"[错误] 加载数据集失败: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="为 SealQA 数据集抓取网页内容")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 道题（默认全部 111 道）")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH), help="输出 JSON 文件路径")
    parser.add_argument("--delay", type=float, default=0.5, help="每道题之间的延迟（秒）")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 续跑：读取已有缓存
    existing = {}
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            existing = {str(item["id"]): item for item in data}
        print(f"发现已有缓存 {len(existing)} 条，将跳过已处理的题目。")

    questions = load_sealqa_dataset(limit=args.limit)
    results = list(existing.values())
    processed_ids = set(existing.keys())

    for i, q in enumerate(questions):
        qid = str(q["id"])
        if qid in processed_ids:
            print(f"[{i+1}/{len(questions)}] 跳过（已缓存）: {q['question'][:60]}...")
            continue

        print(f"\n[{i+1}/{len(questions)}] 处理题目: {q['question'][:80]}")
        print(f"  答案: {q['answer']}")
        print(f"  来源 URL 数量: {len(q['urls'])}")

        sources = []
        for url in q["urls"]:
            print(f"  → 抓取: {url[:80]}...")
            result = fetch_url_content(url)
            if result["error"]:
                print(f"    [失败] {result['error']}")
            else:
                print(f"    [成功] 获取 {len(result['snippet'])} 字符")
            sources.append(result)

        entry = {
            "id": q["id"],
            "question": q["question"],
            "answer": q["answer"],
            "freshness": q["freshness"],
            "question_types": q["question_types"],
            "sources": sources,
        }
        results.append(entry)

        # 每题后立即保存，防止中断丢失进度
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        if args.delay > 0 and i < len(questions) - 1:
            time.sleep(args.delay)

    print(f"\n✅ 完成！共 {len(results)} 条结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
