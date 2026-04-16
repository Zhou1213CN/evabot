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
    若 URL 含有 anchor（#section），则智能定位并只提取该小节内容（±20000字符窗口）。
    返回 {"url": ..., "snippet": ...}
    """
    from bs4 import BeautifulSoup

    # 分离 anchor（处理 #:~:text= 和普通 # 两种形式）
    if "#:~:text=" in url:
        clean_url, text_fragment = url.split("#:~:text=", 1)
        anchor = None
    elif "#" in url:
        clean_url, anchor = url.split("#", 1)
        text_fragment = None
    else:
        clean_url = url
        anchor = None
        text_fragment = None

    try:
        resp = requests.get(clean_url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        html = resp.text

        # ── 优先：用 anchor 定位具体小节 ──────────────────────────
        if anchor:
            soup = BeautifulSoup(html, "html.parser")
            target = soup.find(id=anchor) or soup.find(attrs={"name": anchor})
            if target:
                # 确定当前锚点所在的标题层级（h2/h3/h4 或非标题元素）
                heading_tags = {"h1", "h2", "h3", "h4", "h5", "h6"}
                anchor_tag = target.name if target.name in heading_tags else None
                # 向上找父标题（锚点可能在 span 内）
                if not anchor_tag:
                    parent = target.parent
                    if parent and parent.name in heading_tags:
                        anchor_tag = parent.name
                        target = parent

                stop_tags = set()
                if anchor_tag:
                    level = int(anchor_tag[1])
                    stop_tags = {f"h{i}" for i in range(1, level + 1)}

                chunks = [str(target)]
                for sibling in target.find_next_siblings():
                    tag = sibling.name
                    # 只在遇到同级或更高级标题时停止
                    if stop_tags and tag in stop_tags:
                        break
                    chunks.append(str(sibling))
                    if sum(len(c) for c in chunks) > 80000:
                        break

                section_html = "\n".join(chunks)
                text = md(section_html, strip=["script", "style", "nav", "footer"])
                lines = [l for l in text.splitlines() if l.strip()]
                snippet = "\n".join(lines)
                if len(snippet) > 500:  # 成功提取到内容
                    return {"url": url, "snippet": snippet, "error": None}
                # 内容太少则回退到 text_fragment 或兜底逻辑

        # ── 备选：用 text fragment 关键词定位段落 ─────────────────
        if text_fragment:
            import urllib.parse
            keyword = urllib.parse.unquote(text_fragment.split("%0A")[0])[:80]
            full_text = md(html, strip=["script", "style", "nav", "footer"])
            lines = [l for l in full_text.splitlines() if l.strip()]
            full_text = "\n".join(lines)
            idx = full_text.lower().find(keyword.lower()[:40])
            if idx != -1:
                start = max(0, idx - 2000)
                snippet = full_text[start: idx + 30000]
                return {"url": url, "snippet": snippet, "error": None}

        # ── 兜底：整页转 Markdown，取前 30000 字符 ────────────────
        text = md(html, strip=["script", "style", "nav", "footer"])
        lines = [l for l in text.splitlines() if l.strip()]
        snippet = "\n".join(lines)[:30000]
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
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 道题（默认全部）")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH), help="输出 JSON 文件路径")
    parser.add_argument("--delay", type=float, default=0.5, help="每道题之间的延迟（秒）")
    parser.add_argument("--include-fast-changing", action="store_true",
                        help="是否包含 fast-changing 题目（默认跳过）")
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

    questions = load_sealqa_dataset()  # 先加载全量，再过滤

    # 默认过滤掉 fast-changing 题目（先过滤，再 limit）
    if not args.include_fast_changing:
        before = len(questions)
        questions = [q for q in questions if q.get("freshness") != "fast-changing"]
        print(f"已过滤 fast-changing 题目（{before - len(questions)} 道），剩余 {len(questions)} 道。")

    if args.limit:
        questions = questions[:args.limit]
        print(f"取前 {args.limit} 道题。")

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
