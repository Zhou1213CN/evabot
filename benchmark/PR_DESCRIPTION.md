# Benchmark: Evabot vs ROMA on SealQA (seal_0)

## 目标

在**完全公平**的条件下对比 Evabot 多智能体架构与 ROMA（DSPy ChainOfThought）在处理冲突网络信息时的性能差异。

## 实验设计（"同卷闭卷考试"）

| 变量 | 设定 |
|------|------|
| 数据集 | [vtllms/sealqa](https://huggingface.co/datasets/vtllms/sealqa)，`seal_0` 子集，共 111 道题 |
| 底层模型 | `deepseek-chat`（两边完全一致） |
| 输入来源 | 同一份 `cached_sources.json`（从数据集自带 URL 抓取的真实网页内容） |
| 评分方式 | 关键实体匹配（数字取精确匹配，文本取最长关键词匹配，忽略大小写） |

> **为什么不直接和论文的 45.9% 比？**
> ROMA 论文使用 GPT-4o-mini 的实时搜索接口，与本地阿里云/DeepSeek 搜出来的原材料完全不同，直接比绝对分数没有意义。本方案让两个架构读同一份缓存网页，公平竞争。

---

## Benchmark 文件说明

| 文件 | 作用 |
|------|------|
| `benchmark/fetch_sources.py` | 阶段一：加载 SealQA 数据集，按 URL anchor 智能提取相关小节，保存为 `cached_sources.json` |
| `benchmark/cached_sources.json` | 共享输入：每道题的网页原文（两个架构读同一份） |
| `benchmark/run_evabot.py` | 阶段二：Evabot 两阶段推理（Worker 提取事实 → Solver 综合答案） |
| `benchmark/run_roma.py` | 阶段三：ROMA DSPy ChainOfThought 单阶段推理 |
| `benchmark/evabot_results.json` | Evabot 5题 pilot 结果 |
| `benchmark/roma_results.json` | ROMA 5题 pilot 结果 |
| `benchmark/PR_DESCRIPTION.md` | 本文档 |

---

## 架构对比

### Evabot（两阶段多智能体）
```
For each source:
  Worker: "从这个网页里提取与问题相关的所有事实"
          → 逐条返回摘要

Solver: "综合以上事实，输出唯一正确答案"
        → FINAL ANSWER: <answer>
```
- 每道题发出 N+1 次 LLM 调用（N = 来源数量）
- Worker 负责噪声过滤，Solver 负责冲突仲裁

### ROMA（DSPy ChainOfThought 单阶段）
```
DSPy ChainOfThought(ConflictAwareQA):
  input:  question + 所有来源拼接成字符串
  output: answer（带内置思维链）
```
- 每道题发出 1 次 LLM 调用
- 依赖 DSPy 的结构化输出格式自动生成推理链

---

## Pilot 结果（5 题，使用智能小节提取 + 修正评分后）

| 指标 | Evabot | ROMA |
|------|--------|------|
| 准确率 | 20% (1/5) | 40% (2/5) |
| 平均耗时 | ~21.6s / 题 | ~27.6s / 题 |
| LLM 调用次数 / 题 | N+1（来源数+1） | 1 |

### 逐题详情

| # | 问题摘要 | 标准答案 | Evabot | ROMA | 难点分析 |
|---|---------|---------|--------|------|---------|
| 1 | Grammy 最多获奖者 | Serban Ghenea | Taylor Swift ❌ | Taylor Swift ❌ | 内容噪音过多，Taylor Swift 出现频次更高，干扰了判断 |
| 2 | NBA 60分+球员数 | 12 players | 5 ❌ | 6 ❌ | 页面无直接计数，需推理统计 |
| 3 | 票房前10最新电影 | Zootopia 2 | 无法确定 ❌ | Barbie ❌ | 时效性问题，Wikipedia 可能未更新 |
| 4 | 诺奖物理学家年龄 | 84 years old | 81 ❌ | **84** ✅ | ROMA CoT 正确推算；Evabot Worker 引入了错误中间结果 |
| 5 | IATA 成员数 | 370 | **370** ✅ | **370** ✅ | 智能小节提取后找到了完整数字（之前截断时只看到352） |

---

## 关键发现

### 1. 内容截断影响显著
- 截断 3000 字时：Q5 两者都答 352（网页局部数字），错误
- 扩大到完整小节提取后：Q5 两者都答对 370（找到了完整总数）
- **结论**：对包含大型表格的 Wikipedia 页面，提取正确的小节比截断更重要

### 2. 更多内容 ≠ 更高准确率（噪音问题）
- Q1 在内容扩充后两者都从 ✅ 变成 ❌
- "Taylor Swift"在 Grammy 页面出现频次远高于"Serban Ghenea"（作为表演者 vs 录音师）
- **结论**：SealQA 的核心挑战是信息优先级判断，不是信息量

### 3. Evabot 更快，ROMA 在推理型题目上更准
- Evabot Worker 分步提取容易在中间阶段引入错误（如 Q4 算出 81 岁）
- ROMA 单阶段 CoT 保留了完整推理上下文，Q4 正确算出 84
- 但 ROMA max_tokens 限制导致部分长内容被截断需要重试，实际耗时更长

### 4. 评分 Bug 已修复
- 原始评分要求所有非停用词都出现在预测中
- 修复后：数字答案只需数字匹配，文本答案取最长关键实体匹配
- ROMA Q4 由此从 ❌ 纠正为 ✅

---

## 复现步骤

```bash
# 1. 安装依赖
pip install datasets requests markdownify beautifulsoup4 openai dspy

# 2. 抓取全部 111 题（首次运行，支持断点续跑）
python benchmark/fetch_sources.py

# 3. 运行 Evabot
python benchmark/run_evabot.py

# 4. 运行 ROMA
python benchmark/run_roma.py
```

> ⚠️ 全量 111 题预计耗时 35-50 分钟，需要 DeepSeek API Key（设置环境变量 `deepseek_key`）

---

## 后续计划

- [ ] 跑全量 111 题，获得统计显著的准确率对比
- [ ] 分题型分析（freshness / conflicting / calculation 等类别）
- [ ] 探索 Evabot 的改进方向：在 Worker 阶段加入可信度权重
