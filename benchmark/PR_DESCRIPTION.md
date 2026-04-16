# Benchmark: Evabot vs ROMA on SealQA (seal_0)

## 目标

在**完全公平**的条件下对比 Evabot 多智能体架构与 ROMA（DSPy ChainOfThought）在处理冲突网络信息时的性能差异。

## 实验设计（"同卷闭卷考试"）

| 变量 | 设定 |
|------|------|
| 数据集 | [vtllms/sealqa](https://huggingface.co/datasets/vtllms/sealqa)，`seal_0` 子集 |
| 有效题目 | **76 道**（过滤掉 35 道 fast-changing 时效性题目，仅保留 never-changing + slow-changing） |
| 底层模型 | `deepseek-chat`（两边完全一致） |
| 输入来源 | 同一份 `cached_sources.json`（按 URL anchor 智能提取 Wikipedia 相关小节） |
| 评分方式 | 三级匹配：否定语义 → 数字精确匹配 → 最长关键实体匹配（忽略大小写） |

> **为什么不直接和论文的 45.9% 比？**
> ROMA 论文使用 GPT-4o-mini 的实时搜索接口，与本地 DeepSeek 搜出来的原材料完全不同，直接比绝对分数没有意义。本方案让两个架构读同一份缓存网页，做"同卷闭卷考试"。

> **为什么过滤 fast-changing 题目？**
> 数据集中 35 道 fast-changing 题目（如"票房前10最新电影"答案是 Zootopia 2，一部 2025 年上映的电影）的答案依赖实时数据，任何模型都无法从 Wikipedia 中得到正确答案，放入对比毫无意义。

---

## Benchmark 文件说明

| 文件 | 作用 |
|------|------|
| `benchmark/fetch_sources.py` | 阶段一：加载 SealQA，按 URL anchor 智能提取相关小节，默认跳过 fast-changing |
| `benchmark/cached_sources.json` | 共享输入：每道题的网页原文（两个架构读同一份） |
| `benchmark/run_evabot.py` | 阶段二：Evabot 架构（Solver 自主决策 + 按需派发 Worker） |
| `benchmark/run_roma.py` | 阶段三：ROMA 架构（DSPy ChainOfThought 单阶段推理） |
| `benchmark/evabot_results.json` | Evabot pilot 结果（含 token 统计） |
| `benchmark/roma_results.json` | ROMA pilot 结果（含 token 统计） |

---

## 架构对比

### Evabot（Solver 自主决策 + 按需派 Worker）
```
Butler 把问题交给 Solver

Solver 主循环（最多 8 轮，带 tool_call）：
  ├── analyze_source(source_id, aspect)
  │     → Worker 执行：精读指定来源，返回事实摘要
  │     → 结果以 tool message 回传给 Solver，Solver 可继续追问
  └── final_answer(answer, reasoning)
        → 提交最终答案，结束循环
```
- **Solver 完全自主**：要不要派 Worker、派几次、关注哪个角度，均由 Solver 的 tool_call 决定
- Worker 调用次数不固定（实测 0~5 次/题）
- 每次 Worker 调用都携带完整 snippet → **Token 消耗随调用次数线性增长**

### ROMA（DSPy ChainOfThought 单阶段）
```
DSPy ChainOfThought(ConflictAwareQA):
  input:  question + 所有来源拼接成字符串
  output: answer（含内置思维链 reasoning）
```
- 每道题仅 1 次 LLM 调用，所有来源一次性输入
- 依赖模型在单次推理中自行识别并解决冲突

---

## Pilot 结果（5 题，never-changing，含 Token 统计）

| 指标 | Evabot（Solver 自主）| ROMA (DSPy CoT) |
|------|---------------------|-----------------|
| 准确率 | **60%** (3/5) | **60%** (3/5) |
| 平均耗时 | 50.5s / 题 | 38.8s / 题 |
| 平均 Worker 调用 | 2.8 次 / 题（0~5不等） | — (单阶段) |
| **平均 Token 消耗** | **33,052 / 题** | **19,061 / 题** |
| **Token 消耗比** | **1.74×** | 1× (基准) |

### 逐题详情（5 题 pilot，never-changing）

| # | 问题摘要 | 标准答案 | Evabot | ROMA | 难点 |
|---|---------|---------|--------|------|------|
| 1 | 2026 FIFA 女足三国主办？ | 2026年不会有女足世界杯 | 加拿大、墨西哥、美国 ❌ | 加拿大、墨西哥、美国 ❌ | false-premise，两者都被题目迷惑 |
| 2 | Spotify 唯一登顶女艺人 | No one | No female artist achieved this ✅ | None ✅ | 否定语义题，两者均正确识别 |
| 3 | 占热带雨林四分之一的是？ | Congolian rainforests | Congolian rainforests ✅ | Congolian rainforests ✅ | 两个来源存在冲突（亚马逊 vs 刚果），均正确判断 |
| 4 | Trump 和 Biden 共同访问的国家数 | 18 | 12 ❌ | 19 ❌ | 需跨两页 Wikipedia 交叉统计，均出现偏差 |
| 5 | 原地稳定排序算法名称 | Block sort | Block sort ✅ | Block sort ✅ | 经典知识题，两者均答对 |

---

## 关键发现

### 1. fast-changing 题目应全部剔除
- 数据集中 35 道 fast-changing 题目（effective_year 2025/2026）依赖实时数据
- 典型例子："票房前10最新电影"答案为 Zootopia 2（2025年上映），任何模型在训练截止后都无法答对
- **数据集有效题目：76 道**（never-changing 34 + slow-changing 42）

### 2. URL Anchor 智能小节提取 > 固定字符截断
- 早期截断 3000 字：Q5（IATA 成员数）两者均答错（仅看到局部的 352）
- 改用 anchor 定位相关小节后：两者均答对（找到完整总数 370）
- **结论**：对 Wikipedia 长页面，提取正确小节比截断更重要

### 3. 更多内容 ≠ 更高准确率（噪音问题）
- 扩大内容后 Q1（Grammy 获奖者）两者从 ✅ 变 ❌
- "Taylor Swift" 在页面出现频次远高于真正答案 "Serban Ghenea"
- **结论**：SealQA 的核心挑战是**信息优先级判断**，而非信息量

### 4. Token 消耗：Evabot 约为 ROMA 的 1.74 倍
- Evabot：Worker 每次调用携带完整 snippet → 多次调用 = 多倍 input token
- ROMA：单次调用，所有来源拼接后一次性送入
- 在准确率相同时，ROMA 的 token 效率更高
- 但 Evabot 的多轮 Worker 策略在复杂推理题上有潜力（Q3 Evabot 派了 4 次 Worker 反而答对）

### 5. 评分器迭代（3 轮修复）
| 版本 | 问题 | 修复 |
|------|------|------|
| v1 | 要求所有非停用词均出现 | → |
| v2 | 数字答案只需数字匹配；文本取最长关键实体 | "84" 匹配 "84 years old" ✅ |
| v3 | 新增否定语义匹配 | "No female artist" 匹配 "No one" ✅ |

### 6. false-premise 题是共同盲区
- Q1（FIFA 女足）：题目问"哪三国主办"，但正确答案是"2026年根本没有女足世界杯"
- 两个架构都被前提误导，没有先验证题目假设
- **结论**：false-premise 题需要专门的前提验证能力

---

## 复现步骤

```bash
# 1. 安装依赖
pip install datasets requests markdownify beautifulsoup4 openai dspy

# 2. 抓取 76 道有效题目（默认跳过 fast-changing，支持断点续跑）
python benchmark/fetch_sources.py

# 3. 运行 Evabot（Solver 自主决策架构）
deepseek_key=<YOUR_KEY> python benchmark/run_evabot.py

# 4. 运行 ROMA（DSPy ChainOfThought）
deepseek_key=<YOUR_KEY> python benchmark/run_roma.py

# 如需包含 fast-changing 题目（不推荐）
python benchmark/fetch_sources.py --include-fast-changing
```

> ⚠️ 全量 76 题预计耗时：Evabot ~60min，ROMA ~20min。建议分别后台运行。

---

## 后续计划

- [ ] 跑全量 76 题，获得统计显著的准确率与 Token 消耗对比
- [ ] 按 question_types 分组分析（advanced_reasoning / entity_disambiguation / false_premise / temporal_tracking）
- [ ] 探索 Evabot 优化方向：Worker 加入可信度权重，减少冗余调用次数
- [ ] 探索 ROMA 优化方向：multi-hop 场景下拆分来源分别推理
