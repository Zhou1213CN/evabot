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
| `benchmark/run_evabot.py` | 阶段二：Evabot 架构（Solver 读取全部来源后自主决策，按需派发 Worker） |
| `benchmark/run_roma.py` | 阶段三：ROMA 架构（DSPy ChainOfThought 单阶段推理） |
| `benchmark/evabot_results.json` | Evabot pilot 结果（含 token 和 LLM 调用次数统计） |
| `benchmark/roma_results.json` | ROMA pilot 结果（含 token 和 LLM 调用次数统计） |

---

## 架构对比

### Evabot（Solver 读源 + 自主决策 + 按需派 Worker）
```
Butler 把问题交给 Solver（同时携带所有 source 完整内容）

Solver 主循环（最多 8 轮，带 tool_call）：
  ├── [直接] final_answer(answer, reasoning)
  │     → 来源清晰时跳过 Worker，直接回答（1 次 LLM 调用）
  ├── analyze_source(source_id, aspect)
  │     → Worker 执行：精读指定来源，返回事实摘要
  │     → 结果以 tool message 回传给 Solver，Solver 可继续追问
  └── final_answer(answer, reasoning)
        → 基于 Worker 反馈提交答案
```
- **Solver 拿到题目时已能看到所有 source 内容**，自主决定直答还是派 Worker
- Worker 调用次数不固定（可为 0，即 Solver 直接回答）
- 统计指标：`llm_calls = solver_calls + worker_calls`

### ROMA（DSPy ChainOfThought 单阶段）
```
DSPy ChainOfThought(ConflictAwareQA):
  input:  question + 所有来源拼接成字符串
  output: answer（含内置思维链 reasoning）
```
- 每道题固定 1 次 LLM 调用，所有来源一次性输入
- 依赖模型在单次推理中自行识别并解决冲突

---

## Pilot 结果（5 题，never-changing，架构修正后）

| 指标 | Evabot（Solver 自主）| ROMA (DSPy CoT) |
|------|---------------------|-----------------|
| **准确率** | **80%** (4/5) ✅ | **60%** (3/5) |
| 平均耗时 | 27.0s / 题 | 9.7s / 题 |
| **平均 LLM 调用** | **4.6 次 / 题**（Solver+Worker） | **1.0 次 / 题** |
| 平均 Worker 调用 | 2.0 次 / 题（0~8 不等） | — (单阶段) |
| **平均 Token 消耗** | **71,251 / 题** | **18,452 / 题** |
| **Token 消耗比** | **3.86×**（含 Q4 异常值） | 1× (基准) |

> ⚠️ Q4 为异常值：Evabot 在该题使用了 15 次 LLM 调用、212K tokens 仍答错，严重拉高平均值。剔除 Q4 后 Evabot 平均 token 为 36,474（约 ROMA 的 1.97×）。

### 逐题详情

| # | 问题摘要 | 标准答案 | Evabot | ROMA | 备注 |
|---|---------|---------|--------|------|------|
| 1 | 2026 FIFA 女足三国主办？ | 2026年不会有女足世界杯 | False premise, no 2026 Women's WC ✅ | 加拿大、墨西哥、美国 ❌ | **架构修正后 Evabot 正确识别 false-premise，ROMA 被题目迷惑** |
| 2 | Spotify 唯一登顶女艺人 | No one | No female artist… Bad Bunny topped ✅ | None ✅ | Evabot 0 Worker，直接回答（1 次 LLM） |
| 3 | 占热带雨林四分之一的是？ | Congolian rainforests | Congolian rainforests ✅ | Congolian rainforests ✅ | Evabot 1 次 Worker 辅助确认 |
| 4 | Trump 和 Biden 共同访问的国家数 | 18 | 17 ❌ | 19 ❌ | 两者均出错；Evabot 消耗 15 LLM 调用、212K token，需加调用上限 |
| 5 | 原地稳定排序算法名称 | Block sort | Block sort ✅ | Block sort ✅ | Evabot 0 Worker，直接回答（1 次 LLM） |

---

## 关键发现

### 1. 架构修正：Solver 先读源，再决策
- 旧架构：Solver 只看 URL 列表，必须派 Worker 才能看到内容 → 强制多 agent
- 新架构：Solver 初始 context 含所有 source 完整内容，可直接回答（0 Worker）或有针对性地派 Worker 深挖
- **效果**：Q2、Q5 降为 1 次 LLM 调用；Q1 false-premise 从 ❌ 变 ✅

### 2. Evabot 准确率从 60% 提升到 80%，超过 ROMA
- 架构修正后 Evabot 正确处理了 Q1 false-premise（ROMA 仍然答错）
- Evabot 在能直接推断的题目上不再浪费 Worker 调用

### 3. Q4 存在"失控"问题（最高优先级待修复）
- Trump/Biden 共同访问国家数：Evabot 用了 15 次 LLM 调用、212K tokens，仍得出错误答案 17
- 根因：需跨两页 Wikipedia 交叉统计，Solver 反复派 Worker 却无法收敛
- **后续**：需要为 MAX_SOLVER_ROUNDS 设置更严格的上限，并加入 early-stop 机制

### 4. LLM 调用次数是关键差异指标
- ROMA：固定 1 次/题
- Evabot：平均 4.6 次/题（最低 1 次，Q4 高达 15 次）
- Token 消耗和延迟均与 LLM 调用次数正相关
- 控制 Worker 调用次数 = 控制成本的核心杠杆

### 5. fast-changing 题目应全部剔除（已实现）
- 35 道 fast-changing 题目依赖实时数据，任何模型均无法从缓存 Wikipedia 答对
- 两个 runner 默认过滤，`--include-fast-changing` 可手动开启

### 6. 评分器三级匹配（稳定）
| 优先级 | 规则 | 示例 |
|--------|------|------|
| 1 | 否定语义匹配 | "No one" → 预测含 "no"/"none"/"nobody" 即对 |
| 2 | 数字精确匹配 | 答案 "18" → 预测含 "18" 即对 |
| 3 | 最长关键实体匹配 | 答案 "Block sort" → 预测含 "block" 即对 |

---

## 复现步骤

```bash
# 1. 安装依赖
pip install datasets requests markdownify beautifulsoup4 openai dspy

# 2. 抓取 76 道有效题目（默认跳过 fast-changing，支持断点续跑）
python benchmark/fetch_sources.py

# 3. 运行 Evabot（Solver 读源后自主决策架构）
deepseek_key=<YOUR_KEY> python benchmark/run_evabot.py

# 4. 运行 ROMA（DSPy ChainOfThought）
deepseek_key=<YOUR_KEY> python benchmark/run_roma.py

# 仅跑前 5 题快速验证
deepseek_key=<YOUR_KEY> python benchmark/run_evabot.py --limit 5
deepseek_key=<YOUR_KEY> python benchmark/run_roma.py --limit 5
```

> ⚠️ 全量 76 题预计耗时：Evabot ~60min，ROMA ~20min。建议分别后台运行。

---

## 后续计划

- [ ] 为 Evabot 加入 early-stop / 最大 Worker 调用上限，解决 Q4 类失控问题
- [ ] 跑全量 76 题，获得统计显著的准确率与 Token 消耗对比
- [ ] 按 question_types 分组分析（advanced_reasoning / entity_disambiguation / false_premise / temporal_tracking）
- [ ] 探索 ROMA 优化方向：multi-hop 场景下拆分来源分别推理
