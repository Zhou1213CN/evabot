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
| `benchmark/run_evabot.py` | 阶段二：Evabot 架构（Butler → Solver 自主决策 → Worker 按需派发） |
| `benchmark/run_roma.py` | 阶段三：ROMA 架构（DSPy ChainOfThought 单阶段推理） |
| `benchmark/evabot_results.json` | Evabot 全量结果（含 token 和 LLM 调用次数统计） |
| `benchmark/roma_results.json` | ROMA 全量结果（含 token 和 LLM 调用次数统计） |

---

## 架构对比

### Evabot（Butler → Solver 自主决策 → Worker 按需派发）
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
- Solver 初始只看 URL 列表，Worker 负责读取具体内容（上下文分割）
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

## 全量结果（76 题，never/slow-changing）

| 指标 | Evabot | ROMA |
|------|--------|------|
| **准确率** | **44.7% (34/76)** ✅ | **32.9% (25/76)** |
| 平均耗时 | 36.9s / 题 | 6.3s / 题 |
| **平均 LLM 调用** | **5.7 次 / 题** | **1.0 次 / 题** |
| 平均 Worker 调用 | 2.4 次 / 题 | — |
| **平均 Token 消耗** | **27,194 / 题** | **10,911 / 题** |
| **Token 消耗比** | **2.49×** | 1× |

### 题目级别对比

| | 数量 |
|--|------|
| Evabot 赢（Evabot ✅ ROMA ❌） | **14 题** |
| Evabot 输（Evabot ❌ ROMA ✅） | **5 题** |
| 平局 | 57 题 |

> Evabot 用 2.49× token 换来了 **+11.8 个百分点**的准确率提升，且在 76 题中净胜 9 题。

### 按 question_types 分析

| 题型 | Evabot | ROMA | Evabot 优势 |
|------|--------|------|------------|
| temporal tracking (6题) | **66.7%** | 50.0% | +16.7% |
| entity/event disambiguation (45题) | **44.4%** | 26.7% | +17.7% |
| cross-lingual reasoning (5题) | 40.0% | 40.0% | 持平 |
| advanced reasoning (55题) | **38.2%** | 32.7% | +5.5% |
| **false-premise (8题)** | **25.0%** | **12.5%** | +12.5%，**两者共同短板** |

---

## 关键发现

### 1. Evabot 多 agent 架构在所有题型上均优于或持平 ROMA
- 最大优势在 entity/event disambiguation（+17.7%）和 temporal tracking（+16.7%）
- 多轮 Worker 调用在需要精确定位特定事实的题目上优势明显
- 仅在 5 题上输给 ROMA，说明 Evabot 架构极少出现"多分析反而答错"的情况

### 2. false-premise 是两者共同短板
- Evabot 25%，ROMA 12.5%，两者都很差
- 根因：两个架构都缺乏主动验证题目前提的机制
- 这是下一步优化的核心方向

### 3. Runaway 问题已解决，根因是 source 为空
- 早期实验中出现 15 次 LLM 调用的失控现象
- 根因：macOS LibreSSL SSL 兼容性问题导致 fetch 静默失败，source 为空时 Solver 反复派 Worker 重试
- **已修复**：fetch 加 SSL fallback + 续跑逻辑区分空/有内容缓存；Worker 遇到空 source 明确返回"不可用，勿重试"

### 4. Source 截断影响 Worker 提取精度
- 所有 source 截断至 30k 字符，计数类题目（如"两人共同访问的国家数"）的完整列表可能被截断
- Worker 只能提取截断范围内的信息，导致计数偏差
- 这是当前架构下影响 advanced reasoning 类题目准确率的主要瓶颈

### 5. Token 效率
- Evabot 平均 5.7 次 LLM 调用 / 题，ROMA 固定 1 次
- 多 agent 的信息质量优势以约 2.49× token 为代价
- 控制 Worker 调用次数是降低成本的核心杠杆

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

# 3. 运行 Evabot（Butler → Solver 自主决策架构）
deepseek_key=<YOUR_KEY> python benchmark/run_evabot.py

# 4. 运行 ROMA（DSPy ChainOfThought）
deepseek_key=<YOUR_KEY> python benchmark/run_roma.py

# 仅跑前 5 题快速验证
deepseek_key=<YOUR_KEY> python benchmark/run_evabot.py --limit 5
deepseek_key=<YOUR_KEY> python benchmark/run_roma.py --limit 5
```

> ⚠️ 全量 76 题实测耗时：Evabot ~47min，ROMA ~8min。

---

## 后续计划

- [ ] 针对 false-premise 题型优化：框架层面加入前提验证机制
- [ ] 解决 source 截断问题：计数类题目需要完整列表，考虑动态扩展 source 窗口
- [ ] 控制 Worker 调用上限，在精度和 token 效率之间取得更好平衡
- [ ] 探索 ROMA 优化方向：multi-hop 场景下拆分来源分别推理
