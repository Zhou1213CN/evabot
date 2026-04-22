---
name: get-online-info
description: 互联网信息获取与外部检索中枢。
---

# 互联网与外部检索 (Get Online Information)

## 模块定位
本模块是所有**涉及互联网公开信息获取**的顶层入口。当任务需要超出大模型自身知识库的时效性数据、新闻动态、公开资料或特定网页内容时，应进入此模块。

## 任务路由与子 Skill 分发
在执行具体的信息获取任务前，请先评估信息来源的具体要求：

1. **通用知识与即时问答**：若需要快速了解某个实体、事件或获取综合性的搜索结果，请优先使用本目录自带的通用聚合搜索工具。
2. **特定垂直网站数据**：若需要抓取特定复杂网页（如深度解析某个具体的 URL 内容）、处理需登录态的网页，或执行针对特定平台的操作，请调用该目录下的专属子 Skill。

## 默认通用搜索工具 (Aliyun IQS)

当前顶层模块内置了一个通用的聚合搜索脚本 `get_online_info.py`。该脚本可完成基础的搜索引擎交互和内容提炼。

### 使用环境与要求
- 系统必须已设置环境变量 `ALIYUN_API_KEY`。

### 工具调用示例
- Linux环境: `python get_online_info.py --data '{"query": "北京天气"}'`
- Windows环境: `python get_online_info.py --data "{\"query\": \"北京天气\"}"`

### 核心参数定义 (JSON 格式传入 --data)
- query: string | 必填 | 搜索问题。建议30字符以内，超出500字符会被截断。
- timeRange: string | 默认: "NoLimit" | 时间范围："OneDay", "OneWeek", "OneMonth", "OneYear", "NoLimit"。
- category: string | 默认: None | 查询分类(如finance, law, medical, internet, tax, news_province, news_center)，多个行业用逗号分隔。一般通用场景，不要指定category，会影响召回效果。
- engineType: string | 默认: "Generic" | 引擎类型："Generic"(标准版,返回约10条),"GenericAdvanced"(增强版,返回40-80条，收费选项), "LiteAdvanced"(轻量版,返回1-50条)。
- city: string | 默认: None | 城市名，如“北京市”。仅对Generic引擎生效。
- ip: string | 默认: None | 位置IP。优先级低于城市，仅对Generic引擎生效。
- mainText: bool | 默认: False | 是否返回长正文。
- markdownText: bool | 默认: False | 是否返回markdown格式正文。
- richMainBody: bool | 默认: False | 是否返回富文本全正文。
- summary: bool | 默认: False | 是否返回增强摘要(收费选项)。
- rerankScore: bool | 默认: True | 是否进行Rerank并返回得分。

## 执行纪律
- **信息提炼**：获取到外部网络数据后，必须进行阅读、清洗和去重，然后按任务要求精准提取目标信息。禁止将原始的长篇代码或无关的网页底栏文本直接抛给上游。
- **信息验证**：交叉比对不同搜索结果的信息一致性。遇到明显冲突的数据，需在最终汇报中客观列出不同来源的差异。
- **失败与降级**：如果默认的搜索工具报错，需如实分析报错信息。若调整关键词多次搜索后仍未能找到有效信息，请明确汇报“未能检索到相关有效信息”，严禁基于大模型本身的旧知识捏造即时数据。
