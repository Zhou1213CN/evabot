---
name: code-developer
description: 软件工程与代码开发中枢,创建市面上不存在的功能时使用。
---

# Code Developer (Software Engineering)

## 模块定位
本模块是所有**代码编写、工程修改与软件测试**的顶层入口。当任务目标不是使用现有软件，而是创造新的逻辑、修复系统 Bug、开发新功能或编写独立脚本时，应进入此模块。

## 任务路由与子 Skill 分发
1. **单文件与简单脚本**：若只需编写或修改独立的轻量级脚本，直接在当前层级调用基础工具（`read_file`, `write_file`, `edit_file`, `exec_command`）完成闭环。
2. **复杂工程与特定框架**：若涉及大型项目或需要完整的工程化构建，需向下派发给专用的子 Skill。

## 强制工作流 (Mandatory Workflow)
在进行代码开发时，必须严格遵守以下执行顺序：
1. **方案确认**：在编写或修改大量代码前，必须先输出架构设计或核心修改思路，并调用 `communicate_with_upstream` 向上游发起确认。得到肯定答复后方可动手。
2. **测试驱动**：代码编写完成后，禁止直接宣告成功。必须调用 `exec_command` 执行代码或运行单元测试进行验证。

---

## 核心执行纪律 (Behavioral Guidelines)

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding
**Don't assume. Don't hide confusion. Surface tradeoffs.**
- State your assumptions explicitly. If uncertain, ask via `communicate_with_upstream`.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.

### 2. Simplicity First
**Minimum code that solves the problem. Nothing speculative.**
- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

### 3. Surgical Changes
**Touch only what you must. Clean up only your own mess.**
When using `edit_file` or `write_file`:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.
- Remove imports/variables/functions that YOUR changes made unused.

### 4. Goal-Driven Execution
**Define success criteria. Loop until verified.**
Transform tasks into verifiable goals and strictly use `exec_command` to verify:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```text
1. [Step] → verify: [exec_command check]
2. [Step] → verify: [exec_command check]