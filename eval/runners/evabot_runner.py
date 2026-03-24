"""
eval/runners/evabot_runner.py

Run benchmark questions through the live EVABot server via WebSocket.

EVABot architecture recap (relevant to evaluation):
  - Butler  → receives user message, dispatches to Solver
  - Solver  → decomposes task, spawns Worker(s) with minimal local context
  - Worker  → executes, may trigger `communicate_with_upstream` (Escalate)
  - Auditor → validates output; failure triggers skill evolution loop

What we track per question:
  - final_answer       : str
  - escalate_count     : int   (# of communicate_with_upstream calls detected)
  - escalate_resolved  : int   (# that were resolved with a better answer)
  - evolution_triggered: bool  (did Auditor fire a skill evolution?)
  - cost               : float (cumulative LLM cost in USD)
  - latency_s          : float
  - task_nodes         : List  (from /api/tasks/{solve_id})
  - raw_messages       : List  (WebSocket stream)
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional

import requests
import websocket  # websocket-client library

from eval.config import (
    EVABOT_BASE_URL,
    EVABOT_WS_URL,
    EVABOT_TIMEOUT_S,
)


class EvaBotRunner:
    """
    Submits a question to the running EVABot server and waits for the
    assistant response. Captures Escalate events and task-tree metrics.
    """

    def __init__(
        self,
        base_url: str = EVABOT_BASE_URL,
        ws_url: str = EVABOT_WS_URL,
        timeout_s: int = EVABOT_TIMEOUT_S,
    ):
        self.base_url  = base_url.rstrip("/")
        self.ws_url    = ws_url.rstrip("/")
        self.timeout_s = timeout_s

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def run(self, prompt: str, context_hint: str = "") -> Dict[str, Any]:
        """
        Submit *prompt* to EVABot and collect results.

        Parameters
        ----------
        prompt       : The benchmark question / task description.
        context_hint : Extra background fed as a preamble (used in
                       SEAL-0 to inject the evidence snippets).

        Returns
        -------
        dict with keys:
          final_answer, escalate_count, escalate_resolved,
          evolution_triggered, cost, latency_s, task_nodes, raw_messages
        """
        channel_id = f"eval_{uuid.uuid4().hex[:12]}"
        full_prompt = f"{context_hint}\n\n{prompt}".strip() if context_hint else prompt

        messages: List[Dict] = []
        assistant_parts: List[str] = []
        escalate_count    = 0
        evolution_triggered = False
        t_start = time.time()

        ws_endpoint = f"{self.ws_url}/ws/chat/{channel_id}"

        try:
            ws = websocket.create_connection(ws_endpoint, timeout=self.timeout_s)
            # Send the user message
            ws.send(json.dumps({"content": full_prompt}))

            deadline = time.time() + self.timeout_s
            while time.time() < deadline:
                ws.settimeout(max(1.0, deadline - time.time()))
                try:
                    raw = ws.recv()
                except websocket.WebSocketTimeoutException:
                    break

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                messages.append(msg)
                sender = msg.get("sender", "")
                content = msg.get("content", "")
                msg_type = msg.get("message_type", "")

                # Collect assistant (butler) reply segments
                if sender in ("butler", "solver") and msg.get("message_role") == "assistant":
                    assistant_parts.append(content)

                # Detect Escalate: Worker calling communicate_with_upstream
                tool_calls = (msg.get("data") or {}).get("tool_calls", [])
                for tc in tool_calls:
                    fn_name = (tc.get("function") or {}).get("name", "")
                    if fn_name == "communicate_with_upstream":
                        escalate_count += 1

                # Detect evolution: Auditor updating worker.md or skill
                if sender == "auditor" and "evolution" in content.lower():
                    evolution_triggered = True
                if sender == "auditor" and "skill" in content.lower() and "upgrad" in content.lower():
                    evolution_triggered = True

                # Stop when Butler sends the final user-facing answer
                if sender == "butler" and msg.get("send_type") == "user" and content.strip():
                    break

            ws.close()
        except Exception as e:
            messages.append({"error": str(e)})

        latency_s = round(time.time() - t_start, 2)

        # ── Pull task tree for detailed metrics ──
        task_nodes: List[Dict] = []
        total_cost = 0.0
        escalate_resolved = 0

        try:
            resp = requests.get(
                f"{self.base_url}/api/tasks",
                timeout=10,
            )
            if resp.ok:
                tasks = resp.json()
                # Find the task matching our channel
                for task in tasks:
                    if task.get("channel_id") == channel_id:
                        solve_id = task["solve_id"]
                        detail = requests.get(
                            f"{self.base_url}/api/tasks/{solve_id}",
                            timeout=10,
                        )
                        if detail.ok:
                            tree = detail.json()
                            task_nodes = tree.get("nodes", [])
                            total_cost = tree.get("total_cost", 0.0)
                            # Count resolved escalations from audit feedback
                            for node in task_nodes:
                                for attempt in node.get("attempts", []):
                                    fb = attempt.get("audit_feedback", "")
                                    if "escalat" in fb.lower() and "success" in fb.lower():
                                        escalate_resolved += 1
                        break
        except Exception:
            pass

        final_answer = " ".join(assistant_parts).strip() or _extract_last_content(messages)

        return {
            "final_answer":        final_answer,
            "escalate_count":      escalate_count,
            "escalate_resolved":   escalate_resolved,
            "evolution_triggered": evolution_triggered,
            "cost":                total_cost or _sum_cost(messages),
            "latency_s":           latency_s,
            "task_nodes":          task_nodes,
            "raw_messages":        messages,
        }

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    def health_check(self) -> bool:
        """Return True if the EVABot server is reachable."""
        try:
            r = requests.get(f"{self.base_url}/", timeout=5)
            return r.status_code == 200
        except Exception:
            return False


def _extract_last_content(messages: List[Dict]) -> str:
    """Fallback: grab the last non-empty content string from messages."""
    for msg in reversed(messages):
        c = msg.get("content", "").strip()
        if c and msg.get("sender") in ("butler", "solver", "worker"):
            return c
    return ""


def _sum_cost(messages: List[Dict]) -> float:
    """Sum cost field from data payloads across all messages."""
    total = 0.0
    for msg in messages:
        data = msg.get("data") or {}
        total += float(data.get("cost", 0.0))
    return round(total, 6)
