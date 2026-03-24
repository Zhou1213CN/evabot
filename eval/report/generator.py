"""
eval/report/generator.py

Generates a self-contained HTML report and a JSON summary from the
benchmark results collected by MetricsCollector.

The report mirrors the three-capability framing from the EVABot paper:
  1. Anti-interference  (local-view vs. global-view accuracy + noise resistance)
  2. Escalate efficacy  (trigger rate, win rate after escalate)
  3. Evolution delta    (accuracy improvement after skill evolution)
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from eval.metrics.collector import MetricsCollector


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────

def generate_report(
    collectors: List[MetricsCollector],
    output_dir: str,
) -> str:
    """
    Build HTML + JSON report from a list of per-benchmark collectors.
    Returns path to the HTML file.
    """
    summaries = [c.summary() for c in collectors]
    all_records = {c.benchmark: c.records for c in collectors}

    # ── JSON dump ──
    json_path = os.path.join(output_dir, "evabot_eval_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"summaries": summaries, "records": all_records},
            f, ensure_ascii=False, indent=2,
        )

    # ── HTML report ──
    html = _build_html(summaries, all_records)
    html_path = os.path.join(output_dir, "evabot_eval_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return html_path


# ─────────────────────────────────────────────────────────────
# HTML builder
# ─────────────────────────────────────────────────────────────

def _build_html(summaries: List[Dict], all_records: Dict) -> str:
    benchmark_sections = "\n".join(
        _benchmark_section(s, all_records.get(s["benchmark"], []))
        for s in summaries
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EVABot Capability Evaluation Report</title>
<style>
  :root {{
    --eva: #6C63FF;
    --base: #FF6B6B;
    --green: #4CAF50;
    --gold: #FFC107;
    --bg: #0F0F1A;
    --card: #1A1A2E;
    --border: #2D2D4E;
    --text: #E0E0F0;
    --muted: #8888AA;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 2rem;
    line-height: 1.6;
  }}
  h1 {{ font-size: 2rem; color: var(--eva); margin-bottom: .25rem; }}
  h2 {{ font-size: 1.4rem; color: var(--eva); margin: 1.5rem 0 .75rem; border-bottom: 1px solid var(--border); padding-bottom: .4rem; }}
  h3 {{ font-size: 1.1rem; color: var(--gold); margin: 1.2rem 0 .5rem; }}
  .subtitle {{ color: var(--muted); margin-bottom: 2rem; font-size: .95rem; }}
  .grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.5rem; }}
  .grid-2 {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-bottom: 1.5rem; }}
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
  }}
  .stat-label {{ font-size: .75rem; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }}
  .stat-value {{ font-size: 2rem; font-weight: 700; margin: .2rem 0; }}
  .stat-sub   {{ font-size: .85rem; color: var(--muted); }}
  .positive {{ color: var(--green); }}
  .negative {{ color: var(--base); }}
  .neutral  {{ color: var(--gold); }}
  .eva-color  {{ color: var(--eva); }}
  .base-color {{ color: var(--base); }}
  .badge {{
    display: inline-block;
    padding: .2rem .6rem;
    border-radius: 9999px;
    font-size: .75rem;
    font-weight: 600;
    margin-left: .5rem;
  }}
  .badge-green {{ background: rgba(76,175,80,.2); color: var(--green); }}
  .badge-red   {{ background: rgba(255,107,107,.2); color: var(--base); }}
  .benchmark-block {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 2rem;
  }}
  table {{ width: 100%; border-collapse: collapse; font-size: .88rem; }}
  th {{ text-align: left; padding: .5rem .75rem; color: var(--muted); font-weight: 600; border-bottom: 1px solid var(--border); }}
  td {{ padding: .5rem .75rem; border-bottom: 1px solid rgba(45,45,78,.5); }}
  tr:hover td {{ background: rgba(108,99,255,.05); }}
  .progress-bar {{ height: 8px; border-radius: 4px; background: var(--border); overflow: hidden; margin-top: .3rem; }}
  .progress-fill {{ height: 100%; border-radius: 4px; }}
  .bar-eva  {{ background: var(--eva); }}
  .bar-base {{ background: var(--base); }}
  .section-intro {{ color: var(--muted); font-size: .9rem; margin-bottom: 1rem; }}
  .capability-tag {{
    display: inline-block; padding: .2rem .7rem; border-radius: 6px;
    font-size: .7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .06em; margin-bottom: .75rem;
  }}
  .cap1 {{ background: rgba(108,99,255,.15); color: var(--eva); }}
  .cap2 {{ background: rgba(255,193,7,.15);  color: var(--gold); }}
  .cap3 {{ background: rgba(76,175,80,.15);  color: var(--green); }}
  footer {{ margin-top: 3rem; color: var(--muted); font-size: .82rem; text-align: center; }}
</style>
</head>
<body>

<h1>EVABot Capability Evaluation</h1>
<p class="subtitle">
  Benchmarks adapted from ROMA (arXiv:2602.01848) &nbsp;|&nbsp;
  Three-axis evaluation: Anti-interference · Escalate Efficacy · Evolution Delta
</p>

{_overview_cards(summaries)}

<h2>Benchmark Results</h2>
{benchmark_sections}

<h2>Methodology</h2>
<div class="card">
<h3>Experimental Design</h3>
<p class="section-intro">
Each benchmark is run in two modes:
<br><br>
<strong style="color:var(--eva)">EVABot (Local View)</strong> —
Each worker node operates on a minimal, task-relevant context slice.
Information gaps trigger the Escalate mechanism.
Failed tasks initiate the skill evolution loop.
<br><br>
<strong style="color:var(--base)">Baseline (Global View)</strong> —
A single LLM call receives all retrieved documents concatenated
into one context window, with no decomposition or escalation.
<br><br>
<strong>Judge</strong>: GPT-4o-mini (same as ROMA paper) for factual accuracy, writing quality, and ablation design scoring.
</p>
<h3>Three Capability Axes</h3>
<table>
<tr><th>Axis</th><th>Metric</th><th>Benchmark(s)</th></tr>
<tr><td>1. Anti-interference</td><td>accuracy_delta = EVABot_acc − Baseline_acc; noise_resistance_score</td><td>SEAL-0, FRAMES, SimpleQA, EQ-Bench, AbGen</td></tr>
<tr><td>2. Escalate Efficacy</td><td>escalate_rate_per_q; escalate_win_rate (% resolved → correct)</td><td>All</td></tr>
<tr><td>3. Evolution Delta</td><td>post_accuracy − pre_accuracy on failed items after skill evolution</td><td>SEAL-0, FRAMES, SimpleQA</td></tr>
</table>
</div>

<footer>Generated by EVABot Evaluation Framework &nbsp;·&nbsp; Based on ROMA (arXiv:2602.01848)</footer>
</body>
</html>
"""


def _overview_cards(summaries: List[Dict]) -> str:
    if not summaries:
        return ""

    # Aggregate across all benchmarks
    ns = [s["n"] for s in summaries if s["n"] > 0]
    total_n = sum(ns)

    eva_accs  = [s["evabot_accuracy"]    for s in summaries if s["n"] > 0]
    bas_accs  = [s["baseline_accuracy"]  for s in summaries if s["n"] > 0]
    avg_eva   = round(sum(eva_accs) / len(eva_accs), 3) if eva_accs else 0
    avg_bas   = round(sum(bas_accs) / len(bas_accs), 3) if bas_accs else 0
    avg_delta = round(avg_eva - avg_bas, 3)

    esc_rates = [s["escalate_rate_per_q"] for s in summaries if s["n"] > 0]
    avg_esc   = round(sum(esc_rates) / len(esc_rates), 3) if esc_rates else 0

    esc_wins  = [s["escalate_win_rate"] for s in summaries if s["n"] > 0]
    avg_win   = round(sum(esc_wins) / len(esc_wins), 3) if esc_wins else 0

    evo_deltas = [s["evolution_delta"] for s in summaries if s.get("evolution_delta") is not None]
    avg_evo = round(sum(evo_deltas) / len(evo_deltas), 3) if evo_deltas else None

    delta_class = "positive" if avg_delta >= 0 else "negative"
    delta_sign  = "+" if avg_delta >= 0 else ""

    return f"""
<div class="grid-3" style="margin-bottom:2rem">
  <div class="card">
    <div class="stat-label">Accuracy Gain (EVABot vs Baseline)</div>
    <div class="stat-value {delta_class}">{delta_sign}{avg_delta*100:.1f}pp</div>
    <div class="stat-sub">EVABot {avg_eva*100:.1f}% &nbsp;vs&nbsp; Baseline {avg_bas*100:.1f}%</div>
    <div class="capability-tag cap1" style="margin-top:.6rem">Capability 1: Anti-interference</div>
  </div>
  <div class="card">
    <div class="stat-label">Escalate Win Rate</div>
    <div class="stat-value neutral">{avg_win*100:.1f}%</div>
    <div class="stat-sub">Avg {avg_esc:.2f} escalations / question</div>
    <div class="capability-tag cap2" style="margin-top:.6rem">Capability 2: Escalate Efficacy</div>
  </div>
  <div class="card">
    <div class="stat-label">Evolution Delta</div>
    <div class="stat-value positive">{"+" + f"{avg_evo*100:.1f}pp" if avg_evo is not None else "N/A"}</div>
    <div class="stat-sub">{"Accuracy improvement after skill evolution" if avg_evo is not None else "Run with --evolution to measure"}</div>
    <div class="capability-tag cap3" style="margin-top:.6rem">Capability 3: Evolution Loop</div>
  </div>
</div>
<div class="card" style="margin-bottom:2rem">
  <table>
    <tr>
      <th>Benchmark</th>
      <th>N</th>
      <th>EVABot Acc</th>
      <th>Baseline Acc</th>
      <th>Δ Accuracy</th>
      <th>Noise Resistance</th>
      <th>Esc Rate</th>
      <th>Esc Win %</th>
      <th>Evo Δ</th>
    </tr>
    {"".join(_summary_row(s) for s in summaries)}
  </table>
</div>
"""


def _summary_row(s: Dict) -> str:
    delta = s.get("accuracy_delta", 0) or 0
    sign  = "+" if delta >= 0 else ""
    cls   = "positive" if delta >= 0 else "negative"
    evo   = s.get("evolution_delta")
    evo_s = f"+{evo*100:.1f}pp" if evo is not None else "—"
    nr_eva = s.get("evabot_noise_resistance", 1.0)
    nr_bas = s.get("baseline_noise_resistance", 1.0)
    nr_delta = nr_eva - nr_bas
    nr_cls = "positive" if nr_delta >= 0 else "negative"
    nr_sign = "+" if nr_delta >= 0 else ""
    return (
        f"<tr>"
        f"<td><strong>{s['benchmark']}</strong></td>"
        f"<td>{s['n']}</td>"
        f"<td class='eva-color'>{s.get('evabot_accuracy',0)*100:.1f}%</td>"
        f"<td class='base-color'>{s.get('baseline_accuracy',0)*100:.1f}%</td>"
        f"<td class='{cls}'>{sign}{delta*100:.1f}pp</td>"
        f"<td class='{nr_cls}'>{nr_sign}{nr_delta*100:.1f}pp ({nr_eva*100:.0f}% vs {nr_bas*100:.0f}%)</td>"
        f"<td>{s.get('escalate_rate_per_q',0):.2f}</td>"
        f"<td>{s.get('escalate_win_rate',0)*100:.1f}%</td>"
        f"<td class='positive'>{evo_s}</td>"
        f"</tr>"
    )


def _benchmark_section(s: Dict, records: List[Dict]) -> str:
    bm = s["benchmark"]
    descriptions = {
        "SEAL-0":  ("Anti-interference on conflicting web evidence",
                    "111 questions with planted misleading sources. "
                    "EVABot's context isolation prevents misleading snippets from corrupting the answer."),
        "FRAMES":  ("Multi-hop factual reasoning across Wikipedia",
                    "824 questions requiring integration of multiple Wikipedia pages. "
                    "EVABot decomposes each hop into an isolated sub-task."),
        "SimpleQA":("Precise single-fact retrieval",
                    "4,326 short factual questions. "
                    "EVABot's minimal-context nodes prevent near-miss distractors from overriding the exact fact."),
        "EQ-Bench": ("Long-form narrative writing (8 turns)",
                     "Multi-turn creative writing. "
                     "EVABot maintains a plan summary at the top node while each chapter sees only the local outline."),
        "AbGen":   ("Rigorous ablation study design",
                    "Ablation design tasks with noisy context. "
                    "EVABot's isolated nodes focus on specific system components, ignoring irrelevant background."),
    }
    title, desc = descriptions.get(bm, (bm, ""))

    # Capability-specific sub-sections
    cap1 = _cap1_block(s, records, bm)
    cap2 = _cap2_block(s, records)
    cap3 = _cap3_block(s, records)

    return f"""
<div class="benchmark-block">
  <h2>{bm} — {title}</h2>
  <p class="section-intro">{desc}</p>
  {cap1}
  {cap2}
  {cap3}
</div>
"""


def _cap1_block(s: Dict, records: List[Dict], bm: str) -> str:
    eva_pct  = s.get("evabot_accuracy",   0) * 100
    bas_pct  = s.get("baseline_accuracy", 0) * 100
    delta    = s.get("accuracy_delta",    0) * 100
    delta_cls = "positive" if delta >= 0 else "negative"
    delta_sign = "+" if delta >= 0 else ""

    nr_eva = s.get("evabot_noise_resistance",    1.0) * 100
    nr_bas = s.get("baseline_noise_resistance",  1.0) * 100

    writing_row = ""
    if s.get("evabot_writing_score") is not None:
        ew = s["evabot_writing_score"]
        bw = s.get("baseline_writing_score", 0) or 0
        writing_row = (
            f"<tr><td>Writing Score (0-40)</td>"
            f"<td class='eva-color'>{ew:.1f}</td>"
            f"<td class='base-color'>{bw:.1f}</td>"
            f"<td class='{'positive' if ew>=bw else 'negative'}'>{'+' if ew>=bw else ''}{ew-bw:.1f}</td></tr>"
        )
    ablation_row = ""
    if s.get("evabot_ablation_score") is not None:
        ea = s["evabot_ablation_score"]
        ba = s.get("baseline_ablation_score", 0) or 0
        ablation_row = (
            f"<tr><td>Ablation Score (4-20)</td>"
            f"<td class='eva-color'>{ea:.1f}</td>"
            f"<td class='base-color'>{ba:.1f}</td>"
            f"<td class='{'positive' if ea>=ba else 'negative'}'>{'+' if ea>=ba else ''}{ea-ba:.1f}</td></tr>"
        )

    return f"""
<div>
  <span class="capability-tag cap1">Capability 1 — Anti-interference: Local View vs Global View</span>
  <div class="grid-2">
    <div>
      <div class="stat-label" style="margin-bottom:.5rem">Accuracy</div>
      <div style="margin-bottom:.4rem">
        <span class="stat-label">EVABot (local)</span>
        <div class="progress-bar"><div class="progress-fill bar-eva" style="width:{eva_pct:.1f}%"></div></div>
        <span class="eva-color" style="font-size:1.3rem;font-weight:700">{eva_pct:.1f}%</span>
      </div>
      <div>
        <span class="stat-label">Baseline (global)</span>
        <div class="progress-bar"><div class="progress-fill bar-base" style="width:{bas_pct:.1f}%"></div></div>
        <span class="base-color" style="font-size:1.3rem;font-weight:700">{bas_pct:.1f}%</span>
      </div>
    </div>
    <div>
      <table>
        <tr><th>Metric</th><th class="eva-color">EVABot</th><th class="base-color">Baseline</th><th>Δ</th></tr>
        <tr><td>Accuracy</td>
            <td class="eva-color">{eva_pct:.1f}%</td>
            <td class="base-color">{bas_pct:.1f}%</td>
            <td class="{delta_cls}">{delta_sign}{delta:.1f}pp</td></tr>
        <tr><td>Noise Resistance</td>
            <td class="eva-color">{nr_eva:.1f}%</td>
            <td class="base-color">{nr_bas:.1f}%</td>
            <td class="{'positive' if nr_eva>=nr_bas else 'negative'}">{'+' if nr_eva>=nr_bas else ''}{nr_eva-nr_bas:.1f}pp</td></tr>
        {writing_row}
        {ablation_row}
      </table>
    </div>
  </div>
</div>"""


def _cap2_block(s: Dict, records: List[Dict]) -> str:
    esc_rate = s.get("escalate_rate_per_q", 0)
    win_rate = s.get("escalate_win_rate",   0) * 100
    total_esc = s.get("total_escalations",  0)
    evo_count = s.get("evolution_triggered_count", 0)

    return f"""
<div style="margin-top:1.2rem">
  <span class="capability-tag cap2">Capability 2 — Escalate Efficacy</span>
  <div class="grid-3">
    <div class="card">
      <div class="stat-label">Escalations triggered</div>
      <div class="stat-value neutral">{total_esc}</div>
      <div class="stat-sub">{esc_rate:.2f} avg per question</div>
    </div>
    <div class="card">
      <div class="stat-label">Win rate after escalate</div>
      <div class="stat-value positive">{win_rate:.1f}%</div>
      <div class="stat-sub">Resolved → correct answer</div>
    </div>
    <div class="card">
      <div class="stat-label">Evolution loops triggered</div>
      <div class="stat-value neutral">{evo_count}</div>
      <div class="stat-sub">Auditor-initiated skill updates</div>
    </div>
  </div>
</div>"""


def _cap3_block(s: Dict, records: List[Dict]) -> str:
    pre  = s.get("pre_evolution_accuracy")
    post = s.get("post_evolution_accuracy")
    evo  = s.get("evolution_delta")

    if pre is None:
        return f"""
<div style="margin-top:1.2rem">
  <span class="capability-tag cap3">Capability 3 — Evolution Delta</span>
  <p class="section-intro" style="margin-top:.5rem">
    Run with <code>--evolution</code> flag to measure pre/post-evolution accuracy.
  </p>
</div>"""

    sign = "+" if evo >= 0 else ""
    cls  = "positive" if evo >= 0 else "negative"
    return f"""
<div style="margin-top:1.2rem">
  <span class="capability-tag cap3">Capability 3 — Evolution Delta</span>
  <div class="grid-3" style="margin-top:.75rem">
    <div class="card">
      <div class="stat-label">Pre-evolution accuracy</div>
      <div class="stat-value base-color">{pre*100:.1f}%</div>
    </div>
    <div class="card">
      <div class="stat-label">Post-evolution accuracy</div>
      <div class="stat-value positive">{post*100:.1f}%</div>
    </div>
    <div class="card">
      <div class="stat-label">Evolution Δ</div>
      <div class="stat-value {cls}">{sign}{evo*100:.1f}pp</div>
      <div class="stat-sub">Skill self-repair improvement</div>
    </div>
  </div>
</div>"""
