"""teachability.py -- Pre-DPO vs post-DPO comparison engine.

Computes teachability metrics by comparing baseline and post-DPO evaluation
results, and generates HTML reports with before/after analysis.
"""

import json
from pathlib import Path
from typing import Optional

from cane_personality.types import TeachabilityResult


def _load_results(path: str) -> dict:
    """Load a baseline or post-DPO JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _compute_traits_from_results(results: list[dict]) -> dict:
    """Compute average criteria scores from raw results."""
    if not results:
        return {}
    totals = {}
    counts = {}
    for r in results:
        cs = r.get("criteria_scores", {})
        for k, v in cs.items():
            totals[k] = totals.get(k, 0) + v
            counts[k] = counts.get(k, 0) + 1
    return {k: totals[k] / counts[k] for k in totals}


def _assign_grade(result: TeachabilityResult) -> str:
    """Assign A-F grade based on teachability metrics."""
    # Weighted score: recovery matters most, regression penalizes, efficiency is bonus
    score = (
        result.failure_recovery_rate * 50
        + max(0, result.net_delta) * 1.0
        - result.regression_rate * 30
        + min(result.dpo_efficiency, 2.0) * 5
    )
    if score >= 40:
        return "A"
    elif score >= 25:
        return "B"
    elif score >= 15:
        return "C"
    elif score >= 5:
        return "D"
    else:
        return "F"


def compute_teachability(
    pre_path: str,
    post_path: str,
    dpo_pairs_path: Optional[str] = None,
) -> TeachabilityResult:
    """Compare pre-DPO and post-DPO results to compute teachability metrics.

    Args:
        pre_path: Path to pre-DPO baseline JSON (from run_baselines.py)
        post_path: Path to post-DPO evaluation JSON (from run_local_baseline.py)
        dpo_pairs_path: Optional path to DPO JSONL to count training pairs
    """
    pre_data = _load_results(pre_path)
    post_data = _load_results(post_path)

    pre_results = pre_data.get("results", [])
    post_results = post_data.get("results", [])

    # Build lookup by question text
    pre_by_q = {r["question"]: r for r in pre_results}
    post_by_q = {r["question"]: r for r in post_results}

    # Find matched questions
    matched_questions = set(pre_by_q.keys()) & set(post_by_q.keys())
    if not matched_questions:
        raise ValueError("No matching questions between pre and post results")

    # Compute per-question deltas
    pre_scores = []
    post_scores = []
    pre_fails = 0
    pre_passes = 0
    recovered = 0
    regressed = 0
    improved = 0
    worsened = 0

    for q in matched_questions:
        pre_s = pre_by_q[q].get("score", 0)
        post_s = post_by_q[q].get("score", 0)
        pre_scores.append(pre_s)
        post_scores.append(post_s)

        pre_fail = pre_s < 50
        post_fail = post_s < 50

        if pre_fail:
            pre_fails += 1
            if not post_fail:
                recovered += 1
        else:
            pre_passes += 1
            if post_fail:
                regressed += 1

        if post_s > pre_s + 5:
            improved += 1
        elif post_s < pre_s - 5:
            worsened += 1

    pre_avg = sum(pre_scores) / len(pre_scores)
    post_avg = sum(post_scores) / len(post_scores)
    net_delta = post_avg - pre_avg

    failure_recovery = recovered / pre_fails if pre_fails > 0 else 1.0
    regression = regressed / pre_passes if pre_passes > 0 else 0.0

    # Per-trait deltas (criteria-level)
    pre_traits = _compute_traits_from_results(
        [pre_by_q[q] for q in matched_questions]
    )
    post_traits = _compute_traits_from_results(
        [post_by_q[q] for q in matched_questions]
    )
    trait_deltas = {
        k: post_traits.get(k, 0) - pre_traits.get(k, 0)
        for k in set(list(pre_traits.keys()) + list(post_traits.keys()))
    }

    # Count DPO pairs
    dpo_count = 0
    if dpo_pairs_path and Path(dpo_pairs_path).exists():
        with open(dpo_pairs_path, "r", encoding="utf-8") as f:
            dpo_count = sum(1 for line in f if line.strip())

    efficiency = net_delta / dpo_count if dpo_count > 0 else 0.0

    model_name = pre_data.get("model_name", Path(pre_path).stem)

    result = TeachabilityResult(
        model_name=model_name,
        pre_avg=pre_avg,
        post_avg=post_avg,
        net_delta=net_delta,
        failure_recovery_rate=failure_recovery,
        regression_rate=regression,
        per_trait_deltas=trait_deltas,
        dpo_pair_count=dpo_count,
        dpo_efficiency=efficiency,
        questions_improved=improved,
        questions_regressed=worsened,
        total_questions=len(matched_questions),
    )
    result.grade = _assign_grade(result)
    return result


def compare_teachability(results: list[TeachabilityResult]) -> dict:
    """Rank models by teachability across dimensions."""
    if not results:
        return {}

    ranked = {
        "by_recovery": sorted(results, key=lambda r: r.failure_recovery_rate, reverse=True),
        "by_net_delta": sorted(results, key=lambda r: r.net_delta, reverse=True),
        "by_efficiency": sorted(results, key=lambda r: r.dpo_efficiency, reverse=True),
        "by_grade": sorted(results, key=lambda r: "ABCDF".index(r.grade)),
    }

    return {
        "rankings": {
            k: [r.model_name for r in v]
            for k, v in ranked.items()
        },
        "results": [r.to_dict() for r in results],
    }


def generate_teachability_html(results: list[TeachabilityResult]) -> str:
    """Generate self-contained HTML teachability report."""
    results_json = json.dumps([r.to_dict() for r in results])

    # Build leaderboard rows
    rows = ""
    for r in sorted(results, key=lambda x: "ABCDF".index(x.grade)):
        grade_color = {
            "A": "#51cf66", "B": "#94d82d", "C": "#ffd43b",
            "D": "#ff922b", "F": "#ff6b6b"
        }.get(r.grade, "#888")
        delta_color = "#51cf66" if r.net_delta >= 0 else "#ff6b6b"
        delta_sign = "+" if r.net_delta >= 0 else ""
        rows += f"""<tr>
            <td style="font-weight:600">{r.model_name}</td>
            <td><span style="color:{grade_color};font-size:20px;font-weight:700">{r.grade}</span></td>
            <td>{r.pre_avg:.1f}</td>
            <td>{r.post_avg:.1f}</td>
            <td style="color:{delta_color};font-weight:600">{delta_sign}{r.net_delta:.1f}</td>
            <td>{r.failure_recovery_rate:.0%}</td>
            <td>{r.regression_rate:.0%}</td>
            <td>{r.dpo_pair_count}</td>
            <td>{r.dpo_efficiency:.2f}</td>
            <td>{r.questions_improved}</td>
            <td>{r.questions_regressed}</td>
        </tr>"""

    # Build per-trait delta table
    all_traits = set()
    for r in results:
        all_traits.update(r.per_trait_deltas.keys())
    trait_headers = "".join(f"<th>{t}</th>" for t in sorted(all_traits))
    trait_rows = ""
    for r in sorted(results, key=lambda x: "ABCDF".index(x.grade)):
        cells = ""
        for t in sorted(all_traits):
            d = r.per_trait_deltas.get(t, 0)
            color = "#51cf66" if d >= 0 else "#ff6b6b"
            sign = "+" if d >= 0 else ""
            cells += f'<td style="color:{color}">{sign}{d:.1f}</td>'
        trait_rows += f"<tr><td style='font-weight:600'>{r.model_name}</td>{cells}</tr>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cane Teachability Report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0a; color: #e0e0e0; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 8px; color: #fff; }}
  h2 {{ font-size: 20px; font-weight: 600; margin: 32px 0 16px; color: #fff; }}
  .subtitle {{ color: #888; font-size: 14px; margin-bottom: 24px; }}
  .card {{ background: #141414; border: 1px solid #222; border-radius: 12px; padding: 24px; margin-bottom: 24px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th {{ text-align: left; padding: 10px 12px; border-bottom: 2px solid #333; color: #888; font-weight: 500; font-size: 12px; text-transform: uppercase; }}
  td {{ padding: 10px 12px; border-bottom: 1px solid #1a1a1a; }}
  tr:hover {{ background: #1a1a1a; }}
  .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  #bar-chart, #radar-chart {{ height: 400px; }}
</style>
</head>
<body>
<div class="container">
  <h1>Teachability Leaderboard</h1>
  <p class="subtitle">Pre-DPO vs Post-DPO performance comparison across models</p>

  <div class="card">
    <h2>Leaderboard</h2>
    <div style="overflow-x:auto">
    <table>
      <tr>
        <th>Model</th><th>Grade</th><th>Pre Avg</th><th>Post Avg</th>
        <th>Delta</th><th>Recovery</th><th>Regression</th>
        <th>DPO Pairs</th><th>Efficiency</th><th>Improved</th><th>Regressed</th>
      </tr>
      {rows}
    </table>
    </div>
  </div>

  <div class="chart-row">
    <div class="card"><div id="bar-chart"></div></div>
    <div class="card"><div id="radar-chart"></div></div>
  </div>

  <div class="card">
    <h2>Per-Trait Deltas</h2>
    <div style="overflow-x:auto">
    <table>
      <tr><th>Model</th>{trait_headers}</tr>
      {trait_rows}
    </table>
    </div>
  </div>
</div>

<script>
const data = {results_json};

// Grouped bar chart: pre vs post
const models = data.map(d => d.model_name);
const pre = data.map(d => d.pre_avg);
const post = data.map(d => d.post_avg);

Plotly.newPlot('bar-chart', [
  {{ x: models, y: pre, name: 'Pre-DPO', type: 'bar', marker: {{ color: '#555' }} }},
  {{ x: models, y: post, name: 'Post-DPO', type: 'bar', marker: {{ color: '#51cf66' }} }},
], {{
  title: {{ text: 'Average Score: Before vs After DPO', font: {{ color: '#fff' }} }},
  barmode: 'group',
  paper_bgcolor: '#141414', plot_bgcolor: '#141414',
  font: {{ color: '#aaa' }},
  xaxis: {{ gridcolor: '#222' }}, yaxis: {{ gridcolor: '#222', range: [0, 100] }},
  legend: {{ font: {{ color: '#ccc' }} }},
}}, {{ responsive: true }});

// Radar chart overlay: pre vs post for each model
if (data.length > 0) {{
  const traits = Object.keys(data[0].per_trait_deltas || {{}});
  const radarTraces = [];
  data.forEach((d, i) => {{
    const colors = ['#748ffc', '#ff6b6b', '#51cf66', '#ffd43b', '#cc5de8', '#20c997'];
    const c = colors[i % colors.length];
    radarTraces.push({{
      type: 'scatterpolar',
      r: traits.map(t => (d.per_trait_deltas[t] || 0)),
      theta: traits,
      fill: 'toself',
      name: d.model_name,
      line: {{ color: c }},
      opacity: 0.6,
    }});
  }});
  Plotly.newPlot('radar-chart', radarTraces, {{
    title: {{ text: 'Trait Deltas (Post - Pre)', font: {{ color: '#fff' }} }},
    paper_bgcolor: '#141414', plot_bgcolor: '#141414',
    font: {{ color: '#aaa' }},
    polar: {{ bgcolor: '#141414', radialaxis: {{ gridcolor: '#333' }}, angularaxis: {{ gridcolor: '#333' }} }},
    legend: {{ font: {{ color: '#ccc' }} }},
  }}, {{ responsive: true }});
}}
</script>
</body>
</html>"""
