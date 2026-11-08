"""
run_teachability.py -- Orchestrate the full teachability pipeline.

For each model: baseline eval -> generate DPO pairs -> DPO train -> post-DPO eval -> compare.

Usage:
    python run_teachability.py qwen25_7b                          # single model
    python run_teachability.py --all                              # all trainable models
    python run_teachability.py --compare                          # compare existing results only
    python run_teachability.py qwen25_7b --skip-train --skip-eval # compare only
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from sparse_personality.teachability import (
    compute_teachability,
    compare_teachability,
    generate_teachability_html,
)

BASE_DIR = Path(__file__).parent
BASELINES_DIR = BASE_DIR / "baselines"
TRAINED_DIR = BASE_DIR / "trained"

# Models that can be DPO-trained locally (must match train_dpo.py)
TRAINABLE_MODELS = {
    "qwen25_7b": "Qwen/Qwen2.5-7B-Instruct",
    "olmo3_7b": "allenai/OLMo-3-7B-Instruct-20251121",
    "gemma3_4b": "google/gemma-3-4b-it",
}

# OpenRouter IDs for baseline evaluation (must match run_baselines.py)
OPENROUTER_MODELS = {
    "qwen25_7b": "qwen/qwen-2.5-7b-instruct",
    "olmo3_7b": "allenai/olmo-3-7b-instruct-20251121",
    "gemma3_4b": "google/gemma-3-4b-it",
}


def run_cmd(cmd: list[str], desc: str) -> bool:
    """Run a subprocess and stream output."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    if result.returncode != 0:
        print(f"\n  FAILED: {desc} (exit code {result.returncode})")
        return False
    return True


def ensure_baseline(model_name: str) -> bool:
    """Ensure pre-DPO baseline exists, run if not."""
    baseline_path = BASELINES_DIR / f"{model_name}.json"
    if baseline_path.exists():
        print(f"  Baseline exists: {baseline_path}")
        return True

    if model_name not in OPENROUTER_MODELS:
        print(f"  No OpenRouter ID for {model_name}, cannot run baseline.")
        print(f"  Add it to OPENROUTER_MODELS in run_teachability.py and run_baselines.py MODELS.")
        return False

    return run_cmd(
        [sys.executable, "run_baselines.py", model_name],
        f"Running baseline for {model_name}",
    )


def ensure_dpo_pairs(model_name: str) -> bool:
    """Check that DPO pairs exist and have content."""
    dpo_path = BASELINES_DIR / f"{model_name}_dpo.jsonl"
    if dpo_path.exists() and dpo_path.stat().st_size > 0:
        with open(dpo_path) as f:
            count = sum(1 for line in f if line.strip())
        print(f"  DPO pairs exist: {dpo_path} ({count} pairs)")
        return count >= 5

    # Re-profile to generate pairs (requires re-running baseline with updated profiler)
    print(f"  DPO pairs missing or empty for {model_name}.")
    print(f"  Re-run baseline to generate pairs: python run_baselines.py {model_name}")
    return False


def train_dpo(model_name: str) -> bool:
    """Run DPO training."""
    adapter_dir = TRAINED_DIR / f"{model_name}-personality"
    if adapter_dir.exists() and any(adapter_dir.glob("*.safetensors")):
        print(f"  Trained adapter exists: {adapter_dir}")
        return True

    return run_cmd(
        [sys.executable, "train_dpo.py", model_name],
        f"DPO training {model_name}",
    )


def eval_post_dpo(model_name: str) -> bool:
    """Run post-DPO evaluation."""
    post_path = BASELINES_DIR / f"{model_name}_dpo.json"
    if post_path.exists():
        print(f"  Post-DPO eval exists: {post_path}")
        return True

    return run_cmd(
        [sys.executable, "run_local_baseline.py", model_name],
        f"Post-DPO evaluation for {model_name}",
    )


def run_pipeline(model_name: str, skip_train: bool = False, skip_eval: bool = False) -> bool:
    """Run the full teachability pipeline for a single model."""
    print(f"\n{'#'*60}")
    print(f"  TEACHABILITY PIPELINE: {model_name}")
    print(f"{'#'*60}")

    # Step 1: Ensure baseline
    if not ensure_baseline(model_name):
        return False

    # Step 2: Ensure DPO pairs
    if not ensure_dpo_pairs(model_name):
        return False

    # Step 3: Train DPO
    if not skip_train:
        if not train_dpo(model_name):
            return False
    else:
        print("  Skipping DPO training (--skip-train)")

    # Step 4: Post-DPO evaluation
    if not skip_eval:
        if not eval_post_dpo(model_name):
            return False
    else:
        print("  Skipping post-DPO evaluation (--skip-eval)")

    return True


def generate_report(model_names: list[str]) -> str:
    """Generate teachability comparison report for all models with both pre and post data."""
    results = []
    for name in model_names:
        pre_path = BASELINES_DIR / f"{name}.json"
        post_path = BASELINES_DIR / f"{name}_dpo.json"
        dpo_path = BASELINES_DIR / f"{name}_dpo.jsonl"

        if not pre_path.exists() or not post_path.exists():
            print(f"  Skipping {name}: missing pre or post data")
            continue

        try:
            result = compute_teachability(
                str(pre_path),
                str(post_path),
                str(dpo_path) if dpo_path.exists() else None,
            )
            results.append(result)
            print(f"  {name}: grade={result.grade}, delta={result.net_delta:+.1f}, "
                  f"recovery={result.failure_recovery_rate:.0%}")
        except Exception as e:
            print(f"  Error computing teachability for {name}: {e}")

    if not results:
        print("\nNo models have both pre and post-DPO data yet.")
        return ""

    # Compare and rank
    comparison = compare_teachability(results)
    comp_path = BASELINES_DIR / "teachability.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Saved: {comp_path}")

    # Generate HTML report
    html = generate_teachability_html(results)
    html_path = BASELINES_DIR / "teachability.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Saved: {html_path}")

    return str(html_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full teachability pipeline")
    parser.add_argument("models", nargs="*", help="Model names to process")
    parser.add_argument("--all", action="store_true", help="Run all trainable models")
    parser.add_argument("--compare", action="store_true", help="Only compare existing results")
    parser.add_argument("--skip-train", action="store_true", help="Skip DPO training step")
    parser.add_argument("--skip-eval", action="store_true", help="Skip post-DPO evaluation step")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.all:
        model_names = list(TRAINABLE_MODELS.keys())
    elif args.models:
        model_names = args.models
    elif args.compare:
        # Find all models that have both pre and post data
        model_names = []
        for f in BASELINES_DIR.glob("*_dpo.json"):
            name = f.stem.replace("_dpo", "")
            if (BASELINES_DIR / f"{name}.json").exists():
                model_names.append(name)
    else:
        print("Provide model names, --all, or --compare. Trainable models:")
        for k, v in TRAINABLE_MODELS.items():
            print(f"  {k}: {v}")
        sys.exit(1)

    if not args.compare:
        for name in model_names:
            success = run_pipeline(name, args.skip_train, args.skip_eval)
            if not success:
                print(f"\n  Pipeline failed for {name}, continuing...")

    # Generate comparison report
    report_path = generate_report(model_names)
    if report_path:
        print(f"\n{'='*60}")
        print(f"  Teachability report: {report_path}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
