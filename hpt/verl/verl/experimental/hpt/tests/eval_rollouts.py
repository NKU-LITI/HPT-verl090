"""Judge recorded rollouts with the original and the current reward, per step.

The original reward is the canonical wrapper before this repo's changes:
``legacy compute_score("<think>\\n" + response, ground_truth)``. The current
reward is ``reward.compute_score``. Recorded length metadata is still passed
for interface compatibility but is not part of the current verdict.

Run from the verifier directory:
    uv run --no-project --with "math-verify[antlr4_13_2]==0.9.0" --with sympy \
        --with pylatexenc --with latex2sympy2_extended --with matplotlib \
        python tests/eval_rollouts.py --data-dir ../rollout_data --out-dir ../rollout_eval
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import types
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

PACKAGE = "verl090_verifier"
VERIFIER_DIR = Path(__file__).resolve().parents[1]


def _import_reward():
    if PACKAGE not in sys.modules:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(VERIFIER_DIR)]
        sys.modules[PACKAGE] = package
    return importlib.import_module(f"{PACKAGE}.reward"), importlib.import_module(f"{PACKAGE}.legacy_entropy_math")


def _judge(fn) -> tuple[float, bool]:
    try:
        return float(bool(fn())), False
    except Exception:
        return 0.0, True


def judge_file(args: tuple[Path, Path, int]) -> Path:
    """Judge one rollout file and cache the per-sample results as JSON lines."""
    path, cache_dir, max_response_length = args
    out = cache_dir / (path.stem + ".jsonl")
    if out.exists():
        return out
    reward, legacy = _import_reward()
    rows = []
    with path.open() as f:
        for line in f:
            item = json.loads(line)
            response, gt = item["response"], item["reward_model"]["ground_truth"]
            extra_info = {reward.RESPONSE_LENGTH_KEY: int(item["response_length"])}
            original, original_error = _judge(
                lambda: legacy.compute_score(reward.legacy_reward_input(response), gt)
            )
            current, current_error = _judge(
                lambda: reward.compute_score(
                    item["data_source"], response, gt, extra_info, max_response_length=max_response_length
                )
            )
            rows.append({
                "global_step": int(item["global_step"]),
                "rollout_type": item["rollout_type"],
                "data_source": item["data_source"],
                "recorded": float(item["reward"]),
                "original": original,
                "current": current,
                "original_error": original_error,
                "current_error": current_error,
            })
    tmp = out.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    tmp.rename(out)
    return out


def summarize(cache_files: list[Path], out_dir: Path) -> list[dict]:
    """Per-step mean reward over train rollouts (validation rollouts are excluded)."""
    groups = defaultdict(list)
    for path in cache_files:
        for line in path.open():
            row = json.loads(line)
            if row["rollout_type"] == "train":
                groups[row["global_step"]].append(row)
    summary = []
    for step, rows in sorted(groups.items()):
        n = len(rows)
        summary.append({
            "global_step": step,
            "n": n,
            "recorded_reward_mean": sum(r["recorded"] for r in rows) / n,
            "original_reward_mean": sum(r["original"] for r in rows) / n,
            "current_reward_mean": sum(r["current"] for r in rows) / n,
            "original_matches_recorded": sum(r["original"] == r["recorded"] for r in rows) / n,
            "original_errors": sum(r["original_error"] for r in rows),
            "current_errors": sum(r["current_error"] for r in rows),
            "only_original": sum(r["original"] > r["current"] for r in rows),
            "only_current": sum(r["current"] > r["original"] for r in rows),
        })
    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return summary


def plot(summary: list[dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [row["global_step"] for row in summary]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(steps, [row["original_reward_mean"] for row in summary], label="original reward")
    ax.plot(steps, [row["current_reward_mean"] for row in summary], label="current reward")
    ax.set_title("Train rollouts: reward mean per step")
    ax.set_xlabel("global step")
    ax.set_ylabel("reward mean")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-response-length", type=int, default=4096,
                        help="Generation limit the rollouts were sampled with")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="Only judge the first N files")
    args = parser.parse_args()

    cache_dir = args.out_dir / "per_sample"
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.data_dir.glob("step_*.jsonl"))[: args.limit]
    tasks = [(path, cache_dir, args.max_response_length) for path in files]
    with Pool(args.workers, maxtasksperchild=1) as pool:
        cache_files = []
        for i, out in enumerate(pool.imap_unordered(judge_file, tasks), 1):
            cache_files.append(out)
            print(f"[{i}/{len(tasks)}] {out.name}", flush=True)

    summary = summarize(sorted(cache_files), args.out_dir)
    plot(summary, args.out_dir / "reward_mean_curves.png")
    n = sum(row["n"] for row in summary)
    mean = lambda key: sum(row[key] * row["n"] for row in summary) / n
    print(f"train samples={n} original={mean('original_reward_mean'):.4f} "
          f"current={mean('current_reward_mean'):.4f} recorded={mean('recorded_reward_mean'):.4f} "
          f"original==recorded={mean('original_matches_recorded'):.4f}")

if __name__ == "__main__":
    main()
