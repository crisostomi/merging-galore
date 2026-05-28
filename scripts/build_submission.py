#!/usr/bin/env python3
"""Build a campaign_submission_v1 artifact from an evaluation results JSON.

Turns the output of scripts/evaluate_multitask_merging.py into the exact
submission payload the Merging Galore campaign projector expects, so publishing
to Flywheel is a copy-paste of the resulting file plus the evidence uploads.

Usage:
    uv run python scripts/build_submission.py \
        --merger tsv \
        --encoder b32 \
        --track improvement \
        --run-name tsv-small-hard10 \
        --blurb "one-line description of the run" \
        --results results/ViT-B-32/tsv/10.json \
        --run-log logs/tsv_b32_hard10.run.log

Writes <results_dir>/campaign_submission_v1.json (override with --out).
"""
import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

ENCODER_TO_MODEL = {"b32": "ViT-B-32", "b16": "ViT-B-16", "l14": "ViT-L-14"}
ENCODER_TO_SETTING = {"b32": "small", "l14": "large"}
HARD10 = [
    "SUN397", "Cars", "RESISC45", "GTSRB", "DTD",
    "Flowers102", "CIFAR100", "Food101", "KMNIST", "EuroSAT",
]
VALID_TRACKS = ["baseline", "improvement", "ablation", "reproduction", "failure_analysis"]


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def default_author() -> str:
    try:
        email = git("config", "user.email")
        if email:
            return email
    except subprocess.CalledProcessError:
        pass
    return ""


def repo_context() -> dict:
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    commit = git("rev-parse", "HEAD")
    parent = git("rev-parse", "HEAD~1")
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    return {"branch": branch, "commit": commit, "parent_commit": parent, "worktree_dirty": dirty}


def class_path_from_config(config_path: Path) -> str:
    for line in config_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("_target_:"):
            return line.split(":", 1)[1].strip()
    raise ValueError(f"No _target_ found in {config_path}")


def build_metrics(results: dict) -> dict:
    avg = results["avg"][0]
    acc_avg = avg["acc/test/avg"]
    norm_avg = avg["normalized_acc/test/avg"]
    per_dataset = {}
    for ds in HARD10:
        row = results[ds][0]
        per_dataset[ds] = {
            "acc": row[f"acc/test/{ds}"],
            "normalized_acc": row[f"normalized_acc/test/{ds}"],
            "loss": row[f"loss/test/{ds}"],
        }
    return {
        "normalized_acc_avg": norm_avg,
        "acc_avg": acc_avg,
        "per_dataset": per_dataset,
        "normalized_acc_pct": round(norm_avg * 100, 2),
        "acc_pct": round(acc_avg * 100, 2),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--merger", required=True, help="merger config name (e.g. tsv, task_arithmetic)")
    p.add_argument("--encoder", required=True, choices=ENCODER_TO_SETTING.keys(), help="b32 (small) or l14 (large)")
    p.add_argument("--track", required=True, choices=VALID_TRACKS)
    p.add_argument("--run-name", required=True)
    p.add_argument("--blurb", required=True)
    p.add_argument("--results", required=True, type=Path, help="path to the eval results JSON")
    p.add_argument("--run-log", required=True, help="repo-relative path to the run log evidence")
    p.add_argument("--pr-url", default=None, help="URL of the associated PR against crisostomi/merging-galore (required for a valid submission)")
    p.add_argument("--author", default=None, help="submission author email (default: git config user.email)")
    p.add_argument("--date", default=dt.date.today().isoformat())
    p.add_argument("--status", default="valid", choices=["valid", "invalid", "draft"])
    p.add_argument("--seed-index", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval-on-val", action="store_true")
    p.add_argument("--extra", nargs="*", default=[], help="extra evidence file paths (repo-relative)")
    p.add_argument("--merger-source", default=None, help="repo-relative merger source path (default: src/model_merging/merger/<merger>_merger.py)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    author = args.author or default_author()
    if not author:
        p.error("--author not given and `git config user.email` is unset")

    if args.status == "valid" and not args.pr_url:
        p.error("a valid submission requires --pr-url (an associated PR against crisostomi/merging-galore)")

    setting = ENCODER_TO_SETTING[args.encoder]
    merger_config = f"conf/merger/{args.merger}.yaml"
    class_path = class_path_from_config(Path(merger_config))
    results = json.loads(args.results.read_text())
    metrics = build_metrics(results)

    # default extra: SVD-based mergers also depend on the structured-merging module
    extra = list(args.extra)
    if not extra and class_path.endswith(("IsotropicMerger", "TaskSingularVectorsMerger")):
        extra = ["src/model_merging/merging/structured.py"]

    submission = {
        "status": args.status,
        "setting": setting,
        "track": args.track,
        "author": author,
        "run_name": args.run_name,
        "blurb": args.blurb,
        "date": args.date,
        "merger": {"name": args.merger, "config": merger_config, "class_path": class_path},
        "metrics": metrics,
        "runtime": {"device": args.device, "eval_on_val": args.eval_on_val, "seed_index": args.seed_index},
        "repo": {**repo_context(), "pr_url": args.pr_url},
        "artifacts": {
            "results_json": str(args.results),
            "merger_source": args.merger_source or f"src/model_merging/merger/{args.merger}_merger.py",
            "merger_config": merger_config,
            "run_log": args.run_log,
            "extra": extra,
        },
    }

    out = args.out or (args.results.parent / "campaign_submission_v1.json")
    out.write_text(json.dumps(submission, indent=2) + "\n")
    print(f"wrote {out}")
    print(f"  setting={setting}  track={args.track}")
    print(f"  norm_acc_pct={metrics['normalized_acc_pct']}  acc_pct={metrics['acc_pct']}")
    print(f"  commit={submission['repo']['commit'][:10]}  dirty={submission['repo']['worktree_dirty']}")
    print(f"  pr_url={submission['repo']['pr_url']}")
    if submission["repo"]["worktree_dirty"]:
        print("  WARNING: worktree is dirty — commit before publishing for a clean provenance trail.")


if __name__ == "__main__":
    main()
