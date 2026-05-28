# Submitting to the Merging Galore campaign

A submission is **one attempt node** on the Flywheel *Merging Galore* campaign, carrying
**two JSON artifacts**. This repo provides `scripts/build_submission.py` to generate the
submission artifact; publishing the node itself is done through the Flywheel MCP tools
(your agent does this for you).

## 1. Run the evaluation

```bash
uv run python scripts/evaluate_multitask_merging.py \
    merger=<your_merger> benchmark=hard10 nn/encoder=b32   # Small (ViT-B-32)
# use nn/encoder=l14 for the Large (ViT-L-14) setting
```

The reported numbers must come from this **unmodified** harness on `benchmark=hard10`.
By default the results JSON is written to `results/<MODEL>/10.json`.

## 2. Build the submission artifact

```bash
uv run python scripts/build_submission.py \
    --merger <your_merger> --encoder <b32|l14> \
    --track <baseline|improvement|ablation|reproduction|failure_analysis> \
    --run-name <short-name> --blurb "<one-line description>" \
    --results results/<MODEL>/10.json \
    --run-log <path-to-run-log>
```

This writes `campaign_submission_v1.json` next to the results, filling in `metrics`
(including the `normalized_acc_pct` / `acc_pct` fields the leaderboard displays and ranks
on), per-dataset metrics, and git provenance (`repo.commit`, etc.). Author defaults to
your `git config user.email`.

## 3. Publish the attempt node (Flywheel MCP)

Ask your agent to "submit to the Merging Galore campaign". It will:

1. Create an attempt node — branch from the nearest conceptual parent, or attach to the
   campaign root.
2. **Set the attempt node's visibility to `public`.** This is required: the campaign's
   `submission_policy.required_visibility` is `public`, so private/unlisted attempt nodes
   fail artifact finalization and never reach the leaderboard.
3. Attach and finalize **exactly two** artifacts:
   - `results.json` — the raw eval output, metadata `campaign_role=results`.
   - `campaign_submission_v1.json` — metadata `campaign_role=submission` and
     `campaign_schema=campaign_submission_v1`.

   The merger source, Hydra config, and run log are referenced by repo-relative path
   inside the submission (and pinned by `repo.commit`); they are **not** uploaded
   separately, so commit and push your merger first for a clean provenance trail.

The leaderboard refreshes automatically once the submission artifact finalizes on a
public attempt node. Read live standings with the `flywheel_get_campaign_snapshot` tool.

## Rules recap

- **No additional training** — weight-space merging only, from the pretrained encoder and
  the per-task fine-tuned checkpoints.
- Pick **one** setting per submission (`small` = ViT-B-32, `large` = ViT-L-14).
- Attached evidence must let another researcher rerun the exact command and recover the
  reported JSON.
- Ranking per setting: `metrics.normalized_acc_pct` (primary), `metrics.acc_pct` (tiebreaker).
