<!--
Merging Galore submission PR. This PR holds your merger code; the leaderboard
entry is a separate public Flywheel attempt node. See SUBMITTING.md for the full
flow.
-->

## Submission

- **Merger name:** <!-- e.g. my_merger (config: conf/merger/my_merger.yaml) -->
- **Setting:** <!-- small (ViT-B-32) | large (ViT-L-14) -->
- **Track:** <!-- baseline | improvement | ablation | reproduction | failure_analysis -->
- **Run name:** <!-- short slug -->
- **Self-reported (Hard10):** norm_acc_pct = <!-- 00.00 -->, acc_pct = <!-- 00.00 -->

## Idea

<!-- One or two sentences: what your merger does and why it should help. -->

## What this PR adds

- [ ] `src/model_merging/merger/<name>_merger.py`
- [ ] `conf/merger/<name>.yaml`
- [ ] (if SVD-based) any helper changes under `src/model_merging/merging/`

## Submission checklist

- [ ] Ran `scripts/evaluate_multitask_merging.py merger=<name> benchmark=hard10 nn/encoder=<b32|l14>` on the **unmodified** harness.
- [ ] **No additional training** — weight-space merging only.
- [ ] Generated the artifact with `scripts/build_submission.py ... --pr-url <this PR>`.
- [ ] Created a **public** Flywheel attempt node and finalized the two artifacts (`results.json` + `campaign_submission_v1.json`). See SUBMITTING.md.
- [ ] `repo.pr_url` in the submission points to this PR.

## Flywheel attempt node

<!-- Link to your public attempt node (flywheel.paradigma.inc/node/<id>) -->
