# Stage 11 Relaxed-Scene Early-Stop Summary (2026-04-12)

## Goal
Stabilize short-nav behavior in Stage 11 while preserving task realism, reduce collision without reintroducing yaw dithering.

## Final selected strategy
- Use relaxed-but-realistic Stage 11 geometry/reward tweaks.
- Stop by performance peak (early-stop) instead of long-horizon continuation.
- Deploy best checkpoint alias.

## Effective config changes (Final Curriculum, Stage 11)
- `defect_single_gap_obstacle_y_jitter`: `0.04 -> 0.02`
- `defect_dwb_curriculum_goal_reach_thresholds[11]`: `0.76 -> 0.80`
- `defect_dwb_curriculum_time_penalties[11]`: `-0.07 -> -0.05`
- `defect_dwb_curriculum_goal_local_ys[11]`: `0.06 -> 0.12`

Reference file:
- `source/isaac_lab_tutorial/isaac_lab_tutorial/tasks/direct/isaac_lab_tutorial/isaac_lab_tutorial_env_cfg.py`

## Runs used in decision
1) Baseline (before relaxed tweaks)
- Log: `tmp/train_stage11_obstacle_xcenter_p020_coll30_danger012_surge4_def2_cap032_from2495345_10k_20260412_202448.log`
- Tail-5: success `33.20%`, collision `38.00%`, timeout `28.80%`
- Debug tail-5: `v_fwd=0.1626`, `yaw_flip=21.72%`

2) Relaxed config quick verify (5k)
- Log: `tmp/train_stage11_relaxed_scene_goaly012_reach080_time005_5k_from2505504_20260412_205255.log`
- Tail-5: success `45.12%`, collision `27.94%`, timeout `26.94%`
- Debug tail-5: `v_fwd=0.1618`, `yaw_flip=21.20%`
- Best single-point collision: `18.0%` (success `54.0%`)

3) Relaxed config continuation (10k)
- Log: `tmp/train_stage11_relaxed_scene_goaly012_reach080_time005_10k_from2510624_20260412_212530.log`
- Tail-5: success `41.54%`, collision `30.44%`, timeout `28.04%`
- Tail-10: success `42.87%`, collision `30.27%`, timeout `26.88%`
- Debug tail-5: `v_fwd=0.1630`, `yaw_flip=21.28%`
- Best single-point collision (from log parsing): step `2514396`, success `48.0%`, collision `22.0%`, timeout `30.0%`

## Decision rationale
- Relative to baseline, relaxed settings deliver robust improvements in success and collision.
- Continued training after 5k showed slight regression in tail averages, indicating diminishing returns and mild rollback behavior.
- Best-point performance is strong enough for engineering deployment and reproducibility.

## Deployment checkpoint aliases
- `logs/resume_aliases/stage11_relaxed_earlystop_best_20260412.pt`
- `logs/resume_aliases/agent_2514396_best_alias.pt`

Note:
- `agent_2514396_best_alias.pt` is an alias name for deployment tracking.
- The trainer saves periodic `agent_*.pt` plus `best_agent.pt`; exact step-named file at `2514396` is not emitted by interval checkpointing.

## Optional next verification
- Symmetric gap replay/eval with `x_center=0.0` for 100 episodes, report:
  - success / collision / timeout
  - `yaw_flip` and `v_fwd`


## Symmetric-gap generalization eval (x_center=0.00)

- Date: 2026-04-12
- Eval script: `scripts/skrl/eval_lidar_nav.py`
- Task: `Isaac-Lab-Tutorial-LidarShortNavDefectSingleObstacleFinalCurriculum-TurtleBot3-Direct-v0`
- Checkpoint: `logs/resume_aliases/agent_2514396.pt`
- Episodes: `100` (num_envs `32`, seed `42`)
- Symmetric override method: temporary Stage-11 `defect_dwb_curriculum_obstacle_x_centers[11]=0.0`, auto-restored after eval
- Log: `tmp/eval_stage11_symmetric_xcenter0_100ep_from2514396_20260412_221243.log`

### Result
- success: `20.0%`
- collision: `8.0%`
- timeout: `72.0%`
- ep_len_mean: `159.85`
- debug mean (from `[DEBUG]` lines, n=6):
  - `v_fwd=0.1345`
  - `yaw_flip=14.93%`

### Interpretation
- Compared to asymmetric Stage-11 run (`tail-5: success 41.54%, collision 30.44%, timeout 28.04%`), symmetric evaluation shows:
  - lower success and higher timeout (policy turns conservative in symmetric geometry)
  - collision remains low
  - yaw-flip stays moderate/low (no severe left-right dithering relapse)
