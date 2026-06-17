"""Helpers for restoring curriculum step from skrl checkpoint paths."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any

_AGENT_STEP_PATTERN = re.compile(r"agent_(\d+)\.pt$")


def infer_resume_step_from_checkpoint(resume_path: str | Path) -> tuple[int | None, str | None]:
    """Infer the environment step counter from a checkpoint path.

    Supports explicit ``agent_<step>.pt`` checkpoints and best-checkpoint recovery by
    aligning ``best_agent.pt`` with the closest sibling ``agent_<step>.pt`` checkpoint
    timestamp in the same directory.
    """

    checkpoint_path = Path(resume_path).expanduser().resolve()
    match = _AGENT_STEP_PATTERN.fullmatch(checkpoint_path.name)
    if match:
        return int(match.group(1)), f"checkpoint filename {checkpoint_path.name}"

    if checkpoint_path.name != "best_agent.pt":
        return None, None

    try:
        best_mtime = checkpoint_path.stat().st_mtime
    except OSError:
        return None, None

    candidates = _collect_agent_step_candidates(checkpoint_path.parent)
    inferred_step, inferred_source = _infer_step_from_candidates(best_mtime, candidates, checkpoint_path.name)
    if inferred_step is not None:
        return inferred_step, inferred_source

    search_root = _infer_checkpoint_search_root(checkpoint_path)
    if search_root is not None:
        inferred_step, inferred_source = _infer_step_from_matching_best_checkpoint(checkpoint_path, search_root)
        if inferred_step is not None:
            return inferred_step, inferred_source

    return None, None


def _collect_agent_step_candidates(search_dir: Path) -> list[tuple[Path, int, float]]:
    candidates: list[tuple[Path, int, float]] = []
    for candidate_path in search_dir.glob("agent_*.pt"):
        match = _AGENT_STEP_PATTERN.fullmatch(candidate_path.name)
        if not match:
            continue
        try:
            candidate_mtime = candidate_path.stat().st_mtime
        except OSError:
            continue
        candidates.append((candidate_path, int(match.group(1)), candidate_mtime))
    return candidates


def _infer_step_from_candidates(
    target_mtime: float, candidates: list[tuple[Path, int, float]], checkpoint_name: str
) -> tuple[int | None, str | None]:
    if not candidates:
        return None, None

    older_or_equal = [candidate for candidate in candidates if candidate[2] <= target_mtime]
    if older_or_equal:
        selected_path, selected_step, _ = max(older_or_equal, key=lambda candidate: candidate[2])
        return selected_step, f"{checkpoint_name} matched to {selected_path.name} by mtime"

    selected_path, selected_step, _ = min(candidates, key=lambda candidate: abs(candidate[2] - target_mtime))
    return selected_step, f"{checkpoint_name} matched to {selected_path.name} by nearest mtime"


def _infer_checkpoint_search_root(checkpoint_path: Path) -> Path | None:
    if checkpoint_path.parent.name == "checkpoints" and len(checkpoint_path.parents) >= 3:
        return checkpoint_path.parents[2]
    if len(checkpoint_path.parents) >= 2:
        return checkpoint_path.parents[1]
    return None


def _infer_step_from_matching_best_checkpoint(
    checkpoint_path: Path, search_root: Path
) -> tuple[int | None, str | None]:
    try:
        checkpoint_size = checkpoint_path.stat().st_size
    except OSError:
        return None, None

    target_hash: str | None = None
    peer_best_paths = sorted(search_root.glob("**/best_agent.pt"))
    peer_best_paths.sort(key=lambda path: path.parent.name != "checkpoints")
    for peer_best_path in peer_best_paths:
        if peer_best_path == checkpoint_path:
            continue
        try:
            if peer_best_path.stat().st_size != checkpoint_size:
                continue
        except OSError:
            continue

        if target_hash is None:
            target_hash = _sha256sum(checkpoint_path)
        if _sha256sum(peer_best_path) != target_hash:
            continue

        inferred_step, inferred_source = infer_resume_step_from_checkpoint(peer_best_path)
        if inferred_step is not None:
            try:
                peer_label = str(peer_best_path.relative_to(search_root))
            except ValueError:
                peer_label = str(peer_best_path)
            if inferred_source:
                return inferred_step, f"{checkpoint_path.name} content-matched to {peer_label}; {inferred_source}"
            return inferred_step, f"{checkpoint_path.name} content-matched to {peer_label}"

    return None, None


def _sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restore_env_curriculum_step(
    env: Any,
    resume_step: int | None,
    *,
    curriculum_start_step: int | None = None,
    source: str | None = None,
    log_prefix: str = "[INFO]",
) -> bool:
    """Restore the common environment step counter and refresh any known curricula."""

    base_env = getattr(env, "unwrapped", env)
    if not hasattr(base_env, "common_step_counter"):
        print(f"{log_prefix} Env has no common_step_counter; skipped curriculum step restore.")
        return False

    target_step = curriculum_start_step if curriculum_start_step is not None else resume_step
    if target_step is None:
        print(f"{log_prefix} No curriculum step available; skipped curriculum step restore.")
        return False

    base_env.common_step_counter = int(max(0, target_step))
    if hasattr(base_env, "_defect_dwb_curriculum_step_origin_override"):
        # Preserve curriculum progress across process restarts for relative-step curricula.
        base_env._defect_dwb_curriculum_step_origin_override = 0
    for method_name in (
        "_apply_defect_dwb_curriculum",
        "_apply_defect_ushape_curriculum",
        "_apply_defect_mppi_curriculum",
    ):
        if not hasattr(base_env, method_name):
            continue
        curriculum_fn = getattr(base_env, method_name)
        try:
            curriculum_fn(force=True)
        except TypeError:
            curriculum_fn()

    source_suffix = f" from {source}" if source else ""
    override_suffix = ""
    if curriculum_start_step is not None:
        override_suffix = f" (curriculum_start_step={int(max(0, curriculum_start_step))})"
    print(
        f"{log_prefix} Restored env common_step_counter{source_suffix}: "
        f"{base_env.common_step_counter}{override_suffix}"
    )
    return True
