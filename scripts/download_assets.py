#!/usr/bin/env python3
"""Download Isaac asset files listed in .collect.mapping.json manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_ROOT = REPO_ROOT / "source" / "isaac_lab_tutorial" / "isaac_lab_tutorial" / "assets"


def _normalize_robot_name(robot: str) -> str:
    key = robot.strip().lower()
    if key in {"jetbot", "jb"}:
        return "jetbot"
    if key in {"turtlebot3_burger", "turtlebot3", "turtlebot", "tb3", "burger"}:
        return "turtlebot3_burger"
    if key == "all":
        return "all"
    raise ValueError(f"Unsupported robot profile: {robot}")


def _manifest_paths(robot: str) -> list[Path]:
    robot_key = _normalize_robot_name(robot)
    if robot_key == "all":
        manifests = sorted(ASSETS_ROOT.glob("*/.collect.mapping.json"))
    elif robot_key == "jetbot":
        manifests = [ASSETS_ROOT / "Jetbot" / ".collect.mapping.json"]
    else:
        manifests = [ASSETS_ROOT / "Turtlebot" / ".collect.mapping.json"]
    return manifests


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _resolve_target(manifest_dir: Path, rel_path: str) -> Path:
    candidate = (manifest_dir / rel_path).resolve()
    base = manifest_dir.resolve()
    if candidate != base and base not in candidate.parents:
        raise ValueError(f"Illegal target path outside manifest directory: {rel_path}")
    return candidate


def _download_to_file(url: str, output_path: Path, timeout: int, retries: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "isaaclab-jetbot-asset-fetcher"})

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response, tmp_path.open("wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            tmp_path.replace(output_path)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt < retries:
                print(f"  [RETRY] {url} ({attempt + 1}/{retries}) due to: {exc}")

    raise RuntimeError(f"Failed to download {url}: {last_error}")


def _load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch robot assets from .collect.mapping.json manifests.")
    parser.add_argument(
        "--robot",
        default="all",
        help="Robot profile: all | jetbot | turtlebot3_burger (supports aliases: jb, tb3, burger).",
    )
    parser.add_argument("--force", action="store_true", help="Re-download files even if target file exists.")
    parser.add_argument(
        "--verify-hash",
        action="store_true",
        help="Verify SHA1 against target_hash/source_hash when present in the manifest.",
    )
    parser.add_argument(
        "--strict-hash",
        action="store_true",
        help="Treat hash mismatch as failure (requires --verify-hash).",
    )
    parser.add_argument("--timeout", type=int, default=60, help="Per-request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=2, help="Retry count for failed downloads.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without downloading.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.strict_hash and not args.verify_hash:
        print("[ERROR] --strict-hash requires --verify-hash")
        return 2

    try:
        manifests = _manifest_paths(args.robot)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 2

    if not manifests:
        print(f"[ERROR] No manifest files found under: {ASSETS_ROOT}")
        return 1

    print(f"[ASSETS] Using assets root: {ASSETS_ROOT}")
    print(f"[ASSETS] Found {len(manifests)} manifest(s)")

    downloaded = 0
    skipped = 0
    failed = 0

    for manifest_path in manifests:
        if not manifest_path.exists():
            print(f"[WARN] Manifest not found: {manifest_path}")
            failed += 1
            continue

        print(f"[ASSETS] Processing manifest: {manifest_path}")
        try:
            manifest = _load_manifest(manifest_path)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [ERROR] Failed to read manifest: {exc}")
            failed += 1
            continue

        records = manifest.get("file_records", [])
        if not isinstance(records, list):
            print("  [ERROR] Invalid manifest format: file_records must be a list")
            failed += 1
            continue

        for record in records:
            source_url = record.get("source_url")
            target_rel = record.get("target_url")
            expected_hash = record.get("target_hash") or record.get("source_hash")

            if not source_url or not target_rel:
                print("  [WARN] Skipping malformed record (missing source_url or target_url)")
                failed += 1
                continue

            try:
                target_path = _resolve_target(manifest_path.parent, target_rel)
            except ValueError as exc:
                print(f"  [ERROR] {exc}")
                failed += 1
                continue

            if target_path.exists() and not args.force:
                if args.verify_hash and expected_hash:
                    actual_hash = _sha1(target_path)
                    if actual_hash != expected_hash:
                        print(f"  [WARN] Hash mismatch, re-downloading: {target_path}")
                    else:
                        print(f"  [SKIP] Exists and hash matches: {target_path}")
                        skipped += 1
                        continue
                else:
                    print(f"  [SKIP] Exists: {target_path}")
                    skipped += 1
                    continue

            if args.dry_run:
                print(f"  [DRY-RUN] Download {source_url} -> {target_path}")
                downloaded += 1
                continue

            print(f"  [GET] {source_url}")
            try:
                _download_to_file(source_url, target_path, timeout=args.timeout, retries=args.retries)
                if args.verify_hash and expected_hash:
                    actual_hash = _sha1(target_path)
                    if actual_hash != expected_hash:
                        msg = (
                            f"  [WARN] Hash mismatch after download: {target_path} "
                            f"(expected {expected_hash}, got {actual_hash})"
                        )
                        if args.strict_hash:
                            print(msg)
                            failed += 1
                            continue
                        print(msg)
                print(f"  [OK] Saved: {target_path}")
                downloaded += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  [ERROR] {exc}")
                failed += 1

    print("")
    print(f"[SUMMARY] downloaded={downloaded}, skipped={skipped}, failed={failed}")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
