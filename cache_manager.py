#!/usr/bin/env python3
"""
cache_manager.py — Clean up accumulated cache and temporary files.

Usage:
    python cache_manager.py              # preview safe-to-delete items (dry-run)
    python cache_manager.py --safe       # delete safe targets (caches, temp files)
    python cache_manager.py --all        # also delete regeneratable data outputs
    python cache_manager.py --dry-run    # preview what --safe or --all would remove
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Always safe to delete (caches, temp files, IDE artefacts)
SAFE_DIR_NAMES = {".ipynb_checkpoints", "__pycache__", ".jupyter"}
SAFE_FILE_NAMES = {".DS_Store", "training_log.jsonl"}
SAFE_FILE_SUFFIXES = {".bak"}
SAFE_DIRS = {"anaconda_projects"}  # relative to project root

# Regeneratable outputs (only with --all)
REGEN_DIRS = {
    Path("data") / "predictions",
    Path("data") / "processed",
}

SKIP_DIRS = {".git"}


def _fmt_size(n_bytes: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n_bytes) < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def _dir_size(p: Path) -> int:
    """Total bytes inside a directory."""
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def collect_targets(include_regen: bool = False):
    """Walk the project tree and return a list of (Path, size_bytes, category) tuples."""
    targets = []

    for item in sorted(PROJECT_ROOT.rglob("*")):
        # Never touch .git
        rel = item.relative_to(PROJECT_ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue

        if item.is_dir():
            if item.name in SAFE_DIR_NAMES:
                targets.append((item, _dir_size(item), "safe"))
            elif str(rel) in SAFE_DIRS:
                targets.append((item, _dir_size(item), "safe"))
        elif item.is_file():
            if item.name in SAFE_FILE_NAMES:
                targets.append((item, item.stat().st_size, "safe"))
            elif item.suffix in SAFE_FILE_SUFFIXES:
                targets.append((item, item.stat().st_size, "safe"))

    if include_regen:
        for rel_dir in REGEN_DIRS:
            full = PROJECT_ROOT / rel_dir
            if full.exists() and full.is_dir():
                targets.append((full, _dir_size(full), "regen"))

    # De-duplicate: remove children whose parents are already listed
    target_set = {t[0] for t in targets}
    deduped = []
    for path, size, cat in targets:
        if not any(p != path and path.is_relative_to(p) for p in target_set):
            deduped.append((path, size, cat))

    return deduped


def preview(targets):
    """Print a table of targets without deleting anything."""
    if not targets:
        print("Nothing to clean.")
        return

    total = 0
    print(f"\n{'Path':<60s} {'Size':>10s}  {'Type'}")
    print("-" * 82)
    for path, size, cat in targets:
        rel = path.relative_to(PROJECT_ROOT)
        label = "safe" if cat == "safe" else "regen (--all)"
        print(f"{str(rel):<60s} {_fmt_size(size):>10s}  {label}")
        total += size
    print("-" * 82)
    print(f"{'Total':<60s} {_fmt_size(total):>10s}")
    print()


def delete_targets(targets):
    """Delete listed targets after user confirmation."""
    if not targets:
        print("Nothing to clean.")
        return

    preview(targets)

    answer = input("Delete the above items? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        return

    freed = 0
    for path, size, _cat in targets:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            freed += size
            print(f"  Deleted: {path.relative_to(PROJECT_ROOT)}")
        except OSError as e:
            print(f"  FAILED:  {path.relative_to(PROJECT_ROOT)} ({e})")

    print(f"\nFreed {_fmt_size(freed)}.")


def main():
    parser = argparse.ArgumentParser(
        description="Clean cache and temp files from the project.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be deleted (no changes)")
    parser.add_argument("--safe", action="store_true",
                        help="Delete safe targets (caches, temp files)")
    parser.add_argument("--all", action="store_true",
                        help="Delete safe + regeneratable data outputs")
    args = parser.parse_args()

    include_regen = args.all

    # Default to dry-run if no action flag is given
    if not args.safe and not args.all:
        args.dry_run = True

    targets = collect_targets(include_regen=include_regen)

    if args.dry_run:
        print("[DRY RUN] The following items would be removed:")
        preview(targets)
    else:
        delete_targets(targets)


if __name__ == "__main__":
    main()
