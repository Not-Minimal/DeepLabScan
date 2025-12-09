#!/usr/bin/env python3
"""
Minimal Roboflow downloader and organizer.

Performs:
  - curl -L "<url>" -o roboflow.zip
  - unzip -o roboflow.zip
  - rm -f roboflow.zip

Additionally: if the extraction produced a `data.yaml` (or train/valid/test)
directly in the project root, this script will move those items into
`data/raw` (or another destination directory specified with --dest).

Usage:
  python scripts/download_roboflow_simple.py
  python scripts/download_roboflow_simple.py --url "<roboflow-url>" --dest data/raw --force

Notes:
- This script expects `curl` and `unzip` to be available on PATH.
- By default it will not overwrite existing destination files unless --force is provided.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

DEFAULT_URL = "https://app.roboflow.com/ds/x1AnsjJNU8?key=42r4vS7RNP"
DEFAULT_ZIP = "roboflow.zip"
DEFAULT_DEST = "data/raw"


def check_program(name: str) -> bool:
    from shutil import which

    return which(name) is not None


def run_command(cmd: str) -> int:
    try:
        result = subprocess.run(cmd, shell=True, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error running command: {e}", file=sys.stderr)
        return 1


def safe_move(src: Path, dst: Path, force: bool = False) -> bool:
    """
    Move src -> dst. If dst exists:
      - if force: remove dst then move
      - else: skip and return False
    Returns True if move performed, False if skipped.
    """
    if not src.exists():
        return False
    if dst.exists():
        if force:
            # remove existing dst
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            else:
                shutil.rmtree(dst)
        else:
            print(
                f"Skipping move of {src} because destination {dst} already exists (use --force to overwrite)"
            )
            return False

    # Ensure parent exists
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(src), str(dst))
        return True
    except Exception as e:
        print(f"Failed to move {src} -> {dst}: {e}", file=sys.stderr)
        return False


def move_project_root_dataset_items(
    project_root: Path, dest_root: Path, force: bool = False
) -> None:
    """
    If the extraction created dataset pieces at the project root (e.g. data.yaml,
    train/, valid/, test/), move them under dest_root so the repo root remains clean.

    Moves:
      - project_root/data.yaml -> dest_root/data.yaml
      - project_root/{train,valid,test} -> dest_root/{train,valid,test}

    Behavior on conflict:
      - If target exists and force==True, overwrite.
      - If target exists and force==False, skip that item.
    """
    items_moved = []

    # Ensure dest exists
    dest_root.mkdir(parents=True, exist_ok=True)

    # Move data.yaml if present at project root
    src_yaml = project_root / "data.yaml"
    if src_yaml.exists():
        dst_yaml = dest_root / "data.yaml"
        ok = safe_move(src_yaml, dst_yaml, force=force)
        if ok:
            items_moved.append(("data.yaml", dst_yaml))

    # Move common splits if present at project root
    for split in ("train", "valid", "test"):
        src_split = project_root / split
        if src_split.exists() and src_split.is_dir():
            dst_split = dest_root / split
            ok = safe_move(src_split, dst_split, force=force)
            if ok:
                items_moved.append((split, dst_split))

    if items_moved:
        print("Moved dataset items into:", dest_root)
        for name, path in items_moved:
            print(f"  - {name} -> {path}")
    else:
        print("No dataset items found at project root to move.")


def find_possible_project_root_dataset(project_root: Path) -> Iterable[Path]:
    """
    Return items (if any) that look like dataset pieces in project root:
      - data.yaml
      - train/, valid/, test/
    """
    candidates = []
    if (project_root / "data.yaml").exists():
        candidates.append(project_root / "data.yaml")
    for s in ("train", "valid", "test"):
        p = project_root / s
        if p.exists():
            candidates.append(p)
    return candidates


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Minimal Roboflow downloader + organize into data/raw if needed"
    )
    p.add_argument(
        "--url", "-u", default=DEFAULT_URL, help="Roboflow curl URL to download"
    )
    p.add_argument(
        "--zip-name", default=DEFAULT_ZIP, help="Temporary zip filename to write"
    )
    p.add_argument(
        "--dest",
        "-d",
        default=DEFAULT_DEST,
        help="Destination directory for dataset (default: data/raw)",
    )
    p.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing files in destination",
    )
    p.add_argument(
        "--no-move-root",
        action="store_true",
        help="Do not attempt to move files from project root into dest",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    url = args.url
    zip_name = args.zip_name
    dest_dir = Path(args.dest)
    force = args.force
    move_root = not args.no_move_root

    # Check tools
    if not check_program("curl"):
        print(
            "Required program 'curl' not found in PATH. Install curl or run the equivalent command manually.",
            file=sys.stderr,
        )
        return 2
    if not check_program("unzip"):
        print(
            "Required program 'unzip' not found in PATH. Install unzip or run the equivalent command manually.",
            file=sys.stderr,
        )
        return 3

    zip_path = Path(zip_name)

    # Download
    print(f"Downloading: {url} -> {zip_path}")
    cmd_download = f'curl -L "{url}" -o "{zip_path}"'
    rc = run_command(cmd_download)
    if rc != 0 or not zip_path.exists():
        print(
            f"Download failed (exit {rc}). ZIP not found at {zip_path}.",
            file=sys.stderr,
        )
        return 4

    # Unzip into current working directory (project root)
    print(f"Unzipping: {zip_path} (overwriting existing files with unzip -o)")
    cmd_unzip = f'unzip -o "{zip_path}"'
    rc = run_command(cmd_unzip)
    if rc != 0:
        print(f"Unzip failed (exit {rc}).", file=sys.stderr)
        # attempt cleanup
        try:
            if zip_path.exists():
                zip_path.unlink()
        except Exception:
            pass
        return 5

    # Remove zip file
    try:
        zip_path.unlink()
        print(f"Removed temporary file: {zip_path}")
    except Exception:
        print(
            f"Could not remove temporary zip: {zip_path}. You can remove it manually.",
            file=sys.stderr,
        )

    # If dataset pieces (data.yaml or train/valid/test) are now in project root, move them into dest
    project_root = Path.cwd()
    if move_root:
        candidates = list(find_possible_project_root_dataset(project_root))
        if candidates:
            print("Detected dataset files/folders at project root.")
            # Ensure dest is inside project and create parent
            dest_dir = project_root / dest_dir
            move_project_root_dataset_items(project_root, dest_dir, force=force)
        else:
            print(
                "No data.yaml or train/valid/test found at project root - nothing moved."
            )

    print("Download and extraction complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
