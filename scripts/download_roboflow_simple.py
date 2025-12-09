#!/usr/bin/env python3
"""
Minimal helper to run the Roboflow curl+unzip sequence.

This script performs the exact sequence:

  curl -L "<url>" -o roboflow.zip
  unzip -o roboflow.zip
  rm -f roboflow.zip

Usage:
  python scripts/download_roboflow_simple.py
  python scripts/download_roboflow_simple.py --url "https://app.roboflow.com/ds/x1AnsjJNU8?key=42r4vS7RNP"

Notes:
- This intentionally keeps behavior very small and predictable.
- It checks that `curl` and `unzip` exist on PATH and will error if they are missing.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_URL = "https://app.roboflow.com/ds/x1AnsjJNU8?key=42r4vS7RNP"
ZIP_NAME = "roboflow.zip"


def check_program(name: str) -> bool:
    return shutil.which(name) is not None


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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Download and unzip Roboflow zip (minimal)."
    )
    parser.add_argument(
        "--url",
        "-u",
        default=DEFAULT_URL,
        help="Roboflow curl URL to download (default is the provided one).",
    )
    parser.add_argument(
        "--zip-name",
        default=ZIP_NAME,
        help=f"Temporary zip filename (default: {ZIP_NAME}).",
    )
    args = parser.parse_args(argv)

    url = args.url
    zip_path = Path(args.zip_name)

    # Check required tools
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

    # Download
    print(f"Downloading: {url}")
    cmd_download = f'curl -L "{url}" -o "{zip_path}"'
    rc = run_command(cmd_download)
    if rc != 0 or not zip_path.exists():
        print(
            f"Download failed (exit {rc}). ZIP not found at {zip_path}.",
            file=sys.stderr,
        )
        return 4

    # Unzip (overwrite -o to replace files if present)
    print(f"Unzipping: {zip_path}")
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

    # Cleanup
    try:
        zip_path.unlink()
        print(f"Removed temporary file: {zip_path}")
    except Exception:
        print(
            f"Could not remove temporary zip: {zip_path}. You can remove it manually.",
            file=sys.stderr,
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
