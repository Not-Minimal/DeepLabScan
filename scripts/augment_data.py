#!/usr/bin/env python3
"""
CLI script to run dataset balancing/augmentation.

Usage:
    python scripts/augment_data.py --data-dir ./data/raw --limit 0.25 --seed 42 --verbose

This script adapts a classroom notebook workflow and delegates the heavy lifting
to `src.data.augmentation.balance_dataset`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure src is importable when running from project root
# (this mirrors the pattern used by other scripts in the repo)
from pathlib import Path as _Path

sys.path.append(str(_Path(__file__).resolve().parent.parent))

try:
    from src.data.augmentation import balance_dataset  # type: ignore
except Exception as e:  # pragma: no cover - defensive import handling for CLI
    # Provide a helpful error message if import fails
    print(
        "Error: no se pudo importar 'src.data.augmentation'. Asegúrate de ejecutar desde el root del proyecto"
    )
    print(f"Import error: {e}")
    raise

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Balance a YOLO dataset using light augmentation (flip, small rotations, brightness)."
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        required=True,
        help="Root of the dataset (must contain data.yaml and train/{images,labels})",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=float,
        default=0.25,
        help="Fraction of new instances allowed relative to original total (default: 0.25)",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging (DEBUG). Default is INFO.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON summary of the operation (useful for CI).",
    )
    return parser.parse_args()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(level)
    # Replace handlers to ensure consistent output when invoked repeatedly
    root.handlers = [handler]


def main() -> int:
    args = _parse_args()
    _setup_logging(args.verbose)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error("El directorio indicado no existe: %s", data_dir)
        return 2

    try:
        logger.info(
            "Iniciando balanceo del dataset en %s (limit=%s, seed=%d)",
            data_dir,
            args.limit,
            args.seed,
        )
        summary = balance_dataset(
            str(data_dir), limit_percent=args.limit, seed=args.seed
        )

        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print("\n=== Balance Summary ===")
            print(f"Total original instances : {summary.get('total_original')}")
            print(f"Global new-instance limit: {summary.get('limit_new')}")
            print(f"Instances generated      : {summary.get('generated')}")
            print("Final counts per class   :")
            final_counts = summary.get("final_counts", {})
            for cid in sorted(final_counts.keys(), key=int):
                print(f"  - Class {cid}: {final_counts[cid]}")
            gen_files = summary.get("generated_files", []) or []
            if gen_files:
                print(f"Generated files (examples): {gen_files[:10]}")
            else:
                print("No files were generated.")
        return 0
    except FileNotFoundError as e:
        logger.error("Fallo por archivo no encontrado: %s", e)
        return 3
    except ValueError as e:
        logger.error("Fallo de validación: %s", e)
        return 4
    except Exception as e:  # pragma: no cover - top level catch to return non-zero
        logger.exception("Error inesperado durante el balanceo: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
