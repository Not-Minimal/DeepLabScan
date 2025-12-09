# DeepLabScan/src/data/augmentation.py
"""
Data augmentation utilities and a small CLI to balance a YOLO dataset.

This module adapts the logic from `tarea_3.py` used in class:
- analyze_dataset(labels_dir, images_dir)
- rotate_bbox(...)
- apply_random_augmentation(...)
- balance_dataset(data_dir, limit_percent=0.25, seed=42)

CLI usage:
    python -m src.data.augmentation --data-dir /path/to/dataset --limit 0.25 --seed 42 --verbose

Assumes YOLO dataset structure:
  data_dir/
    data.yaml
    train/
      images/
      labels/
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def analyze_dataset(
    labels_dir: str, images_dir: str
) -> Tuple[Counter, Dict[str, List[int]]]:
    """
    Analyze YOLO labels in `labels_dir`.

    Args:
        labels_dir: directory containing .txt label files (YOLO: class x y w h)
        images_dir: directory where images live (jpg/png/jpeg) - used to map label files to image filenames

    Returns:
        conteo: Counter mapping class_id -> total instances
        img_map: dict mapping image_filename -> list[class_ids_in_image]
    """
    labels_path = Path(labels_dir)
    images_path = Path(images_dir)

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_path}")
    if not images_path.exists():
        raise FileNotFoundError(f"Images directory not found: {images_path}")

    conteo = Counter()
    img_composition: Dict[str, List[int]] = {}

    for lf in labels_path.iterdir():
        if lf.suffix.lower() != ".txt":
            continue
        stem = lf.stem

        # find the corresponding image file (prefer jpg, png, jpeg)
        img_file = None
        for ext in (".jpg", ".png", ".jpeg"):
            cand = images_path / (stem + ext)
            if cand.exists():
                img_file = cand.name
                break
        if img_file is None:
            # no matching image found; skip
            logger.debug("No image found for label %s, skipping", lf)
            continue

        classes_in_file = []
        try:
            with lf.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cid = int(parts[0])
                    except ValueError:
                        continue
                    conteo[cid] += 1
                    classes_in_file.append(cid)
        except Exception as e:
            logger.warning("Error reading label file %s: %s", lf, e)
            continue

        if classes_in_file:
            img_composition[img_file] = classes_in_file

    return conteo, img_composition


def rotate_bbox(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    image_width: int,
    image_height: int,
    angle_degrees: float,
) -> Tuple[float, float, float, float, int, int]:
    """
    Rotate a bounding box defined in pixel coordinates and return the
    axis-aligned bbox that contains the rotated corners, along with the
    new (rotated) image size (expand=True behavior).

    Returns:
        nx1, ny1, nx2, ny2, new_w, new_h
    """
    corners = np.array(
        [[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]], dtype=float
    )
    cx, cy = image_width / 2.0, image_height / 2.0

    # emulate PIL rotate expand=True to get new dims
    temp_img = Image.new("RGB", (image_width, image_height))
    rotated_temp = temp_img.rotate(angle_degrees, expand=True)
    new_w, new_h = rotated_temp.size

    theta = np.deg2rad(angle_degrees)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    new_pts = []
    for x, y in corners:
        x_rel, y_rel = x - cx, y - cy
        x_r = x_rel * cos_t - y_rel * sin_t
        y_r = x_rel * sin_t + y_rel * cos_t
        new_x = x_r + new_w / 2.0
        new_y = y_r + new_h / 2.0
        new_pts.append([new_x, new_y])

    new_pts = np.array(new_pts)
    nx1, ny1 = float(new_pts[:, 0].min()), float(new_pts[:, 1].min())
    nx2, ny2 = float(new_pts[:, 0].max()), float(new_pts[:, 1].max())

    return nx1, ny1, nx2, ny2, int(new_w), int(new_h)


def apply_random_augmentation(
    image: Image.Image, boxes: List[List[float]], w: int, h: int
) -> Tuple[Image.Image, List[List[float]]]:
    """
    Apply light random augmentations to an image and update YOLO-normalized boxes.

    boxes: list of [class_id, x_center_norm, y_center_norm, w_norm, h_norm]

    Returns:
        augmented_image (PIL.Image), new_boxes in the same normalized format
    """
    aug_img = image.copy()
    aug_boxes = [b[:] for b in boxes]
    curr_w, curr_h = w, h

    # Horizontal flip with 50% chance
    if random.random() < 0.5:
        aug_img = aug_img.transpose(Image.FLIP_LEFT_RIGHT)
        for i in range(len(aug_boxes)):
            aug_boxes[i][1] = 1.0 - aug_boxes[i][1]

    # Small rotation (-10, 10)
    angle = random.uniform(-10.0, 10.0)
    if abs(angle) > 1e-6:
        temp_img = Image.new("RGB", (curr_w, curr_h))
        new_w, new_h = temp_img.rotate(angle, expand=True).size
        aug_img = aug_img.rotate(angle, expand=True)

        new_boxes = []
        for box in aug_boxes:
            cid, xc, yc, bw, bh = box
            x_px = xc * curr_w
            y_px = yc * curr_h
            w_px = bw * curr_w
            h_px = bh * curr_h

            x1 = x_px - w_px / 2.0
            y1 = y_px - h_px / 2.0
            x2 = x_px + w_px / 2.0
            y2 = y_px + h_px / 2.0

            nx1, ny1, nx2, ny2, rw, rh = rotate_bbox(
                x1, y1, x2, y2, curr_w, curr_h, angle
            )

            n_xc = ((nx1 + nx2) / 2.0) / rw
            n_yc = ((ny1 + ny2) / 2.0) / rh
            n_bw = (nx2 - nx1) / rw
            n_bh = (ny2 - ny1) / rh

            # Validate reasonable values
            if n_bw <= 0 or n_bh <= 0:
                continue
            if not (0.0 <= n_xc <= 1.0 and 0.0 <= n_yc <= 1.0):
                # the box might be completely outside
                continue

            new_boxes.append(
                [
                    int(cid),
                    float(np.clip(n_xc, 0.0, 1.0)),
                    float(np.clip(n_yc, 0.0, 1.0)),
                    float(np.clip(n_bw, 0.0, 1.0)),
                    float(np.clip(n_bh, 0.0, 1.0)),
                ]
            )

        aug_boxes = new_boxes
        curr_w, curr_h = int(rw), int(rh)

    # Brightness jitter
    if random.random() < 0.5:
        enh = ImageEnhance.Brightness(aug_img)
        aug_img = enh.enhance(random.uniform(0.7, 1.3))

    return aug_img, aug_boxes


def balance_dataset(data_dir: str, limit_percent: float = 0.25, seed: int = 42) -> Dict:
    """
    Balance (augment) minority classes up to the majority class count,
    but respecting a global budget `limit_percent` of new instances.

    Args:
        data_dir: root of the dataset that contains data.yaml
        limit_percent: fraction (0.0-1.0) of new instances relative to total original
        seed: RNG seed for reproducibility

    Returns:
        summary dict with counts and list of generated files
    """
    random.seed(seed)

    root = Path(data_dir)
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"No data.yaml found in {root}")

    with yaml_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_images = root / "train" / "images"
    train_labels = root / "train" / "labels"

    if not train_images.exists() or not train_labels.exists():
        raise FileNotFoundError(
            f"Expected structure: {root}/train/images and {root}/train/labels"
        )

    conteo, img_map = analyze_dataset(str(train_labels), str(train_images))
    if not conteo:
        raise ValueError(
            "No labels were found to analyze in the training labels directory."
        )

    # Majority class
    id_major = max(conteo, key=conteo.get)
    count_major = conteo[id_major]
    total_original = sum(conteo.values())
    limit_new = int(total_original * float(limit_percent))

    logger.info("Majority class: %s (%d instances)", id_major, count_major)
    logger.info(
        "Total original instances: %d, global new-instance limit: %d",
        total_original,
        limit_new,
    )

    generated = 0
    generated_files = []
    conteo_actual = conteo.copy()

    classes_to_augment = [c for c in sorted(conteo_actual.keys()) if c != id_major]

    # Iterate classes and create new images until each matches majority or budget exhausted
    for cid in classes_to_augment:
        if generated >= limit_new:
            break

        target = count_major
        if conteo_actual[cid] >= target:
            continue

        # candidates: images that include this class
        candidates = [img for img, classes in img_map.items() if cid in classes]
        candidates.sort(key=lambda x: (id_major in img_map[x], len(img_map[x])))

        if not candidates:
            logger.info("No candidate images found for class %s, skipping", cid)
            continue

        cand_idx = 0
        while conteo_actual[cid] < target and generated < limit_new:
            img_filename = candidates[cand_idx % len(candidates)]
            cand_idx += 1

            src_img_path = train_images / img_filename
            src_lbl_path = train_labels / (Path(img_filename).stem + ".txt")

            try:
                img = Image.open(src_img_path).convert("RGB")
                w, h = img.size
                boxes = []
                with src_lbl_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        try:
                            boxes.append(
                                [int(parts[0])] + [float(p) for p in parts[1:]]
                            )
                        except ValueError:
                            continue

                aug_img, aug_boxes = apply_random_augmentation(img, boxes, w, h)
                if not aug_boxes:
                    # nothing valid after augmentation
                    continue

                suffix = random.randint(10000, 99999)
                new_base = f"{Path(img_filename).stem}_aug_{suffix}"
                out_img_path = train_images / (new_base + ".jpg")
                out_lbl_path = train_labels / (new_base + ".txt")

                # Save image and labels
                aug_img.save(out_img_path)
                with out_lbl_path.open("w", encoding="utf-8") as f:
                    for b in aug_boxes:
                        f.write(
                            f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n"
                        )

                added = 0
                for b in aug_boxes:
                    conteo_actual[int(b[0])] += 1
                    added += 1

                generated += added
                generated_files.append((str(out_img_path), str(out_lbl_path)))
                logger.debug("Generated %s (+%d instances)", out_img_path, added)

            except Exception as e:
                logger.warning("Error processing candidate %s: %s", img_filename, e)
                continue

            # safety break if too many iterations (avoid infinite loop)
            if cand_idx > len(candidates) * 20 and generated == 0:
                # If we've cycled many times and couldn't generate anything, break to next class
                logger.info(
                    "Too many failed attempts generating for class %s, moving on.", cid
                )
                break

    summary = {
        "total_original": int(total_original),
        "limit_new": int(limit_new),
        "generated": int(generated),
        "final_counts": {int(k): int(v) for k, v in conteo_actual.items()},
        "generated_files": [g[0] for g in generated_files],
    }

    return summary


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)
    else:
        # replace handlers to ensure consistent formatting
        root.handlers = [handler]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Balance YOLO dataset via light augmentation."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root of dataset (contains data.yaml)",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=0.25,
        help="Fraction of new instances allowed relative to original total (default 0.25)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--verbose", action="store_true", help="Verbose logging (DEBUG)"
    )
    parser.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON summary"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _setup_logging(args.verbose)

    try:
        summary = balance_dataset(
            args.data_dir, limit_percent=args.limit, seed=args.seed
        )
        if args.json:
            print(json.dumps(summary, indent=2, ensure_ascii=False))
        else:
            print("\n=== Balance Summary ===")
            print(f"Total original instances : {summary['total_original']}")
            print(f"Global new-instance limit: {summary['limit_new']}")
            print(f"Instances generated      : {summary['generated']}")
            print("Final counts per class   :")
            for cid, cnt in sorted(summary["final_counts"].items()):
                print(f"  - Class {cid}: {cnt}")
            print(f"Generated files (examples): {summary['generated_files'][:10]}")
    except Exception as e:
        logger.exception("Error running augmentation/balance: %s", e)
        raise
