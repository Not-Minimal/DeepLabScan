#!/usr/bin/env python3
"""
download_roboflow_zip.py

Descarga un ZIP (por ejemplo, el que proporciona Roboflow con `curl -L "<url>" > roboflow.zip`)
y organiza su contenido en la estructura esperada por el proyecto:

  <dest>/
    data.yaml
    train/
      images/
      labels/
    valid/
      images/
      labels/
    test/
      images/
      labels/

Características:
- Descarga vía URL (urllib), descomprime el ZIP y "aplana" si el ZIP contiene una carpeta raíz.
- Mueve el contenido al destino (por defecto: data/raw).
- Si falta `data.yaml`, puede generarlo automáticamente a partir de las etiquetas (opción `--generate-data-yaml`).
- Opciones: --keep-zip, --force (sobrescribir destino), --dry-run, --verbose.

Uso:
    python scripts/download_roboflow_zip.py --url "https://app.roboflow.com/ds/XXX?key=YYY"
    python scripts/download_roboflow_zip.py --url "<url>" --dest data/raw --generate-data-yaml --verbose

Este script no requiere paquetes externos.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

logger = logging.getLogger("download_roboflow_zip")


def download_url_to_file(url: str, out_path: Path, verbose: bool = False) -> None:
    logger.info("Descargando URL: %s -> %s", url, out_path)
    req = urllib.request.Request(url, headers={"User-Agent": "DeepLabScan/1.0"})
    with urllib.request.urlopen(req) as resp:
        # Guardar streaming para evitar cargar todo en memoria
        with out_path.open("wb") as f:
            shutil.copyfileobj(resp, f)
    logger.info("Descarga completada: %s", out_path)


def unzip_to_dir(zip_path: Path, target_dir: Path, overwrite: bool = False) -> None:
    logger.info("Descomprimiendo %s a %s", zip_path, target_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Extraer en un directorio temporal dentro de target_dir para luego mover si es necesario
        with tempfile.TemporaryDirectory(dir=str(target_dir.parent)) as td:
            tmpdir = Path(td) / "roboflow_unzip_tmp"
            tmpdir.mkdir(parents=True, exist_ok=True)
            zf.extractall(path=str(tmpdir))

            # Si tmpdir contiene una única carpeta top-level, mover su contenido
            top_items = list(tmpdir.iterdir())
            if len(top_items) == 1 and top_items[0].is_dir():
                extracted_root = top_items[0]
                logger.debug("ZIP contiene carpeta raíz: %s", extracted_root.name)
            else:
                extracted_root = tmpdir

            # Mover contenido de extracted_root -> target_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            for item in extracted_root.iterdir():
                dest = target_dir / item.name
                if dest.exists():
                    if overwrite:
                        if dest.is_dir():
                            shutil.rmtree(dest)
                        else:
                            dest.unlink()
                    else:
                        # intentar fusionar si ambos son directorios
                        if item.is_dir() and dest.is_dir():
                            logger.debug("Fusionando directorio %s -> %s", item, dest)
                            for sub in item.iterdir():
                                sub_dest = dest / sub.name
                                if sub_dest.exists():
                                    if sub_dest.is_dir():
                                        shutil.rmtree(sub_dest)
                                    else:
                                        sub_dest.unlink()
                                if sub.is_dir():
                                    shutil.copytree(sub, sub_dest)
                                else:
                                    shutil.copy2(sub, sub_dest)
                            continue
                        else:
                            raise FileExistsError(
                                f"Destino ya existe: {dest} (use --force para sobrescribir)"
                            )
                # mover/renombrar
                if item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
    logger.info("Descompresión finalizada.")


def discover_dataset_layout(root: Path) -> Dict[str, bool]:
    """
    Revisa la estructura y devuelve un dict indicando presencia de:
      - data_yaml
      - train/images, train/labels
      - valid/images, valid/labels
      - test/images, test/labels
    """
    report = {
        "data_yaml": (root / "data.yaml").exists(),
        "train_images": (root / "train" / "images").exists(),
        "train_labels": (root / "train" / "labels").exists(),
        "valid_images": (root / "valid" / "images").exists(),
        "valid_labels": (root / "valid" / "labels").exists(),
        "test_images": (root / "test" / "images").exists(),
        "test_labels": (root / "test" / "labels").exists(),
    }
    return report


def scan_label_classes(labels_dir: Path) -> Tuple[Set[int], Dict[str, List[int]]]:
    """
    Escanea archivos .txt tipo YOLO en labels_dir y devuelve:
      - set de class_ids encontrados (asumiendo que el primer token es entero)
      - map imagen -> lista de class_ids (imagen filename)
    """
    classes: Set[int] = set()
    img_map: Dict[str, List[int]] = {}
    if not labels_dir.exists():
        return classes, img_map

    for f in labels_dir.glob("*.txt"):
        try:
            with f.open("r", encoding="utf-8") as fh:
                cids = []
                for line in fh:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cid = int(parts[0])
                    except Exception:
                        # no es entero; ignorar
                        continue
                    classes.add(cid)
                    cids.append(cid)
                if cids:
                    # buscar imagen filename candidates
                    base = f.stem
                    # preferir jpg
                    for ext in (".jpg", ".png", ".jpeg"):
                        candidate = labels_dir.parent / "images" / (base + ext)
                        if candidate.exists():
                            img_map[candidate.name] = cids
                            break
                    else:
                        # fallback: usar base + .jpg (archivo puede no existir en disco)
                        img_map[base] = cids
        except Exception as e:
            logger.debug("Error leyendo label %s: %s", f, e)
    return classes, img_map


def generate_data_yaml(
    root: Path,
    inferred_classes: Set[int],
    inferred_names: Union[None, List[str]] = None,
) -> None:
    """
    Crea un data.yaml básico en root con rutas relativas y nombres inferidos.
    - Si inferred_names es None, genera names en formato ['class_0','class_1',...]
    """
    train_images = "train/images"
    val_images = "valid/images"
    test_images = "test/images"
    if not (root / "train").exists():
        logger.warning(
            "No se encontró carpeta 'train' para generar data.yaml correctamente."
        )

    if inferred_classes:
        max_idx = max(inferred_classes)
        nc = max_idx + 1
        # construir names array
        if inferred_names:
            names = inferred_names
        else:
            names = [f"class_{i}" for i in range(nc)]
    else:
        # fallback: intentar contar imágenes para inferir nc = 1
        nc = 1
        names = ["class_0"]

    data = {
        "train": train_images,
        "val": val_images,
        "test": test_images,
        "nc": nc,
        "names": names,
    }

    yaml_path = root / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        # write simple YAML without adding extra dependencies
        import yaml as _yaml

        _yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    logger.info("Se creó data.yaml en %s (nc=%d)", yaml_path, nc)


def tidy_move_into_place(
    src_root: Path, dest_root: Path, overwrite: bool = False
) -> None:
    """
    Si el contenido fue descomprimido en src_root y queremos moverlo a dest_root.
    Realiza un merge seguro.
    """
    dest_root.mkdir(parents=True, exist_ok=True)
    for item in src_root.iterdir():
        dest = dest_root / item.name
        if dest.exists():
            if overwrite:
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            else:
                # intentar fusionar directorios
                if item.is_dir() and dest.is_dir():
                    for sub in item.iterdir():
                        sub_dest = dest / sub.name
                        if sub_dest.exists():
                            if sub_dest.is_dir():
                                shutil.rmtree(sub_dest)
                            else:
                                sub_dest.unlink()
                        if sub.is_dir():
                            shutil.copytree(sub, sub_dest)
                        else:
                            shutil.copy2(sub, sub_dest)
                    continue
                else:
                    raise FileExistsError(
                        f"Destino ya existe: {dest} (use --force para sobrescribir)"
                    )
        # mover/copy
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)


def find_single_subdir(path: Path) -> Union[Path, None]:
    """
    Si `path` contiene exactamente un subdirectorio y nada más (o sólo __MACOSX/),
    devuelve esa subcarpeta; en caso contrario devuelve None.
    """
    children = [p for p in path.iterdir() if p.name != "__MACOSX"]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return None


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Descargar y organizar ZIP de Roboflow en data/raw (o destino indicado)."
    )
    parser.add_argument(
        "--url",
        "-u",
        required=True,
        help="URL para descargar el ZIP (Roboflow share URL).",
    )
    parser.add_argument(
        "--dest",
        "-d",
        default="data/raw",
        help="Directorio destino donde colocar dataset (por defecto: data/raw).",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Conservar el ZIP descargado en el destino.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescribir contenido existente en destino.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simular sin escribir (descarga sí, pero no mover ni crear YAML).",
    )
    parser.add_argument(
        "--generate-data-yaml",
        action="store_true",
        help="Si falta data.yaml, generar uno inferido a partir de labels.",
    )
    parser.add_argument("--verbose", action="store_true", help="Mostrar logs DEBUG.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    dest = Path(args.dest).resolve()

    # Crear carpeta temporal para la operación
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        zip_path = td_path / "roboflow_download.zip"

        try:
            download_url_to_file(args.url, zip_path, verbose=args.verbose)
        except Exception as e:
            logger.error("Error descargando la URL: %s", e)
            return 10

        # Verificar que sea un ZIP válido
        try:
            with zipfile.ZipFile(zip_path, "r") as _:
                pass
        except zipfile.BadZipFile:
            logger.error("El archivo descargado no es un ZIP válido.")
            return 11

        # Si dry-run: informar contenido del ZIP y salir
        if args.dry_run:
            with zipfile.ZipFile(zip_path, "r") as zf:
                logger.info("Contenido del ZIP (primeras 50 entradas):")
                for i, name in enumerate(zf.namelist()):
                    if i >= 50:
                        logger.info("... y más")
                        break
                    logger.info(" - %s", name)
            logger.info("Dry-run completado. No se hicieron cambios en el filesystem.")
            if args.keep_zip:
                out_zip = dest / "roboflow.zip"
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(zip_path, out_zip)
                logger.info("ZIP guardado en: %s", out_zip)
            return 0

        # Descomprimir en directorio temporal y luego mover/planchar al destino
        tmp_extract_dir = td_path / "extracted"
        tmp_extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            unzip_to_dir(zip_path, tmp_extract_dir, overwrite=args.force)
        except Exception as e:
            logger.error("Error descomprimiendo: %s", e)
            return 12

        # Si tmp_extract_dir contiene una única carpeta que es el proyecto, usarla
        single = find_single_subdir(tmp_extract_dir)
        if single:
            logger.debug("ZIP tenía carpeta raíz; usando %s como fuente", single)
            src_root = single
        else:
            src_root = tmp_extract_dir

        logger.debug("Contenido fuente para mover: %s", list(src_root.iterdir())[:20])

        # Mover a destino
        if dest.exists() and not args.force:
            # comprobar si parece ya un dataset válido -> si es así, abortar a menos que force
            layout = discover_dataset_layout(dest)
            if (
                layout.get("data_yaml")
                or layout.get("train_images")
                or layout.get("train_labels")
            ):
                logger.error(
                    "Destino %s ya contiene archivos (parece un dataset). Use --force para sobrescribir o elegir otro destino.",
                    dest,
                )
                return 13

        if args.force and dest.exists():
            logger.info("--force: limpiando destino %s", dest)
            shutil.rmtree(dest)

        if not args.dry_run:
            try:
                tidy_move_into_place(src_root, dest, overwrite=args.force)
            except Exception as e:
                logger.error("Error moviendo archivos al destino: %s", e)
                return 14

        # Mantener zip si se indicó
        if args.keep_zip:
            out_zip = dest / "roboflow.zip"
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(zip_path, out_zip)
            logger.info("ZIP guardado en: %s", out_zip)

        # Verificar estructura y crear data.yaml si falta y se pidió
        layout = discover_dataset_layout(dest)
        logger.info("Estructura detectada en destino: %s", json.dumps(layout, indent=2))
        if not layout["data_yaml"]:
            logger.warning("No se encontró data.yaml en el destino.")
            if args.generate_data_yaml:
                # tratar de inferir clases a partir de train/labels
                labels_dir = dest / "train" / "labels"
                classes, img_map = scan_label_classes(labels_dir)
                logger.info(
                    "Clases inferidas: %s", sorted(list(classes)) if classes else "N/A"
                )
                if classes:
                    if not args.dry_run:
                        generate_data_yaml(dest, classes)
                        logger.info("data.yaml creado en %s", dest / "data.yaml")
                else:
                    # si no se pudieron inferir clases, crear un data.yaml mínimo
                    logger.warning(
                        "No se pudieron inferir clases desde las etiquetas; se creará data.yaml mínimo con nc=1."
                    )
                    if not args.dry_run:
                        generate_data_yaml(dest, set())
            else:
                logger.info(
                    "Si quieres generar un data.yaml automáticamente, vuelve a ejecutar con --generate-data-yaml"
                )

        logger.info("Operación completada. Dataset organizado en: %s", dest)
        logger.info(
            "Verifica que el directorio contiene data.yaml y train/valid/test con images/labels."
        )

    return 0


if __name__ == "__main__":
    exit_code = main(sys.argv[1:])
    sys.exit(exit_code)
