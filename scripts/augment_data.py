#!/usr/bin/env python3
"""
Script de Aumentación de Datos para Balanceo de Dataset YOLO

Adaptado del notebook tarea_3.py usado en clases.

Este script:
1. Analiza el dataset (conteo de clases, identificar mayoría/minoría)
2. Genera gráfico de distribución inicial
3. Aplica aumentación de datos (flip, rotación, brillo) hasta balancear clases
4. Respeta límite del 25% de nuevas instancias
5. Genera gráfico de distribución final

Uso:
    python scripts/augment_data.py --data-dir data/raw
    python scripts/augment_data.py --data-dir data/raw --limit 0.25 --seed 42
"""

import argparse
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageEnhance

# Importar matplotlib y seaborn para gráficos
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️  matplotlib/seaborn/pandas no disponibles. No se generarán gráficos.")


def analizar_dataset(labels_path, images_path):
    """
    Analiza el dataset y retorna:
    1. Conteo total por clase
    2. Un mapa de {imagen_filename: [lista_de_clases_en_imagen]}
    """
    conteo = Counter()
    img_composition = defaultdict(list)

    for filename in os.listdir(labels_path):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(labels_path, filename)
        img_name = os.path.splitext(filename)[0]

        # Buscar imagen correspondiente
        img_file = None
        for ext in [".jpg", ".png", ".jpeg"]:
            img_path = os.path.join(images_path, img_name + ext)
            if os.path.exists(img_path):
                img_file = img_name + ext
                break

        if not img_file:
            continue

        with open(filepath, "r") as f:
            classes_in_file = []
            for line in f:
                try:
                    class_id = int(line.strip().split()[0])
                    conteo[class_id] += 1
                    classes_in_file.append(class_id)
                except:
                    continue

            if classes_in_file:
                img_composition[img_file] = classes_in_file

    return conteo, img_composition


def rotate_bbox(x_min, y_min, x_max, y_max, image_width, image_height, angle):
    """
    Rota una bounding box y retorna las nuevas coordenadas.
    """
    corners = np.array([[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]])
    original_center_x, original_center_y = image_width / 2, image_height / 2

    temp_img = Image.new("RGB", (image_width, image_height))
    rotated_temp = temp_img.rotate(angle, expand=True)
    new_w, new_h = rotated_temp.size

    rad_angle = np.deg2rad(angle)
    cos_a, sin_a = np.cos(rad_angle), np.sin(rad_angle)

    new_corners = []
    for x, y in corners:
        x_t, y_t = x - original_center_x, y - original_center_y
        x_r = x_t * cos_a - y_t * sin_a
        y_r = x_t * sin_a + y_t * cos_a
        new_corners.append([x_r + new_w / 2, y_r + new_h / 2])

    new_corners = np.array(new_corners)
    return (
        np.min(new_corners[:, 0]),
        np.min(new_corners[:, 1]),
        np.max(new_corners[:, 0]),
        np.max(new_corners[:, 1]),
        new_w,
        new_h,
    )


def apply_random_augmentation(image, boxes, w, h):
    """
    Aplica aumentaciones aleatorias (flip, rotación, brillo) a imagen y boxes.
    boxes: lista de [class_id, x_center, y_center, width, height] normalizados
    """
    aug_img = image.copy()
    aug_boxes = [b[:] for b in boxes]
    curr_w, curr_h = w, h

    # Flip Horizontal
    if random.random() < 0.5:
        aug_img = aug_img.transpose(Image.FLIP_LEFT_RIGHT)
        for i in range(len(aug_boxes)):
            aug_boxes[i][1] = 1 - aug_boxes[i][1]

    # Rotación leve (-10 a 10 grados)
    angle = random.uniform(-10, 10)
    if abs(angle) > 1:
        temp_img = Image.new("RGB", (curr_w, curr_h))
        new_w, new_h = temp_img.rotate(angle, expand=True).size
        aug_img = aug_img.rotate(angle, expand=True)

        new_boxes = []
        for box in aug_boxes:
            cid, xc, yc, bw, bh = box
            x_px, y_px = xc * curr_w, yc * curr_h
            w_px, h_px = bw * curr_w, bh * curr_h
            x1, y1 = x_px - w_px / 2, y_px - h_px / 2
            x2, y2 = x_px + w_px / 2, y_px + h_px / 2

            nx1, ny1, nx2, ny2, _, _ = rotate_bbox(
                x1, y1, x2, y2, curr_w, curr_h, angle
            )

            n_xc = ((nx1 + nx2) / 2) / new_w
            n_yc = ((ny1 + ny2) / 2) / new_h
            n_bw = (nx2 - nx1) / new_w
            n_bh = (ny2 - ny1) / new_h

            # Validar límites
            if 0 < n_bw <= 1 and 0 < n_bh <= 1:
                new_boxes.append(
                    [cid, np.clip(n_xc, 0, 1), np.clip(n_yc, 0, 1), n_bw, n_bh]
                )

        aug_boxes = new_boxes
        curr_w, curr_h = new_w, new_h

    # Color/Brillo
    if random.random() < 0.5:
        aug_img = ImageEnhance.Brightness(aug_img).enhance(random.uniform(0.7, 1.3))

    return aug_img, aug_boxes


def plot_distribution(
    conteo, class_names_map, title, count_mayoritaria=None, output_path=None
):
    """Genera gráfico de distribución de clases."""
    if not PLOTTING_AVAILABLE:
        return

    df = pd.DataFrame(conteo.items(), columns=["Class_ID", "Count"])
    df["Class_Name"] = df["Class_ID"].map(class_names_map)
    df = df.sort_values("Class_ID")

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="Class_Name",
        y="Count",
        data=df,
        hue="Class_Name",
        palette="viridis",
        legend=False,
    )
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    if count_mayoritaria:
        plt.axhline(
            y=count_mayoritaria,
            color="r",
            linestyle="--",
            label="Clase Mayoritaria Original",
        )
        plt.legend()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Gráfico guardado en: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Aumentación de datos para balanceo de dataset YOLO"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directorio raíz del dataset (debe contener data.yaml y train/)",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=0.25,
        help="Límite de aumentación como fracción del total original (default: 0.25 = 25%%)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Semilla aleatoria para reproducibilidad"
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Guardar gráficos en disco en lugar de mostrarlos",
    )

    args = parser.parse_args()

    # Configurar seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Rutas
    PATH_BASE = Path(args.data_dir)
    DATA_YAML_PATH = PATH_BASE / "data.yaml"
    LABELS_PATH = PATH_BASE / "train" / "labels"
    IMAGES_PATH = PATH_BASE / "train" / "images"

    # Validar estructura
    if not DATA_YAML_PATH.exists():
        print(f"❌ Error: No se encontró {DATA_YAML_PATH}")
        sys.exit(1)

    if not LABELS_PATH.exists() or not IMAGES_PATH.exists():
        print(
            f"❌ Error: Estructura de dataset inválida. Se requiere train/images y train/labels"
        )
        sys.exit(1)

    print("=" * 60)
    print("ANÁLISIS Y AUMENTACIÓN DE DATOS")
    print("=" * 60)

    # Cargar data.yaml
    with open(DATA_YAML_PATH, "r") as f:
        data_yaml = yaml.safe_load(f)

    class_names = data_yaml["names"]
    class_names_map = {i: name for i, name in enumerate(class_names)}
    print(f"\nClases detectadas: {class_names_map}")

    # Análisis inicial
    print("\n[1/4] Analizando dataset inicial...")
    conteo_inicial, mapa_imagenes = analizar_dataset(str(LABELS_PATH), str(IMAGES_PATH))

    if not conteo_inicial:
        print(
            "❌ Error: No se encontraron etiquetas. Revisa la estructura del dataset."
        )
        sys.exit(1)

    # Identificar clase mayoritaria
    id_clase_mayoritaria = max(conteo_inicial, key=conteo_inicial.get)
    count_mayoritaria = conteo_inicial[id_clase_mayoritaria]

    print(f"\n--- Breve Análisis ---")
    print(
        f"Clase Mayoritaria: ID {id_clase_mayoritaria} ({class_names_map.get(id_clase_mayoritaria)}) -> {count_mayoritaria} instancias"
    )

    total_instancias_originales = sum(conteo_inicial.values())
    LIMITE_AUMENTACION = int(total_instancias_originales * args.limit)

    print(f"Total instancias originales: {total_instancias_originales}")
    print(
        f"Límite de aumentación ({args.limit * 100:.0f}%): Máximo {LIMITE_AUMENTACION} nuevas instancias permitidas"
    )

    # Gráfico inicial
    print("\n[2/4] Generando gráfico de distribución inicial...")
    if args.save_plots:
        plot_path = PATH_BASE / "distribution_initial.png"
        plot_distribution(
            conteo_inicial,
            class_names_map,
            "Distribución de Clases (Inicial)",
            count_mayoritaria,
            output_path=plot_path,
        )
    else:
        plot_distribution(
            conteo_inicial,
            class_names_map,
            "Distribución de Clases (Inicial)",
            count_mayoritaria,
        )

    # Proceso de aumentación
    print("\n[3/4] Iniciando proceso de balanceo...")
    instancias_generadas_total = 0
    conteo_actual = conteo_inicial.copy()

    clases_a_aumentar = [c for c in conteo_actual if c != id_clase_mayoritaria]

    for id_clase in clases_a_aumentar:
        if instancias_generadas_total >= LIMITE_AUMENTACION:
            print("⚠️  Se alcanzó el límite global de seguridad. Deteniendo.")
            break

        objetivo = count_mayoritaria
        actual = conteo_actual[id_clase]
        faltan = objetivo - actual

        if faltan <= 0:
            continue

        print(
            f"\nProcesando Clase {class_names_map[id_clase]} (Faltan aprox: {faltan})"
        )

        candidatos = [
            img for img, clases in mapa_imagenes.items() if id_clase in clases
        ]
        candidatos.sort(
            key=lambda x: (
                id_clase_mayoritaria in mapa_imagenes[x],
                len(mapa_imagenes[x]),
            )
        )

        if not candidatos:
            print(f"  ⚠️  No hay imágenes base para la clase {id_clase}. Saltando.")
            continue

        idx_candidato = 0
        while conteo_actual[id_clase] < objetivo:
            if instancias_generadas_total >= LIMITE_AUMENTACION:
                print("  ⚠️  Límite alcanzado durante el proceso.")
                break

            img_filename = candidatos[idx_candidato % len(candidatos)]
            idx_candidato += 1

            src_img_path = IMAGES_PATH / img_filename
            src_lbl_path = LABELS_PATH / (Path(img_filename).stem + ".txt")

            try:
                # Leer imagen y etiquetas
                img = Image.open(src_img_path).convert("RGB")
                w, h = img.size
                boxes = []
                with open(src_lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            boxes.append(
                                [int(parts[0])] + [float(p) for p in parts[1:]]
                            )

                # Aplicar aumentación
                aug_img, aug_boxes = apply_random_augmentation(img, boxes, w, h)

                if not aug_boxes:
                    continue

                # Guardar imagen y etiquetas aumentadas
                suffix = random.randint(10000, 99999)
                new_base = f"{Path(img_filename).stem}_aug_{suffix}"

                aug_img.save(IMAGES_PATH / (new_base + ".jpg"))

                with open(LABELS_PATH / (new_base + ".txt"), "w") as f:
                    for box in aug_boxes:
                        f.write(
                            f"{int(box[0])} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n"
                        )

                # Actualizar conteos
                instancias_agregadas = 0
                for box in aug_boxes:
                    cid = int(box[0])
                    conteo_actual[cid] += 1
                    instancias_agregadas += 1

                instancias_generadas_total += instancias_agregadas

                if idx_candidato % 50 == 0:
                    print(
                        f"  → Actual: {conteo_actual[id_clase]} / Meta: {objetivo} (Global: +{instancias_generadas_total})"
                    )

            except Exception as e:
                print(f"  ⚠️  Error procesando {img_filename}: {e}")
                continue

    print(f"\n=== FINALIZADO ===")
    print(f"Total instancias agregadas: {instancias_generadas_total}")
    print("Conteo final:", dict(conteo_actual))

    # Gráfico final
    print("\n[4/4] Generando gráfico de distribución final...")
    if args.save_plots:
        plot_path = PATH_BASE / "distribution_final.png"
        plot_distribution(
            conteo_actual,
            class_names_map,
            "Distribución de Clases Post-Aumentación",
            count_mayoritaria,
            output_path=plot_path,
        )
    else:
        plot_distribution(
            conteo_actual,
            class_names_map,
            "Distribución de Clases Post-Aumentación",
            count_mayoritaria,
        )

    print("\n" + "=" * 60)
    print("✓ PROCESO COMPLETADO")
    print("=" * 60)
    print(f"\nDataset aumentado guardado en: {PATH_BASE / 'train'}")
    print("\nPróximo paso:")
    print(f"  python scripts/train.py --data-dir {PATH_BASE}")


if __name__ == "__main__":
    main()
