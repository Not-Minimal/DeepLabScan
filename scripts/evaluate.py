#!/usr/bin/env python3
"""
Script de Evaluación YOLO - Adaptado de tarea_3.py

Este script evalúa un modelo YOLO entrenado y muestra/guarda visualizaciones:
1. Carga el modelo entrenado
2. Ejecuta validación con model.val()
3. Muestra métricas (Precision, Recall, mAP, F1)
4. Visualiza gráficos generados (confusion matrix, curves, etc.)

Uso:
    python scripts/evaluate.py --weights runs/detect/train/weights/best.pt
    python scripts/evaluate.py --weights runs/detect/train/weights/best.pt --data-dir data/raw
    python scripts/evaluate.py --weights best.pt --device cuda --save-plots
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from ultralytics import YOLO
except ImportError:
    print("❌ Error: Se requieren ultralytics y matplotlib")
    print("Instala con: pip install ultralytics matplotlib")
    sys.exit(1)


def find_validation_results(weights_path):
    """
    Encuentra el directorio de resultados de validación más reciente.
    Busca en runs/detect/val* o en la estructura relativa a los pesos.
    """
    weights_path = Path(weights_path)

    # Intentar encontrar directorio de validación relativo a los pesos
    # Estructura típica: runs/detect/train/weights/best.pt
    # Validación: runs/detect/val o runs/detect/val2, etc.

    if weights_path.parent.name == "weights":
        # Subir dos niveles: weights -> train -> detect
        detect_dir = weights_path.parent.parent.parent
        if detect_dir.exists():
            # Buscar directorios val* ordenados por fecha
            val_dirs = sorted(
                detect_dir.glob("val*"), key=lambda x: x.stat().st_mtime, reverse=True
            )
            if val_dirs:
                return val_dirs[0]

    # Fallback: buscar en runs/detect/val*
    runs_detect = Path("runs/detect")
    if runs_detect.exists():
        val_dirs = sorted(
            runs_detect.glob("val*"), key=lambda x: x.stat().st_mtime, reverse=True
        )
        if val_dirs:
            return val_dirs[0]

    return None


def display_or_save_plot(image_path, title, save_dir=None, show=True):
    """
    Muestra o guarda un gráfico desde un archivo de imagen.
    """
    if not os.path.exists(image_path):
        print(f"  ⚠️  No se encontró: {image_path}")
        return False

    try:
        img = mpimg.imread(image_path)

        if show:
            # Mostrar en pantalla
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")
            plt.tight_layout()
            plt.show()

        if save_dir:
            # Guardar copia en directorio de salida
            save_path = Path(save_dir) / Path(image_path).name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.title(title)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  ✓ Guardado: {save_path}")

        return True
    except Exception as e:
        print(f"  ⚠️  Error al procesar {image_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Evaluación de modelo YOLO")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Ruta a los pesos del modelo (.pt)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directorio del dataset (contiene data.yaml). Si no se especifica, se buscará automáticamente",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo (cpu, cuda, mps, 0, 1). Auto-detect si no se especifica",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Guardar gráficos en results/evaluation/",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="No mostrar gráficos en pantalla (útil en servidores sin display)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Split a evaluar (default: val)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("EVALUACIÓN DE MODELO YOLO")
    print("=" * 60)

    # Verificar que existen los pesos
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"\n❌ Error: No se encontraron los pesos: {args.weights}")
        print("\nAsegúrate de haber entrenado el modelo primero:")
        print("  python scripts/train.py --data-dir data/raw --epochs 15")
        sys.exit(1)

    print(f"\n[1/4] Cargando modelo...")
    print(f"  - Pesos: {weights_path}")

    try:
        model = YOLO(str(weights_path))
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        sys.exit(1)

    # Buscar data.yaml
    data_yaml = None
    if args.data_dir:
        data_yaml = Path(args.data_dir) / "data.yaml"
        if not data_yaml.exists():
            print(f"❌ Error: No se encontró {data_yaml}")
            sys.exit(1)
    else:
        # Intentar encontrar data.yaml automáticamente
        possible_paths = [
            Path("data/raw/data.yaml"),
            Path("data.yaml"),
            weights_path.parent.parent.parent / "data.yaml",
        ]
        for path in possible_paths:
            if path.exists():
                data_yaml = path
                break

        if not data_yaml:
            print("❌ Error: No se encontró data.yaml")
            print("\nEspecifica el directorio del dataset:")
            print("  python scripts/evaluate.py --weights best.pt --data-dir data/raw")
            sys.exit(1)

    print(f"  - Dataset: {data_yaml}")

    # Detectar dispositivo si no se especificó
    if args.device:
        device = args.device
    else:
        import torch

        if torch.cuda.is_available():
            device = 0
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"  - Dispositivo: {device}")
    print(f"  - Split: {args.split}")

    # Evaluar modelo
    print(f"\n[2/4] Ejecutando validación...")
    print("-" * 60)

    try:
        metrics = model.val(
            data=str(data_yaml),
            split=args.split,
            device=device,
            plots=True,
            save_json=True,
            verbose=True,
        )

        print("-" * 60)
    except Exception as e:
        print(f"❌ Error durante la evaluación: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Mostrar métricas principales
    print(f"\n[3/4] Resumen de Métricas")
    print("=" * 60)

    # Extraer métricas del objeto results
    try:
        # Las métricas están en metrics.results_dict o metrics.box
        if hasattr(metrics, "results_dict"):
            results = metrics.results_dict
        elif hasattr(metrics, "box"):
            results = {
                "metrics/precision(B)": metrics.box.p
                if hasattr(metrics.box, "p")
                else 0,
                "metrics/recall(B)": metrics.box.r if hasattr(metrics.box, "r") else 0,
                "metrics/mAP50(B)": metrics.box.map50
                if hasattr(metrics.box, "map50")
                else 0,
                "metrics/mAP50-95(B)": metrics.box.map
                if hasattr(metrics.box, "map")
                else 0,
            }
        else:
            results = {}

        # Intentar extraer valores
        precision = results.get("metrics/precision(B)", 0)
        recall = results.get("metrics/recall(B)", 0)
        map50 = results.get("metrics/mAP50(B)", 0)
        map50_95 = results.get("metrics/mAP50-95(B)", 0)

        print(f"Precision:      {precision:.4f}")
        print(f"Recall:         {recall:.4f}")
        print(f"mAP@0.5:        {map50:.4f}")
        print(f"mAP@0.5:0.95:   {map50_95:.4f}")

        # Calcular F1-Score
        if precision > 0 or recall > 0:
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )
            print(f"F1-Score:       {f1:.4f}")

        print("=" * 60)

        # Interpretación
        print("\n📊 Interpretación:")
        if map50 >= 0.9:
            print("  ✓ mAP@0.5 Excelente (≥0.9): Detección muy precisa")
        elif map50 >= 0.7:
            print("  ✓ mAP@0.5 Buena (0.7-0.9): Detección confiable")
        elif map50 >= 0.5:
            print(
                "  ⚠️  mAP@0.5 Aceptable (0.5-0.7): Puede mejorar con más entrenamiento"
            )
        else:
            print(
                "  ❌ mAP@0.5 Baja (<0.5): Se recomienda más entrenamiento o revisar datos"
            )

        if precision < 0.5:
            print(
                "  ⚠️  Precision baja: Muchos falsos positivos (aumenta confidence threshold)"
            )
        if recall < 0.5:
            print(
                "  ⚠️  Recall bajo: Muchos falsos negativos (disminuye confidence threshold)"
            )

    except Exception as e:
        print(f"⚠️  No se pudieron extraer todas las métricas: {e}")

    # Visualizar resultados
    print(f"\n[4/4] Visualizaciones")
    print("=" * 60)

    # Buscar directorio de validación
    val_dir = find_validation_results(weights_path)

    if not val_dir:
        print("⚠️  No se encontró directorio de validación con resultados")
        print("   Los gráficos pueden estar en runs/detect/val*/")
    else:
        print(f"Directorio de resultados: {val_dir}")

        save_dir = Path("results/evaluation") if args.save_plots else None
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"Guardando gráficos en: {save_dir}")

        show_plots = not args.no_show

        # Lista de gráficos comunes generados por YOLO
        plots = [
            ("confusion_matrix.png", "Matriz de Confusión"),
            ("confusion_matrix_normalized.png", "Matriz de Confusión Normalizada"),
            ("F1_curve.png", "Curva F1-Score"),
            ("P_curve.png", "Curva de Precision"),
            ("R_curve.png", "Curva de Recall"),
            ("PR_curve.png", "Curva Precision-Recall"),
        ]

        found_plots = False
        for plot_file, title in plots:
            plot_path = val_dir / plot_file
            if display_or_save_plot(
                plot_path, title, save_dir=save_dir, show=show_plots
            ):
                found_plots = True

        # Buscar visualizaciones de predicciones si existen
        viz_dir = val_dir / "visualizations"
        if viz_dir.exists():
            viz_images = list(viz_dir.glob("*.jpg")) + list(viz_dir.glob("*.png"))
            if viz_images:
                print(
                    f"\n📸 Visualizaciones de predicciones: {len(viz_images)} imágenes"
                )
                print(f"   Ubicación: {viz_dir}")

                # Mostrar algunas ejemplos si se pidió
                if show_plots and len(viz_images) > 0:
                    print("   Mostrando primeras 3 predicciones...")
                    for img_path in viz_images[:3]:
                        display_or_save_plot(
                            img_path,
                            f"Predicción: {img_path.name}",
                            save_dir=save_dir,
                            show=True,
                        )

        if not found_plots:
            print("⚠️  No se encontraron gráficos de validación")
            print(f"   Verifica manualmente en: {val_dir}")

    print("\n" + "=" * 60)
    print("✓ EVALUACIÓN COMPLETADA")
    print("=" * 60)

    if val_dir:
        print(f"\n📁 Resultados completos en: {val_dir}")
    if args.save_plots:
        print(f"📁 Gráficos guardados en: results/evaluation/")

    print("\n📖 Próximos pasos:")
    print(
        f"  - Ver todos los gráficos en: {val_dir if val_dir else 'runs/detect/val*/'}"
    )
    print(
        f"  - Hacer predicciones: python scripts/predict.py --weights {args.weights} --source <imagen>"
    )


if __name__ == "__main__":
    main()
