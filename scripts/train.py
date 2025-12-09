#!/usr/bin/env python3
"""
Script de Entrenamiento YOLO - Adaptado de tarea_3.py

Este script implementa el flujo de entrenamiento de la clase:
1. Detecta automáticamente GPU/CPU disponible
2. Carga modelo YOLO preentrenado
3. Entrena con el dataset aumentado
4. Guarda resultados y pesos del modelo
5. Registra resultados en Excel para comparación

Uso:
    python scripts/train.py --data-dir data/raw --epochs 15
    python scripts/train.py --data-dir data/raw --model yolo11n.pt --epochs 50 --imgsz 640
"""

import argparse
import os
import sys
import time
from pathlib import Path

try:
    import torch
    from ultralytics import YOLO
except ImportError:
    print("❌ Error: Se requieren ultralytics y torch.")
    print("Instala con: pip install ultralytics torch torchvision")
    sys.exit(1)

# Importar Excel Logger
try:
    from excel_logger import get_logger
except ImportError:
    print("⚠️  Excel logger no disponible. Los resultados no se guardarán en Excel.")
    get_logger = None


def check_device():
    """
    Detecta y retorna el dispositivo disponible (GPU/CPU).
    """
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ GPU disponible. Usando: {gpu_name}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("✓ Apple Silicon GPU (MPS) disponible. Usando: MPS")
    else:
        device = "cpu"
        print("⚠️  No hay GPU disponible. Usando: CPU")

    return device


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de modelo YOLO")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directorio del dataset (debe contener data.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Modelo YOLO a usar (default: yolo11n.pt). Opciones: yolov8n.pt, yolo11n.pt, etc.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Número de épocas de entrenamiento (default: 15)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Tamaño de las imágenes durante entrenamiento (default: 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Tamaño del batch. -1 para auto-batch (default: -1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo a usar (cpu, cuda, mps, 0, 1, etc). Auto-detect si no se especifica",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Directorio donde guardar resultados (default: runs/detect)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="train",
        help="Nombre del experimento (default: train)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Épocas de paciencia para early stopping (default: 50)",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Usar pesos preentrenados (recomendado)",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Notas adicionales sobre el experimento",
    )
    parser.add_argument(
        "--no-excel",
        action="store_true",
        help="No guardar resultados en Excel",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ENTRENAMIENTO DE MODELO YOLO")
    print("=" * 60)

    # Validar que existe data.yaml
    data_dir = Path(args.data_dir)
    data_yaml_path = data_dir / "data.yaml"

    if not data_yaml_path.exists():
        print(f"\n❌ Error: No se encontró {data_yaml_path}")
        print("\nAsegúrate de:")
        print("  1. Haber descargado el dataset")
        print("  2. Haber ejecutado aumentación de datos (si aplica)")
        print(f"  3. Que exista el archivo data.yaml en {data_dir}")
        sys.exit(1)

    print(f"\n[1/4] Configuración")
    print(f"  - Dataset: {data_yaml_path}")
    print(f"  - Modelo: {args.model}")
    print(f"  - Épocas: {args.epochs}")
    print(f"  - Tamaño imagen: {args.imgsz}")
    print(f"  - Batch: {'auto' if args.batch == -1 else args.batch}")

    # Detectar dispositivo
    print(f"\n[2/4] Detectando dispositivo...")
    if args.device:
        device = args.device
        print(f"  - Usando dispositivo especificado: {device}")
    else:
        device = check_device()

    # Cargar modelo
    print(f"\n[3/4] Cargando modelo YOLO...")
    try:
        # Si el modelo existe localmente, usarlo; si no, YOLO lo descargará
        if Path(args.model).exists():
            print(f"  - Cargando modelo local: {args.model}")
            model = YOLO(args.model)
        else:
            print(f"  - Descargando modelo preentrenado: {args.model}")
            model = YOLO(args.model)
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        print("\nModelos disponibles comunes:")
        print("  - yolov8n.pt (nano, más rápido)")
        print("  - yolov8s.pt (small)")
        print("  - yolov8m.pt (medium)")
        print("  - yolo11n.pt (YOLO11 nano)")
        sys.exit(1)

    # Entrenar
    print(f"\n[4/4] Iniciando entrenamiento...")
    print("=" * 60)

    start_time = time.time()

    try:
        train_results = model.train(
            data=str(data_yaml_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            project=args.project,
            name=args.name,
            patience=args.patience,
            pretrained=args.pretrained,
            verbose=True,
        )

        end_time = time.time()
        duration_minutes = (end_time - start_time) / 60

        print("\n" + "=" * 60)
        print("✓ ENTRENAMIENTO COMPLETADO")
        print("=" * 60)

        # Construir ruta de salida
        output_dir = Path(args.project) / args.name
        weights_path = output_dir / "weights" / "best.pt"

        # Buscar el directorio train más reciente si no existe
        if not weights_path.exists():
            train_dirs = sorted(
                Path(args.project).glob("train*"),
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            if train_dirs:
                output_dir = train_dirs[0]
                weights_path = output_dir / "weights" / "best.pt"

        print(f"\n📁 Resultados guardados en: {output_dir}")
        if weights_path.exists():
            print(f"🏆 Mejor modelo: {weights_path}")

        # Extraer métricas de los resultados
        try:
            # Leer resultados del CSV generado por YOLO
            results_csv = output_dir / "results.csv"
            if results_csv.exists():
                import pandas as pd

                df_results = pd.read_csv(results_csv)
                # Obtener última fila (mejores resultados)
                last_row = df_results.iloc[-1]

                # Extraer métricas (las columnas pueden variar según versión de YOLO)
                best_map50 = last_row.get("metrics/mAP50(B)", 0)
                best_map50_95 = last_row.get("metrics/mAP50-95(B)", 0)
                best_precision = last_row.get("metrics/precision(B)", 0)
                best_recall = last_row.get("metrics/recall(B)", 0)
                final_loss = last_row.get("train/box_loss", 0)

                # Si no están disponibles, buscar nombres alternativos
                if best_map50 == 0:
                    for col in df_results.columns:
                        if "map50" in col.lower() and "95" not in col.lower():
                            best_map50 = last_row[col]
                            break

                if best_map50_95 == 0:
                    for col in df_results.columns:
                        if "map" in col.lower() and "95" in col.lower():
                            best_map50_95 = last_row[col]
                            break

                if best_precision == 0:
                    for col in df_results.columns:
                        if "precision" in col.lower():
                            best_precision = last_row[col]
                            break

                if best_recall == 0:
                    for col in df_results.columns:
                        if "recall" in col.lower():
                            best_recall = last_row[col]
                            break

                print("\n📊 Métricas finales:")
                print(f"  - mAP@0.5: {best_map50:.4f}")
                print(f"  - mAP@0.5:0.95: {best_map50_95:.4f}")
                print(f"  - Precision: {best_precision:.4f}")
                print(f"  - Recall: {best_recall:.4f}")
                print(f"  - Duración: {duration_minutes:.2f} minutos")

                # Guardar en Excel
                if not args.no_excel and get_logger is not None:
                    try:
                        logger = get_logger()
                        logger.log_training(
                            experiment_name=args.name,
                            model=args.model,
                            dataset=str(data_yaml_path),
                            epochs=args.epochs,
                            batch=args.batch if args.batch != -1 else "auto",
                            imgsz=args.imgsz,
                            device=str(device),
                            duration_minutes=duration_minutes,
                            best_map50=float(best_map50),
                            best_map50_95=float(best_map50_95),
                            best_precision=float(best_precision),
                            best_recall=float(best_recall),
                            final_loss=float(final_loss),
                            weights_path=str(weights_path),
                            notes=args.notes,
                        )
                    except Exception as e:
                        print(f"\n⚠️  No se pudieron guardar resultados en Excel: {e}")

            else:
                print("\n⚠️  No se encontró archivo de resultados CSV")

        except Exception as e:
            print(f"\n⚠️  No se pudieron extraer métricas: {e}")

        print("\n📊 Próximos pasos:")
        print(f"  1. Evaluar modelo:")
        print(f"     python scripts/evaluate.py --weights {weights_path}")
        print(f"\n  2. Hacer predicciones:")
        print(
            f"     python scripts/predict.py --weights {weights_path} --source <imagen>"
        )
        print(f"\n  3. Ver métricas en: {output_dir}")
        print(f"\n  4. Ver comparación en Excel: results/experiment_results.xlsx")

    except KeyboardInterrupt:
        print("\n\n⚠️  Entrenamiento interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error durante entrenamiento: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
