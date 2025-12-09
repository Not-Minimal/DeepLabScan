#!/usr/bin/env python3
"""
Script de Entrenamiento YOLO - Adaptado de tarea_3.py

Este script implementa el flujo de entrenamiento de la clase:
1. Detecta automáticamente GPU/CPU disponible
2. Carga modelo YOLO preentrenado
3. Entrena con el dataset aumentado
4. Guarda resultados y pesos del modelo

Uso:
    python scripts/train.py --data-dir data/raw --epochs 15
    python scripts/train.py --data-dir data/raw --model yolo11n.pt --epochs 50 --imgsz 640
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import torch
    from ultralytics import YOLO
except ImportError:
    print("❌ Error: Se requieren ultralytics y torch.")
    print("Instala con: pip install ultralytics torch torchvision")
    sys.exit(1)


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

        print("\n" + "=" * 60)
        print("✓ ENTRENAMIENTO COMPLETADO")
        print("=" * 60)

        # Construir ruta de salida
        output_dir = Path(args.project) / args.name
        weights_path = output_dir / "weights" / "best.pt"

        print(f"\n📁 Resultados guardados en: {output_dir}")
        if weights_path.exists():
            print(f"🏆 Mejor modelo: {weights_path}")
        else:
            # Buscar el directorio train más reciente
            train_dirs = sorted(
                Path(args.project).glob("train*"),
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            if train_dirs:
                latest_weights = train_dirs[0] / "weights" / "best.pt"
                if latest_weights.exists():
                    print(f"🏆 Mejor modelo: {latest_weights}")

        print("\n📊 Próximos pasos:")
        print(f"  1. Evaluar modelo:")
        print(f"     python scripts/evaluate.py --weights {weights_path}")
        print(f"\n  2. Hacer predicciones:")
        print(
            f"     python scripts/predict.py --weights {weights_path} --source <imagen>"
        )
        print(f"\n  3. Ver métricas en: {output_dir}")

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
