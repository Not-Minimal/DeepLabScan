#!/usr/bin/env python3
"""
Script de Predicción YOLO - Adaptado de tarea_3.py

Este script realiza predicciones con un modelo YOLO entrenado:
1. Carga el modelo entrenado
2. Realiza predicciones sobre imágenes/videos/webcam
3. Muestra y guarda resultados con bounding boxes
4. Imprime información de detecciones

Uso:
    python scripts/predict.py --weights runs/detect/train/weights/best.pt --source imagen.jpg
    python scripts/predict.py --weights best.pt --source imagenes/
    python scripts/predict.py --weights best.pt --source video.mp4
    python scripts/predict.py --weights best.pt --source 0  # webcam
"""

import argparse
import sys
from pathlib import Path

try:
    import cv2 as cv
    import matplotlib.pyplot as plt
    from ultralytics import YOLO
except ImportError:
    print("❌ Error: Se requieren ultralytics, opencv-python y matplotlib")
    print("Instala con: pip install ultralytics opencv-python matplotlib")
    sys.exit(1)


def display_prediction(result, show=True, save_path=None):
    """
    Muestra o guarda una predicción visualizada.
    Similar al flujo de tarea_3.py donde se usa cv.imshow o plt.imshow
    """
    # result.plot() devuelve imagen con cajas dibujadas (BGR)
    im_array = result.plot()

    if show:
        # Convertir BGR a RGB para matplotlib
        im_rgb = cv.cvtColor(im_array, cv.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 8))
        plt.imshow(im_rgb)
        plt.axis("off")
        plt.title(f"Detecciones: {Path(result.path).name}")
        plt.tight_layout()
        plt.show()

    if save_path:
        # Guardar en formato OpenCV (BGR)
        cv.imwrite(str(save_path), im_array)
        return save_path

    return None


def main():
    parser = argparse.ArgumentParser(description="Predicción con modelo YOLO")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Ruta a los pesos del modelo (.pt)",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Fuente: ruta a imagen, directorio, video, URL o 0 para webcam",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold para NMS (default: 0.7)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Tamaño de imagen para inferencia (default: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo (cpu, cuda, mps, 0, 1). Auto-detect si no se especifica",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Guardar resultados (default: True)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/predict",
        help="Directorio de salida (default: runs/predict)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="exp",
        help="Nombre del experimento (default: exp)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Mostrar predicciones en ventanas interactivas",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Guardar resultados en archivos .txt (formato YOLO)",
    )
    parser.add_argument(
        "--save-conf",
        action="store_true",
        help="Guardar confidence en archivos .txt",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Máximo de detecciones por imagen (default: 300)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("PREDICCIÓN CON MODELO YOLO")
    print("=" * 60)

    # Verificar pesos
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"\n❌ Error: No se encontraron los pesos: {args.weights}")
        print("\nAsegúrate de haber entrenado el modelo primero:")
        print("  python scripts/train.py --data-dir data/raw --epochs 15")
        sys.exit(1)

    print(f"\n[1/3] Cargando modelo...")
    print(f"  - Pesos: {weights_path}")

    try:
        model = YOLO(str(weights_path))
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        sys.exit(1)

    # Verificar fuente
    source = args.source
    if source != "0" and not source.startswith("http"):
        source_path = Path(source)
        if not source_path.exists():
            print(f"\n❌ Error: No se encontró la fuente: {source}")
            sys.exit(1)

        # Determinar tipo de fuente
        if source_path.is_dir():
            print(f"  - Fuente: Directorio ({source})")
        elif source_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
            print(f"  - Fuente: Video ({source})")
        else:
            print(f"  - Fuente: Imagen ({source})")
    elif source == "0":
        print(f"  - Fuente: Webcam")
    else:
        print(f"  - Fuente: URL ({source})")

    # Detectar dispositivo
    if args.device:
        device = args.device
    else:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"  - Dispositivo: {device}")
    print(f"  - Confidence: {args.conf}")
    print(f"  - IoU: {args.iou}")

    # Realizar predicciones
    print(f"\n[2/3] Realizando predicciones...")
    print("-" * 60)

    try:
        results = model.predict(
            source=source,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=device,
            max_det=args.max_det,
            save=args.save,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            project=args.project,
            name=args.name,
            verbose=True,
        )

        print("-" * 60)
    except Exception as e:
        print(f"❌ Error durante la predicción: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Procesar y mostrar resultados
    print(f"\n[3/3] Resumen de Resultados")
    print("=" * 60)

    total_detections = 0
    total_images = 0
    class_counts = {}

    for result in results:
        total_images += 1

        if hasattr(result, "boxes") and len(result.boxes) > 0:
            boxes = result.boxes
            n_detections = len(boxes)
            total_detections += n_detections

            print(f"\n📄 {Path(result.path).name}")
            print(f"   Detecciones: {n_detections}")

            # Contar por clase
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                conf = float(box.conf[0])

                # Coordenadas de la caja
                xyxy = box.xyxy[0].cpu().numpy()

                if cls_name not in class_counts:
                    class_counts[cls_name] = 0
                class_counts[cls_name] += 1

                print(
                    f"   - {cls_name}: confianza={conf:.3f}, bbox=[{xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f}]"
                )

            # Mostrar predicción si se solicitó
            if args.show:
                display_prediction(result, show=True)

    print("\n" + "=" * 60)
    print("📊 RESUMEN GENERAL")
    print("=" * 60)
    print(f"Total de imágenes procesadas: {total_images}")
    print(f"Total de detecciones: {total_detections}")

    if class_counts:
        print("\nDetecciones por clase:")
        for cls_name, count in sorted(class_counts.items()):
            print(f"  - {cls_name}: {count}")
    else:
        print("\n⚠️  No se realizaron detecciones (prueba reducir --conf)")

    # Información de salida
    if args.save:
        output_dir = Path(args.project) / args.name
        print(f"\n📁 Resultados guardados en: {output_dir}")

        # Listar algunos archivos generados
        if output_dir.exists():
            saved_images = list(output_dir.glob("*.jpg")) + list(
                output_dir.glob("*.png")
            )
            if saved_images:
                print(f"   ✓ Imágenes con predicciones: {len(saved_images)}")

            if args.save_txt:
                saved_labels = list(output_dir.glob("labels/*.txt"))
                if saved_labels:
                    print(f"   ✓ Archivos de labels: {len(saved_labels)}")

    print("\n" + "=" * 60)
    print("✓ PREDICCIÓN COMPLETADA")
    print("=" * 60)

    # Consejos
    if total_detections == 0:
        print("\n💡 Consejos:")
        print("  - No se detectó nada. Prueba reducir --conf (ej: --conf 0.1)")
        print("  - Verifica que el modelo fue entrenado con clases similares")
    elif total_detections > total_images * 10:
        print("\n💡 Consejos:")
        print("  - Muchas detecciones. Si hay falsos positivos, aumenta --conf")
    else:
        print("\n💡 Todo parece correcto!")
        print("  - Para ajustar sensibilidad: modifica --conf")
        print("  - Para filtrar detecciones superpuestas: ajusta --iou")


if __name__ == "__main__":
    main()
