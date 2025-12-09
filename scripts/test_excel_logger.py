#!/usr/bin/env python3
"""
Script de Prueba para Excel Logger

Este script demuestra el funcionamiento del sistema de logging en Excel
generando datos de ejemplo para entrenamiento, evaluación y predicción.

Uso:
    python scripts/test_excel_logger.py
    python scripts/test_excel_logger.py --excel-path results/test_results.xlsx
"""

import argparse
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

try:
    from excel_logger import ExcelLogger
except ImportError:
    print("❌ Error: No se pudo importar excel_logger")
    print("Asegúrate de estar en el directorio correcto")
    sys.exit(1)


def generate_random_metrics():
    """Genera métricas aleatorias pero realistas"""
    # Métricas base realistas
    base_map50 = random.uniform(0.6, 0.95)
    base_precision = random.uniform(0.65, 0.92)
    base_recall = random.uniform(0.60, 0.88)

    # mAP@0.5:0.95 es típicamente menor que mAP@0.5
    map50_95 = base_map50 * random.uniform(0.65, 0.80)

    return {
        "map50": round(base_map50, 4),
        "map50_95": round(map50_95, 4),
        "precision": round(base_precision, 4),
        "recall": round(base_recall, 4),
    }


def test_training_logs(logger, num_experiments=5):
    """Prueba el logging de entrenamientos"""
    print("\n" + "=" * 80)
    print("PROBANDO LOGGING DE ENTRENAMIENTOS")
    print("=" * 80)

    models = ["yolo11n.pt", "yolo11s.pt", "yolov8n.pt", "yolov8s.pt"]
    devices = ["cuda", "mps", "cpu"]

    for i in range(num_experiments):
        metrics = generate_random_metrics()

        model = random.choice(models)
        device = random.choice(devices)
        epochs = random.choice([15, 30, 50, 100])
        batch = random.choice([8, 16, 32, -1])
        imgsz = random.choice([416, 640, 800])
        duration = random.uniform(15.0, 120.0)

        exp_name = f"train_exp_{i + 1}_{model.split('.')[0]}"

        print(f"\n[{i + 1}/{num_experiments}] Registrando: {exp_name}")
        print(f"  - Modelo: {model}")
        print(f"  - Épocas: {epochs}")
        print(f"  - mAP@0.5: {metrics['map50']:.4f}")

        success = logger.log_training(
            experiment_name=exp_name,
            model=model,
            dataset="data/raw/data.yaml",
            epochs=epochs,
            batch=batch if batch != -1 else "auto",
            imgsz=imgsz,
            device=device,
            duration_minutes=duration,
            best_map50=metrics["map50"],
            best_map50_95=metrics["map50_95"],
            best_precision=metrics["precision"],
            best_recall=metrics["recall"],
            final_loss=random.uniform(0.05, 0.25),
            weights_path=f"runs/detect/{exp_name}/weights/best.pt",
            notes=f"Experimento de prueba #{i + 1} - {epochs} épocas con {model}",
        )

        if success:
            print(f"  ✓ Guardado exitosamente")
        else:
            print(f"  ✗ Error al guardar")

    print("\n✓ Prueba de entrenamientos completada")


def test_evaluation_logs(logger, num_experiments=5):
    """Prueba el logging de evaluaciones"""
    print("\n" + "=" * 80)
    print("PROBANDO LOGGING DE EVALUACIONES")
    print("=" * 80)

    splits = ["val", "test"]
    devices = ["cuda", "mps", "cpu"]

    for i in range(num_experiments):
        metrics = generate_random_metrics()

        split = random.choice(splits)
        device = random.choice(devices)
        classes = random.randint(2, 10)

        exp_name = f"eval_exp_{i + 1}_{split}"

        print(f"\n[{i + 1}/{num_experiments}] Registrando: {exp_name}")
        print(f"  - Split: {split}")
        print(f"  - mAP@0.5: {metrics['map50']:.4f}")
        print(
            f"  - F1-Score: {2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']):.4f}"
        )

        success = logger.log_evaluation(
            experiment_name=exp_name,
            weights_path=f"runs/detect/train_{i}/weights/best.pt",
            dataset="data/raw/data.yaml",
            split=split,
            device=device,
            precision=metrics["precision"],
            recall=metrics["recall"],
            map50=metrics["map50"],
            map50_95=metrics["map50_95"],
            classes_detected=classes,
            visualizations_count=random.randint(5, 20),
            notes=f"Evaluación en {split} set - {classes} clases detectadas",
        )

        if success:
            print(f"  ✓ Guardado exitosamente")
        else:
            print(f"  ✗ Error al guardar")

    print("\n✓ Prueba de evaluaciones completada")


def test_prediction_logs(logger, num_experiments=5):
    """Prueba el logging de predicciones"""
    print("\n" + "=" * 80)
    print("PROBANDO LOGGING DE PREDICCIONES")
    print("=" * 80)

    sources = ["test_image.jpg", "video.mp4", "test_dir/", "0"]
    devices = ["cuda", "mps", "cpu"]

    class_names = ["clase_A", "clase_B", "clase_C", "clase_D"]

    for i in range(num_experiments):
        source = random.choice(sources)
        device = random.choice(devices)
        conf = random.uniform(0.15, 0.45)
        iou = random.uniform(0.4, 0.7)

        total_images = random.randint(1, 50)
        total_detections = random.randint(0, total_images * 10)

        # Generar detecciones por clase
        num_classes = random.randint(1, len(class_names))
        class_counts = {}
        remaining = total_detections

        for j in range(num_classes):
            if j == num_classes - 1:
                count = remaining
            else:
                count = random.randint(0, remaining)
                remaining -= count

            if count > 0:
                class_counts[class_names[j]] = count

        exp_name = f"predict_exp_{i + 1}"

        print(f"\n[{i + 1}/{num_experiments}] Registrando: {exp_name}")
        print(f"  - Source: {source}")
        print(f"  - Total imágenes: {total_images}")
        print(f"  - Total detecciones: {total_detections}")
        print(f"  - Confidence: {conf:.2f}")

        success = logger.log_prediction(
            experiment_name=exp_name,
            weights_path=f"runs/detect/train_{i}/weights/best.pt",
            source=source,
            confidence=conf,
            iou=iou,
            device=device,
            total_images=total_images,
            total_detections=total_detections,
            class_counts=class_counts,
            output_dir=f"runs/predict/{exp_name}",
            notes=f"Predicción en {source} - {total_detections} detecciones totales",
        )

        if success:
            print(f"  ✓ Guardado exitosamente")
        else:
            print(f"  ✗ Error al guardar")

    print("\n✓ Prueba de predicciones completada")


def test_summary_and_analysis(logger):
    """Prueba las funciones de análisis"""
    print("\n" + "=" * 80)
    print("PROBANDO FUNCIONES DE ANÁLISIS")
    print("=" * 80)

    # Obtener resumen
    print("\n1. Resumen de experimentos:")
    df = logger.get_summary_dataframe()
    print(f"   Total de experimentos: {len(df)}")

    if len(df) > 0:
        print(f"   Tipos: {df['Tipo'].unique().tolist()}")
        print(f"   Modelos: {df['Modelo'].unique().tolist()}")

    # Mejor modelo por diferentes métricas
    print("\n2. Mejores modelos por métrica:")

    metrics_to_test = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1-Score"]

    for metric in metrics_to_test:
        best = logger.get_best_model(metric=metric)
        if best:
            print(f"\n   Mejor {metric}:")
            print(f"   - Experimento: {best.get('Experimento', 'N/A')}")
            print(f"   - Valor: {best.get(metric, 'N/A')}")

    # Mostrar últimos experimentos
    print("\n3. Últimos 5 experimentos:")
    logger.print_summary(last_n=5)

    print("\n✓ Prueba de análisis completada")


def main():
    parser = argparse.ArgumentParser(
        description="Prueba del sistema de logging en Excel"
    )

    parser.add_argument(
        "--excel-path",
        type=str,
        default="results/test_experiment_results.xlsx",
        help="Ruta al archivo Excel de prueba",
    )

    parser.add_argument(
        "--num-train",
        type=int,
        default=5,
        help="Número de experimentos de entrenamiento a generar (default: 5)",
    )

    parser.add_argument(
        "--num-eval",
        type=int,
        default=5,
        help="Número de experimentos de evaluación a generar (default: 5)",
    )

    parser.add_argument(
        "--num-predict",
        type=int,
        default=5,
        help="Número de experimentos de predicción a generar (default: 5)",
    )

    parser.add_argument(
        "--skip-train", action="store_true", help="Saltar pruebas de entrenamiento"
    )

    parser.add_argument(
        "--skip-eval", action="store_true", help="Saltar pruebas de evaluación"
    )

    parser.add_argument(
        "--skip-predict", action="store_true", help="Saltar pruebas de predicción"
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Eliminar archivo de prueba existente antes de empezar",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PRUEBA DEL SISTEMA DE EXCEL LOGGER")
    print("=" * 80)
    print(f"\nArchivo Excel: {args.excel_path}")

    # Limpiar archivo si se solicita
    excel_path = Path(args.excel_path)
    if args.clean and excel_path.exists():
        excel_path.unlink()
        print(f"\n✓ Archivo de prueba anterior eliminado")

    # Crear logger
    print("\nInicializando logger...")
    logger = ExcelLogger(str(excel_path))
    print("✓ Logger inicializado")

    # Ejecutar pruebas
    try:
        if not args.skip_train:
            test_training_logs(logger, num_experiments=args.num_train)

        if not args.skip_eval:
            test_evaluation_logs(logger, num_experiments=args.num_eval)

        if not args.skip_predict:
            test_prediction_logs(logger, num_experiments=args.num_predict)

        # Análisis y resumen
        test_summary_and_analysis(logger)

        print("\n" + "=" * 80)
        print("✓ TODAS LAS PRUEBAS COMPLETADAS EXITOSAMENTE")
        print("=" * 80)

        print(f"\n📊 Archivo Excel generado: {excel_path}")
        print(f"📁 Abre el archivo para ver los resultados")

        print("\n💡 Próximos pasos:")
        print(f"  1. Abrir Excel: {excel_path}")
        print(
            f"  2. Ver análisis: python scripts/view_results.py --excel-path {excel_path}"
        )
        print(f"  3. Comparar: Revisar las diferentes hojas del Excel")

    except KeyboardInterrupt:
        print("\n\n⚠️  Prueba interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error durante las pruebas: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
