#!/usr/bin/env python3
"""
Script de Evaluación YOLO

Evalúa un modelo YOLO entrenado y genera métricas.

Uso:
    python scripts/evaluate.py --weights runs/train/exp/weights/best.pt
    python scripts/evaluate.py --weights best.pt --data data.yaml --device cuda
"""

import argparse
import yaml
from pathlib import Path
import sys

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent))

from ultralytics import YOLO
from src.evaluation import MetricsCalculator
from src.utils import ResultsVisualizer


def parse_args():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Evaluar modelo YOLO')
    
    # Modelo
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Ruta a los pesos del modelo (.pt)'
    )
    
    # Datos
    parser.add_argument(
        '--data',
        type=str,
        help='Ruta al archivo data.yaml'
    )
    
    # Evaluación
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['val', 'test'],
        help='Conjunto a evaluar (val o test)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Dispositivo (cpu, cuda, mps, 0, 1, etc.)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.7,
        help='IoU threshold para NMS'
    )
    
    # Salida
    parser.add_argument(
        '--save-report',
        type=str,
        help='Ruta para guardar reporte de evaluación'
    )
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Guardar gráficas de métricas'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/evaluation',
        help='Directorio para guardar resultados'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Carga configuración desde archivo YAML."""
    config_file = Path(config_path)
    if not config_file.exists():
        return {}
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    """Función principal."""
    args = parse_args()
    
    print("="*60)
    print("EVALUACIÓN DE MODELO YOLO")
    print("="*60)
    
    # Verificar pesos
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"\n❌ Error: No se encontraron los pesos: {args.weights}")
        sys.exit(1)
    
    print(f"\n1. Cargando modelo: {args.weights}")
    model = YOLO(args.weights)
    
    # Configurar datos
    data_yaml = args.data
    if not data_yaml:
        # Intentar encontrar data.yaml
        possible_paths = [
            Path('data/raw/data.yaml'),
            Path('data.yaml'),
            weights_path.parent.parent.parent / 'data.yaml'
        ]
        for path in possible_paths:
            if path.exists():
                data_yaml = str(path)
                break
    
    if not data_yaml or not Path(data_yaml).exists():
        print("\n❌ Error: No se encontró data.yaml")
        print("Especifica la ruta con --data")
        sys.exit(1)
    
    print(f"2. Usando datos: {data_yaml}")
    print(f"3. Evaluando en conjunto: {args.split}")
    print(f"4. Dispositivo: {args.device}")
    print("="*60 + "\n")
    
    # Crear calculador de métricas
    calculator = MetricsCalculator(model, data_yaml)
    
    # Evaluar
    metrics = calculator.evaluate(
        split=args.split,
        device=args.device,
        verbose=True
    )
    
    # Guardar reporte
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = args.save_report or str(output_dir / 'evaluation_report.txt')
    report = calculator.generate_report(save_path=report_path)
    print("\n" + report)
    
    # Crear visualizaciones
    if args.save_plots:
        print("\n5. Generando visualizaciones...")
        visualizer = ResultsVisualizer()
        
        # Buscar archivos de resultados
        val_dir = weights_path.parent.parent
        
        # Resumen de métricas
        visualizer.create_results_summary(
            metrics,
            save_path=output_dir / 'metrics_summary.png',
            show=False
        )
        print(f"   ✓ Resumen de métricas: {output_dir / 'metrics_summary.png'}")
        
        # Gráficas de entrenamiento si existen
        results_csv = val_dir / 'results.csv'
        if results_csv.exists():
            visualizer.plot_training_metrics(
                str(results_csv),
                save_path=output_dir / 'training_metrics.png',
                show=False
            )
            print(f"   ✓ Métricas de entrenamiento: {output_dir / 'training_metrics.png'}")
        
        # Matriz de confusión si existe
        confusion_matrix = val_dir / 'confusion_matrix.png'
        if confusion_matrix.exists():
            visualizer.plot_confusion_matrix(
                str(confusion_matrix),
                save_path=output_dir / 'confusion_matrix_viz.png',
                show=False
            )
            print(f"   ✓ Matriz de confusión: {output_dir / 'confusion_matrix_viz.png'}")
    
    # Interpretación
    print("\n" + "="*60)
    print("INTERPRETACIÓN DE RESULTADOS")
    print("="*60)
    print(f"mAP@0.5: {calculator.interpret_map(metrics['map50'])}")
    print(f"mAP@0.5:0.95: {calculator.interpret_map(metrics['map50_95'])}")
    print("="*60)
    
    # F1-Score
    f1 = calculator.calculate_f1_score(metrics['precision'], metrics['recall'])
    print(f"\nF1-Score: {f1:.4f}")
    
    # Recomendaciones
    print("\nRECOMENDACIONES:")
    if metrics['precision'] < 0.5:
        print("  ⚠️  Precision baja: muchos falsos positivos")
        print("     → Aumenta el confidence threshold")
        print("     → Añade más datos de entrenamiento")
    if metrics['recall'] < 0.5:
        print("  ⚠️  Recall bajo: muchos falsos negativos")
        print("     → Disminuye el confidence threshold")
        print("     → Revisa el etiquetado de datos")
    if metrics['map50_95'] < 0.3:
        print("  ⚠️  mAP bajo: el modelo necesita más entrenamiento")
        print("     → Aumenta el número de épocas")
        print("     → Ajusta la aumentación de datos")
        print("     → Considera usar un modelo más grande")
    
    if metrics['map50_95'] >= 0.7:
        print("  ✓ ¡Excelente rendimiento! El modelo está listo para producción.")
    
    print("\n" + "="*60)
    print("✓ EVALUACIÓN COMPLETADA")
    print("="*60)
    print(f"\nResultados guardados en: {output_dir}")


if __name__ == '__main__':
    main()
