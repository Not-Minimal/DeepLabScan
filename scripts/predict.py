#!/usr/bin/env python3
"""
Script de Predicción YOLO

Realiza predicciones con un modelo YOLO entrenado.

Uso:
    python scripts/predict.py --weights best.pt --source images/
    python scripts/predict.py --weights best.pt --source video.mp4
    python scripts/predict.py --weights best.pt --source 0  # webcam
"""

import argparse
from pathlib import Path
import sys

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent))

from ultralytics import YOLO
from src.utils import ResultsVisualizer


def parse_args():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Realizar predicciones con YOLO')
    
    # Modelo
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Ruta a los pesos del modelo (.pt)'
    )
    
    # Entrada
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Fuente de entrada (imagen, directorio, video, URL, 0 para webcam)'
    )
    
    # Parámetros de predicción
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (0.0-1.0)'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.7,
        help='IoU threshold para NMS (0.0-1.0)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Tamaño de las imágenes'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Dispositivo (cpu, cuda, mps, 0, 1, etc.)'
    )
    parser.add_argument(
        '--max-det',
        type=int,
        default=300,
        help='Máximo de detecciones por imagen'
    )
    
    # Salida
    parser.add_argument(
        '--save',
        action='store_true',
        default=True,
        help='Guardar resultados'
    )
    parser.add_argument(
        '--save-txt',
        action='store_true',
        help='Guardar resultados en formato txt'
    )
    parser.add_argument(
        '--save-conf',
        action='store_true',
        help='Guardar confianza en archivos txt'
    )
    parser.add_argument(
        '--save-crop',
        action='store_true',
        help='Guardar detecciones recortadas'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='runs/predict',
        help='Directorio del proyecto'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='exp',
        help='Nombre del experimento'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Mostrar resultados'
    )
    parser.add_argument(
        '--show-labels',
        action='store_true',
        default=True,
        help='Mostrar etiquetas en las predicciones'
    )
    parser.add_argument(
        '--show-conf',
        action='store_true',
        default=True,
        help='Mostrar confianza en las predicciones'
    )
    
    # Opciones adicionales
    parser.add_argument(
        '--line-width',
        type=int,
        help='Ancho de línea de las cajas'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualizar características del modelo'
    )
    parser.add_argument(
        '--augment',
        action='store_true',
        help='Usar aumentación en predicción (TTA)'
    )
    parser.add_argument(
        '--agnostic-nms',
        action='store_true',
        help='NMS agnóstico a clases'
    )
    
    return parser.parse_args()


def main():
    """Función principal."""
    args = parse_args()
    
    print("="*60)
    print("PREDICCIÓN CON MODELO YOLO")
    print("="*60)
    
    # Verificar pesos
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"\n❌ Error: No se encontraron los pesos: {args.weights}")
        sys.exit(1)
    
    print(f"\n1. Cargando modelo: {args.weights}")
    model = YOLO(args.weights)
    
    # Verificar fuente
    source = args.source
    if source != '0' and not source.startswith('http'):
        source_path = Path(source)
        if not source_path.exists():
            print(f"\n❌ Error: No se encontró la fuente: {source}")
            sys.exit(1)
    
    print(f"2. Fuente de entrada: {source}")
    print(f"3. Configuración:")
    print(f"   - Confidence: {args.conf}")
    print(f"   - IoU: {args.iou}")
    print(f"   - Dispositivo: {args.device}")
    print(f"   - Tamaño imagen: {args.imgsz}")
    print("="*60 + "\n")
    
    # Realizar predicciones
    print("Realizando predicciones...")
    results = model.predict(
        source=source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        max_det=args.max_det,
        save=args.save,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_crop=args.save_crop,
        project=args.project,
        name=args.name,
        show=args.show,
        show_labels=args.show_labels,
        show_conf=args.show_conf,
        line_width=args.line_width,
        visualize=args.visualize,
        augment=args.augment,
        agnostic_nms=args.agnostic_nms,
        verbose=True
    )
    
    # Resumen de resultados
    print("\n" + "="*60)
    print("RESUMEN DE PREDICCIONES")
    print("="*60)
    
    total_detections = 0
    for result in results:
        if hasattr(result, 'boxes'):
            n_detections = len(result.boxes)
            total_detections += n_detections
            if n_detections > 0:
                print(f"\nImagen: {result.path}")
                print(f"Detecciones: {n_detections}")
                
                # Contar por clase
                classes = result.boxes.cls.cpu().numpy()
                for cls_id in set(classes):
                    count = (classes == cls_id).sum()
                    cls_name = result.names[int(cls_id)]
                    print(f"  - {cls_name}: {count}")
    
    print(f"\nTotal de detecciones: {total_detections}")
    
    if args.save:
        print(f"\n✓ Resultados guardados en: {args.project}/{args.name}")
        if args.save_txt:
            print(f"✓ Etiquetas guardadas en formato txt")
        if args.save_crop:
            print(f"✓ Detecciones recortadas guardadas")
    
    print("\n" + "="*60)
    print("✓ PREDICCIÓN COMPLETADA")
    print("="*60)
    
    # Mostrar estadísticas adicionales
    if total_detections > 0:
        print("\nESTADÍSTICAS:")
        all_confidences = []
        for result in results:
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                all_confidences.extend(result.boxes.conf.cpu().numpy().tolist())
        
        if all_confidences:
            import numpy as np
            print(f"  - Confianza promedio: {np.mean(all_confidences):.3f}")
            print(f"  - Confianza máxima: {np.max(all_confidences):.3f}")
            print(f"  - Confianza mínima: {np.min(all_confidences):.3f}")
    
    print("\nCONSEJOS:")
    print("  - Si hay muchos falsos positivos, aumenta --conf")
    print("  - Si faltan detecciones, disminuye --conf")
    print("  - Para mejorar detección en bordes, usa --augment")
    print("  - Para video en tiempo real, reduce --imgsz")


if __name__ == '__main__':
    main()
