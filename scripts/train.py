#!/usr/bin/env python3
"""
Script de Entrenamiento YOLO

Entrena un modelo YOLO con datos desde Roboflow.

Uso:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --epochs 100 --batch 16 --imgsz 640
"""

import argparse
import yaml
from pathlib import Path
import sys

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import YOLOModel, YOLOTrainer
from src.data import DataAugmentation


def parse_args():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Entrenar modelo YOLO')
    
    # Configuración
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Ruta al archivo de configuración'
    )
    
    # Datos
    parser.add_argument(
        '--data',
        type=str,
        help='Ruta al archivo data.yaml (sobrescribe config)'
    )
    
    # Modelo
    parser.add_argument(
        '--model',
        type=str,
        help='Modelo YOLO (yolov8n.pt, yolov8s.pt, etc.)'
    )
    parser.add_argument(
        '--weights',
        type=str,
        help='Ruta a pesos pre-entrenados personalizados'
    )
    
    # Hiperparámetros
    parser.add_argument(
        '--epochs',
        type=int,
        help='Número de épocas'
    )
    parser.add_argument(
        '--batch',
        type=int,
        help='Tamaño del batch'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        help='Tamaño de las imágenes'
    )
    parser.add_argument(
        '--device',
        type=str,
        help='Dispositivo (cpu, cuda, mps, 0, 1, etc.)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        help='Épocas de patience para early stopping'
    )
    
    # Proyecto
    parser.add_argument(
        '--project',
        type=str,
        help='Directorio del proyecto'
    )
    parser.add_argument(
        '--name',
        type=str,
        help='Nombre del experimento'
    )
    
    # Aumentación
    parser.add_argument(
        '--augmentation',
        type=str,
        choices=['default', 'light', 'heavy'],
        help='Nivel de aumentación de datos'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """
    Carga configuración desde archivo YAML.
    
    Args:
        config_path (str): Ruta al archivo de configuración
    
    Returns:
        dict: Configuración
    """
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"⚠️  Archivo de configuración no encontrado: {config_path}")
        print("Usando configuración por defecto")
        return {}
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    """Función principal."""
    args = parse_args()
    
    # Cargar configuración
    config = load_config(args.config)
    
    print("="*60)
    print("ENTRENAMIENTO DE MODELO YOLO")
    print("="*60)
    
    # Configurar modelo
    model_name = args.model or config.get('model', {}).get('name', 'yolov8n.pt')
    weights_path = args.weights
    
    print(f"\n1. Inicializando modelo: {model_name}")
    yolo_model = YOLOModel(
        model_name=model_name,
        pretrained=True,
        weights_path=weights_path
    )
    
    # Configurar datos
    data_yaml = args.data
    if not data_yaml:
        # Buscar data.yaml en el directorio de datos
        data_path = Path(config.get('paths', {}).get('data', 'data/raw'))
        possible_paths = [
            data_path / 'data.yaml',
            Path('data/raw/data.yaml'),
            Path('data.yaml')
        ]
        for path in possible_paths:
            if path.exists():
                data_yaml = str(path)
                break
    
    if not data_yaml or not Path(data_yaml).exists():
        print("\n❌ Error: No se encontró data.yaml")
        print("Por favor:")
        print("  1. Descarga datos desde Roboflow")
        print("  2. Especifica la ruta con --data")
        sys.exit(1)
    
    print(f"2. Usando datos: {data_yaml}")
    
    # Configurar aumentación
    augmentation_params = None
    if args.augmentation:
        print(f"3. Configurando aumentación: {args.augmentation}")
        if args.augmentation == 'light':
            augmentation_params = DataAugmentation.get_light_augmentation()
        elif args.augmentation == 'heavy':
            augmentation_params = DataAugmentation.get_heavy_augmentation()
        else:
            augmentation_params = DataAugmentation.get_default_augmentation()
    elif 'augmentation' in config:
        augmentation_params = config['augmentation']
    
    # Configurar entrenamiento
    training_config = config.get('training', {})
    
    epochs = args.epochs or training_config.get('epochs', 100)
    batch = args.batch or training_config.get('batch', 16)
    imgsz = args.imgsz or training_config.get('imgsz', 640)
    device = args.device or training_config.get('device', 'cpu')
    patience = args.patience or training_config.get('patience', 50)
    project = args.project or training_config.get('project', 'runs/train')
    name = args.name or training_config.get('name', 'exp')
    
    print(f"\n4. Iniciando entrenamiento:")
    print(f"   - Épocas: {epochs}")
    print(f"   - Batch: {batch}")
    print(f"   - Tamaño imagen: {imgsz}")
    print(f"   - Dispositivo: {device}")
    print(f"   - Patience: {patience}")
    print("="*60 + "\n")
    
    # Entrenar
    trainer = YOLOTrainer(yolo_model.get_model(), data_yaml)
    
    # Hiperparámetros adicionales
    extra_params = {
        k: v for k, v in training_config.items()
        if k not in ['epochs', 'batch', 'imgsz', 'device', 'patience', 'project', 'name']
    }
    
    results = trainer.train(
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=patience,
        project=project,
        name=name,
        augmentation_params=augmentation_params,
        **extra_params
    )
    
    print("\n" + "="*60)
    print("✓ ENTRENAMIENTO COMPLETADO")
    print("="*60)
    print(f"\nResultados guardados en: {project}/{name}")
    print(f"Pesos del modelo: {project}/{name}/weights/best.pt")
    print("\nPróximos pasos:")
    print(f"  1. Evaluar: python scripts/evaluate.py --weights {project}/{name}/weights/best.pt")
    print(f"  2. Predecir: python scripts/predict.py --weights {project}/{name}/weights/best.pt --source path/to/images")


if __name__ == '__main__':
    main()
