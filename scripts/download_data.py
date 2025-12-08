#!/usr/bin/env python3
"""
Script para Descargar Datos desde Roboflow

Descarga un dataset desde Roboflow para usar con YOLO.

Uso:
    python scripts/download_data.py --workspace mi_workspace --project mi_proyecto --version 1
    python scripts/download_data.py --config configs/config.yaml
"""

import argparse
import yaml
from pathlib import Path
import sys

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import RoboflowDataLoader


def parse_args():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Descargar datos desde Roboflow')
    
    # Configuración
    parser.add_argument(
        '--config',
        type=str,
        help='Ruta al archivo de configuración'
    )
    
    # Roboflow
    parser.add_argument(
        '--workspace',
        type=str,
        help='Workspace de Roboflow'
    )
    parser.add_argument(
        '--project',
        type=str,
        help='Proyecto de Roboflow'
    )
    parser.add_argument(
        '--version',
        type=int,
        help='Versión del dataset'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='API key de Roboflow (opcional, se puede usar .env)'
    )
    
    # Salida
    parser.add_argument(
        '--location',
        type=str,
        default='./data/raw',
        help='Directorio donde guardar el dataset'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='yolov8',
        help='Formato del dataset'
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
    print("DESCARGA DE DATOS DESDE ROBOFLOW")
    print("="*60)
    
    # Obtener parámetros
    workspace = args.workspace
    project = args.project
    version = args.version
    api_key = args.api_key
    
    # Si se proporciona config, cargar de ahí
    if args.config:
        config = load_config(args.config)
        roboflow_config = config.get('roboflow', {})
        
        workspace = workspace or roboflow_config.get('workspace')
        project = project or roboflow_config.get('project')
        version = version or roboflow_config.get('version', 1)
    
    # Verificar parámetros requeridos
    if not workspace or not project:
        print("\n❌ Error: Se requieren workspace y project")
        print("\nOpciones:")
        print("  1. Usa --workspace y --project")
        print("  2. Configura roboflow en configs/config.yaml")
        print("\nEjemplo:")
        print("  python scripts/download_data.py --workspace mi_workspace --project mi_proyecto --version 1")
        sys.exit(1)
    
    print(f"\nParámetros:")
    print(f"  - Workspace: {workspace}")
    print(f"  - Proyecto: {project}")
    print(f"  - Versión: {version}")
    print(f"  - Destino: {args.location}")
    print(f"  - Formato: {args.format}")
    print("="*60 + "\n")
    
    try:
        # Crear loader
        print("Conectando con Roboflow...")
        loader = RoboflowDataLoader(
            workspace=workspace,
            project=project,
            version=version,
            api_key=api_key
        )
        
        # Descargar dataset
        print("\nDescargando dataset...")
        dataset_location = loader.download_dataset(
            location=args.location,
            format=args.format
        )
        
        # Obtener información
        info = loader.get_dataset_info()
        
        print("\n" + "="*60)
        print("✓ DESCARGA COMPLETADA")
        print("="*60)
        print(f"\nDataset guardado en: {dataset_location}")
        print("\nInformación del dataset:")
        print(f"  - Workspace: {info['workspace']}")
        print(f"  - Proyecto: {info['name']}")
        print(f"  - Versión: {info['version']}")
        
        # Buscar data.yaml
        data_yaml = Path(dataset_location) / 'data.yaml'
        if data_yaml.exists():
            print(f"\n✓ Archivo data.yaml encontrado: {data_yaml}")
            print("\nPróximos pasos:")
            print(f"  1. Entrenar: python scripts/train.py --data {data_yaml}")
            print(f"  2. O actualiza configs/config.yaml con la ruta del dataset")
        else:
            print("\n⚠️  Advertencia: No se encontró data.yaml")
            print("Verifica la estructura del dataset descargado")
        
        # Mostrar estructura esperada
        print("\nEstructura esperada del dataset:")
        print(f"{args.location}/")
        print("  ├── data.yaml")
        print("  ├── train/")
        print("  │   ├── images/")
        print("  │   └── labels/")
        print("  ├── valid/")
        print("  │   ├── images/")
        print("  │   └── labels/")
        print("  └── test/")
        print("      ├── images/")
        print("      └── labels/")
        
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nConsejos:")
        print("  - Verifica que tu API key sea correcta")
        print("  - Asegúrate de tener acceso al workspace y proyecto")
        print("  - Crea un archivo .env con ROBOFLOW_API_KEY=tu_api_key")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
