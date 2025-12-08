"""
YOLO Trainer Module

Módulo para entrenamiento del modelo YOLO.
"""

from pathlib import Path
from typing import Optional, Dict, Any
from ultralytics import YOLO


class YOLOTrainer:
    """
    Clase para entrenar modelos YOLO.
    
    Args:
        model: Objeto YOLO a entrenar
        data_yaml: Ruta al archivo data.yaml con configuración del dataset
    """
    
    def __init__(
        self,
        model: YOLO,
        data_yaml: str
    ):
        self.model = model
        self.data_yaml = data_yaml
        self.results = None
    
    def train(
        self,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        device: str = 'cpu',
        patience: int = 50,
        save: bool = True,
        project: str = 'runs/train',
        name: str = 'exp',
        augmentation_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        Entrena el modelo YOLO.
        
        Args:
            epochs (int): Número de épocas de entrenamiento
            imgsz (int): Tamaño de las imágenes
            batch (int): Tamaño del batch
            device (str): Dispositivo (cpu, cuda, mps, 0, 1, etc.)
            patience (int): Épocas sin mejora antes de early stopping
            save (bool): Guardar checkpoints
            project (str): Directorio del proyecto
            name (str): Nombre del experimento
            augmentation_params (dict): Parámetros de aumentación
            **kwargs: Argumentos adicionales para el entrenamiento
        
        Returns:
            Results: Resultados del entrenamiento
        """
        print(f"Iniciando entrenamiento con:")
        print(f"  - Épocas: {epochs}")
        print(f"  - Tamaño de imagen: {imgsz}")
        print(f"  - Batch size: {batch}")
        print(f"  - Dispositivo: {device}")
        
        # Preparar argumentos de entrenamiento
        train_args = {
            'data': self.data_yaml,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'device': device,
            'patience': patience,
            'save': save,
            'project': project,
            'name': name,
            'verbose': True,
            **kwargs
        }
        
        # Agregar parámetros de aumentación si se proporcionan
        if augmentation_params:
            train_args.update(augmentation_params)
        
        # Entrenar
        self.results = self.model.train(**train_args)
        
        print(f"\n✓ Entrenamiento completado!")
        print(f"  Resultados guardados en: {project}/{name}")
        
        return self.results
    
    def get_results(self) -> Optional[Any]:
        """
        Retorna los resultados del entrenamiento.
        
        Returns:
            Results: Resultados del último entrenamiento
        """
        return self.results
    
    def save_model(self, path: str) -> None:
        """
        Guarda el modelo entrenado.
        
        Args:
            path (str): Ruta donde guardar el modelo
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        print(f"Modelo guardado en: {path}")
    
    @staticmethod
    def get_training_tips() -> Dict[str, str]:
        """
        Retorna consejos para el entrenamiento.
        
        Returns:
            dict: Consejos de entrenamiento
        """
        return {
            'epochs': 'Comienza con 100 épocas, ajusta según early stopping',
            'batch': 'Usa el batch más grande que permita tu GPU (16, 32, 64)',
            'imgsz': '640 es el estándar, aumenta a 1280 para objetos pequeños',
            'patience': '50 épocas es razonable para evitar sobreentrenamiento',
            'device': 'Usa cuda si tienes GPU NVIDIA, mps para Apple Silicon',
            'augmentation': 'Ajusta según tu dataset, más aumentación para menos datos',
            'learning_rate': 'YOLOv8 usa lr adaptativo, raramente necesitas ajustarlo',
            'optimizer': 'AdamW es el default y funciona bien en la mayoría de casos'
        }
