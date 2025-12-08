"""
YOLO Model Module

Módulo para configuración y manejo del modelo YOLO.
"""

from pathlib import Path
from typing import Optional, Union
from ultralytics import YOLO


class YOLOModel:
    """
    Wrapper para el modelo YOLO de Ultralytics.
    
    Args:
        model_name (str): Nombre del modelo (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        pretrained (bool): Si usar pesos pre-entrenados
        weights_path (Optional[str]): Ruta a pesos personalizados
    """
    
    def __init__(
        self,
        model_name: str = 'yolov8n.pt',
        pretrained: bool = True,
        weights_path: Optional[str] = None
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        
        # Cargar modelo
        if weights_path:
            print(f"Cargando modelo desde: {weights_path}")
            self.model = YOLO(weights_path)
        elif pretrained:
            print(f"Cargando modelo pre-entrenado: {model_name}")
            self.model = YOLO(model_name)
        else:
            print(f"Creando modelo nuevo: {model_name}")
            self.model = YOLO(model_name)
    
    def get_model(self) -> YOLO:
        """
        Retorna el objeto del modelo YOLO.
        
        Returns:
            YOLO: Objeto del modelo
        """
        return self.model
    
    def info(self) -> None:
        """
        Muestra información del modelo.
        """
        self.model.info()
    
    @staticmethod
    def get_available_models() -> dict:
        """
        Retorna información de modelos YOLO disponibles.
        
        Returns:
            dict: Información de modelos
        """
        return {
            'yolov8n.pt': {
                'name': 'YOLOv8 Nano',
                'params': '3.2M',
                'size': 'Muy pequeño',
                'speed': 'Muy rápido',
                'use_case': 'Dispositivos móviles, edge computing'
            },
            'yolov8s.pt': {
                'name': 'YOLOv8 Small',
                'params': '11.2M',
                'size': 'Pequeño',
                'speed': 'Rápido',
                'use_case': 'Balance velocidad-precisión'
            },
            'yolov8m.pt': {
                'name': 'YOLOv8 Medium',
                'params': '25.9M',
                'size': 'Medio',
                'speed': 'Moderado',
                'use_case': 'Aplicaciones generales'
            },
            'yolov8l.pt': {
                'name': 'YOLOv8 Large',
                'params': '43.7M',
                'size': 'Grande',
                'speed': 'Lento',
                'use_case': 'Alta precisión'
            },
            'yolov8x.pt': {
                'name': 'YOLOv8 XLarge',
                'params': '68.2M',
                'size': 'Muy grande',
                'speed': 'Muy lento',
                'use_case': 'Máxima precisión'
            }
        }
