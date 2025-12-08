"""
Data Augmentation Module

Módulo para aumentación de datos en entrenamiento YOLO.
"""

from typing import Dict, Any


class DataAugmentation:
    """
    Configuración de aumentación de datos para entrenamiento YOLO.
    
    YOLOv8 incluye aumentación por defecto, esta clase permite
    personalizar los parámetros.
    """
    
    @staticmethod
    def get_default_augmentation() -> Dict[str, Any]:
        """
        Retorna configuración por defecto de aumentación.
        
        Returns:
            dict: Parámetros de aumentación
        """
        return {
            'hsv_h': 0.015,  # Variación de tono (hue)
            'hsv_s': 0.7,    # Variación de saturación
            'hsv_v': 0.4,    # Variación de brillo (value)
            'degrees': 0.0,  # Rotación aleatoria
            'translate': 0.1,  # Traslación aleatoria
            'scale': 0.5,    # Escala aleatoria
            'shear': 0.0,    # Transformación de corte
            'perspective': 0.0,  # Perspectiva aleatoria
            'flipud': 0.0,   # Flip vertical
            'fliplr': 0.5,   # Flip horizontal
            'mosaic': 1.0,   # Aumentación Mosaic
            'mixup': 0.0,    # Aumentación MixUp
            'copy_paste': 0.0  # Copy-Paste aumentación
        }
    
    @staticmethod
    def get_light_augmentation() -> Dict[str, Any]:
        """
        Retorna configuración ligera de aumentación.
        
        Returns:
            dict: Parámetros de aumentación ligera
        """
        return {
            'hsv_h': 0.01,
            'hsv_s': 0.5,
            'hsv_v': 0.3,
            'degrees': 0.0,
            'translate': 0.05,
            'scale': 0.3,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 0.5,
            'mixup': 0.0,
            'copy_paste': 0.0
        }
    
    @staticmethod
    def get_heavy_augmentation() -> Dict[str, Any]:
        """
        Retorna configuración intensiva de aumentación.
        
        Returns:
            dict: Parámetros de aumentación intensiva
        """
        return {
            'hsv_h': 0.02,
            'hsv_s': 0.9,
            'hsv_v': 0.6,
            'degrees': 10.0,
            'translate': 0.2,
            'scale': 0.9,
            'shear': 2.0,
            'perspective': 0.001,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.1,
            'copy_paste': 0.1
        }
