"""
Roboflow Data Loader Module

Este módulo maneja la descarga y preparación de datos desde Roboflow.
"""

import os
from pathlib import Path
from typing import Optional
from roboflow import Roboflow
from dotenv import load_dotenv


class RoboflowDataLoader:
    """
    Carga datos desde Roboflow para entrenamiento YOLO.
    
    Args:
        workspace (str): Nombre del workspace en Roboflow
        project (str): Nombre del proyecto en Roboflow
        version (int): Versión del dataset
        api_key (Optional[str]): API key de Roboflow (si no está en .env)
    """
    
    def __init__(
        self,
        workspace: str,
        project: str,
        version: int = 1,
        api_key: Optional[str] = None
    ):
        self.workspace = workspace
        self.project = project
        self.version = version
        
        # Cargar API key
        load_dotenv()
        self.api_key = api_key or os.getenv('ROBOFLOW_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "API key no encontrada. Proporciona api_key o configura "
                "ROBOFLOW_API_KEY en .env"
            )
        
        # Inicializar Roboflow
        self.rf = Roboflow(api_key=self.api_key)
        self.workspace_obj = self.rf.workspace(self.workspace)
        self.project_obj = self.workspace_obj.project(self.project)
        self.dataset = None
    
    def download_dataset(
        self,
        location: str = "./data/raw",
        format: str = "yolov8"
    ) -> str:
        """
        Descarga el dataset desde Roboflow.
        
        Args:
            location (str): Ruta donde guardar el dataset
            format (str): Formato del dataset (default: yolov8)
        
        Returns:
            str: Ruta al dataset descargado
        """
        # Crear directorio si no existe
        Path(location).mkdir(parents=True, exist_ok=True)
        
        # Descargar dataset
        print(f"Descargando dataset {self.project} v{self.version}...")
        self.dataset = self.project_obj.version(self.version).download(
            model_format=format,
            location=location
        )
        
        print(f"Dataset descargado en: {self.dataset.location}")
        return self.dataset.location
    
    def get_dataset_info(self) -> dict:
        """
        Obtiene información del dataset.
        
        Returns:
            dict: Información del dataset
        """
        if not self.dataset:
            raise ValueError("Primero descarga el dataset con download_dataset()")
        
        return {
            'location': self.dataset.location,
            'name': self.project,
            'version': self.version,
            'workspace': self.workspace
        }
