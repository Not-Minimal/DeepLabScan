"""
Results Visualization Module

Módulo para visualizar resultados de detección y métricas.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple, Any
from ultralytics import YOLO


class ResultsVisualizer:
    """
    Visualiza resultados de detección y métricas de entrenamiento.
    """
    
    def __init__(self):
        """Inicializa el visualizador."""
        self.colors = self._generate_colors(80)  # Para 80 clases de COCO
        sns.set_style('whitegrid')
    
    @staticmethod
    def _generate_colors(n: int) -> List[Tuple[int, int, int]]:
        """
        Genera colores aleatorios para las clases.
        
        Args:
            n (int): Número de colores
        
        Returns:
            list: Lista de colores en formato BGR
        """
        np.random.seed(42)
        colors = []
        for _ in range(n):
            colors.append((
                int(np.random.randint(0, 255)),
                int(np.random.randint(0, 255)),
                int(np.random.randint(0, 255))
            ))
        return colors
    
    def plot_predictions(
        self,
        image_path: str,
        results: Any,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> np.ndarray:
        """
        Visualiza predicciones en una imagen.
        
        Args:
            image_path (str): Ruta a la imagen
            results: Resultados de predicción de YOLO
            save_path (str): Ruta para guardar imagen
            show (bool): Mostrar imagen
        
        Returns:
            np.ndarray: Imagen con predicciones
        """
        # Cargar imagen
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Dibujar predicciones
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Obtener coordenadas
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Dibujar caja
                color = self.colors[cls % len(self.colors)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Dibujar etiqueta
                label = f"{result.names[cls]}: {conf:.2f}"
                (w, h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                )
                cv2.rectangle(
                    img, (x1, y1 - 20), (x1 + w, y1), color, -1
                )
                cv2.putText(
                    img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
                )
        
        # Mostrar
        if show:
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title('Predicciones YOLO')
            plt.tight_layout()
            plt.show()
        
        # Guardar
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, img_bgr)
            print(f"Imagen guardada en: {save_path}")
        
        return img
    
    def plot_training_metrics(
        self,
        results_csv: str,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Visualiza métricas de entrenamiento desde CSV.
        
        Args:
            results_csv (str): Ruta al archivo results.csv
            save_path (str): Ruta para guardar figura
            show (bool): Mostrar figura
        """
        import pandas as pd
        
        # Leer CSV
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        ax = axes[0, 0]
        if 'train/box_loss' in df.columns:
            ax.plot(df['epoch'], df['train/box_loss'], label='Box Loss', marker='o')
        if 'train/cls_loss' in df.columns:
            ax.plot(df['epoch'], df['train/cls_loss'], label='Class Loss', marker='s')
        if 'train/dfl_loss' in df.columns:
            ax.plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss', marker='^')
        ax.set_xlabel('Época')
        ax.set_ylabel('Loss')
        ax.set_title('Pérdidas de Entrenamiento')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Precision y Recall
        ax = axes[0, 1]
        if 'metrics/precision(B)' in df.columns:
            ax.plot(df['epoch'], df['metrics/precision(B)'], 
                   label='Precision', marker='o', color='blue')
        if 'metrics/recall(B)' in df.columns:
            ax.plot(df['epoch'], df['metrics/recall(B)'], 
                   label='Recall', marker='s', color='green')
        ax.set_xlabel('Época')
        ax.set_ylabel('Valor')
        ax.set_title('Precision y Recall')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # mAP
        ax = axes[1, 0]
        if 'metrics/mAP50(B)' in df.columns:
            ax.plot(df['epoch'], df['metrics/mAP50(B)'], 
                   label='mAP@0.5', marker='o', color='orange')
        if 'metrics/mAP50-95(B)' in df.columns:
            ax.plot(df['epoch'], df['metrics/mAP50-95(B)'], 
                   label='mAP@0.5:0.95', marker='s', color='red')
        ax.set_xlabel('Época')
        ax.set_ylabel('mAP')
        ax.set_title('Mean Average Precision')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning Rate
        ax = axes[1, 1]
        lr_cols = [col for col in df.columns if 'lr' in col.lower()]
        for col in lr_cols:
            ax.plot(df['epoch'], df[col], label=col, marker='o')
        ax.set_xlabel('Época')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Tasa de Aprendizaje')
        if lr_cols:
            ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfica guardada en: {save_path}")
        
        # Mostrar
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_confusion_matrix(
        self,
        confusion_matrix_path: str,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Visualiza matriz de confusión desde imagen generada por YOLO.
        
        Args:
            confusion_matrix_path (str): Ruta a confusion_matrix.png
            save_path (str): Ruta para guardar figura
            show (bool): Mostrar figura
        """
        img = cv2.imread(confusion_matrix_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Matriz de Confusión', fontsize=16, pad=20)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Matriz de confusión guardada en: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_results_summary(
        self,
        metrics: dict,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Crea un resumen visual de métricas.
        
        Args:
            metrics (dict): Diccionario de métricas
            save_path (str): Ruta para guardar figura
            show (bool): Mostrar figura
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Preparar datos
        metric_names = ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95']
        values = [
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('map50', 0),
            metrics.get('map50_95', 0)
        ]
        
        # Crear barras
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        bars = ax.barh(metric_names, values, color=colors, alpha=0.7)
        
        # Añadir valores en las barras
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(value + 0.02, i, f'{value:.3f}', 
                   va='center', fontweight='bold')
        
        ax.set_xlabel('Valor', fontsize=12)
        ax.set_title('Resumen de Métricas de Evaluación', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Resumen guardado en: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
