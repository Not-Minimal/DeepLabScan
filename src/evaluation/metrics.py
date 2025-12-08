"""
Metrics Calculator Module

Módulo para calcular métricas de evaluación (Precision, Recall, mAP).
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class MetricsCalculator:
    """
    Calcula métricas de evaluación para modelos YOLO.
    
    Args:
        model: Modelo YOLO entrenado
        data_yaml (str): Ruta al archivo data.yaml
    """
    
    def __init__(
        self,
        model: YOLO,
        data_yaml: str
    ):
        self.model = model
        self.data_yaml = data_yaml
        self.results = None
    
    def evaluate(
        self,
        split: str = 'val',
        device: str = 'cpu',
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evalúa el modelo en el conjunto de validación/test.
        
        Args:
            split (str): Conjunto a evaluar ('val' o 'test')
            device (str): Dispositivo para evaluación
            verbose (bool): Mostrar información detallada
        
        Returns:
            dict: Diccionario con todas las métricas
        """
        print(f"Evaluando modelo en conjunto: {split}")
        
        # Validar modelo
        self.results = self.model.val(
            data=self.data_yaml,
            split=split,
            device=device,
            verbose=verbose
        )
        
        # Extraer métricas
        metrics = self._extract_metrics()
        
        if verbose:
            self._print_metrics(metrics)
        
        return metrics
    
    def _extract_metrics(self) -> Dict[str, Any]:
        """
        Extrae métricas de los resultados.
        
        Returns:
            dict: Métricas extraídas
        """
        metrics = {
            'precision': float(self.results.box.mp),  # Mean Precision
            'recall': float(self.results.box.mr),     # Mean Recall
            'map50': float(self.results.box.map50),   # mAP@0.5
            'map50_95': float(self.results.box.map),  # mAP@0.5:0.95
            'fitness': float(self.results.fitness),    # Fitness score
        }
        
        # Métricas por clase si existen
        if hasattr(self.results.box, 'ap_class_index'):
            metrics['per_class'] = {
                'precision': self.results.box.p.tolist() if hasattr(self.results.box, 'p') else [],
                'recall': self.results.box.r.tolist() if hasattr(self.results.box, 'r') else [],
                'ap50': self.results.box.ap50.tolist() if hasattr(self.results.box, 'ap50') else [],
                'ap': self.results.box.ap.tolist() if hasattr(self.results.box, 'ap') else [],
            }
        
        return metrics
    
    def _print_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Imprime métricas de forma legible.
        
        Args:
            metrics (dict): Diccionario de métricas
        """
        print("\n" + "="*50)
        print("MÉTRICAS DE EVALUACIÓN")
        print("="*50)
        print(f"Precision:        {metrics['precision']:.4f}")
        print(f"Recall:           {metrics['recall']:.4f}")
        print(f"mAP@0.5:          {metrics['map50']:.4f}")
        print(f"mAP@0.5:0.95:     {metrics['map50_95']:.4f}")
        print(f"Fitness Score:    {metrics['fitness']:.4f}")
        print("="*50 + "\n")
    
    def calculate_precision(self, tp: int, fp: int) -> float:
        """
        Calcula Precision.
        
        Precision = TP / (TP + FP)
        
        Args:
            tp (int): True Positives
            fp (int): False Positives
        
        Returns:
            float: Valor de precision
        """
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)
    
    def calculate_recall(self, tp: int, fn: int) -> float:
        """
        Calcula Recall.
        
        Recall = TP / (TP + FN)
        
        Args:
            tp (int): True Positives
            fn (int): False Negatives
        
        Returns:
            float: Valor de recall
        """
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)
    
    def calculate_f1_score(self, precision: float, recall: float) -> float:
        """
        Calcula F1-Score.
        
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        
        Args:
            precision (float): Valor de precision
            recall (float): Valor de recall
        
        Returns:
            float: F1-Score
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_iou(
        self,
        box1: Tuple[float, float, float, float],
        box2: Tuple[float, float, float, float]
    ) -> float:
        """
        Calcula Intersection over Union (IoU) entre dos bounding boxes.
        
        Args:
            box1: (x1, y1, x2, y2) primera caja
            box2: (x1, y1, x2, y2) segunda caja
        
        Returns:
            float: Valor de IoU
        """
        # Calcular área de intersección
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calcular áreas de cada caja
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Calcular unión
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def generate_report(
        self,
        save_path: Optional[str] = None
    ) -> str:
        """
        Genera un reporte completo de evaluación.
        
        Args:
            save_path (str): Ruta donde guardar el reporte
        
        Returns:
            str: Reporte en formato texto
        """
        if not self.results:
            raise ValueError("Primero ejecuta evaluate()")
        
        metrics = self._extract_metrics()
        
        report = []
        report.append("="*60)
        report.append("REPORTE DE EVALUACIÓN - MODELO YOLO")
        report.append("="*60)
        report.append("")
        report.append("MÉTRICAS GENERALES:")
        report.append("-"*60)
        report.append(f"  Precision (Media):       {metrics['precision']:.4f}")
        report.append(f"  Recall (Media):          {metrics['recall']:.4f}")
        report.append(f"  F1-Score:                {self.calculate_f1_score(metrics['precision'], metrics['recall']):.4f}")
        report.append(f"  mAP@0.5:                 {metrics['map50']:.4f}")
        report.append(f"  mAP@0.5:0.95:            {metrics['map50_95']:.4f}")
        report.append("")
        
        if 'per_class' in metrics:
            report.append("MÉTRICAS POR CLASE:")
            report.append("-"*60)
            for i, (p, r, ap50, ap) in enumerate(zip(
                metrics['per_class']['precision'],
                metrics['per_class']['recall'],
                metrics['per_class']['ap50'],
                metrics['per_class']['ap']
            )):
                report.append(f"  Clase {i}:")
                report.append(f"    Precision: {p:.4f}")
                report.append(f"    Recall:    {r:.4f}")
                report.append(f"    AP@0.5:    {ap50:.4f}")
                report.append(f"    AP@0.5:0.95: {ap:.4f}")
                report.append("")
        
        report.append("="*60)
        
        report_text = "\n".join(report)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Reporte guardado en: {save_path}")
        
        return report_text
    
    @staticmethod
    def interpret_map(map_value: float) -> str:
        """
        Interpreta el valor de mAP.
        
        Args:
            map_value (float): Valor de mAP
        
        Returns:
            str: Interpretación del valor
        """
        if map_value >= 0.9:
            return "Excelente"
        elif map_value >= 0.7:
            return "Muy bueno"
        elif map_value >= 0.5:
            return "Bueno"
        elif map_value >= 0.3:
            return "Aceptable"
        else:
            return "Necesita mejora"
