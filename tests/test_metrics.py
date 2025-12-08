"""
Tests for metrics calculation
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation import MetricsCalculator


class TestMetricsCalculator:
    """Tests for MetricsCalculator class"""
    
    def test_init(self):
        """Test initialization"""
        mock_model = Mock()
        calculator = MetricsCalculator(mock_model, 'data.yaml')
        
        assert calculator.model == mock_model
        assert calculator.data_yaml == 'data.yaml'
        assert calculator.results is None
    
    def test_calculate_precision(self):
        """Test precision calculation"""
        mock_model = Mock()
        calculator = MetricsCalculator(mock_model, 'data.yaml')
        
        # TP=80, FP=20 -> Precision = 80/100 = 0.8
        precision = calculator.calculate_precision(tp=80, fp=20)
        assert precision == 0.8
        
        # TP=100, FP=0 -> Precision = 1.0
        precision = calculator.calculate_precision(tp=100, fp=0)
        assert precision == 1.0
        
        # TP=0, FP=0 -> Precision = 0.0
        precision = calculator.calculate_precision(tp=0, fp=0)
        assert precision == 0.0
    
    def test_calculate_recall(self):
        """Test recall calculation"""
        mock_model = Mock()
        calculator = MetricsCalculator(mock_model, 'data.yaml')
        
        # TP=80, FN=20 -> Recall = 80/100 = 0.8
        recall = calculator.calculate_recall(tp=80, fn=20)
        assert recall == 0.8
        
        # TP=100, FN=0 -> Recall = 1.0
        recall = calculator.calculate_recall(tp=100, fn=0)
        assert recall == 1.0
        
        # TP=0, FN=0 -> Recall = 0.0
        recall = calculator.calculate_recall(tp=0, fn=0)
        assert recall == 0.0
    
    def test_calculate_f1_score(self):
        """Test F1-score calculation"""
        mock_model = Mock()
        calculator = MetricsCalculator(mock_model, 'data.yaml')
        
        # Precision=0.8, Recall=0.8 -> F1 = 0.8
        f1 = calculator.calculate_f1_score(precision=0.8, recall=0.8)
        assert f1 == 0.8
        
        # Precision=1.0, Recall=0.5 -> F1 = 2*(1.0*0.5)/(1.0+0.5) = 0.667
        f1 = calculator.calculate_f1_score(precision=1.0, recall=0.5)
        assert abs(f1 - 0.6666666666666666) < 1e-10
        
        # Precision=0.0, Recall=0.0 -> F1 = 0.0
        f1 = calculator.calculate_f1_score(precision=0.0, recall=0.0)
        assert f1 == 0.0
    
    def test_calculate_iou(self):
        """Test IoU calculation"""
        mock_model = Mock()
        calculator = MetricsCalculator(mock_model, 'data.yaml')
        
        # Perfect overlap
        box1 = (0, 0, 10, 10)
        box2 = (0, 0, 10, 10)
        iou = calculator.calculate_iou(box1, box2)
        assert iou == 1.0
        
        # No overlap
        box1 = (0, 0, 10, 10)
        box2 = (20, 20, 30, 30)
        iou = calculator.calculate_iou(box1, box2)
        assert iou == 0.0
        
        # Partial overlap
        box1 = (0, 0, 10, 10)  # Area = 100
        box2 = (5, 5, 15, 15)  # Area = 100
        # Intersection = (5,5,10,10) = 25
        # Union = 100 + 100 - 25 = 175
        # IoU = 25/175 = 0.142857...
        iou = calculator.calculate_iou(box1, box2)
        expected = 25.0 / 175.0
        assert abs(iou - expected) < 1e-6
    
    def test_calculate_iou_touching_boxes(self):
        """Test IoU with touching boxes"""
        mock_model = Mock()
        calculator = MetricsCalculator(mock_model, 'data.yaml')
        
        # Boxes touching at edge
        box1 = (0, 0, 10, 10)
        box2 = (10, 0, 20, 10)
        iou = calculator.calculate_iou(box1, box2)
        assert iou == 0.0
    
    def test_interpret_map(self):
        """Test mAP interpretation"""
        # Test excellent
        assert MetricsCalculator.interpret_map(0.95) == "Excelente"
        assert MetricsCalculator.interpret_map(0.9) == "Excelente"
        
        # Test very good
        assert MetricsCalculator.interpret_map(0.85) == "Muy bueno"
        assert MetricsCalculator.interpret_map(0.7) == "Muy bueno"
        
        # Test good
        assert MetricsCalculator.interpret_map(0.6) == "Bueno"
        assert MetricsCalculator.interpret_map(0.5) == "Bueno"
        
        # Test acceptable
        assert MetricsCalculator.interpret_map(0.4) == "Aceptable"
        assert MetricsCalculator.interpret_map(0.3) == "Aceptable"
        
        # Test needs improvement
        assert MetricsCalculator.interpret_map(0.2) == "Necesita mejora"
        assert MetricsCalculator.interpret_map(0.1) == "Necesita mejora"
    
    @patch('src.evaluation.metrics.Path')
    def test_generate_report(self, mock_path):
        """Test report generation"""
        mock_model = Mock()
        calculator = MetricsCalculator(mock_model, 'data.yaml')
        
        # Mock results
        mock_results = Mock()
        mock_box = Mock()
        mock_box.mp = 0.85
        mock_box.mr = 0.80
        mock_box.map50 = 0.82
        mock_box.map = 0.75
        mock_results.box = mock_box
        mock_results.fitness = 0.78
        
        calculator.results = mock_results
        
        report = calculator.generate_report()
        
        assert "REPORTE DE EVALUACIÓN" in report
        assert "0.8500" in report  # precision
        assert "0.8000" in report  # recall
        assert "0.8200" in report  # map50
        assert "0.7500" in report  # map50_95
    
    def test_generate_report_without_results_raises_error(self):
        """Test report generation without evaluation raises ValueError"""
        mock_model = Mock()
        calculator = MetricsCalculator(mock_model, 'data.yaml')
        
        with pytest.raises(ValueError, match="Primero ejecuta evaluate"):
            calculator.generate_report()
