"""
Tests for YOLO model and trainer
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models import YOLOModel, YOLOTrainer


class TestYOLOModel:
    """Tests for YOLOModel class"""
    
    @patch('src.models.yolo_model.YOLO')
    def test_init_with_pretrained(self, mock_yolo):
        """Test initialization with pretrained model"""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        yolo_model = YOLOModel(model_name='yolov8n.pt', pretrained=True)
        
        assert yolo_model.model_name == 'yolov8n.pt'
        assert yolo_model.pretrained is True
        mock_yolo.assert_called_once_with('yolov8n.pt')
    
    @patch('src.models.yolo_model.YOLO')
    def test_init_with_custom_weights(self, mock_yolo):
        """Test initialization with custom weights"""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        yolo_model = YOLOModel(
            model_name='yolov8n.pt',
            weights_path='custom.pt'
        )
        
        mock_yolo.assert_called_once_with('custom.pt')
    
    @patch('src.models.yolo_model.YOLO')
    def test_get_model(self, mock_yolo):
        """Test get_model method"""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        yolo_model = YOLOModel(model_name='yolov8n.pt')
        model = yolo_model.get_model()
        
        assert model == mock_model
    
    @patch('src.models.yolo_model.YOLO')
    def test_info(self, mock_yolo):
        """Test info method"""
        mock_model = Mock()
        mock_model.info = Mock()
        mock_yolo.return_value = mock_model
        
        yolo_model = YOLOModel(model_name='yolov8n.pt')
        yolo_model.info()
        
        mock_model.info.assert_called_once()
    
    def test_get_available_models(self):
        """Test get_available_models static method"""
        models = YOLOModel.get_available_models()
        
        assert 'yolov8n.pt' in models
        assert 'yolov8s.pt' in models
        assert 'yolov8m.pt' in models
        assert 'yolov8l.pt' in models
        assert 'yolov8x.pt' in models
        
        # Check structure
        for model_name, info in models.items():
            assert 'name' in info
            assert 'params' in info
            assert 'size' in info
            assert 'speed' in info
            assert 'use_case' in info


class TestYOLOTrainer:
    """Tests for YOLOTrainer class"""
    
    def test_init(self):
        """Test initialization"""
        mock_model = Mock()
        trainer = YOLOTrainer(mock_model, 'data.yaml')
        
        assert trainer.model == mock_model
        assert trainer.data_yaml == 'data.yaml'
        assert trainer.results is None
    
    @patch('src.models.trainer.Path')
    def test_train(self, mock_path):
        """Test train method"""
        mock_model = Mock()
        mock_results = Mock()
        mock_model.train.return_value = mock_results
        
        trainer = YOLOTrainer(mock_model, 'data.yaml')
        results = trainer.train(
            epochs=50,
            imgsz=640,
            batch=16,
            device='cpu'
        )
        
        assert results == mock_results
        assert trainer.results == mock_results
        mock_model.train.assert_called_once()
    
    def test_train_with_augmentation(self):
        """Test train with augmentation parameters"""
        mock_model = Mock()
        mock_results = Mock()
        mock_model.train.return_value = mock_results
        
        trainer = YOLOTrainer(mock_model, 'data.yaml')
        augmentation_params = {
            'hsv_h': 0.015,
            'fliplr': 0.5
        }
        
        results = trainer.train(
            epochs=50,
            augmentation_params=augmentation_params
        )
        
        # Check that augmentation params were passed
        call_kwargs = mock_model.train.call_args[1]
        assert 'hsv_h' in call_kwargs
        assert 'fliplr' in call_kwargs
    
    def test_get_results(self):
        """Test get_results method"""
        mock_model = Mock()
        trainer = YOLOTrainer(mock_model, 'data.yaml')
        
        assert trainer.get_results() is None
        
        mock_results = Mock()
        trainer.results = mock_results
        assert trainer.get_results() == mock_results
    
    @patch('src.models.trainer.Path')
    def test_save_model(self, mock_path):
        """Test save_model method"""
        mock_model = Mock()
        mock_model.save = Mock()
        
        trainer = YOLOTrainer(mock_model, 'data.yaml')
        trainer.save_model('model.pt')
        
        mock_model.save.assert_called_once_with('model.pt')
    
    def test_get_training_tips(self):
        """Test get_training_tips static method"""
        tips = YOLOTrainer.get_training_tips()
        
        assert 'epochs' in tips
        assert 'batch' in tips
        assert 'imgsz' in tips
        assert 'patience' in tips
        assert 'device' in tips
        assert 'augmentation' in tips
        
        # Check that tips are strings
        for key, value in tips.items():
            assert isinstance(value, str)
            assert len(value) > 0
