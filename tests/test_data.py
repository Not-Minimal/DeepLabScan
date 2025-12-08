"""
Tests for data loading and preprocessing
"""

import pytest
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data import RoboflowDataLoader, DataAugmentation


class TestRoboflowDataLoader:
    """Tests for RoboflowDataLoader class"""
    
    def test_init_with_api_key(self):
        """Test initialization with API key"""
        loader = RoboflowDataLoader(
            workspace="test_workspace",
            project="test_project",
            version=1,
            api_key="test_api_key"
        )
        assert loader.workspace == "test_workspace"
        assert loader.project == "test_project"
        assert loader.version == 1
        assert loader.api_key == "test_api_key"
    
    def test_init_without_api_key_raises_error(self):
        """Test initialization without API key raises ValueError"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key no encontrada"):
                RoboflowDataLoader(
                    workspace="test_workspace",
                    project="test_project",
                    version=1
                )
    
    @patch('src.data.loader.Roboflow')
    def test_download_dataset(self, mock_roboflow):
        """Test dataset download"""
        # Mock Roboflow
        mock_dataset = Mock()
        mock_dataset.location = "/path/to/dataset"
        
        mock_version = Mock()
        mock_version.download.return_value = mock_dataset
        
        mock_project = Mock()
        mock_project.version.return_value = mock_version
        
        mock_workspace = Mock()
        mock_workspace.project.return_value = mock_project
        
        mock_rf = Mock()
        mock_rf.workspace.return_value = mock_workspace
        mock_roboflow.return_value = mock_rf
        
        # Test
        loader = RoboflowDataLoader(
            workspace="test_workspace",
            project="test_project",
            version=1,
            api_key="test_api_key"
        )
        
        location = loader.download_dataset(location="./test_data")
        
        assert location == "/path/to/dataset"
        mock_version.download.assert_called_once()
    
    @patch('src.data.loader.Roboflow')
    def test_get_dataset_info(self, mock_roboflow):
        """Test getting dataset info"""
        # Setup mock
        mock_rf = Mock()
        mock_roboflow.return_value = mock_rf
        
        loader = RoboflowDataLoader(
            workspace="test_workspace",
            project="test_project",
            version=1,
            api_key="test_api_key"
        )
        
        # Mock dataset
        loader.dataset = Mock()
        loader.dataset.location = "/path/to/dataset"
        
        info = loader.get_dataset_info()
        
        assert info['location'] == "/path/to/dataset"
        assert info['name'] == "test_project"
        assert info['version'] == 1
        assert info['workspace'] == "test_workspace"
    
    @patch('src.data.loader.Roboflow')
    def test_get_dataset_info_without_download_raises_error(self, mock_roboflow):
        """Test getting dataset info without downloading raises ValueError"""
        mock_rf = Mock()
        mock_roboflow.return_value = mock_rf
        
        loader = RoboflowDataLoader(
            workspace="test_workspace",
            project="test_project",
            version=1,
            api_key="test_api_key"
        )
        
        with pytest.raises(ValueError, match="Primero descarga el dataset"):
            loader.get_dataset_info()


class TestDataAugmentation:
    """Tests for DataAugmentation class"""
    
    def test_get_default_augmentation(self):
        """Test default augmentation parameters"""
        params = DataAugmentation.get_default_augmentation()
        
        assert 'hsv_h' in params
        assert 'hsv_s' in params
        assert 'hsv_v' in params
        assert 'fliplr' in params
        assert params['hsv_h'] == 0.015
        assert params['fliplr'] == 0.5
    
    def test_get_light_augmentation(self):
        """Test light augmentation parameters"""
        params = DataAugmentation.get_light_augmentation()
        
        assert 'hsv_h' in params
        assert 'mosaic' in params
        assert params['hsv_h'] < DataAugmentation.get_default_augmentation()['hsv_h']
        assert params['mosaic'] == 0.5
    
    def test_get_heavy_augmentation(self):
        """Test heavy augmentation parameters"""
        params = DataAugmentation.get_heavy_augmentation()
        
        assert 'hsv_h' in params
        assert 'degrees' in params
        assert 'mixup' in params
        assert params['degrees'] > 0
        assert params['mixup'] > 0
    
    def test_augmentation_params_have_required_fields(self):
        """Test all augmentation configs have required fields"""
        required_fields = [
            'hsv_h', 'hsv_s', 'hsv_v',
            'degrees', 'translate', 'scale',
            'fliplr', 'mosaic'
        ]
        
        for method in [
            DataAugmentation.get_default_augmentation,
            DataAugmentation.get_light_augmentation,
            DataAugmentation.get_heavy_augmentation
        ]:
            params = method()
            for field in required_fields:
                assert field in params, f"{field} missing in {method.__name__}"
