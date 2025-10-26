"""
Unit tests for EchoForgeConfig
"""
import pytest
import tempfile
import os
import yaml
from src.agents.echoForge.config import EchoForgeConfig


class TestEchoForgeConfig:
    """Test cases for EchoForgeConfig class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = EchoForgeConfig()
        
        assert config.mode == "copilot"
        assert config.llm_model == "gpt-4"
        assert config.llm_temperature == 0.7
        assert config.max_conversation_history == 10
        assert config.confidence_threshold == 0.7
        assert config.data_dir == "data/echoForge"
        assert config.logs_dir == "logs"
        assert config.cache_dir == "cache"
    
    def test_config_from_file_exists(self):
        """Test loading config from existing file"""
        # Create temporary config file
        config_data = {
            "mode": "copilot",
            "llm_model": "gpt-3.5-turbo",
            "llm_temperature": 0.5,
            "max_conversation_history": 20,
            "confidence_threshold": 0.8,
            "data_dir": "custom_data",
            "logs_dir": "custom_logs",
            "cache_dir": "custom_cache"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = EchoForgeConfig.from_file(temp_path)
            
            assert config.mode == "copilot"
            assert config.llm_model == "gpt-3.5-turbo"
            assert config.llm_temperature == 0.5
            assert config.max_conversation_history == 20
            assert config.confidence_threshold == 0.8
            assert config.data_dir == "custom_data"
            assert config.logs_dir == "custom_logs"
            assert config.cache_dir == "custom_cache"
        finally:
            os.unlink(temp_path)
    
    def test_config_from_file_not_exists(self):
        """Test loading config from non-existent file returns defaults"""
        config = EchoForgeConfig.from_file("non_existent_config.yaml")
        
        # Should return default values
        assert config.mode == "copilot"
        assert config.llm_model == "gpt-4"
        assert config.llm_temperature == 0.7
    
    def test_save_to_file(self):
        """Test saving config to file"""
        config = EchoForgeConfig()
        config.mode = "copilot"
        config.llm_model = "gpt-3.5-turbo"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "test_config.yaml")
            config.save_to_file(config_path)
            
            # Verify file was created
            assert os.path.exists(config_path)
            
            # Verify content
            with open(config_path, 'r') as f:
                saved_data = yaml.safe_load(f)
            
            assert saved_data["mode"] == "copilot"
            assert saved_data["llm_model"] == "gpt-3.5-turbo"
            assert saved_data["llm_temperature"] == 0.7
    
    def test_save_to_file_creates_directory(self):
        """Test that save_to_file creates parent directories if they don't exist"""
        config = EchoForgeConfig()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested directory path
            nested_dir = os.path.join(temp_dir, "nested", "subdir")
            config_path = os.path.join(nested_dir, "config.yaml")
            
            config.save_to_file(config_path)
            
            # Verify directory was created and file exists
            assert os.path.exists(nested_dir)
            assert os.path.exists(config_path)
    
    def test_config_roundtrip(self):
        """Test saving and loading config maintains all values"""
        original_config = EchoForgeConfig()
        original_config.mode = "copilot"
        original_config.llm_model = "custom-model"
        original_config.llm_temperature = 0.9
        original_config.max_conversation_history = 15
        original_config.confidence_threshold = 0.6
        original_config.data_dir = "test_data"
        original_config.logs_dir = "test_logs"
        original_config.cache_dir = "test_cache"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "roundtrip_config.yaml")
            
            # Save config
            original_config.save_to_file(config_path)
            
            # Load config
            loaded_config = EchoForgeConfig.from_file(config_path)
            
            # Verify all values match
            assert loaded_config.mode == original_config.mode
            assert loaded_config.llm_model == original_config.llm_model
            assert loaded_config.llm_temperature == original_config.llm_temperature
            assert loaded_config.max_conversation_history == original_config.max_conversation_history
            assert loaded_config.confidence_threshold == original_config.confidence_threshold
            assert loaded_config.data_dir == original_config.data_dir
            assert loaded_config.logs_dir == original_config.logs_dir
            assert loaded_config.cache_dir == original_config.cache_dir
