"""
Unit tests for EchoForgeAgent
"""
import pytest
import tempfile
import yaml
import os
from unittest.mock import patch, MagicMock
from src.agents.echoForge.agent import EchoForgeAgent


class TestEchoForgeAgent:
    """Test cases for EchoForgeAgent class"""
    
    @patch('src.agents.echoForge.memory.OpenAIEmbeddings')
    @patch('src.agents.echoForge.agent.ChatOpenAI')
    def test_agent_init_with_valid_modes(self, mock_chat_openai, mock_embeddings):
        """Test agent initialization with valid modes"""
        # Mock the OpenAI dependencies
        mock_chat_openai.return_value = MagicMock()
        mock_embeddings.return_value = MagicMock()
        
        # Test copilot mode
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({"mode": "copilot"}, f)
            temp_path = f.name
        
        try:
            agent = EchoForgeAgent(temp_path)
            assert agent.config.mode == "copilot"
            assert agent.graph is not None
        finally:
            os.unlink(temp_path)
        
        # Test echo mode
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({"mode": "echo"}, f)
            temp_path = f.name
        
        try:
            agent = EchoForgeAgent(temp_path)
            assert agent.config.mode == "echo"
            assert agent.graph is not None
        finally:
            os.unlink(temp_path)
    
    @patch('src.agents.echoForge.memory.OpenAIEmbeddings')
    @patch('src.agents.echoForge.agent.ChatOpenAI')
    def test_agent_init_with_invalid_mode(self, mock_chat_openai, mock_embeddings):
        """Test agent initialization with invalid mode raises error"""
        # Mock the OpenAI dependencies
        mock_chat_openai.return_value = MagicMock()
        mock_embeddings.return_value = MagicMock()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({"mode": "invalid_mode"}, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                EchoForgeAgent(temp_path)
            
            assert "Invalid mode 'invalid_mode'" in str(exc_info.value)
            assert "Must be one of: copilot, echo" in str(exc_info.value)
        finally:
            os.unlink(temp_path)
    
    @patch('src.agents.echoForge.memory.OpenAIEmbeddings')
    @patch('src.agents.echoForge.agent.ChatOpenAI')
    def test_agent_init_with_empty_mode(self, mock_chat_openai, mock_embeddings):
        """Test agent initialization with empty mode raises error"""
        # Mock the OpenAI dependencies
        mock_chat_openai.return_value = MagicMock()
        mock_embeddings.return_value = MagicMock()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({"mode": ""}, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                EchoForgeAgent(temp_path)
            
            assert "Invalid mode ''" in str(exc_info.value)
        finally:
            os.unlink(temp_path)
    
    @patch('src.agents.echoForge.memory.OpenAIEmbeddings')
    @patch('src.agents.echoForge.agent.ChatOpenAI')
    def test_agent_init_with_none_mode(self, mock_chat_openai, mock_embeddings):
        """Test agent initialization with None mode raises error"""
        # Mock the OpenAI dependencies
        mock_chat_openai.return_value = MagicMock()
        mock_embeddings.return_value = MagicMock()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({"mode": None}, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                EchoForgeAgent(temp_path)
            
            assert "Invalid mode 'None'" in str(exc_info.value)
        finally:
            os.unlink(temp_path)
    
    @patch('src.agents.echoForge.memory.FAISS')
    @patch('src.agents.echoForge.memory.OpenAIEmbeddings')
    @patch('src.agents.echoForge.agent.ChatOpenAI')
    def test_agent_chat_method(self, mock_chat_openai, mock_embeddings, mock_faiss):
        """Test agent chat method returns None (interactive mode)"""
        # Mock the OpenAI dependencies
        mock_chat_openai.return_value = MagicMock()
        mock_embeddings.return_value = MagicMock()
        mock_faiss.from_documents.return_value = MagicMock()
        
        agent = EchoForgeAgent()
        response = agent.chat()
        
        # Chat method returns None for interactive mode
        assert response is None
    
    @patch('src.agents.echoForge.memory.FAISS')
    @patch('src.agents.echoForge.memory.OpenAIEmbeddings')
    @patch('src.agents.echoForge.agent.ChatOpenAI')
    def test_agent_components_initialized(self, mock_chat_openai, mock_embeddings, mock_faiss):
        """Test that all agent components are properly initialized"""
        # Mock the OpenAI dependencies
        mock_chat_openai.return_value = MagicMock()
        mock_embeddings.return_value = MagicMock()
        mock_faiss.from_documents.return_value = MagicMock()
        
        agent = EchoForgeAgent()
        
        # Check config
        assert agent.config is not None
        assert hasattr(agent.config, 'mode')
        
        # Check memory
        assert agent.memory is not None
        
        # Check prompts
        assert agent.copilot_prompts is not None
        
        # Check graph
        assert agent.graph is not None
    
    @patch('src.agents.echoForge.memory.FAISS')
    @patch('src.agents.echoForge.memory.OpenAIEmbeddings')
    @patch('src.agents.echoForge.agent.ChatOpenAI')
    def test_agent_with_default_config(self, mock_chat_openai, mock_embeddings, mock_faiss):
        """Test agent initialization with default config file"""
        # Mock the OpenAI dependencies
        mock_chat_openai.return_value = MagicMock()
        mock_embeddings.return_value = MagicMock()
        mock_faiss.from_documents.return_value = MagicMock()
        
        # This should work with the default config/echoforge.yaml
        agent = EchoForgeAgent()
        assert agent.config.mode in ["copilot", "echo"]
        assert agent.graph is not None
