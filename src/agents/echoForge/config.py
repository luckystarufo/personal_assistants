"""
EchoForge Configuration Management
"""
from dataclasses import dataclass
from typing import Dict, Any
import yaml
import os


@dataclass
class EchoForgeConfig:
    """Configuration for EchoForge agent"""
    mode: str = "copilot"  # "copilot", "echo"
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.7
    max_conversation_history: int = 10
    confidence_threshold: float = 0.7
    data_dir: str = "data/echoForge"
    logs_dir: str = "logs"
    cache_dir: str = "cache"
    
    @classmethod
    def from_file(cls, config_path: str) -> 'EchoForgeConfig':
        """Load configuration from YAML file"""
        print(f"[CONFIG] Loading config from {config_path}")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
            return cls(**data)
        else:
            print(f"[CONFIG] Config file not found, using defaults")
            return cls()
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to YAML file"""
        print(f"[CONFIG] Saving config to {config_path}")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
