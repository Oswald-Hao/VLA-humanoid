"""
Configuration for VLA Robot Learning Framework
"""
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np


class Config:
    """Configuration class for VLA framework"""
    
    # Project structure
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    TRAJECTORIES_DIR = PROJECT_ROOT / "trajectories"
    MODELS_DIR = PROJECT_ROOT / "models"
    LOGS_DIR = PROJECT_ROOT / "logs"
    CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
    
    # MuJoCo Configuration
    MUJOCO = {
        "model_path": PROJECT_ROOT / "models" / "biped_s45" / "xml" / "biped_s45.xml",
        "timestep": 0.02,  # 50Hz
        "gravity": [0, 0, -9.81],
        "integrator": "Euler",
    }
    
    # Robot Configuration (Kuavo S45)
    ROBOT = {
        "num_joints": 26,
        "joint_limits": {
            "min": np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,
                             -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,
                             -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,
                             -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,
                             -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,
                             -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,
                             -2.8973, -1.7628, -2.8973, -3.0718, -2.8973]),
            "max": np.array([2.8973, 1.7628, 2.8973, 3.0718, 2.8973, 3.7525, 2.8973,
                             2.8973, 1.7628, 2.8973, 3.0718, 2.8973, 3.7525, 2.8973,
                             2.8973, 1.7628, 2.8973, 3.0718, 2.8973, 3.7525, 2.8973,
                             2.8973, 1.7628, 2.8973, 3.0718, 2.8973, 3.7525, 2.8973,
                             2.8973, 1.7628, 2.8973, 3.0718, 2.8973, 3.7525, 2.8973,
                             2.8973, 1.7628, 2.8973, 3.0718, 2.8973, 3.7525, 2.8973,
                             2.8973, 1.7628, 2.8973, 3.0718, 2.8973])
        },
        "actuator_limits": {
            "min": -87.0,
            "max": 87.0
        }
    }
    
    # Perception Configuration
    PERCEPTION = {
        "image_size": (224, 224),
        "image_channels": 3,
        "camera_height": 1.5,
        "camera_distance": 2.0,
        "camera_angle": 45.0,
        "state_dim": 59,  # joints (26) + velocities (26) + body_pos (3) + body_quat (4)
    }
    
    # Simulation Configuration
    SIMULATION = {
        "enabled": True,
        "frequency": 50.0,  # Hz
        "max_duration": 10.0,  # seconds
        "real_time": True,
    }
    
    # Instructions Configuration
    INSTRUCTIONS = [
        "前进", "后退"
    ]
    
    # Skills Configuration
    SKILLS = {
        "types": ["walk", "turn", "wave", "bow", "stand", "custom"],
        "default_duration": 5.0,
        "default_frequency": 25.0,
    }
    
    # Algorithm Configuration
    ALGORITHM = {
        # System-2 Planner
        "planner": {
            "instruction_vocab_size": 1000,
            "instruction_embed_dim": 128,
            "lstm_hidden_dim": 128,
            "obs_feature_dim": 384,  # CNN 256 + MLP 128
            "hidden_dims": [256, 128],
            "num_skills": 6,
            "param_dim": 32,
        },
        
        # System-1 Executor
        "executor": {
            "plan_embed_dim": 64,
            "hidden_dims": [256, 128],
        },
        
        # Training
        "training": {
            "learning_rate": 1e-3,
            "batch_size": 8,
            "num_epochs": 500,
            "weight_decay": 1e-4,
            "dropout": 0.1,
            "lambda_param": 0.1,  # MSE loss weight
            "early_stopping_patience": 10,
            "save_all_checkpoints": False,  # 是否保存所有checkpoint
        }
    }
    
    # Data Pipeline Configuration
    DATA = {
        "train_split": 0.7,
        "val_split": 0.15,
        "test_split": 0.15,
        "augmentation": {
            "enabled": True,
            "rotation_range": 15,
            "erase_prob": 0.3,
            "erase_scale": (0.02, 0.33),
            "noise_std": 0.01,
        },
        "cleaning": {
            "enabled": True,
            "velocity_threshold": 10.0,
            "position_threshold": 5.0,
        }
    }
    
    # Training Configuration
    TRAINING = {
        "device": "auto",  # auto, cpu, cuda
        "num_workers": 4,
        "pin_memory": True,
        "shuffle": True,
        "random_seed": 42,
        "checkpoint_frequency": 5,  # epochs
    }
    
    # Model Configuration
    MODEL = {
        "save_format": "pth",
        "save_best_only": True,
        "export_formats": ["torchscript", "onnx"],
    }
    
    # Visualization Configuration
    VISUALIZATION = {
        "enabled": True,
        "tensorboard_dir": LOGS_DIR / "tensorboard",
        "plot_dir": LOGS_DIR / "plots",
        "save_plots": True,
        "plot_frequency": 1,  # epochs
    }
    
    # Logging Configuration
    LOGGING = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": LOGS_DIR / "training.log",
        "max_file_size": "10MB",
        "backup_count": 5,
    }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        directories = [
            cls.DATA_DIR,
            cls.TRAJECTORIES_DIR,
            cls.MODELS_DIR,
            cls.LOGS_DIR,
            cls.CHECKPOINTS_DIR,
            cls.VISUALIZATION["tensorboard_dir"],
            cls.VISUALIZATION["plot_dir"],
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_device(cls) -> str:
        """Get training device"""
        if cls.TRAINING["device"] == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        else:
            return cls.TRAINING["device"]
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert to dictionary"""
        config_dict = {}
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and not callable(value):
                if isinstance(value, Path):
                    config_dict[key] = str(value)
                elif isinstance(value, dict):
                    config_dict[key] = cls._convert_dict(value)
                else:
                    config_dict[key] = value
        return config_dict
    
    @classmethod
    def _convert_dict(cls, d: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively convert dictionary"""
        converted = {}
        for key, value in d.items():
            if isinstance(value, Path):
                converted[key] = str(value)
            elif isinstance(value, dict):
                converted[key] = cls._convert_dict(value)
            elif isinstance(value, (list, tuple)):
                converted[key] = [cls._convert_item(item) for item in value]
            else:
                converted[key] = value
        return converted
    
    @classmethod
    def _convert_item(cls, item: Any) -> Any:
        """Convert single item"""
        if isinstance(item, Path):
            return str(item)
        elif isinstance(item, dict):
            return cls._convert_dict(item)
        elif isinstance(item, (list, tuple)):
            return [cls._convert_item(i) for item in item]
        else:
            return item


# Global configuration instance
config = Config()

# Create necessary directories
config.create_directories()

# Expose common attributes at module level for backward compatibility
create_directories = config.create_directories
get_device = config.get_device
TRAINING = config.TRAINING
TRAJECTORIES_DIR = config.TRAJECTORIES_DIR
MODELS_DIR = config.MODELS_DIR
PERCEPTION = config.PERCEPTION