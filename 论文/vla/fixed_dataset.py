#!/usr/bin/env python3
"""
修复版轨迹数据集
解决索引越界问题，支持直接动作预测训练
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import json
import numpy as np
from PIL import Image
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import sys
from collections import defaultdict
import random

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config import config


class FixedTrajectoryDataset(Dataset):
    """
    修复版轨迹数据集 - 解决索引越界问题
    
    主要修复：
    - 正确处理数据集大小和索引
    - 支持直接动作预测训练
    - 改进数据预处理一致性
    """
    
    def __init__(self, data_dir: str, split: str = 'train', 
                 augment: bool = True, max_seq_len: int = 20,
                 sequence_length: int = 16,
                 history_window_size: int = 5,
                 predict_absolute_angles: bool = True,
                 trajectories: List[Dict[str, Any]] = None,
                 vocab: Dict[str, int] = None):
        """
        初始化修复版轨迹数据集
        
        Args:
            data_dir: 数据目录路径
            split: 数据分割 ('train', 'val', 'test')
            augment: 是否进行数据增强
            max_seq_len: 最大序列长度
            sequence_length: 序列长度
            history_window_size: 历史窗口大小
            predict_absolute_angles: 是否预测绝对角度
            trajectories: 预加载的轨迹数据（用于数据分割）
            vocab: 预构建的词汇表（用于数据分割）
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.augment = augment
        self.max_seq_len = max_seq_len
        self.sequence_length = sequence_length
        self.history_window_size = history_window_size
        self.predict_absolute_angles = predict_absolute_angles
        
        # 加载轨迹数据
        if trajectories is not None:
            self.trajectories = trajectories
        else:
            self.trajectories = self._load_trajectories()
        
        # 构建词汇表
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)
        
        # 图像预处理
        self.image_transform = self._get_image_transform()
        
        # 统计信息
        self.skill_counts = self._count_skills()
        self._compute_data_statistics()
        
        # 创建序列索引 - 这是修复的关键
        self.sequence_indices = self._create_sequence_indices()
        
        print(f"Loaded {len(self.trajectories)} trajectories for {split} split")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Skill distribution: {self.skill_counts}")
        print(f"Total sequences: {len(self.sequence_indices)}")
        print(f"History window size: {self.history_window_size}")
        print(f"Predict absolute angles: {self.predict_absolute_angles}")
        if hasattr(self, 'action_stats'):
            print(f"Action stats - Mean: {self.action_stats['mean']:.4f}, Std: {self.action_stats['std']:.4f}")
    
    def _create_sequence_indices(self) -> List[Tuple[int, int]]:
        """创建序列索引列表，避免索引越界"""
        indices = []
        
        for traj_idx, trajectory in enumerate(self.trajectories):
            observations = trajectory.get('observations', [])
            actions = trajectory.get('actions', [])
            
            # 确保观测和动作数量匹配
            min_frames = min(len(observations), len(actions))
            
            if min_frames < self.sequence_length:
                # 如果轨迹不够长，跳过或使用填充
                continue
            
            # 计算可以生成多少个序列
            num_sequences = min_frames - self.sequence_length + 1
            
            # 限制每个轨迹的序列数量，避免某些轨迹过多
            max_sequences_per_traj = 20
            num_sequences = min(num_sequences, max_sequences_per_traj)
            
            for seq_idx in range(num_sequences):
                indices.append((traj_idx, seq_idx))
        
        return indices
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.sequence_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取序列数据样本 - 修复版
        
        Returns:
            Dict[str, torch.Tensor]: 包含以下键的字典：
                - instruction_ids: 指令ID
                - state_sequence: 状态序列 [seq_len, state_dim]
                - action_sequences: 动作序列 [seq_len, action_dim]
                - skill_id: 技能ID
        """
        if idx >= len(self.sequence_indices):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.sequence_indices)}")
        
        # 获取轨迹和序列索引
        traj_idx, seq_idx = self.sequence_indices[idx]
        
        trajectory = self.trajectories[traj_idx]
        observations = trajectory.get('observations', [])
        actions = trajectory.get('actions', [])
        
        # 处理指令
        instruction = trajectory.get('instruction', '')
        skill_id = trajectory.get('skill_id', 'unknown')
        instruction_id = self._instruction_to_id(instruction, skill_id)
        instruction_tensor = torch.tensor(instruction_id, dtype=torch.long)
        
        # 提取序列数据
        start_idx = seq_idx
        end_idx = start_idx + self.sequence_length
        
        # 确保索引在有效范围内
        end_idx = min(end_idx, len(observations), len(actions))
        actual_seq_len = end_idx - start_idx
        
        # 处理状态序列
        state_sequence = []
        action_sequence = []
        
        for i in range(actual_seq_len):
            obs_idx = start_idx + i
            if obs_idx < len(observations) and obs_idx < len(actions):
                obs = observations[obs_idx]
                action = actions[obs_idx]
                
                # 处理状态
                if 'joint_pos' in obs and len(obs['joint_pos']) >= 26:
                    joint_pos = np.array(obs['joint_pos'][:26], dtype=np.float32)
                    
                    # 添加时间戳
                    t_norm = obs_idx / max(len(observations) - 1, 1)
                    state = np.concatenate([joint_pos, [t_norm]])
                    
                    # 填充到60维
                    if len(state) < 60:
                        state = np.pad(state, (0, 60 - len(state)), 'constant')
                    
                    state_sequence.append(state)
                
                # 处理动作
                if len(action) >= 26:
                    action_array = np.array(action[:26], dtype=np.float32)
                    
                    # 数据增强
                    if self.augment and self.split == 'train':
                        action_array = self._augment_action(action_array)
                    
                    action_sequence.append(action_array)
        
        # 转换为张量
        if state_sequence:
            state_tensor = torch.stack([torch.tensor(s, dtype=torch.float32) for s in state_sequence])
        else:
            state_tensor = torch.zeros(self.sequence_length, 60, dtype=torch.float32)
        
        if action_sequence:
            action_tensor = torch.stack([torch.tensor(a, dtype=torch.float32) for a in action_sequence])
        else:
            action_tensor = torch.zeros(self.sequence_length, 26, dtype=torch.float32)
        
        # 填充到固定长度
        if state_tensor.shape[0] < self.sequence_length:
            padding = self.sequence_length - state_tensor.shape[0]
            state_tensor = F.pad(state_tensor, (0, 0, 0, padding), 'constant', 0)
            action_tensor = F.pad(action_tensor, (0, 0, 0, padding), 'constant', 0)
        
        # 返回数据字典
        return {
            'instruction_ids': instruction_tensor,
            'state_sequence': state_tensor,
            'action_sequences': action_tensor,
            'skill_id': instruction_tensor  # 使用相同的ID
        }
    
    def _load_trajectories(self) -> List[Dict[str, Any]]:
        """加载轨迹数据"""
        trajectories = []
        
        # 遍历数据目录
        for file_path in self.data_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    trajectory = json.load(f)
                
                # 验证数据格式
                if 'observations' in trajectory and 'actions' in trajectory:
                    trajectories.append(trajectory)
                    
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                continue
        
        return trajectories
    
    def _build_vocab(self) -> Dict[str, int]:
        """构建指令词汇表"""
        skills = set()
        for trajectory in self.trajectories:
            skill_id = trajectory.get('skill_id', 'unknown')
            skills.add(skill_id)
        
        return {skill: idx for idx, skill in enumerate(sorted(skills))}
    
    def _instruction_to_id(self, instruction: str, skill_id: str) -> int:
        """将指令转换为ID"""
        return self.vocab.get(skill_id, 0)
    
    def _count_skills(self) -> Dict[int, int]:
        """统计技能分布"""
        skill_counts = defaultdict(int)
        for trajectory in self.trajectories:
            skill_id = trajectory.get('skill_id', 'unknown')
            skill_counts[self.vocab.get(skill_id, 0)] += 1
        return dict(skill_counts)
    
    def _compute_data_statistics(self):
        """计算数据统计信息"""
        all_actions = []
        
        for trajectory in self.trajectories:
            actions = trajectory.get('actions', [])
            for action in actions:
                if len(action) >= 26:
                    all_actions.append(action[:26])
        
        if all_actions:
            all_actions = np.array(all_actions)
            self.action_stats = {
                'mean': float(np.mean(all_actions)),
                'std': float(np.std(all_actions)),
                'min': float(np.min(all_actions)),
                'max': float(np.max(all_actions))
            }
        else:
            self.action_stats = {
                'mean': 0.0,
                'std': 1.0,
                'min': -1.0,
                'max': 1.0
            }
    
    def _get_image_transform(self):
        """获取图像预处理变换"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _augment_action(self, action: np.ndarray) -> np.ndarray:
        """数据增强 - 动作序列"""
        # 添加高斯噪声
        noise = np.random.normal(0, 0.01, action.shape)
        action = action + noise
        
        # 随机缩放
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.9, 1.1)
            action = action * scale
        
        # 随机时间偏移（对序列中的某些帧）
        if np.random.random() < 0.2:
            offset = np.random.normal(0, 0.005, action.shape)
            action = action + offset
        
        return action