#!/usr/bin/env python3
"""
VLA项目统一训练文件 - 包含所有训练相关的代码
符合CRITICAL PRINCIPLE的两层架构：指令分类 + 动作预测
集成了动作质量验证系统
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import json
from typing import Dict, Optional, List, Any
import sys
import argparse

# 导入必要的模块
from fixed_dataset import FixedTrajectoryDataset
from action_quality_validator import ActionQualityValidator, TrainingQualityMonitor

# ==============================================
# 1. 核心模型架构 - 关键关节专注的ACT模型
# ==============================================

class KeyJointACTGenerator(nn.Module):
    """
    关键关节专注的ACT生成器
    核心思想：自动识别每个指令的关键关节，重点学习这些关节的变化
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # 基础参数
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.num_instructions = config['num_instructions']
        self.hidden_dim = config['hidden_dim']
        self.trajectory_length = 32  # 回退：从128改回32，原始长度
        self.dropout = config['dropout']
        
        # 关键关节数量 - 每个指令重点关注的前N个关节
        self.key_joints_per_instruction = config.get('key_joints_per_instruction', 16)  # 阶段3：从8增加到16，支持通用动作框架
        
        
        # 第一层：指令分类器
        self.instruction_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, self.num_instructions)
        )
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # 指令嵌入
        self.instruction_embedding = nn.Embedding(self.num_instructions, 64)
        
        # 时间编码
        self.time_encoding = nn.Sequential(
            nn.Linear(1, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, self.hidden_dim // 4)
        )
        
        # 时序编码器 - 使用Transformer更好地处理时序依赖
        temporal_input_size = self.hidden_dim + 64 + self.hidden_dim // 4  # state + instruction + time
        
        # 关节重要性分析器 - 为每个指令分析关节重要性
        self.joint_importance_analyzer = nn.Sequential(
            nn.Linear(self.hidden_dim + 64, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.action_dim),
            nn.Sigmoid()  # 输出每个关节的重要性权重
        )
        
        # 第二层：指令专用的关键关节预测器
        self.key_joint_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(temporal_input_size, 256),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(128, self.key_joints_per_instruction)  # 只预测关键关节
            ) for _ in range(self.num_instructions)
        ])
        
        # 完整关节输出层 - 从关键关节扩展到所有关节
        self.full_joint_expander = nn.Sequential(
            nn.Linear(self.key_joints_per_instruction, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.action_dim)
        )
        
        # 时序编码器
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=temporal_input_size,
                nhead=8,
                dim_feedforward=self.hidden_dim,
                dropout=self.dropout,
                batch_first=True
            ),
            num_layers=3
        )
        
        
        
        # 动作历史上下文支持 - 更长的历史长度
        self.history_length = config.get('history_length', 128)  # 历史步数，128步提供更充分的上下文
        self.history_encoding = nn.Sequential(
            nn.Linear(self.action_dim * self.history_length, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化 - 使用更保守的初始化策略"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # 使用更小的初始化范围，防止早期训练不稳定
                if module.weight.dim() >= 2:  # 确保至少是2维张量
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.05)
            elif isinstance(module, nn.TransformerEncoderLayer):
                for param_name, param in module.named_parameters():
                    if 'weight' in param_name and param.dim() >= 2:
                        nn.init.xavier_uniform_(param, gain=0.5)
                    elif 'bias' in param_name:
                        nn.init.constant_(param, 0.0)
    
    def forward(self, start_states, instruction_ids, target_actions=None, action_history=None):
        """前向传播 - 支持动作历史上下文"""
        batch_size = start_states.size(0)
        device = start_states.device
        
        # 状态编码
        state_encoded = self.state_encoder(start_states)
        
        # 第一层：指令分类
        instruction_logits = self.instruction_classifier(state_encoded)
        
        # 指令嵌入
        instruction_emb = self.instruction_embedding(instruction_ids)
        
        # 处理动作历史上下文
        if action_history is not None:
            # 编码动作历史
            history_encoded = self.history_encoding(action_history)
            # 将历史信息融合到状态编码中
            state_encoded = state_encoded + history_encoded
        
        # 分析关节重要性
        joint_importance_input = torch.cat([state_encoded, instruction_emb], dim=-1)
        joint_importance = self.joint_importance_analyzer(joint_importance_input)
        
        # 时间编码
        time_steps = torch.linspace(0, 1, self.trajectory_length, device=device)
        time_embed = self.time_encoding(time_steps.unsqueeze(-1)).unsqueeze(0).expand(batch_size, -1, -1)
        
        # 准备时序输入
        state_expanded = state_encoded.unsqueeze(1).expand(-1, self.trajectory_length, -1)
        instruction_expanded = instruction_emb.unsqueeze(1).expand(-1, self.trajectory_length, -1)
        
        temporal_input = torch.cat([state_expanded, instruction_expanded, time_embed], dim=-1)
        
        # 时序编码
        temporal_output = self.temporal_encoder(temporal_input)
        
        # 第二层：指令专用的关键关节预测
        key_joint_actions = []
        for i in range(batch_size):
            instruction_id = instruction_ids[i].item()
            predictor = self.key_joint_predictors[instruction_id]
            
            # 预测关键关节
            key_action = predictor(temporal_output[i])
            key_joint_actions.append(key_action)
        
        key_joint_actions = torch.stack(key_joint_actions, dim=0)
        
        # 扩展到完整关节输出
        full_joint_actions = []
        for t in range(self.trajectory_length):
            key_at_t = key_joint_actions[:, t, :]
            full_at_t = self.full_joint_expander(key_at_t)
            full_joint_actions.append(full_at_t)
        
        predicted_actions = torch.stack(full_joint_actions, dim=1)
        
        return predicted_actions, instruction_logits, joint_importance, key_joint_actions
    

# ==============================================
# 2. 数据集类
# ==============================================

class KeyJointDataset(Dataset):
    """关键关节数据集 - 平衡采样策略"""
    
    def __init__(self, data_dir: str, file_names: list, trajectory_length: int = 32):
        self.data_dir = data_dir
        self.file_names = file_names
        self.trajectory_length = trajectory_length
        
        # 指令映射
        self.instruction_to_id = {'wave': 0, 'welcome': 1}
        
        # 使用改进的平衡采样策略
        self.samples = self._balanced_sampling_strategy()
        
        print(f"采样结果: 总样本数 {len(self.samples)}")
        instruction_counts = {}
        for sample in self.samples:
            instruction = sample[0]  # 第一个元素始终是指令
            instruction_counts[instruction] = instruction_counts.get(instruction, 0) + 1
        for instruction, count in instruction_counts.items():
            print(f"  {instruction}: {count} 个样本")
    
    def _balanced_sampling_strategy(self):
        """增强的采样策略 - 支持任意起始位置训练"""
        samples = []
        
        # 按指令分组文件
        instruction_files = {'wave': [], 'welcome': []}
        for file_name in self.file_names:
            if 'wave' in file_name:
                instruction_files['wave'].append(file_name)
            elif 'welcome' in file_name:
                instruction_files['welcome'].append(file_name)
        
        # 对每个指令类型进行采样
        for instruction, files in instruction_files.items():
            instruction_samples = []
            
            for file_name in files:
                trajectory_path = os.path.join(self.data_dir, file_name)
                with open(trajectory_path, 'r') as f:
                    data = json.load(f)
                
                observations = data['observations']
                total_frames = len(observations)
                
                # 根据动作特性调整采样策略
                if instruction == 'wave':
                    # 挥手动作：周期性较长，使用较稀疏采样
                    # 目标：每个文件约60个样本
                    target_samples = 60
                    if total_frames > self.trajectory_length:
                        step = max(1, (total_frames - self.trajectory_length) // target_samples)
                        start_indices = list(range(0, total_frames - self.trajectory_length, step))
                    else:
                        start_indices = [0]
                elif instruction == 'welcome':
                    # 抱拳动作：时间较短，使用较密集采样
                    # 目标：每个文件约50个样本
                    target_samples = 50
                    if total_frames > self.trajectory_length:
                        step = max(1, (total_frames - self.trajectory_length) // target_samples)
                        start_indices = list(range(0, total_frames - self.trajectory_length, step))
                    else:
                        start_indices = [0]
                else:
                    # 默认策略
                    if total_frames > self.trajectory_length:
                        start_indices = list(range(0, total_frames - self.trajectory_length, 10))
                    else:
                        start_indices = [0]
                
                for start_idx in start_indices:
                    instruction_samples.append((instruction, file_name, start_idx))
            
            samples.extend(instruction_samples)
        
        # 新增：添加随机起始位置扰动，增强泛化能力
        enhanced_samples = self._add_arbitrary_start_samples(samples)
        
        return enhanced_samples
    
    def _add_arbitrary_start_samples(self, base_samples):
        """添加简化的任意起始位置样本"""
        enhanced_samples = []
        
        for sample in base_samples:
            instruction, file_name, start_idx = sample
            
            # 添加原始样本
            enhanced_samples.append(sample)
            
            # 只对10%的原始样本添加任意起始样本（更低比例）
            if np.random.random() < 0.1:
                # 加载完整轨迹数据
                trajectory_path = os.path.join(self.data_dir, file_name)
                with open(trajectory_path, 'r') as f:
                    data = json.load(f)
                
                observations = data['observations']
                joint_positions = np.array([obs['joint_pos'] for obs in observations])
                total_frames = len(joint_positions)
                
                if total_frames > self.trajectory_length * 2:
                    # 只生成1个任意起始样本
                    mid_start = self.trajectory_length
                    mid_end = total_frames - self.trajectory_length
                    
                    if mid_end > mid_start:
                        # 真正的任意起始位置
                        arbitrary_start = np.random.randint(mid_start, mid_end)
                        
                        # 获取真实的当前状态
                        current_state = joint_positions[arbitrary_start]
                        
                        # 获取从当前状态开始的目标轨迹
                        target_trajectory = joint_positions[arbitrary_start:arbitrary_start + self.trajectory_length]
                        
                        # 确保目标轨迹长度正确
                        if len(target_trajectory) == self.trajectory_length:
                            # 创建状态转移学习样本
                            enhanced_samples.append((
                                instruction, file_name, arbitrary_start, 
                                tuple(current_state), 'state_transfer'
                            ))
        
        return enhanced_samples
    
    def _compute_normalized_action_pattern(self, trajectory, start_state):
        """计算归一化的动作模式 - 增强模型对动作本质的学习"""
        # 计算相对于起始位置的位移
        displacements = trajectory - start_state
        
        # 计算动作的幅度范围
        joint_ranges = np.max(np.abs(displacements), axis=0)
        joint_ranges[joint_ranges == 0] = 1.0  # 避免除零
        
        # 归一化位移模式
        normalized_pattern = displacements / joint_ranges
        
        # 添加时间进度信息
        time_progress = np.linspace(0, 1, len(trajectory))
        time_encoding = np.sin(time_progress * np.pi)  # 使用正弦函数编码时间进度
        
        # 组合归一化模式和时间信息
        enhanced_pattern = np.concatenate([
            normalized_pattern.flatten(),
            time_encoding
        ])
        
        return enhanced_pattern
    
    def __len__(self):
        return len(self.samples)
    
    def analyze_joint_importance(self, joint_positions):
        """分析关节重要性 - 基于变化幅度和范围的综合评估"""
        # 计算每个关节的标准差作为变化量指标
        joint_std = np.std(joint_positions, axis=0)
        
        # 计算每个关节的变化范围
        joint_range = np.max(joint_positions, axis=0) - np.min(joint_positions, axis=0)
        
        # 综合评估：变化量 + 变化范围
        joint_importance_raw = 0.7 * joint_std + 0.3 * joint_range
        
        # 使用softmax进行归一化，保持区分度
        if np.max(joint_importance_raw) > 0:
            # 使用指数函数增强重要关节的权重
            joint_importance = np.exp(joint_importance_raw * 2) / np.sum(np.exp(joint_importance_raw * 2))
        else:
            joint_importance = np.ones_like(joint_importance_raw) / len(joint_importance_raw)
        
        return joint_importance
    
    def sample_trajectory_segment(self, joint_positions, start_idx, trajectory_length):
        """采样轨迹段"""
        total_frames = len(joint_positions)
        
        if total_frames > trajectory_length:
            # 确保不超出范围
            end_idx = min(start_idx + trajectory_length, total_frames)
            segment = joint_positions[start_idx:end_idx]
            
            # 如果长度不够，填充最后一帧
            if len(segment) < trajectory_length:
                last_frame = segment[-1]
                padding = np.tile(last_frame, (trajectory_length - len(segment), 1))
                segment = np.vstack([segment, padding])
        else:
            # 数据不够长，使用整个轨迹并填充
            segment = joint_positions
            if len(segment) < trajectory_length:
                last_frame = segment[-1]
                padding = np.tile(last_frame, (trajectory_length - len(segment), 1))
                segment = np.vstack([segment, padding])
        
        return segment[:trajectory_length]
    
    def _generate_action_history(self, joint_positions, start_idx, history_length=128, current_state=None):
        """生成动作历史上下文 - 支持任意起始位置"""
        # 如果提供了当前状态（用于任意起始位置推理）
        if current_state is not None:
            # 计算相对于原始轨迹的偏移
            original_start = joint_positions[0] if len(joint_positions) > 0 else current_state
            state_offset = current_state - original_start
            
            # 获取历史轨迹段
            history_start = max(0, start_idx - history_length)
            history_segment = joint_positions[history_start:start_idx]
            
            # 对历史轨迹应用相同的偏移
            if len(history_segment) > 0:
                history_segment = history_segment + state_offset
            
            # 历史不足时用偏移后的初始状态填充
            if len(history_segment) < history_length:
                padded_start = original_start + state_offset
                padding = np.tile(padded_start, (history_length - len(history_segment), 1))
                history_segment = np.vstack([padding, history_segment])
        else:
            # 原有逻辑：用于正常训练
            history_start = max(0, start_idx - history_length)
            history_segment = joint_positions[history_start:start_idx]
            
            if len(history_segment) < history_length:
                if len(history_segment) > 0:
                    initial_state = joint_positions[0]
                    padding = np.tile(initial_state, (history_length - len(history_segment), 1))
                    history_segment = np.vstack([padding, history_segment])
                else:
                    history_segment = np.zeros((history_length, joint_positions.shape[1]))
        
        # 确保历史长度正确
        history_segment = history_segment[:history_length]
        
        # 展平历史数据
        return history_segment.flatten()
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 处理不同类型的样本
        if len(sample) == 5 and sample[4] == 'state_transfer':
            # 状态转移学习样本
            instruction, file_name, start_idx, current_state_tuple, sample_type = sample
            is_state_transfer = True
            current_state = np.array(current_state_tuple)
        else:
            # 原始样本
            if len(sample) == 4:
                instruction, file_name, start_idx, random_offset = sample
                is_enhanced = True
                is_state_transfer = False
                current_state = None
            else:
                instruction, file_name, start_idx = sample
                is_enhanced = False
                is_state_transfer = False
                current_state = None
        
        # 加载数据
        trajectory_path = os.path.join(self.data_dir, file_name)
        with open(trajectory_path, 'r') as f:
            data = json.load(f)
        
        # 提取关节数据
        observations = data['observations']
        joint_positions = np.array([obs['joint_pos'] for obs in observations])
        
        if is_state_transfer:
            # 状态转移学习：使用真实的当前状态和目标轨迹
            start_state = current_state
            target_trajectory = joint_positions[start_idx:start_idx + self.trajectory_length]
            target_actions = target_trajectory
            
            # 生成基于真实历史的动作历史
            action_history = self._generate_action_history(
                joint_positions, start_idx, 128, current_state
            )
            
        else:
            # 原始训练逻辑
            trajectory = self.sample_trajectory_segment(joint_positions, start_idx, self.trajectory_length)
            start_state = trajectory[0]
            target_actions = trajectory
            action_history = self._generate_action_history(joint_positions, start_idx)
        
        # 分析关节重要性
        joint_importance = self.analyze_joint_importance(target_actions)
        
        # 获取指令ID
        instruction_id = self.instruction_to_id[instruction]
        
        return {
            'start_state': start_state.astype(np.float32),
            'target_actions': target_actions.astype(np.float32),
            'instruction_id': np.array(instruction_id, dtype=np.int64),
            'joint_importance': joint_importance.astype(np.float32),
            'action_history': action_history.astype(np.float32),
            'file_name': file_name,
            'start_idx': start_idx,
            'is_state_transfer': is_state_transfer,
            'sample_type': sample_type if is_state_transfer else 'normal'
        }

# ==============================================
# 3. 损失函数 - 关键关节专注的损失函数
# ==============================================

class KeyJointACTLoss(nn.Module):
    """关键关节专注的ACT损失函数 - 增强版，防止轨迹回溯"""
    
    def __init__(self, 
                 classification_weight=10.0, 
                 diversity_weight=5.0,
                 key_joint_weight=5.0,
                 stability_weight=0.01,
                 unidirectional_weight=0.02,  # 极低权重：几乎不影响训练
                 continuity_weight=0.01,  # 极低权重：几乎不影响训练
                 goal_directed_weight=0.015,  # 极低权重：几乎不影响训练
                 instruction_target_variances=None):  # 支持不同指令的目标变化量
        super().__init__()
        self.classification_weight = classification_weight
        self.diversity_weight = diversity_weight
        self.key_joint_weight = key_joint_weight
        self.stability_weight = stability_weight
        self.unidirectional_weight = unidirectional_weight
        self.continuity_weight = continuity_weight
        self.goal_directed_weight = goal_directed_weight
        
        # 基于真实数据统计的目标变化量
        if instruction_target_variances is None:
            self.instruction_target_variances = {
                'wave': 0.108,  # 基于wave数据的平均变化量
                'welcome': 0.111  # 基于welcome数据的平均变化量
            }
        else:
            self.instruction_target_variances = instruction_target_variances
        
        # 指令ID到名称的映射
        self.id_to_instruction = {0: 'wave', 1: 'welcome'}
    
    def compute_key_joint_variance(self, actions, joint_importance, instruction_ids):
        """计算关键关节的变化量"""
        batch_size, seq_len, action_dim = actions.shape
        
        # 计算每个关节的变化量（方差）
        joint_variance = torch.var(actions, dim=1)  # [batch_size, action_dim]
        
        # 根据重要性加权
        weighted_variance = joint_variance * joint_importance
        
        # 计算每个样本的整体变化量（跨关节平均）
        overall_variance = torch.mean(weighted_variance, dim=1)  # [batch_size]
        
        return overall_variance
    
    def compute_unidirectional_loss(self, actions, start_states):
        """计算单向递进损失 - 防止明显的轨迹回溯，改进版"""
        batch_size, seq_len, action_dim = actions.shape
        
        if seq_len <= 3:
            return torch.tensor(0.0, device=actions.device)
        
        # 计算每一步相对于起始状态的位移
        start_expanded = start_states.unsqueeze(1).expand(-1, seq_len, -1)
        displacement_from_start = actions - start_expanded
        
        # 主要改进：不是强制单调递增，而是惩罚明显的回溯
        # 计算短期的位移变化，避免惩罚正常的周期性动作
        window_size = min(5, seq_len // 2)
        
        # 计算每个关节的短期趋势
        short_term_regression = 0.0
        count = 0
        
        for i in range(window_size, seq_len - window_size):
            # 计算当前窗口与前一个窗口的平均位置
            current_window = actions[:, i:i+window_size, :]
            prev_window = actions[:, i-window_size:i, :]
            
            current_mean = torch.mean(current_window, dim=1)  # [batch_size, action_dim]
            prev_mean = torch.mean(prev_window, dim=1)  # [batch_size, action_dim]
            
            # 计算位移变化
            displacement_change = current_mean - prev_mean
            
            # 只有当变化方向与整体动作方向相反时才惩罚
            # 使用整体趋势作为参考
            overall_trend = actions[:, -window_size:, :] - actions[:, :window_size, :]
            overall_direction = torch.mean(overall_trend, dim=(1, 2))  # [batch_size]
            
            # 计算局部变化与整体趋势的一致性
            local_change_magnitude = torch.norm(displacement_change, dim=-1)  # [batch_size]
            
            # 只惩罚幅度较大的回溯，避免影响正常的微小调整
            regression_threshold = 0.1  # 调整这个阈值
            large_regression = torch.relu(local_change_magnitude - regression_threshold)
            
            short_term_regression += torch.mean(large_regression)
            count += 1
        
        if count > 0:
            regression_penalty = short_term_regression / count
        else:
            regression_penalty = torch.tensor(0.0, device=actions.device)
        
        return regression_penalty
    
    def compute_continuity_loss(self, actions):
        """计算连续性约束损失 - 防止突然的跳跃式回溯，改进版"""
        batch_size, seq_len, action_dim = actions.shape
        
        if seq_len <= 3:
            return torch.tensor(0.0, device=actions.device)
        
        # 计算一阶差分（速度）
        first_diff = torch.diff(actions, dim=1)  # [batch_size, seq_len-1, action_dim]
        
        # 计算二阶差分（加速度）
        second_diff = torch.diff(first_diff, dim=1)  # [batch_size, seq_len-2, action_dim]
        
        # 改进：只惩罚极端的加速度，允许正常的动作变化
        acceleration_magnitude = torch.norm(second_diff, dim=-1)  # [batch_size, seq_len-2]
        
        # 设置合理的加速度阈值，只惩罚超出正常范围的突变
        acceleration_threshold = 0.5  # 调整这个阈值
        extreme_acceleration = torch.relu(acceleration_magnitude - acceleration_threshold)
        acceleration_penalty = torch.mean(extreme_acceleration)
        
        # 改进方向突变检测，只惩罚剧烈的方向改变
        velocity_norms = torch.norm(first_diff, dim=-1, keepdim=True)
        normalized_velocities = first_diff / (velocity_norms + 1e-8)
        
        # 计算相邻速度向量的点积
        velocity_dot_products = torch.sum(
            normalized_velocities[:, :-1] * normalized_velocities[:, 1:], dim=-1
        )
        
        # 只惩罚剧烈的方向改变（点积 < -0.5，即角度 > 120度）
        severe_direction_change = torch.relu(-0.5 - velocity_dot_products)
        direction_change_penalty = torch.mean(severe_direction_change)
        
        return acceleration_penalty + direction_change_penalty
    
    def compute_goal_directed_loss(self, actions, target_actions):
        """计算终点导向损失 - 引导轨迹朝着目标方向发展，简化版"""
        batch_size, seq_len, action_dim = actions.shape
        
        if seq_len < 4:
            return torch.tensor(0.0, device=actions.device)
        
        # 简化：只比较轨迹终点与目标终点的距离
        predicted_final = actions[:, -1, :]  # [batch_size, action_dim]
        target_final = target_actions[:, -1, :]  # [batch_size, action_dim]
        
        # 计算终点误差，但只惩罚较大的误差
        final_error = torch.norm(predicted_final - target_final, dim=-1)  # [batch_size]
        
        # 设置合理的误差阈值，只惩罚严重偏离目标的情况
        error_threshold = 0.5  # 调整这个阈值
        large_error = torch.relu(final_error - error_threshold)
        
        # 平均惩罚
        misalignment_penalty = torch.mean(large_error)
        
        return misalignment_penalty
    
       
    def forward(self, predicted_actions, target_actions, instruction_logits, instruction_ids, 
                joint_importance, key_joint_actions, start_states=None):
        """计算损失 - 简化版，支持任意起始位置"""
        
        batch_size = predicted_actions.size(0)
        # 使用传入的真实起始状态，如果未提供则使用目标轨迹起始状态
        if start_states is None:
            start_states = target_actions[:, 0]  # 向后兼容
        
        # 简化的任意起点损失：确保轨迹从正确的起始状态开始
        start_alignment_loss = torch.mean(torch.norm(
            predicted_actions[:, 0, :] - start_states, dim=-1
        ))
        
        # 1. 基础动作预测损失
        action_loss = nn.functional.mse_loss(predicted_actions, target_actions)
        
        # 2. 指令分类损失
        classification_loss = nn.functional.cross_entropy(instruction_logits, instruction_ids)
        
        # 3. 关键关节专注损失 - 极简版本
        # 计算每个关节的方差
        joint_variance = torch.var(predicted_actions, dim=1)  # [batch_size, action_dim]
        overall_variance = torch.mean(joint_variance, dim=1)  # [batch_size]
        
        # 使用固定的目标变化量，避免复杂数值计算
        target_variance = 0.1
        key_joint_loss = torch.mean(torch.abs(overall_variance - target_variance))
        
        # 4. 时序稳定性损失 - 极简版本
        if predicted_actions.shape[1] > 1:
            action_diff = torch.diff(predicted_actions, dim=1)
            stability_loss = torch.mean(torch.abs(action_diff))
        else:
            stability_loss = torch.tensor(0.0, device=predicted_actions.device)
        
        # 5. 指令模式多样性损失 - 极简版本
        diversity_loss = torch.tensor(0.0, device=predicted_actions.device)
        unique_instructions = torch.unique(instruction_ids)
        if len(unique_instructions) > 1:
            pattern_means = []
            for instr_id in unique_instructions:
                mask = instruction_ids == instr_id
                if mask.any():
                    pattern_mean = torch.mean(predicted_actions[mask], dim=(0, 1))
                    pattern_means.append(pattern_mean)
            
            if len(pattern_means) > 1:
                pattern_means = torch.stack(pattern_means)
                pattern_diff_matrix = torch.pdist(pattern_means, p=2)
                diversity_loss = torch.mean(torch.relu(0.5 - pattern_diff_matrix))
        
        # 6. 新增：单向递进损失 - 防止轨迹回溯
        unidirectional_loss = self.compute_unidirectional_loss(predicted_actions, start_states)
        
        # 7. 新增：连续性约束损失 - 防止突然的跳跃式回溯
        continuity_loss = self.compute_continuity_loss(predicted_actions)
        
        # 8. 新增：终点导向损失 - 引导轨迹朝着目标方向发展
        goal_directed_loss = self.compute_goal_directed_loss(predicted_actions, target_actions)
        
        # 综合损失 - 包含简化的任意起点损失
        losses = [
            action_loss,
            self.classification_weight * classification_loss,
            self.key_joint_weight * key_joint_loss,
            self.diversity_weight * diversity_loss,
            self.stability_weight * stability_loss,
            self.unidirectional_weight * unidirectional_loss,
            self.continuity_weight * continuity_loss,
            self.goal_directed_weight * goal_directed_loss,
            0.1 * start_alignment_loss  # 简化的任意起点损失，小权重避免干扰
        ]
        
        # 检查每个损失项是否有效
        valid_losses = []
        for i, loss in enumerate(losses):
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: 损失项{i}数值异常: {loss.item()}, 替换为0")
                valid_losses.append(torch.tensor(0.0, device=loss.device))
            else:
                valid_losses.append(loss)
        
        total_loss = torch.stack(valid_losses).sum()
        
        return {
            'total_loss': total_loss,
            'action_loss': action_loss,
            'classification_loss': classification_loss,
            'key_joint_loss': key_joint_loss,
            'diversity_loss': diversity_loss,
            'stability_loss': stability_loss,
            'unidirectional_loss': unidirectional_loss,
            'continuity_loss': continuity_loss,
            'goal_directed_loss': goal_directed_loss,
            'start_alignment_loss': start_alignment_loss,
            'key_joint_variance': torch.mean(overall_variance)
        }

# ==============================================
# 4. 训练函数
# ==============================================

def train_key_joint_act_model(args=None):
    """训练关键关节专注的ACT模型"""
    print("开始训练关键关节专注的ACT模型...")
    print("核心特性：自动识别关键关节，专注于变化大的关节学习")
    print("预测后32步，使用Transformer时序模型")
    
    # 模型配置 - 使用更保守的配置，支持动作历史上下文
    model_config = {
        'state_dim': 26,
        'action_dim': 26,
        'num_instructions': 2,
        'hidden_dim': args.hidden_dim if args else 256,
        'trajectory_length': 32,  # 回退：从128改回32，原始长度
        'dropout': args.dropout if args else 0.2,  # 降低dropout，防止早期训练不稳定
        'key_joints_per_instruction': 16,  # 阶段3：从8增加到16，支持通用动作框架
        'predict_differences': False,  # 直接预测绝对位置
        'signal_amplification': 1.0,  # 不使用信号放大
        'history_length': 128  # 动作历史上下文长度，128步提供更充分的上下文
    }
    
    # 训练配置 - 平衡性能和稳定性，新增防回溯损失，支持命令行参数
    training_config = {
        'learning_rate': args.learning_rate if args else 1e-4,  # 适中的学习率
        'weight_decay': 1e-4,  # 适中的权重衰减
        'batch_size': args.batch_size if args else 64,  # 恢复较大的batch size以提高GPU利用率
        'epochs': args.epochs if args else 2000,
        'classification_weight': 1.0,  # 降低分类权重
        'diversity_weight': 0.5,  # 降低多样性权重
        'key_joint_weight': 0.5,  # 降低关键关节权重
        'stability_weight': 0.05,  # 降低稳定性权重
        'unidirectional_weight': args.unidirectional_weight if args else 0.03,  # 新增：单向递进权重 - 防止轨迹回溯 (降低76%)
        'continuity_weight': args.continuity_weight if args else 0.015,  # 新增：连续性约束权重 - 防止跳跃式回溯 (降低76%)
        'goal_directed_weight': args.goal_directed_weight if args else 0.024  # 新增：终点导向权重 - 引导轨迹方向 (降低76%)
    }
    
    # 数据准备
    data_dir = '/root/kuavo_ws/src/vla/trajectories'
    file_names = [f'wave_{i:03d}.json' for i in range(1, 9)] + [f'welcome_{i:03d}.json' for i in range(1, 9)]
    
    print(f"使用数据文件: {len(file_names)} 个")
    print(f"模型配置: {model_config}")
    print(f"训练配置: {training_config}")
    
    # 数据集
    dataset = KeyJointDataset(
        data_dir, file_names, 
        trajectory_length=model_config['trajectory_length']
    )
    
    # 创建模型和数据加载
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据集划分
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 优化数据加载以提高GPU利用率
    train_loader = DataLoader(
        train_dataset, 
        batch_size=training_config['batch_size'], 
        shuffle=True,
        num_workers=4,  # 使用多个工作进程加载数据
        pin_memory=True if device.type == 'cuda' else False,  # 固定内存加速GPU传输
        persistent_workers=True  # 保持工作进程活跃
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=training_config['batch_size'], 
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True
    )
    
    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    print(f"数据加载优化: num_workers=4, pin_memory={device.type == 'cuda'}")
    
    # 创建模型
    model = KeyJointACTGenerator(model_config).to(device)
    
    # 启用自动混合精度训练提高GPU性能
    scaler = torch.amp.GradScaler('cuda', init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000) if device.type == 'cuda' else None
    print(f"自动混合精度: {'启用' if scaler else '禁用'}")
    if scaler:
        print(f"  初始缩放因子: {scaler.get_scale()}")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数数量: {total_params:,}")
    
    # 损失函数 - 增强版，包含防回溯损失
    loss_fn = KeyJointACTLoss(
        classification_weight=training_config['classification_weight'],
        diversity_weight=training_config['diversity_weight'],
        key_joint_weight=training_config['key_joint_weight'],
        stability_weight=training_config['stability_weight'],
        unidirectional_weight=training_config['unidirectional_weight'],
        continuity_weight=training_config['continuity_weight'],
        goal_directed_weight=training_config['goal_directed_weight']
    )
    
    # 优化器 - 使用更稳定的优化器，启用GPU优化
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        betas=(0.9, 0.999),
        fused=True  # 使用融合优化器提高GPU性能
    )
    
    # 学习率调度器 - 使用更保守的学习率调度
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.8,  # 更温和的降低
        patience=100,  # 更长的耐心
        verbose=True,
        min_lr=1e-7  # 更低的最小学习率
    )
    
    # 创建保存目录 - 使用命令行参数指定的模型名称
    model_name = args.model_name if args else 'anti_regression'
    save_dir = f'./checkpoints/{model_name}'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"开始训练...")
    
    # 初始化质量监控系统
    quality_validator = ActionQualityValidator()
    quality_monitor = TrainingQualityMonitor(quality_validator)
    print("质量监控系统已初始化")
    
    # 训练循环
    best_val_loss = float('inf')
    patience = 150  # 增加耐心值
    patience_counter = 0
    
    # 用于跟踪训练稳定性
    val_loss_history = []
    
    # 启用cuDNN benchmark模式提高GPU性能
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print("启用cuDNN benchmark模式")
    
    for epoch in range(training_config['epochs']):
        # 训练阶段
        model.train()
        train_losses = []
        train_action_losses = []
        train_classification_losses = []
        train_key_joint_losses = []
        train_diversity_losses = []
        train_stability_losses = []
        train_unidirectional_losses = []  # 新增：单向递进损失
        train_continuity_losses = []      # 新增：连续性约束损失
        train_goal_directed_losses = []   # 新增：终点导向损失
        train_accuracies = []
        train_key_joint_variances = []
        
        for batch in train_loader:
            start_states = batch['start_state'].to(device)
            target_actions = batch['target_actions'].to(device)
            instruction_ids = batch['instruction_id'].to(device)
            action_history = batch['action_history'].to(device)
            
            # 使用自动混合精度训练
            if scaler:
                with torch.amp.autocast('cuda'):
                    # 前向传播 - 包含动作历史上下文
                    predicted_actions, instruction_logits, joint_importance, key_joint_actions = model(
                        start_states, instruction_ids, target_actions, action_history
                    )
                    
                    # 计算损失 - 使用真实的起始状态
                    loss_dict = loss_fn(
                        predicted_actions, target_actions, instruction_logits, instruction_ids,
                        joint_importance, key_joint_actions, start_states
                    )
                
                # 质量监控 - 每10个batch监控一次
                if len(train_losses) % 10 == 0:
                    batch_quality = quality_monitor.monitor_batch(
                        predicted_actions, target_actions, instruction_ids
                    )
                
                # 简单的反向传播
                total_loss = loss_dict['total_loss']
                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                
                # 梯度裁剪 - 必须在unscale之前进行
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # CPU模式
                # 前向传播 - 包含动作历史上下文
                predicted_actions, instruction_logits, joint_importance, key_joint_actions = model(
                    start_states, instruction_ids, target_actions, action_history
                )
                
                # 计算损失
                loss_dict = loss_fn(
                    predicted_actions, target_actions, instruction_logits, instruction_ids,
                    joint_importance, key_joint_actions
                )
                
                # 质量监控 - 每10个batch监控一次
                if len(train_losses) % 10 == 0:
                    batch_quality = quality_monitor.monitor_batch(
                        predicted_actions, target_actions, instruction_ids
                    )
                
                # 反向传播
                optimizer.zero_grad()
                loss_dict['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            
            # 记录损失
            total_loss = loss_dict['total_loss'].item()
            train_losses.append(total_loss)
            train_action_losses.append(loss_dict['action_loss'].item())
            train_classification_losses.append(loss_dict['classification_loss'].item())
            train_key_joint_losses.append(loss_dict['key_joint_loss'].item())
            train_diversity_losses.append(loss_dict['diversity_loss'].item())
            train_stability_losses.append(loss_dict['stability_loss'].item())
            train_unidirectional_losses.append(loss_dict['unidirectional_loss'].item())  # 新增
            train_continuity_losses.append(loss_dict['continuity_loss'].item())          # 新增
            train_goal_directed_losses.append(loss_dict['goal_directed_loss'].item())   # 新增
            train_key_joint_variances.append(loss_dict['key_joint_variance'].item())
            
            # 计算分类准确率
            predicted_ids = torch.argmax(instruction_logits, dim=1)
            accuracy = (predicted_ids == instruction_ids).float().mean().item()
            train_accuracies.append(accuracy)
        
        # 验证阶段
        model.eval()
        val_losses = []
        val_action_losses = []
        val_classification_losses = []
        val_key_joint_losses = []
        val_diversity_losses = []
        val_stability_losses = []
        val_unidirectional_losses = []  # 新增：单向递进损失
        val_continuity_losses = []      # 新增：连续性约束损失
        val_goal_directed_losses = []   # 新增：终点导向损失
        val_accuracies = []
        val_key_joint_variances = []
        
        with torch.no_grad():
            for batch in val_loader:
                start_states = batch['start_state'].to(device)
                target_actions = batch['target_actions'].to(device)
                instruction_ids = batch['instruction_id'].to(device)
                action_history = batch['action_history'].to(device)
                
                predicted_actions, instruction_logits, joint_importance, key_joint_actions = model(
                    start_states, instruction_ids, target_actions, action_history
                )
                
                loss_dict = loss_fn(
                    predicted_actions, target_actions, instruction_logits, instruction_ids,
                    joint_importance, key_joint_actions
                )
                
                # 检查数值稳定性
                val_total_loss = loss_dict['total_loss'].item()
                if not torch.isnan(loss_dict['total_loss']) and not np.isnan(val_total_loss) and not np.isinf(val_total_loss):
                    val_losses.append(val_total_loss)
                    val_action_losses.append(loss_dict['action_loss'].item())
                    val_classification_losses.append(loss_dict['classification_loss'].item())
                    val_key_joint_losses.append(loss_dict['key_joint_loss'].item())
                    val_diversity_losses.append(loss_dict['diversity_loss'].item())
                    val_stability_losses.append(loss_dict['stability_loss'].item())
                    val_unidirectional_losses.append(loss_dict['unidirectional_loss'].item())  # 新增
                    val_continuity_losses.append(loss_dict['continuity_loss'].item())          # 新增
                    val_goal_directed_losses.append(loss_dict['goal_directed_loss'].item())   # 新增
                    val_key_joint_variances.append(loss_dict['key_joint_variance'].item())
                    
                    # 计算分类准确率
                    predicted_ids = torch.argmax(instruction_logits, dim=1)
                    accuracy = (predicted_ids == instruction_ids).float().mean().item()
                    val_accuracies.append(accuracy)
        
        # 计算平均损失 - 添加空列表检查，包含新的防回溯损失
        avg_train_loss = np.mean(train_losses) if train_losses else float('nan')
        avg_train_action = np.mean(train_action_losses) if train_action_losses else float('nan')
        avg_train_classification = np.mean(train_classification_losses) if train_classification_losses else float('nan')
        avg_train_key_joint = np.mean(train_key_joint_losses) if train_key_joint_losses else float('nan')
        avg_train_diversity = np.mean(train_diversity_losses) if train_diversity_losses else float('nan')
        avg_train_stability = np.mean(train_stability_losses) if train_stability_losses else float('nan')
        avg_train_unidirectional = np.mean(train_unidirectional_losses) if train_unidirectional_losses else float('nan')  # 新增
        avg_train_continuity = np.mean(train_continuity_losses) if train_continuity_losses else float('nan')              # 新增
        avg_train_goal_directed = np.mean(train_goal_directed_losses) if train_goal_directed_losses else float('nan')     # 新增
        avg_train_accuracy = np.mean(train_accuracies) if train_accuracies else float('nan')
        avg_train_key_variance = np.mean(train_key_joint_variances) if train_key_joint_variances else float('nan')
        
        avg_val_loss = np.mean(val_losses) if val_losses else float('nan')
        avg_val_action = np.mean(val_action_losses) if val_action_losses else float('nan')
        avg_val_classification = np.mean(val_classification_losses) if val_classification_losses else float('nan')
        avg_val_key_joint = np.mean(val_key_joint_losses) if val_key_joint_losses else float('nan')
        avg_val_diversity = np.mean(val_diversity_losses) if val_diversity_losses else float('nan')
        avg_val_stability = np.mean(val_stability_losses) if val_stability_losses else float('nan')
        avg_val_unidirectional = np.mean(val_unidirectional_losses) if val_unidirectional_losses else float('nan')          # 新增
        avg_val_continuity = np.mean(val_continuity_losses) if val_continuity_losses else float('nan')                    # 新增
        avg_val_goal_directed = np.mean(val_goal_directed_losses) if val_goal_directed_losses else float('nan')             # 新增
        avg_val_accuracy = np.mean(val_accuracies) if val_accuracies else float('nan')
        avg_val_key_variance = np.mean(val_key_joint_variances) if val_key_joint_variances else float('nan')
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        
        # 记录验证损失历史
        val_loss_history.append(avg_val_loss)
        
        # 质量监控 - 每个epoch结束后生成质量报告
        epoch_quality = quality_monitor.monitor_epoch(epoch)
        
        # 打印进度 - 每20个epoch打印一次，包含新的防回溯损失
        if epoch % 20 == 0:
            quality_info = f"Quality: {epoch_quality.get('overall_quality', 0.0):.3f}" if epoch_quality else ""
            print(f"Epoch {epoch+1}/{training_config['epochs']} - "
                  f"Train Total: {avg_train_loss:.4f}, Val Total: {avg_val_loss:.4f}, "
                  f"Val Action: {avg_val_action:.4f}, Val Class: {avg_val_classification:.4f}, "
                  f"Val KeyJoint: {avg_val_key_joint:.4f}, Val Div: {avg_val_diversity:.4f}, "
                  f"Val Stability: {avg_val_stability:.4f}, "
                  f"Val Uni: {avg_val_unidirectional:.4f}, Val Cont: {avg_val_continuity:.4f}, "
                  f"Val Goal: {avg_val_goal_directed:.4f}, Acc: {avg_val_accuracy:.3f}, "
                  f"KeyVar: {avg_val_key_variance:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}, "
                  f"{quality_info}")
            
            # 每100个epoch打印详细质量报告
            if epoch % 100 == 0 and epoch > 0:
                print("\n" + "="*50)
                print("详细质量报告:")
                print(quality_monitor.get_quality_report())
                print("="*50 + "\n")
        
        # 保存最佳模型 - 只有当验证损失真正改善时才保存
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # 计算数据集统计信息
            dataset_stats = {
                'state_mean': np.zeros(26),
                'state_std': np.ones(26),
                'action_mean': np.zeros(26),
                'action_std': np.ones(26)
            }
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_config': model_config,
                'training_config': training_config,
                'best_val_loss': best_val_loss,
                'val_accuracy': avg_val_accuracy,
                'val_key_joint_variance': avg_val_key_variance,
                'norm_stats': dataset_stats
            }
            
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
            print(f"  保存最佳模型 - Val Loss: {best_val_loss:.4f}, Acc: {avg_val_accuracy:.3f}")
        else:
            patience_counter += 1
        
        # 早停 - 只有在连续多个epoch都没有改善时才停止
        if patience_counter >= patience and len(val_loss_history) > 100:
            # 检查最近的趋势是否真的稳定
            recent_losses = val_loss_history[-50:]
            if np.mean(recent_losses) > np.mean(val_loss_history[-100:-50]):
                print(f"早停: {patience} 轮验证损失没有改善，且呈上升趋势")
                break
    
    print("训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型参数数量: {total_params:,}")
    
    # 保存最终质量报告
    final_quality_report = quality_monitor.get_quality_report()
    print("\n" + "="*50)
    print("最终质量报告:")
    print(final_quality_report)
    print("="*50)
    
    # 保存质量历史数据
    try:
        quality_monitor.save_quality_history(os.path.join(save_dir, 'quality_history.json'))
        print("质量历史数据保存成功")
    except Exception as e:
        print(f"保存质量历史数据时出错: {e}")
        # 不影响训练继续进行
    
    return model, model_config, training_config

# ==============================================
# 5. 主函数
# ==============================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='VLA项目关键关节专注ACT训练')
    parser.add_argument('--model_name', type=str, default='anti_regression', 
                       help='模型保存目录名称 (默认: anti_regression)')
    parser.add_argument('--epochs', type=int, default=2000, 
                       help='训练轮数 (默认: 2000)')
    parser.add_argument('--batch_size', type=int, default=64, 
                       help='批次大小 (默认: 64)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                       help='学习率 (默认: 1e-4)')
    parser.add_argument('--hidden_dim', type=int, default=256, 
                       help='隐藏层维度 (默认: 256)')
    parser.add_argument('--dropout', type=float, default=0.2, 
                       help='Dropout率 (默认: 0.2)')
    parser.add_argument('--unidirectional_weight', type=float, default=0.03, 
                       help='单向递进损失权重 (降低76%: 从0.125降到0.03，极低约束)')
    parser.add_argument('--continuity_weight', type=float, default=0.015, 
                       help='连续性约束损失权重 (降低76%: 从0.0625降到0.015，极低约束)')
    parser.add_argument('--goal_directed_weight', type=float, default=0.024, 
                       help='终点导向损失权重 (降低76%: 从0.1降到0.024，极低约束)')
    parser.add_argument('--no_anti_regression', action='store_true', 
                       help='完全禁用防回溯损失，恢复到原始版本')
    return parser.parse_args()

def main():
    """主训练函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 如果选择完全禁用防回溯损失，设置权重为0
    if args.no_anti_regression:
        args.unidirectional_weight = 0.0
        args.continuity_weight = 0.0
        args.goal_directed_weight = 0.0
        print("🚫 防回溯损失已完全禁用，恢复到原始训练模式")
    
    print("=" * 60)
    print("VLA项目关键关节专注ACT训练 - 增强防回溯版")
    print("=" * 60)
    print("核心特性：")
    print("1. 自动识别每个指令的关键关节")
    print("2. 专注于变化大的关节学习")
    print("3. 使用Transformer时序模型预测后32步")
    print("4. 稳定的训练策略，避免早期震荡")
    print("5. 阶段3：扩展关键关节数量（8→16），支持通用动作框架")
    print("=" * 60)
    print(f"训练参数:")
    print(f"  模型名称: {args.model_name}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  隐藏层维度: {args.hidden_dim}")
    print(f"  Dropout率: {args.dropout}")
    print(f"  防回溯损失权重: 单向({args.unidirectional_weight}), 连续({args.continuity_weight}), 终点({args.goal_directed_weight}) [优化: 降低76%极低约束]")
    print(f"  关键关节数量: 16 [阶段3: 从8增加到16]")
    print("=" * 60)
    
    try:
        model, model_config, training_config = train_key_joint_act_model(args)
        
        print("\n" + "=" * 60)
        print("训练完成！")
        print("=" * 60)
        print(f"模型配置: {model_config}")
        print(f"训练配置: {training_config}")
        print(f"模型已保存到: ./checkpoints/{args.model_name}/best_model.pth")
        print("\n使用方法：")
        print(f"python simple_act_train.py --model_name {args.model_name} --epochs {args.epochs}")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


