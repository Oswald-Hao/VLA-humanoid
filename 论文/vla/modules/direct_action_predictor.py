#!/usr/bin/env python3
"""
直接动作预测网络 - 改进版本，直接预测动作序列而不是噪声
基于Transformer架构，专门针对平滑的机器人动作序列优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import math


class ActionPredictor(nn.Module):
    """
    直接动作预测器 - 使用Transformer架构直接预测动作序列
    专门优化用于平滑的机器人动作序列预测
    """
    
    def __init__(self, 
                 action_dim: int = 26,
                 chunk_length: int = 16,
                 state_dim: int = 60,
                 visual_feature_dim: int = 512,
                 instruction_embed_dim: int = 64,
                 condition_dim: int = 256,
                 hidden_dim: int = 256,  # 减少隐藏维度以防止过拟合
                 num_heads: int = 4,      # 减少注意力头数
                 num_layers: int = 4):     # 减少Transformer层数
        super().__init__()
        
        self.action_dim = action_dim
        self.chunk_length = chunk_length
        self.state_dim = state_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        
        # 条件编码器
        self.condition_encoder = ConditionEncoder(
            state_dim=state_dim,
            visual_feature_dim=visual_feature_dim,
            instruction_embed_dim=instruction_embed_dim,
            condition_dim=condition_dim
        )
        
        # 条件投影层 - 将条件维度投影到隐藏维度
        self.condition_projection = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3)  # 增加dropout率
        )
        
        # 动作序列编码器
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3)  # 增加dropout率
        )
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout=0.2, max_len=chunk_length)  # 增加dropout
        
        # Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,  # 减少前馈网络维度
            dropout=0.3,  # 增加dropout率
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 动作预测头 - 直接预测动作序列
        self.action_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),  # 减少维度
            nn.GELU(),
            nn.Dropout(0.3),  # 增加dropout率
            nn.Linear(hidden_dim // 2, action_dim)
            # 不使用激活函数，让网络自由学习动作范围
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重 - 使用He初始化以改善梯度流动"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, 
                state: torch.Tensor,
                visual_features: Optional[torch.Tensor] = None,
                instruction_embed: Optional[torch.Tensor] = None,
                target_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 状态信息 [batch_size, state_dim]
            visual_features: 视觉特征 [batch_size, visual_feature_dim]
            instruction_embed: 指令嵌入 [batch_size, instruction_embed_dim]
            target_actions: 目标动作 [batch_size, chunk_length, action_dim] (可选，用于训练)
            
        Returns:
            预测的动作序列 [batch_size, chunk_length, action_dim]
        """
        batch_size = state.shape[0]
        device = state.device
        
        # 条件编码
        condition = self.condition_encoder(state, visual_features, instruction_embed)
        
        # 将条件投影到隐藏维度
        condition = self.condition_projection(condition)
        
        # 生成初始动作序列
        if target_actions is not None:
            # 训练时使用目标动作
            action_emb = self.action_encoder(target_actions)
        else:
            # 推理时生成零序列或从条件中生成
            action_emb = torch.zeros(batch_size, self.chunk_length, self.hidden_dim, device=device)
        
        # 添加条件信息到每一帧
        condition_expanded = condition.unsqueeze(1).expand(-1, self.chunk_length, -1)
        action_emb = action_emb + condition_expanded
        
        # 添加位置编码
        action_emb = self.pos_encoding(action_emb)
        
        # 通过Transformer
        transformer_out = self.transformer(action_emb)
        
        # 预测动作序列
        predicted_actions = self.action_predictor(transformer_out)
        
        return predicted_actions


class PositionalEncoding(nn.Module):
    """
    位置编码 - 为动作序列添加位置信息
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ConditionEncoder(nn.Module):
    """
    条件编码器 - 编码状态、视觉和任务信息
    """
    
    def __init__(self, 
                 state_dim: int = 60,
                 visual_feature_dim: int = 512,
                 instruction_embed_dim: int = 64,
                 condition_dim: int = 256):
        super().__init__()
        
        self.state_dim = state_dim
        self.visual_feature_dim = visual_feature_dim
        self.instruction_embed_dim = instruction_embed_dim
        self.condition_dim = condition_dim
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),  # 增加dropout率
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3)   # 增加dropout率
        )
        
        # 视觉特征编码器（可选）
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_feature_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),  # 增加dropout率
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3)   # 增加dropout率
        ) if visual_feature_dim > 0 else None
        
        # 指令编码器
        self.instruction_encoder = nn.Sequential(
            nn.Linear(instruction_embed_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.3),  # 增加dropout率
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.3)   # 增加dropout率
        )
        
        # 条件融合
        visual_dim = 128 if self.visual_encoder is not None else 0
        total_input_dim = 128 + visual_dim + 64
        self.condition_fusion = nn.Sequential(
            nn.Linear(total_input_dim, condition_dim),
            nn.LayerNorm(condition_dim),
            nn.GELU(),
            nn.Dropout(0.3),  # 增加dropout率
            nn.Linear(condition_dim, condition_dim),
            nn.LayerNorm(condition_dim)
        )
        
        # 保存期望的输入维度
        self.expected_input_dim = total_input_dim
        self.visual_dim = visual_dim
    
    def forward(self, 
                state: torch.Tensor,
                visual_features: Optional[torch.Tensor] = None,
                instruction_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 状态信息 [batch_size, state_dim]
            visual_features: 视觉特征 [batch_size, visual_feature_dim]
            instruction_embed: 指令嵌入 [batch_size, instruction_embed_dim]
            
        Returns:
            条件向量 [batch_size, condition_dim]
        """
        # 处理状态维度不匹配问题
        if state.shape[1] != self.state_dim:
            if state.shape[1] < self.state_dim:
                padding = self.state_dim - state.shape[1]
                state = F.pad(state, (0, padding))
            else:
                state = state[:, :self.state_dim]
        
        # 编码状态
        state_features = self.state_encoder(state)
        
        # 编码视觉特征（如果有）
        if self.visual_encoder is not None and visual_features is not None:
            visual_features_encoded = self.visual_encoder(visual_features)
        else:
            if self.visual_dim > 0:
                visual_features_encoded = torch.zeros(state.shape[0], self.visual_dim, device=state.device)
            else:
                visual_features_encoded = torch.zeros(state.shape[0], 0, device=state.device)
        
        # 编码指令
        instruction_features = torch.zeros(state.shape[0], 64, device=state.device)
        if instruction_embed is not None:
            instruction_features = self.instruction_encoder(instruction_embed)
        
        # 融合条件信息
        condition_input = torch.cat([state_features, visual_features_encoded, instruction_features], dim=-1)
        condition = self.condition_fusion(condition_input)
        
        return condition


class DirectActionModel(nn.Module):
    """
    直接动作预测模型 - 支持指令分类和动作序列预测
    """
    
    def __init__(self, 
                 action_dim: int = 26,
                 chunk_length: int = 16,
                 state_dim: int = 60,
                 visual_feature_dim: int = 512,
                 instruction_vocab_size: int = 2,
                 instruction_embed_dim: int = 64,
                 condition_dim: int = 256,
                 hidden_dim: int = 512):
        super().__init__()
        
        self.action_dim = action_dim
        self.chunk_length = chunk_length
        self.state_dim = state_dim
        self.instruction_vocab_size = instruction_vocab_size
        
        # 指令嵌入层
        self.instruction_embedding = nn.Embedding(instruction_vocab_size, instruction_embed_dim)
        
        # 动作预测器
        self.action_predictor = ActionPredictor(
            action_dim=action_dim,
            chunk_length=chunk_length,
            state_dim=state_dim,
            visual_feature_dim=visual_feature_dim,
            instruction_embed_dim=instruction_embed_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim
        )
        
        # 指令分类器 - 第一层任务
        self.instruction_classifier = nn.Sequential(
            nn.Linear(condition_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, instruction_vocab_size)
        )
        
        # 移除平滑约束模块 - 允许自然的动作变化
        # self.smoothness_constraint = SmoothnessConstraint(action_dim=action_dim)
    
    def forward(self, 
                instruction_ids: torch.Tensor,
                state: torch.Tensor,
                visual_features: Optional[torch.Tensor] = None,
                target_actions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            instruction_ids: 指令ID [batch_size]
            state: 状态信息 [batch_size, state_dim]
            visual_features: 视觉特征 [batch_size, visual_feature_dim]
            target_actions: 目标动作 [batch_size, chunk_length, action_dim] (可选)
            
        Returns:
            包含预测结果的字典
        """
        # 指令嵌入
        instruction_embed = self.instruction_embedding(instruction_ids)
        
        # 获取条件信息用于分类
        condition = self.action_predictor.condition_encoder(state, visual_features, instruction_embed)
        
        # 指令分类
        instruction_logits = self.instruction_classifier(condition)
        
        # 动作序列预测
        predicted_actions = self.action_predictor(
            state=state,
            visual_features=visual_features,
            instruction_embed=instruction_embed,
            target_actions=target_actions
        )
        
        return {
            'instruction_logits': instruction_logits,
            'predicted_actions': predicted_actions,
            'condition': condition
        }
    
    def generate_action_sequence(self, 
                                 instruction_ids: torch.Tensor,
                                 state: torch.Tensor,
                                 visual_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        生成动作序列
        
        Args:
            instruction_ids: 指令ID [batch_size]
            state: 状态信息 [batch_size, state_dim]
            visual_features: 视觉特征 [batch_size, visual_feature_dim]
            
        Returns:
            生成的动作序列 [batch_size, chunk_length, action_dim]
        """
        # 指令嵌入
        instruction_embed = self.instruction_embedding(instruction_ids)
        
        # 直接生成动作序列
        action_sequence = self.action_predictor(
            state=state,
            visual_features=visual_features,
            instruction_embed=instruction_embed,
            target_actions=None
        )
        
        # 移除平滑约束 - 直接返回原始预测
        # action_sequence = self.smoothness_constraint(action_sequence)
        
        return action_sequence


class SmoothnessConstraint(nn.Module):
    """
    平滑约束模块 - 大幅放宽约束以允许动态变化
    """
    
    def __init__(self, action_dim: int, max_diff: float = 0.5):  # 大幅增加允许的变化幅度
        super().__init__()
        self.action_dim = action_dim
        self.max_diff = max_diff
    
    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        应用非常宽松的平滑约束
        
        Args:
            actions: 动作序列 [batch_size, chunk_length, action_dim]
            
        Returns:
            平滑后的动作序列
        """
        batch_size, chunk_length, action_dim = actions.shape
        
        if chunk_length <= 1:
            return actions
        
        # 只限制极端变化，允许合理的动态变化
        diffs = torch.diff(actions, dim=1)
        
        # 使用更宽松的限制 - 只限制极端的跳跃
        diffs = torch.clamp(diffs, -self.max_diff, self.max_diff)
        
        # 重建动作序列
        smooth_actions = actions.clone()
        for t in range(1, chunk_length):
            smooth_actions[:, t, :] = smooth_actions[:, t-1, :] + diffs[:, t-1, :]
        
        return smooth_actions


def create_direct_action_model(config: Dict[str, Any] = None) -> DirectActionModel:
    """
    创建直接动作预测模型
    
    Args:
        config: 配置字典
        
    Returns:
        直接动作预测模型实例
    """
    if config is None:
        config = {}
    
    return DirectActionModel(
        action_dim=config.get('action_dim', 26),
        chunk_length=config.get('chunk_length', 16),
        state_dim=config.get('state_dim', 60),
        visual_feature_dim=config.get('visual_feature_dim', 512),
        instruction_vocab_size=config.get('instruction_vocab_size', 2),
        instruction_embed_dim=config.get('instruction_embed_dim', 64),
        condition_dim=config.get('condition_dim', 256),
        hidden_dim=config.get('hidden_dim', 512)
    )


def test_direct_action_model():
    """测试直接动作预测模型"""
    print("测试直接动作预测模型...")
    
    # 创建模型
    model = create_direct_action_model({
        'action_dim': 26,
        'chunk_length': 16,
        'state_dim': 60,
        'instruction_vocab_size': 2,
        'hidden_dim': 256
    })
    
    # 测试数据
    batch_size = 4
    state_dim = 60
    action_dim = 26
    chunk_length = 16
    
    instruction_ids = torch.tensor([0, 1, 0, 1])
    state = torch.randn(batch_size, state_dim)
    target_actions = torch.randn(batch_size, chunk_length, action_dim)
    
    # 前向传播
    outputs = model(instruction_ids, state, target_actions=target_actions)
    print(f"指令分类logits形状: {outputs['instruction_logits'].shape}")
    print(f"预测动作序列形状: {outputs['predicted_actions'].shape}")
    
    # 生成动作序列
    generated_actions = model.generate_action_sequence(instruction_ids, state)
    print(f"生成动作序列形状: {generated_actions.shape}")
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数数: {total_params:,}")
    
    print("直接动作预测模型测试完成！")


if __name__ == "__main__":
    test_direct_action_model()