#!/usr/bin/env python3
"""
流畅的ACT推理节点 - 恢复到原来的工作版本
核心逻辑：
1. 生成完整轨迹序列
2. 连续执行轨迹
3. 执行完毕后重新生成
"""

import rospy
import json
import time
import math
import numpy as np
import os
import sys
import argparse
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from std_msgs.msg import Float64MultiArray, String
from sensor_msgs.msg import JointState
from message_filters import Subscriber, ApproximateTimeSynchronizer
from std_srvs.srv import Trigger, TriggerResponse
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import TransformStamped
import json
import os
from ros_vla_language.msg import VLACommand

# 提前导入手臂控制相关的消息类型
try:
    from kuavo_msgs.msg import armTargetPoses
    from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest
    HAS_ARM_MSGS = True
except ImportError:
    rospy.logwarn("无法导入 kuavo_msgs，手臂控制功能将被禁用")
    HAS_ARM_MSGS = False

class KeyJointACTGenerator(nn.Module):
    """关键关节专注的ACT生成器 - 与训练脚本完全一致"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # 基础参数
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.num_instructions = config['num_instructions']
        self.hidden_dim = config['hidden_dim']
        self.trajectory_length = config['trajectory_length']
        self.dropout = config['dropout']
        
        # 关键关节数量 - 每个指令重点关注的前N个关节
        self.key_joints_per_instruction = config.get('key_joints_per_instruction', 8)
        
        # 差分预测标志
        self.predict_differences = config.get('predict_differences', False)
        
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
        
        # 损失权重
        self.classification_weight = 10.0
        self.diversity_weight = 5.0
        
        # 信号放大参数
        self.signal_amplification = config.get('signal_amplification', 1.0)
        
        # 动作历史上下文支持 - 与训练代码保持一致
        self.history_length = config.get('history_length', 128)  # 128步历史上下文
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
            
            # 预测关键关节 - 处理整个序列
            sequence_output = temporal_output[i]  # [sequence_length, hidden_dim]
            key_action = predictor(sequence_output)  # [sequence_length, key_joints_per_instruction]
            key_joint_actions.append(key_action)
        
        key_joint_actions = torch.stack(key_joint_actions, dim=0)  # [batch_size, sequence_length, key_joints_per_instruction]
        
        # 扩展到完整关节输出
        full_joint_actions = []
        for t in range(self.trajectory_length):
            key_at_t = key_joint_actions[:, t, :]
            full_at_t = self.full_joint_expander(key_at_t)
            full_joint_actions.append(full_at_t)
        
        predicted_actions = torch.stack(full_joint_actions, dim=1)
        
        return predicted_actions, instruction_logits, joint_importance, key_joint_actions

class SmoothACTInferenceNode:
    """流畅的ACT推理节点 - 恢复到原来的工作版本"""
    
    def __init__(self, model_path: str, config: dict):
        """初始化推理节点"""
        self.model_path = model_path
        self.config = config
        
        # 核心状态变量
        self.is_running = False  # 等待服务调用开始轨迹生成
        self.inference_frequency = config.get('inference_frequency', 30.0)
        self.trajectory_length = 32  # 测试：使用64步推理，与128步训练不匹配
        self.current_trajectory_step = 0
        
        # 动作完成检测机制
        self.initial_position = None  # 记录初始位置
        self.is_action_completed = False  # 动作是否完成
        self.action_start_time = None  # 动作开始时间
        
        # 动作状态管理
        self.action_state = "ready"  # ready -> executing -> completed
        
                                              
        # 轨迹管理 - 预生成模式实现平滑衔接
        self.trajectory_buffer = []
        self.is_generating = False
        self.next_trajectory_buffer = []  # 下一段轨迹缓存
        self.trajectory_blend_steps = 8  # 轨迹混合步数
        self.lookahead_trigger = 0.88  # 改为88%触发，在完整性和预生成时间之间找到平衡
        
        # 动作历史管理
        self.action_history_buffer = []
        self.max_history_length = 128
        
        # 执行周期检测 - 新增变量
        self.motion_velocity_history = []  # 运动速度历史
        self.motion_acceleration_history = []  # 运动加速度历史
        self.completion_detection_window = 16  # 完成检测窗口大小
        
        # 性能监控统计
        self.performance_stats = {
            'trajectory_segments': 0,
            'total_points_executed': 0,
            'repetitive_segments_detected': 0,
            'start_time': None,
            'total_distance_traveled': 0.0,
            'last_position_for_distance': None
        }
        
        # 控制参数
        self.instruction = config.get('instruction', 'wave')
        self.instruction_source = config.get('instruction_source', 'manual')
        self.control_mode = config.get('control_mode', 'arm')
        self.publish_commands = config.get('publish_commands', True)
        
        # 轨迹截断控制 - 新增配置
        self.enable_truncation = config.get('enable_truncation', True)  # 默认启用截断
        
        # 指令状态管理
        self.last_instruction_time = rospy.Time.now()
        self.instruction_change_threshold = rospy.Duration(1.0)  # 指令改变最小间隔1秒
        
        # 模型和标准化参数
        self.model = None
        self.norm_stats = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 动作维度信息
        self.action_dim = 26
        
        # 指令映射
        self.instruction_to_id = {'wave': 0, 'welcome': 1, 'sayhi': 2, 'thumbsup': 3}
        rospy.loginfo(f"指令映射: {self.instruction_to_id}")
        rospy.loginfo(f"当前指令: {self.instruction} -> ID: {self.instruction_to_id.get(self.instruction, 'unknown')}")
        
        # TF变换监听器 - 用于获取末端执行器的实际位置
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.base_frame = "base_link"  # 基座坐标系
        self.left_hand_frame = "zarm_l7_end_effector"  # 左手末端坐标系
        self.right_hand_frame = "zarm_r7_end_effector"  # 右手末端坐标系
        self.initial_left_hand_pos = None  # 左手初始位置
        self.initial_right_hand_pos = None  # 右手初始位置
        
        # 加载默认初始位置
        default_joint, default_left_tf, default_right_tf = self._load_default_position()
        self.default_joint_position = default_joint
        self.default_left_hand_pos = default_left_tf
        self.default_right_hand_pos = default_right_tf
        
        if self.default_joint_position is not None:
            rospy.loginfo("已加载默认初始位置")
        else:
            rospy.logwarn("未找到默认初始位置配置文件")
        
        # 机器人状态
        self.current_joint_positions = None
        
        # 初始化模型
        self._load_model()
        
        # 初始化ROS接口
        self._setup_ros_interfaces()
        
        # 如果是手臂控制模式，设置手臂控制模式
        if self.control_mode == 'arm':
            self._setup_arm_control()
        
        rospy.loginfo("ACT推理节点初始化完成")
        rospy.loginfo(f"指令: {self.instruction}")
        rospy.loginfo(f"指令来源: {self.instruction_source}")
        rospy.loginfo(f"控制模式: {self.control_mode}")
        rospy.loginfo(f"推理频率: {self.inference_frequency}Hz")
    
    def _load_model(self):
        """加载训练好的模型"""
        try:
            rospy.loginfo(f"加载模型: {self.model_path}")
            
            # 加载checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # 获取配置和标准化参数
            if 'model_config' in checkpoint:
                model_config = checkpoint['model_config']
            else:
                model_config = checkpoint['config']
            
            if 'norm_stats' in checkpoint:
                self.norm_stats = checkpoint['norm_stats']
            else:
                # 使用默认的标准化参数
                self.norm_stats = {
                    'state_mean': np.zeros(26),
                    'state_std': np.ones(26),
                    'action_mean': np.zeros(26),
                    'action_std': np.ones(26)
                }
            
            rospy.loginfo(f"模型配置: {model_config}")
            rospy.loginfo(f"标准化参数键: {list(self.norm_stats.keys())}")
            
            # 创建模型
            self.model = KeyJointACTGenerator(model_config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # 保持16步轨迹长度，不使用模型的32步设置
            # self.trajectory_length = self.model.trajectory_length  # 注释掉这行
            rospy.loginfo(f"使用16步轨迹长度（模型原始长度: {self.model.trajectory_length}）")
            
            rospy.loginfo("模型加载成功")
            rospy.loginfo(f"模型设备: {self.device}")
            
                        
        except Exception as e:
            rospy.logerr(f"加载模型失败: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            raise
    
    def _setup_ros_interfaces(self):
        """设置ROS接口"""
        try:
            # 订阅机器人状态话题
            self.joint_state_sub = rospy.Subscriber(
                '/humanoid_controller/optimizedState_mrt/joint_pos',
                Float64MultiArray,
                self._joint_state_callback
            )
            
            # 指令话题订阅器（仅在topic模式下启用）
            if self.instruction_source == 'topic':
                self.instruction_sub = rospy.Subscriber(
                    '/vla_control/command',
                    VLACommand,
                    self._instruction_callback
                )
                rospy.loginfo("话题模式 - 订阅 /vla_control/command 话题获取指令")
            
            # 动作命令发布器
            if self.control_mode == 'arm':
                if HAS_ARM_MSGS:
                    self.arm_target_pub = rospy.Publisher(
                        '/kuavo_arm_target_poses',
                        armTargetPoses,
                        queue_size=10
                    )
                    rospy.loginfo("手臂控制模式 - 发布手臂目标姿态")
                else:
                    rospy.logerr("无法创建手臂发布器：缺少 kuavo_msgs")
            
            # 推理控制服务
            self.start_service = rospy.Service(
                '/smooth_act_inference/start',
                Trigger,
                self._start_callback
            )
            
            self.stop_service = rospy.Service(
                '/smooth_act_inference/stop',
                Trigger,
                self._stop_callback
            )
            
            rospy.loginfo("ROS接口设置完成")
            rospy.loginfo(f"指令模式: {self.instruction_source}")
            if self.instruction_source == 'topic':
                rospy.loginfo("  指令话题: /vla_control/command")
                rospy.loginfo("  支持的指令: wave, welcome, sayhi, thumbsup, none")
            rospy.loginfo("控制服务:")
            rospy.loginfo("  开始推理: rosservice call /smooth_act_inference/start")
            rospy.loginfo("  停止推理: rosservice call /smooth_act_inference/stop")
            
        except Exception as e:
            rospy.logerr(f"设置ROS接口失败: {e}")
            raise
    
    def _setup_arm_control(self):
        """设置手臂控制模式"""
        if not HAS_ARM_MSGS:
            rospy.logerr("无法设置手臂控制模式：缺少 kuavo_msgs")
            return
            
        try:
            # 等待手臂控制模式服务
            rospy.wait_for_service('/arm_traj_change_mode', timeout=5.0)
            
            # 创建服务客户端
            change_mode = rospy.ServiceProxy('/arm_traj_change_mode', changeArmCtrlMode)
            
            # 创建请求
            req = changeArmCtrlModeRequest()
            req.control_mode = 2  # EXTERN_CONTROL (外部控制模式)
            
            # 调用服务
            res = change_mode(req)
            
            if res.result:
                rospy.loginfo("手臂控制模式已设置为: EXTERN_CONTROL (外部控制)")
            else:
                rospy.logerr(f"设置手臂控制模式失败: {res.message}")
                
        except Exception as e:
            rospy.logerr(f"设置手臂控制模式时出错: {e}")
            rospy.logwarn("手臂可能不会响应外部控制命令")
    
    def _joint_state_callback(self, msg: Float64MultiArray):
        """关节状态回调"""
        self.current_joint_positions = np.array(msg.data[:26])
        rospy.logdebug(f"接收到关节位置数据，前3个关节: {self.current_joint_positions[:3]}")
    
    def _instruction_callback(self, msg: VLACommand):
        """指令话题回调"""
        try:
            rospy.loginfo(f"📢 收到指令回调: instruction='{msg.instruction}', is_running={self.is_running}")
            
            current_time = rospy.Time.now()
            
            # 检查时间间隔，防止指令变化过于频繁
            if current_time - self.last_instruction_time < self.instruction_change_threshold:
                rospy.logdebug(f"指令变化过于频繁，忽略: {msg.instruction}")
                return
            
            new_instruction = msg.instruction.strip().lower()
            
            rospy.loginfo(f"📢 指令详情: new_instruction='{new_instruction}', current_instruction='{self.instruction}'")
            
            # 验证指令有效性
            valid_instructions = ['wave', 'welcome', 'sayhi', 'thumbsup', 'none']
            if new_instruction not in valid_instructions:
                rospy.logwarn(f"收到无效指令: {new_instruction}，支持的指令: {valid_instructions}")
                return
            
            # 检查指令是否真的发生了变化
            if new_instruction != self.instruction:
                rospy.loginfo(f"指令更新: {self.instruction} -> {new_instruction}")
                
                # 更新指令
                old_instruction = self.instruction
                self.instruction = new_instruction
                self.last_instruction_time = current_time
                
                # 重置动作状态，准备执行新的指令
                self._reset_action_state()
                
                # 启动推理
                if not self.is_running:
                    rospy.loginfo(f"接收到指令: {self.instruction}，开始推理...")
                    self.is_running = True
                else:
                    rospy.loginfo(f"指令已更新为: {self.instruction}，继续推理...")
                
                # 如果是none指令，停止当前动作
                if new_instruction == 'none':
                    rospy.loginfo("收到none指令，停止当前动作")
                    self.is_action_completed = True
                    self.action_state = "completed"
                    self.trajectory_buffer = []
                    self.next_trajectory_buffer = []
                    self.current_trajectory_step = 0
            else:
                rospy.loginfo(f"📢 指令未变化，但仍然检查是否需要启动推理: {self.instruction}")
                # 如果指令相同但推理未启动，也启动推理
                if not self.is_running:
                    rospy.loginfo(f"指令相同但推理未启动，强制启动推理: {self.instruction}")
                    self.is_running = True
                
        except Exception as e:
            rospy.logerr(f"处理指令回调失败: {e}")
    
    def _reset_action_state(self):
        """重置动作状态，准备执行新指令"""
        rospy.loginfo("重置动作状态，准备执行新指令")
        
        # 清空轨迹缓冲区
        self.trajectory_buffer = []
        self.next_trajectory_buffer = []
        self.current_trajectory_step = 0
        
        # 重置状态变量
        self.initial_position = None
        self.initial_left_hand_pos = None
        self.initial_right_hand_pos = None
        self.initial_tf_time = None
        self.action_start_time = None
        self.max_distance_reached = 0.0
        self.is_action_completed = False
        self.action_state = "ready"
        
        # 重置计数器
        if hasattr(self, 'completion_counter'):
            self.completion_counter = 0
        
        # 重置动作历史
        self.action_history_buffer = []
        self.motion_velocity_history = []
        self.motion_acceleration_history = []
        
        rospy.loginfo("动作状态重置完成")
    
    def _load_default_position(self):
        """加载默认初始位置"""
        config_file = os.path.join(os.path.dirname(__file__), 'default_initial_position.json')
        
        if not os.path.exists(config_file):
            rospy.logwarn(f"默认位置配置文件不存在: {config_file}")
            return None, None, None
            
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            joint_positions = np.array(config['joint_positions'])
            rospy.loginfo(f"加载默认关节位置: {len(joint_positions)}个关节")
            rospy.loginfo(f"前3个关节: {joint_positions[:3]}")
            
            # 加载TF位置
            tf_positions = config.get('tf_positions', {})
            if tf_positions:
                left_hand_pos = np.array(tf_positions['left_hand'])
                right_hand_pos = np.array(tf_positions['right_hand'])
                rospy.loginfo(f"加载默认TF位置 - 左手: [{left_hand_pos[0]:.3f}, {left_hand_pos[1]:.3f}, {left_hand_pos[2]:.3f}]")
                rospy.loginfo(f"加载默认TF位置 - 右手: [{right_hand_pos[0]:.3f}, {right_hand_pos[1]:.3f}, {right_hand_pos[2]:.3f}]")
                return joint_positions, left_hand_pos, right_hand_pos
            else:
                rospy.logwarn("配置文件中没有TF位置信息")
                return joint_positions, None, None
            
        except Exception as e:
            rospy.logerr(f"加载默认位置失败: {e}")
            return None, None, None
    
    def _generate_action_history(self, current_position):
        """生成动作历史上下文 - 使用当前位置填充"""
        if len(self.action_history_buffer) == 0:
            # 如果没有历史数据，使用当前位置填充
            history_data = np.tile(current_position, (self.max_history_length, 1))
        else:
            # 获取历史数据
            history_data = np.array(self.action_history_buffer[-self.max_history_length:])
            
            # 如果历史不够长，用当前位置填充
            if len(history_data) < self.max_history_length:
                padding = np.tile(current_position, (self.max_history_length - len(history_data), 1))
                history_data = np.vstack([padding, history_data])
        
        # 确保历史长度正确
        history_data = history_data[:self.max_history_length]
        
        # 展平历史数据
        return history_data.flatten()
    
    def _update_action_history(self, action):
        """更新动作历史缓冲区"""
        # 将新动作添加到历史缓冲区
        self.action_history_buffer.append(action.copy())
        
        # 保持历史缓冲区在合理范围内
        if len(self.action_history_buffer) > self.max_history_length * 2:
            self.action_history_buffer = self.action_history_buffer[-self.max_history_length * 2:]
    
        
        
        
    def _should_regenerate_trajectory(self):
        """判断是否需要重新生成轨迹 - 预生成模式实现无缝衔接"""
        # 如果正在生成，跳过
        if self.is_generating:
            return False
        
        # 如果动作已完成，不再生成
        if self.is_action_completed:
            return False
        
        # 如果轨迹缓冲区为空，生成主轨迹
        if len(self.trajectory_buffer) == 0:
            return True
        
        # 预生成模式：当当前轨迹执行到75%时，开始生成下一段轨迹
        progress_ratio = self.current_trajectory_step / len(self.trajectory_buffer)
        
        print(f"🔍 checking regeneration: progress={progress_ratio:.2f}, step={self.current_trajectory_step}/{len(self.trajectory_buffer)}, next_buffer={len(self.next_trajectory_buffer)}")
        
        # 条件1：当执行到88%且没有下一段轨迹时，预生成
        if progress_ratio >= self.lookahead_trigger and len(self.next_trajectory_buffer) == 0:
            print(f"🎯 PRE-GENERATION TRIGGER: {progress_ratio*100:.0f}% completed, generating next trajectory!")
            rospy.loginfo(f"预生成模式：当前轨迹执行{progress_ratio*100:.0f}%，开始生成下一段轨迹")
            return True
        
        # 条件2：当前轨迹完全执行完毕且有下一段轨迹，直接切换
        if self.current_trajectory_step >= len(self.trajectory_buffer) and len(self.next_trajectory_buffer) > 0:
            print(f"🔄 SWITCHING: Current trajectory completed, switching to pre-generated trajectory")
            rospy.loginfo("轨迹执行完毕，切换到预生成的下一段轨迹")
            self._switch_to_next_trajectory()
            return False
        
        # 条件3：当前轨迹完全执行完毕但没有下一段轨迹（异常情况），立即生成
        if self.current_trajectory_step >= len(self.trajectory_buffer) and len(self.next_trajectory_buffer) == 0:
            print(f"🚨 EMERGENCY: No pre-generated trajectory available!")
            rospy.logwarn("紧急情况：没有预生成轨迹，立即生成新轨迹")
            return True
        
        return False
    
    def _switch_to_next_trajectory(self):
        """切换到下一段轨迹 - 预生成模式实现无缝切换"""
        rospy.loginfo(f"开始切换到预生成的下一段轨迹 - 当前步={self.current_trajectory_step}")
        
        if len(self.next_trajectory_buffer) == 0:
            rospy.logwarn("尝试切换到空的下一段轨迹")
            return
        
        # 记录切换前的最后位置
        last_position = None
        if len(self.trajectory_buffer) > 0 and self.current_trajectory_step > 0:
            last_position = np.array(self.trajectory_buffer[self.current_trajectory_step - 1])
            rospy.loginfo(f"切换前最后位置: {last_position[:3]}")
        elif len(self.trajectory_buffer) > 0:
            last_position = np.array(self.trajectory_buffer[-1])
            rospy.loginfo(f"切换前轨迹终点: {last_position[:3]}")
        
        # 将下一段轨迹设为主轨迹
        self.trajectory_buffer = self.next_trajectory_buffer.copy()
        self.next_trajectory_buffer = []
        self.current_trajectory_step = 0
        
        rospy.loginfo(f"轨迹切换完成 - 新轨迹长度={len(self.trajectory_buffer)}")
        if len(self.trajectory_buffer) > 0:
            rospy.loginfo(f"新轨迹起始位置: {self.trajectory_buffer[0][:3]}")
            
            # 验证衔接的平滑性
            if last_position is not None:
                connection_distance = np.linalg.norm(last_position[:6] - np.array(self.trajectory_buffer[0])[:6])
                rospy.loginfo(f"衔接距离检查: {connection_distance:.6f}m")
                if connection_distance > 0.01:
                    rospy.logwarn(f"轨迹衔接距离较大: {connection_distance:.6f}m，可能导致停顿")
    
    def _blend_trajectories(self, current_end, next_start, blend_steps):
        """改进的轨迹混合 - 考虑速度平滑过渡"""
        if blend_steps <= 0:
            return next_start
        
        # 创建平滑的混合权重（使用缓动函数）
        t = np.linspace(0.0, 1.0, blend_steps)
        # 使用sin函数创建平滑过渡，避免线性插值的生硬感
        smooth_weights = 0.5 * (1 - np.cos(t * np.pi))  # cosine插值
        
        # 确保长度足够
        current_end_extended = np.tile(current_end, (blend_steps, 1))
        next_start_extended = np.tile(next_start, (blend_steps, 1))
        
        # 平滑混合
        blended_trajectory = []
        for i in range(blend_steps):
            alpha = smooth_weights[i]
            blended_step = (1 - alpha) * current_end_extended[i] + alpha * next_start_extended[i]
            blended_trajectory.append(blended_step)
        
        return np.array(blended_trajectory)
    
    def _generate_next_trajectory_seamlessly(self, current_position):
        """生成下一段轨迹 - 使用最简单的衔接逻辑"""
        try:
            rospy.loginfo("开始生成下一段轨迹")
            
            # 使用机器人实际位置
            if self.current_joint_positions is not None:
                actual_current_position = self.current_joint_positions
                rospy.loginfo("使用机器人实际位置生成下一段轨迹")
            else:
                actual_current_position = current_position
                rospy.logwarn("无法获取实际位置，使用传入位置")
            
            # 生成下一段轨迹
            next_trajectory = self._generate_trajectory_from_model(actual_current_position)
            
            # 简单的轨迹衔接 - 暂时禁用重复检测
            if len(self.trajectory_buffer) > 0:
                # 获取当前轨迹的最后位置
                current_last_position = self.trajectory_buffer[-1]
                next_first_position = next_trajectory[0]
                
                rospy.loginfo(f"轨迹衔接检查:")
                rospy.loginfo(f"  当前轨迹终点: {current_last_position[:3]}")
                rospy.loginfo(f"  下一段轨迹起点: {next_first_position[:3]}")
                rospy.loginfo(f"  起点距离: {np.linalg.norm(current_last_position[:8] - next_first_position[:8]):.6f}")
                
                # 暂时跳过重复检测，直接使用原始轨迹
                cleaned_next_trajectory = next_trajectory
                
                # 创建平滑过渡轨迹
                blend_steps = 12  # 固定12步混合
                transition_trajectory = self._blend_trajectories(
                    current_last_position, cleaned_next_trajectory[0], blend_steps
                )
                
                # 组合轨迹：过渡轨迹 + 下一段轨迹
                self.next_trajectory_buffer = transition_trajectory.tolist() + cleaned_next_trajectory.tolist()
                rospy.loginfo(f"轨迹衔接完成 - 过渡长度={blend_steps}, 总长度={len(self.next_trajectory_buffer)}")
            else:
                self.next_trajectory_buffer = next_trajectory.tolist()
                rospy.loginfo(f"直接使用新生成轨迹 - 长度={len(self.next_trajectory_buffer)}")
            
        except Exception as e:
            rospy.logerr(f"生成下一段轨迹失败: {e}")
            self.next_trajectory_buffer = []
    
    def _predict_trajectory_end(self):
        """预测当前轨迹的结束位置 - 使用更准确的预测方法"""
        if len(self.trajectory_buffer) == 0:
            return self.current_joint_positions.copy()
        
        # 如果已经执行到88%以上，直接使用轨迹的实际终点
        # 这样可以避免预测误差，减少回溯问题
        if self.current_trajectory_step >= int(len(self.trajectory_buffer) * 0.88):
            # 88%以后直接使用真实终点，不再预测
            actual_end = np.array(self.trajectory_buffer[-1])
            rospy.loginfo(f"轨迹结束预测: 88%后使用真实终点={actual_end[:3]}")
            return actual_end
        
        # 如果已经执行了一部分，基于实际执行进度预测
        if self.current_trajectory_step > 0:
            # 使用已执行轨迹的趋势来预测剩余部分的结束位置
            executed_portion = np.array(self.trajectory_buffer[:self.current_trajectory_step])
            
            if len(executed_portion) >= 3:
                # 计算执行部分的移动趋势
                movement_trend = executed_portion[-1] - executed_portion[0]
                
                # 预测结束位置：当前位置 + 趋势的适当延伸
                current_pos = executed_portion[-1]
                predicted_end = current_pos + movement_trend * 0.1  # 更保守的预测
                
                rospy.loginfo(f"轨迹结束预测: 当前位置={current_pos[:3]}, 预测结束={predicted_end[:3]}")
                return predicted_end
        
        # 默认使用轨迹的最后位置
        return np.array(self.trajectory_buffer[-1])
    
        
      
    def _update_performance_stats(self, current_position):
        """更新性能统计和距离跟踪"""
        # 初始化开始时间
        if self.performance_stats['start_time'] is None:
            self.performance_stats['start_time'] = rospy.Time.now()
        
        # 更新执行的点数
        self.performance_stats['total_points_executed'] += 1
        
        # 计算累积移动距离
        if self.performance_stats['last_position_for_distance'] is not None:
            step_distance = np.linalg.norm(current_position[:6] - self.performance_stats['last_position_for_distance'][:6])
            self.performance_stats['total_distance_traveled'] += step_distance
        
        self.performance_stats['last_position_for_distance'] = current_position.copy()
        
        # 每1000步报告一次性能统计
        if self.performance_stats['total_points_executed'] % 1000 == 0:
            elapsed_time = (rospy.Time.now() - self.performance_stats['start_time']).to_sec()
            if elapsed_time > 0:
                avg_distance_per_point = self.performance_stats['total_distance_traveled'] / self.performance_stats['total_points_executed']
                rospy.loginfo("=== 性能统计报告 ===")
                rospy.loginfo(f"  执行点数: {self.performance_stats['total_points_executed']}")
                rospy.loginfo(f"  累积距离: {self.performance_stats['total_distance_traveled']:.6f}m")
                rospy.loginfo(f"  平均每点距离: {avg_distance_per_point:.6f}m")
                rospy.loginfo(f"  执行时间: {elapsed_time:.1f}s")
                rospy.loginfo(f"  轨迹段数: {self.performance_stats['trajectory_segments']}")
                if hasattr(self, 'truncation_stats'):
                    rospy.loginfo(f"  截断次数: {self.truncation_stats.get('count', 0)}")
    
    def _generate_new_trajectory(self):
        """生成新的轨迹 - 预生成模式实现无缝衔接"""
        try:
            if self.current_joint_positions is None:
                rospy.logwarn("当前关节位置未知，无法生成轨迹")
                return
            
            if self.is_generating:
                rospy.logdebug("正在生成轨迹，跳过")
                return
            
            self.is_generating = True
            current_robot_position = self.current_joint_positions.copy()
            
            # 检查动作是否已经完成（防止无限重复）
            if self._is_action_completed():
                rospy.loginfo(f"动作 '{self.instruction}' 已完成，停止生成新轨迹")
                self._stop_current_action()
                self.is_generating = False
                return

            # 判断是生成主轨迹还是预生成下一段轨迹
            if len(self.trajectory_buffer) == 0:
                rospy.loginfo("生成主轨迹段 - 从当前位置开始")
                
                # 记录动作初始位置
                self.initial_position = current_robot_position.copy()
                self.action_start_time = rospy.Time.now()
                rospy.loginfo(f"记录动作初始位置: {self.initial_position[:3]}")
                
                # 生成主轨迹
                predicted_actions = self._generate_trajectory_from_model(current_robot_position)
                
                print(f"🔍 DEBUG: 生成主轨迹，长度={len(predicted_actions)}")
                                
                # 设置轨迹缓冲区
                self.trajectory_buffer = predicted_actions.tolist()
                self.current_trajectory_step = 0
                
                rospy.loginfo(f"主轨迹生成完成 - 长度: {len(self.trajectory_buffer)}")
                
            else:
                # 预生成模式：生成下一段轨迹并存储到next_trajectory_buffer
                rospy.loginfo("预生成模式：生成下一段轨迹")
                
                # 预测当前轨迹的结束位置
                predicted_end_position = self._predict_trajectory_end()
                
                # 基于预测的结束位置生成下一段轨迹
                next_predicted_actions = self._generate_trajectory_from_model(predicted_end_position)
                
                # 智能重复检测和截断
                if len(self.trajectory_buffer) > 0:
                    # 获取当前轨迹的最后几个点
                    last_trajectory_end = np.array(self.trajectory_buffer[-min(10, len(self.trajectory_buffer)):])
                    
                    # 应用重复检测和截断
                    print(f"🔍 DEBUG: 预生成轨迹截断，原长度={len(next_predicted_actions)}")
                    final_next_trajectory = self._remove_trajectory_repetition(next_predicted_actions, last_trajectory_end)
                    print(f"🔍 DEBUG: 预生成轨迹截断完成，新长度={len(final_next_trajectory)}")
                    
                    rospy.loginfo(f"预生成轨迹重复检测: 原长度={len(next_predicted_actions)}, 截断后长度={len(final_next_trajectory)}")
                else:
                    final_next_trajectory = next_predicted_actions
                
                # 存储到next_trajectory_buffer，不立即使用
                self.next_trajectory_buffer = final_next_trajectory.tolist()
                
                rospy.loginfo(f"下一段轨迹预生成完成 - 长度: {len(self.next_trajectory_buffer)}")
                rospy.loginfo("预生成轨迹已就绪，等待当前轨迹执行到75%时自动切换")
                
                # 更新轨迹段统计
                self.performance_stats['trajectory_segments'] += 1
            
            self.is_generating = False
            
        except Exception as e:
            rospy.logerr(f"生成轨迹失败: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            self.is_generating = False
    
    def _generate_trajectory_from_model(self, current_position):
        """直接使用模型预测轨迹，不做任何扩展"""
        try:
            # 生成动作历史上下文
            action_history = self._generate_action_history(current_position)
            
            # 获取指令ID
            instruction_id = self.instruction_to_id.get(self.instruction, 0)
            # 设置当前指令ID用于轨迹连接逻辑
            self.current_instruction_id = instruction_id
            
            # 标准化起始状态
            start_state_norm = (current_position - self.norm_stats['state_mean']) / self.norm_stats['state_std']
            
            # 转换为tensor
            start_state_tensor = torch.FloatTensor(start_state_norm).unsqueeze(0).to(self.device)
            instruction_id_tensor = torch.LongTensor([instruction_id]).to(self.device)
            action_history_tensor = torch.FloatTensor(action_history).unsqueeze(0).to(self.device)
            
            # 使用模型预测轨迹
            with torch.no_grad():
                outputs = self.model(start_state_tensor, instruction_id_tensor, action_history=action_history_tensor)
                predicted_actions_norm = outputs[0]  # 第一个输出是预测动作
            
            # 反标准化
            predicted_actions_full = predicted_actions_norm.cpu().numpy()[0] * self.norm_stats['action_std'] + self.norm_stats['action_mean']
            
            # 只取前16步，因为我们现在使用16步轨迹
            predicted_actions = predicted_actions_full[:self.trajectory_length]
            
            rospy.loginfo(f"轨迹截取: {predicted_actions_full.shape[0]}步 -> {predicted_actions.shape[0]}步")
            
            # 总是应用轨迹对齐 - 确保轨迹起点连续性，消除速度突变
            start_distance = np.linalg.norm(predicted_actions[0][:6] - current_position[:6])
            rospy.loginfo(f"轨迹对齐检查 - 起点偏差: {start_distance:.6f}")
            
            rospy.loginfo("应用轨迹对齐确保连续性")
            aligned_trajectory = self._align_trajectory_to_position(predicted_actions, current_position)
            
            # 应用轨迹平滑处理，减少速度突变
            smoothed_trajectory = self._smooth_trajectory_speed(aligned_trajectory)
            return smoothed_trajectory
            
        except Exception as e:
            rospy.logerr(f"模型预测失败: {e}")
            # 返回简单的静态轨迹
            static_trajectory = np.tile(current_position, (self.model.trajectory_length, 1))
            return static_trajectory
    
    def _align_trajectory_to_position(self, trajectory, current_position):
        """轨迹对齐：智能处理轨迹起点与实际位置不匹配的问题"""
        if len(trajectory) == 0:
            return trajectory
            
        # 使用关键关节数量进行对齐
        key_joints = 8
        
        # 计算轨迹中每个点与当前位置的距离
        distances = []
        for i, pose in enumerate(trajectory):
            pose_key = pose[:key_joints]
            current_key = current_position[:key_joints]
            distance = np.linalg.norm(pose_key - current_key)
            distances.append(distance)
        
        # 找到距离最近的点
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]
        
        rospy.loginfo(f"轨迹对齐：")
        rospy.loginfo(f"  轨迹长度: {len(trajectory)}")
        rospy.loginfo(f"  最近点索引: {min_distance_idx}")
        rospy.loginfo(f"  最近点距离: {min_distance:.6f}")
        rospy.loginfo(f"  轨迹起点距离: {distances[0]:.6f}")
        
        # 智能轨迹对齐：考虑截断后的轨迹长度
        if min_distance_idx > 0 and min_distance > 0.05:  # 基础截断条件
            # 计算截断后的轨迹长度
            truncated_length = len(trajectory) - min_distance_idx
            
            # 只有在截断后轨迹足够长时才截断
            if truncated_length >= 8:
                rospy.loginfo(f"  截断轨迹：跳过前{min_distance_idx}个点，保留{truncated_length}个点")
                return trajectory[min_distance_idx:]
            else:
                rospy.loginfo(f"  截断后轨迹过短({truncated_length}个点)，保持原轨迹")
                return trajectory
        else:
            rospy.loginfo(f"  无需截断，保持原轨迹")
            return trajectory
    
    def _update_motion_analysis(self, current_position):
        """更新运动分析数据"""
        # 计算速度：当前位置与历史中最后一个位置的差值
        if len(self.action_history_buffer) >= 1:
            dt = 1.0 / self.inference_frequency
            # 用当前位置减去历史中最后一个位置（即上一个位置）
            velocity = (current_position - self.action_history_buffer[-1]) / dt
            self.motion_velocity_history.append(velocity)
            
            # 保持历史长度
            if len(self.motion_velocity_history) > self.max_history_length:
                self.motion_velocity_history = self.motion_velocity_history[-self.max_history_length:]
            
            # 计算加速度（速度差值除以时间间隔）
            if len(self.motion_velocity_history) >= 2:
                acceleration = (self.motion_velocity_history[-1] - self.motion_velocity_history[-2]) / dt
                self.motion_acceleration_history.append(acceleration)
                
                # 保持历史长度
                if len(self.motion_acceleration_history) > self.max_history_length:
                    self.motion_acceleration_history = self.motion_acceleration_history[-self.max_history_length:]
    
        
    def _intelligent_action_completion(self, recent_trajectory, current_position):
        """改进的动作完成检测 - 防止中间误判"""
        rospy.loginfo(f"执行动作完成检测")
        
        if not hasattr(self, 'initial_position') or self.initial_position is None:
            rospy.loginfo("初始位置未设置，跳过检测")
            return False
        
        # 1. 防误判：开始5秒不检测（基于真实时间）
        if self.action_start_time is None:
            rospy.loginfo("动作开始时间未设置，跳过检测")
            return False
            
        elapsed_time = (rospy.Time.now() - self.action_start_time).to_sec()
        if elapsed_time < 5.0:
            rospy.loginfo(f"执行时间不足5秒({elapsed_time:.1f}s)，跳过检测")
            return False
        
        # 2. 计算末端关节距离 - 简化方法，只判断末端位置
        # 末端关节：左手末端（关节17），右手末端（关节25）
        # 这是判断是否回到初始位置的最直观方法
        
        # 获取当前指令类型
        current_instruction = getattr(self, 'current_instruction_id', 0)
        instruction_map = {0: 'wave', 1: 'welcome', 2: 'sayhi', 3: 'thumbsup'}
        action_type = instruction_map.get(current_instruction, 'wave')
        
        # 统一的阈值设置 - 不区分动作类型
        distance_threshold = 0.15  # 统一距离阈值（提高到0.15）
        
        # 使用简化的TF坐标计算距离
        try:
            # 直接获取基座到末端的TF变换（不依赖复杂的相对计算）
            current_time = rospy.Time.now()
            
            # 获取左手末端位置
            left_transform = self.tf_buffer.lookup_transform(
                self.base_frame, self.left_hand_frame, rospy.Time(0), rospy.Duration(0.1))
            current_left_pos = np.array([
                left_transform.transform.translation.x,
                left_transform.transform.translation.y,
                left_transform.transform.translation.z
            ])
            
            # 获取右手末端位置
            right_transform = self.tf_buffer.lookup_transform(
                self.base_frame, self.right_hand_frame, rospy.Time(0), rospy.Duration(0.1))
            current_right_pos = np.array([
                right_transform.transform.translation.x,
                right_transform.transform.translation.y,
                right_transform.transform.translation.z
            ])
            
            # 设置初始位置（使用默认TF位置或当前位置）
            if self.initial_left_hand_pos is None:
                if self.default_left_hand_pos is not None and self.default_right_hand_pos is not None:
                    # 直接使用预设的默认TF位置
                    self.initial_left_hand_pos = self.default_left_hand_pos.copy()
                    self.initial_right_hand_pos = self.default_right_hand_pos.copy()
                    rospy.loginfo("使用预设的默认TF位置")
                else:
                    # 使用当前位置作为初始位置
                    self.initial_left_hand_pos = current_left_pos.copy()
                    self.initial_right_hand_pos = current_right_pos.copy()
                    rospy.loginfo("使用当前位置作为初始TF位置")
                
                self.action_start_time = rospy.Time.now()
                rospy.loginfo(f"动作开始时间: {self.action_start_time.to_sec()}")
            
            # 直接计算欧几里得距离
            left_distance = np.linalg.norm(current_left_pos - self.initial_left_hand_pos)
            right_distance = np.linalg.norm(current_right_pos - self.initial_right_hand_pos)
            
            # 调试信息
            rospy.loginfo(f"  左手TF位置: [{current_left_pos[0]:.3f}, {current_left_pos[1]:.3f}, {current_left_pos[2]:.3f}]")
            rospy.loginfo(f"  右手TF位置: [{current_right_pos[0]:.3f}, {current_right_pos[1]:.3f}, {current_right_pos[2]:.3f}]")
            
            # 显示初始位置来源
            if self.default_left_hand_pos is not None and self.default_right_hand_pos is not None:
                rospy.loginfo(f"  左手初始位置(默认): [{self.initial_left_hand_pos[0]:.3f}, {self.initial_left_hand_pos[1]:.3f}, {self.initial_left_hand_pos[2]:.3f}]")
                rospy.loginfo(f"  右手初始位置(默认): [{self.initial_right_hand_pos[0]:.3f}, {self.initial_right_hand_pos[1]:.3f}, {self.initial_right_hand_pos[2]:.3f}]")
            else:
                rospy.loginfo(f"  左手初始位置(实时): [{self.initial_left_hand_pos[0]:.3f}, {self.initial_left_hand_pos[1]:.3f}, {self.initial_left_hand_pos[2]:.3f}]")
                rospy.loginfo(f"  右手初始位置(实时): [{self.initial_right_hand_pos[0]:.3f}, {self.initial_right_hand_pos[1]:.3f}, {self.initial_right_hand_pos[2]:.3f}]")
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"TF变换失败，使用关节角度估算: {e}")
            # 降级到简化的关节角度计算
            left_distance = abs(current_position[17] - self.initial_position[17]) if self.initial_position is not None else 0
            right_distance = abs(current_position[25] - self.initial_position[25]) if self.initial_position is not None else 0
            rospy.loginfo(f"  降级模式 - 左手关节17距离: {left_distance:.6f}")
            rospy.loginfo(f"  降级模式 - 右手关节25距离: {right_distance:.6f}")
        
        # 设置初始位置（使用默认位置或当前位置）
        if self.initial_position is None:
            if self.default_joint_position is not None:
                # 使用预设的默认初始位置
                self.initial_position = self.default_joint_position.copy()
                rospy.loginfo("使用预设的默认初始位置")
            else:
                # 如果没有预设位置，使用当前位置作为初始位置
                self.initial_position = current_position.copy()
                rospy.loginfo(f"使用当前位置作为初始位置: {len(self.initial_position)}个关节")
            
            self.action_start_time = rospy.Time.now()
            rospy.loginfo(f"动作开始时间: {self.action_start_time.to_sec()}")
        
        # 计算关节统计信息
        if self.initial_position is not None:
            joint_changes = np.abs(current_position - self.initial_position)
            active_joint_count = np.sum(joint_changes > 0.01)
            avg_change = np.mean(joint_changes)
            max_change = np.max(joint_changes)
            std_change = np.std(joint_changes)
            total_motion = np.sum(joint_changes)
            significant_joints = [(i, change) for i, change in enumerate(joint_changes) if change > 0.01]
        else:
            joint_changes = np.zeros(len(current_position))
            active_joint_count = 0
            avg_change = 0.0
            max_change = 0.0
            std_change = 0.0
            total_motion = 0.0
            significant_joints = []
        
        # 详细日志
        rospy.loginfo(f"  关节统计: 总数={len(joint_changes)}, 活跃={active_joint_count}")
        rospy.loginfo(f"  角度变化: 平均={avg_change:.4f}, 最大={max_change:.4f}, 标准差={std_change:.4f}")
        rospy.loginfo(f"  运动指标: 总运动量={total_motion:.4f}")
        rospy.loginfo(f"  显著关节: {len(significant_joints)}个")
        rospy.loginfo(f"  估算距离: {left_distance:.6f}")
        
        # 显示默认位置和当前位置的对比（前5个关节）
        if self.default_joint_position is not None:
            default_top5 = self.default_joint_position[:5]
            current_top5 = current_position[:5]
            comparison = [f"关节{i}:{d:.3f}->{c:.3f}" for i, (d, c) in enumerate(zip(default_top5, current_top5))]
            rospy.loginfo(f"  默认vs当前: {', '.join(comparison)}")
        
        # 显示变化最大的3个关节
        if significant_joints:
            top_joints = sorted(significant_joints, key=lambda x: x[1], reverse=True)[:3]
            top_joint_info = [f"关节{idx}({change:.3f})" for idx, change in top_joints]
            rospy.loginfo(f"  主要活动关节: {', '.join(top_joint_info)}")
        
        # 通用判断逻辑：需要两个手都回到初始位置附近才认为完成
        distance = max(right_distance, left_distance)  # 使用较大的距离作为整体判断
        primary_hand = "右手" if right_distance >= left_distance else "左手"
        primary_distance = right_distance if right_distance >= left_distance else left_distance
        
        # 两个手都需要满足距离要求才认为位置正确
        position_ok = (right_distance < distance_threshold) and (left_distance < distance_threshold)
        
        rospy.loginfo(f"  动作持续时间: {(rospy.Time.now() - self.action_start_time).to_sec():.1f}s" if hasattr(self, 'action_start_time') and self.action_start_time else "  动作持续时间: 未知")
        rospy.loginfo(f"  主要判断手: {primary_hand} (距离: {primary_distance:.6f})")
        rospy.loginfo(f"  两手都满足<0.12: {position_ok}")
        
        rospy.loginfo(f"距离计算:")
        rospy.loginfo(f"  动作类型: {action_type}")
        rospy.loginfo(f"  统一距离阈值: {distance_threshold}")
        rospy.loginfo(f"  计算距离: {distance:.6f} (取双手最大值)")
        rospy.loginfo(f"  执行时间: {elapsed_time:.1f}s")
        
        # 条件2: 末端速度足够低（如果有的话）
        velocity_ok = True
        if len(self.motion_velocity_history) > 0:
            # 只检测末端关节速度（关节17和25）
            velocity_vector = self.motion_velocity_history[-1]
            right_hand_velocity = abs(velocity_vector[25])  # 右手末端速度
            left_hand_velocity = abs(velocity_vector[17])   # 左手末端速度
            # 使用两个末端速度的最大值作为判断标准
            end_effector_velocity = max(right_hand_velocity, left_hand_velocity)
            velocity_ok = end_effector_velocity < 0.10  # 严格速度阈值，确保真正停止
            rospy.loginfo(f"  末端速度: 左手={left_hand_velocity:.6f}, 右手={right_hand_velocity:.6f}, 满足: {velocity_ok}")
        rospy.loginfo(f"  距离检查: {distance:.3f} < {distance_threshold} = {position_ok}")
        rospy.loginfo(f"  位置检查: {position_ok}")
        
        # 条件3: 至少移动过（避免一开始就停止）
        has_moved = True
        if hasattr(self, 'max_distance_reached'):
            has_moved = self.max_distance_reached > 0.05
        else:
            self.max_distance_reached = distance
        
        if distance > self.max_distance_reached:
            self.max_distance_reached = distance
            
        # DEBUG: 记录最后一次距离，用于32步完成检查
        self.last_distance_to_initial = distance
            
        rospy.loginfo(f"  最大距离: {self.max_distance_reached:.6f}m, 已移动: {has_moved}")
        
        # 定期报告轨迹截断统计
        if hasattr(self, 'truncation_stats') and self.truncation_stats['count'] > 0:
            if not hasattr(self, 'last_stats_report') or (rospy.Time.now() - self.last_stats_report).to_sec() > 30:
                rospy.loginfo("=== 轨迹截断统计报告 ===")
                rospy.loginfo(f"  截断次数: {self.truncation_stats['count']}")
                rospy.loginfo(f"  总跳过点数: {self.truncation_stats['total_skipped']}")
                rospy.loginfo(f"  平均每次跳过: {self.truncation_stats['total_skipped'] / self.truncation_stats['count']:.1f} 个点")
                self.last_stats_report = rospy.Time.now()
        
        # 4. 连续检测机制：需要连续5步都满足条件才停止
        current_conditions_ok = position_ok and velocity_ok and has_moved
        
        if not hasattr(self, 'completion_counter'):
            self.completion_counter = 0
        
        if current_conditions_ok:
            self.completion_counter += 1
            rospy.loginfo(f"  完成计数: {self.completion_counter}/5")
        else:
            self.completion_counter = 0
            rospy.loginfo(f"  条件不满足，重置计数器")
        
        # 5. 最终判断：连续5步满足条件且执行超过5秒
        should_complete = (self.completion_counter >= 5) and (elapsed_time > 5.0)
        
        if should_complete:
            rospy.loginfo("*** 检测到动作完成 ***")
            rospy.loginfo(f"条件: 连续5步满足 (末端位置正确+速度<0.08+已移动过) + 执行超过5秒")
            rospy.loginfo(f"详细: 末端距离{distance:.3f}<{distance_threshold} (双手最大值)")
            rospy.loginfo(f"动作类型: {action_type}")
            # 重置状态，准备下一个动作循环
            self.initial_position = None
            self.initial_left_hand_pos = None  # 重置左手初始位置
            self.initial_right_hand_pos = None  # 重置右手初始位置
            self.initial_tf_time = None  # 重置初始TF时间戳
            self.action_start_time = None
            self.max_distance_reached = 0.0
            self.completion_counter = 0
            # 重置完成状态，允许重新生成轨迹
            self.is_action_completed = False
            self.action_state = "ready"
            rospy.loginfo("重置状态，准备下一个动作循环")
        
        return should_complete
    
    
    def _remove_trajectory_repetition(self, new_trajectory, last_trajectory_end):
        """智能检测并移除新轨迹中与前一轨迹重复的部分 - 改进版本"""
        # 检查是否启用截断
        if not self.enable_truncation:
            rospy.loginfo("轨迹截断已禁用，返回完整轨迹")
            return new_trajectory
            
        print(f"🚀 TRUNCATION FUNCTION CALLED!")
        print(f"🚀 New trajectory length: {len(new_trajectory)}")
        print(f"🚀 Last trajectory length: {len(last_trajectory_end)}")
        
        if len(new_trajectory) == 0 or len(last_trajectory_end) == 0:
            print(f"🚀 Empty trajectory, returning early")
            return new_trajectory
        
        print(f"🚀 Starting improved repetition detection...")
        rospy.loginfo("=== 开始改进的重复检测 ===")
        
        # 获取关键位置信息
        current_pos = new_trajectory[0]
        last_pos = last_trajectory_end[-1]  # 前一轨迹的最后一个位置
        second_last_pos = last_trajectory_end[-2] if len(last_trajectory_end) >= 2 else last_trajectory_end[0]
        
        # 计算关键距离
        initial_distance = np.linalg.norm(current_pos[:6] - last_pos[:6])
        last_step_distance = np.linalg.norm(last_pos[:6] - second_last_pos[:6]) if len(last_trajectory_end) >= 2 else 0.0
        
        rospy.loginfo(f"关键信息:")
        rospy.loginfo(f"  新轨迹起点距前一终点: {initial_distance:.6f}")
        rospy.loginfo(f"  前一轨迹最后一步距离: {last_step_distance:.6f}")
        rospy.loginfo(f"  新轨迹长度: {len(new_trajectory)}")
        rospy.loginfo(f"  前一轨迹长度: {len(last_trajectory_end)}")
        
        skip_count = 0
        
        # 策略1：改进的回退检测 - 针对不同指令特性优化
        # 分析新轨迹前几个点的移动方向
        movement_directions = []
        
        # 获取当前指令信息
        current_instruction = "unknown"
        if hasattr(self, 'current_instruction_id'):
            instruction_map = {0: 'wave', 1: 'welcome', 2: 'sayhi', 3: 'thumbsup'}
            current_instruction = instruction_map.get(self.current_instruction_id, 'unknown')
        
        # 根据指令特性调整检测参数
        if current_instruction == 'welcome':
            # welcome动作可能包含向身体靠近的合理动作，使用更宽松的标准
            backtrack_threshold = 0.002  # 增大阈值，减少误判
            max_backtrack_points = 5  # 允许更多"回退"点
            detection_range = 8  # 检测范围减小
            rospy.loginfo(f"🎯 Welcome指令检测：使用宽松回退标准")
        else:
            # wave动作使用标准检测
            backtrack_threshold = 0.001
            max_backtrack_points = 3
            detection_range = 12
            rospy.loginfo(f"🎯 Wave指令检测：使用标准回退标准")
        
        for i in range(1, min(detection_range, len(new_trajectory))):
            # 计算相对于前一轨迹终点的距离变化
            dist_to_last = np.linalg.norm(new_trajectory[i][:6] - last_pos[:6])
            
            # 计算相对于新轨迹起点的移动
            movement_from_start = np.linalg.norm(new_trajectory[i][:6] - current_pos[:6])
            
            # 判断移动方向：使用动态阈值
            if dist_to_last < initial_distance - backtrack_threshold:
                direction = "回退"
            elif dist_to_last > initial_distance + backtrack_threshold:
                direction = "前进"
            else:
                direction = "保持"
            
            movement_directions.append({
                'index': i,
                'dist_to_last': dist_to_last,
                'movement_from_start': movement_from_start,
                'direction': direction
            })
            
            # 只在debug模式显示详细信息
            if i <= 5:  # 只显示前5个点的详细信息
                rospy.loginfo(f"  点{i}: {direction}, 距前一终点={dist_to_last:.6f}, 移动距离={movement_from_start:.6f}")
        
        # 分析移动模式
        backtrack_points = [m for m in movement_directions if m['direction'] == '回退']
        forward_points = [m for m in movement_directions if m['direction'] == '前进']
        
        rospy.loginfo(f"移动模式分析: 回退点={len(backtrack_points)}, 前进点={len(forward_points)}")
        print(f"🔍 BACKTRACK ANALYSIS: {len(backtrack_points)} backtrack points, {len(forward_points)} forward points")
        
        # 使用指令特定的阈值进行截断决策
        if len(backtrack_points) >= max_backtrack_points:  # 使用指令特定的阈值
            print(f"🎯 DETECTED BACKTRACK PATTERN for {current_instruction}!")
            rospy.loginfo(f"检测到{current_instruction}指令的回退模式，开始自适应截断！")
            
            # 根据指令特性调整截断策略
            if current_instruction == 'welcome':
                # welcome动作：更保守的截断策略
                if len(backtrack_points) >= 6:
                    # 严重回退：跳过前1/4
                    skip_count = min(6, len(new_trajectory) // 4)
                    print(f"🚀 SEVERE BACKTRACK (Welcome): Skipping first QUARTER of trajectory ({skip_count} points)")
                elif len(backtrack_points) >= 4:
                    # 中等回退：跳过前1/6
                    skip_count = min(4, len(new_trajectory) // 6)
                    print(f"🚀 MEDIUM BACKTRACK (Welcome): Skipping first SIXTH of trajectory ({skip_count} points)")
                else:
                    # 轻微回退：跳过前1/8
                    skip_count = min(3, len(new_trajectory) // 8)
                    print(f"🚀 MINOR BACKTRACK (Welcome): Skipping first EIGHTH of trajectory ({skip_count} points)")
            else:
                # wave动作：使用原来的激进截断策略
                if len(backtrack_points) >= 8:
                    # 严重回退：直接跳过前一半的轨迹
                    skip_count = min(16, len(new_trajectory) // 2)
                    print(f"🚀 SEVERE BACKTRACK (Wave): Skipping first HALF of trajectory ({skip_count} points)")
                elif len(backtrack_points) >= 5:
                    # 中等回退：跳过前1/3
                    skip_count = min(10, len(new_trajectory) // 3)
                    print(f"🚀 MEDIUM BACKTRACK (Wave): Skipping first THIRD of trajectory ({skip_count} points)")
                else:
                    # 轻微回退：跳过前1/4
                    skip_count = min(6, len(new_trajectory) // 4)
                    print(f"🚀 MINOR BACKTRACK (Wave): Skipping first QUARTER of trajectory ({skip_count} points)")
            
            rospy.loginfo(f"指令特异性截断: {current_instruction} - {len(backtrack_points)}个回退点 -> 跳过前{skip_count}个点")
        
        # 策略2：检测周期性重复模式
        if skip_count == 0:
            rospy.loginfo("策略2：检测周期性重复模式...")
            
            # 检查新轨迹是否有"来回摆动"的模式
            for i in range(min(8, len(new_trajectory)-5)):
                # 分析这个区域的移动模式
                region_distances = []
                for j in range(i, min(i+5, len(new_trajectory))):
                    if j > 0:
                        step_dist = np.linalg.norm(new_trajectory[j][:6] - new_trajectory[j-1][:6])
                        region_distances.append(step_dist)
                
                if len(region_distances) >= 3:
                    # 计算移动的一致性
                    avg_movement = np.mean(region_distances)
                    movement_variance = np.var(region_distances)
                    
                    rospy.loginfo(f"  区域{i}-{i+len(region_distances)-1}: 平均移动={avg_movement:.6f}, 方差={movement_variance:.8f}")
                    
                    # 如果移动很小且方差很小，可能是重复性动作
                    if avg_movement < 0.002 and movement_variance < 0.000001:
                        rospy.loginfo(f"  检测到可能的重复区域，跳过前{i}个点")
                        skip_count = i
                        break
        
        # 策略3：与前一轨迹的结尾模式比较
        if skip_count == 0 and len(last_trajectory_end) >= 5:
            rospy.loginfo("策略3：与前一轨迹结尾模式比较...")
            
            # 获取前一轨迹最后5步的移动模式
            last_pattern = []
            for i in range(max(0, len(last_trajectory_end)-5), len(last_trajectory_end)-1):
                step_movement = np.linalg.norm(last_trajectory_end[i+1][:6] - last_trajectory_end[i][:6])
                last_pattern.append(step_movement)
            
            # 在新轨迹中寻找相似的模式
            for start_idx in range(min(10, len(new_trajectory)-5)):
                new_pattern = []
                for i in range(start_idx, min(start_idx+5, len(new_trajectory)-1)):
                    step_movement = np.linalg.norm(new_trajectory[i+1][:6] - new_trajectory[i][:6])
                    new_pattern.append(step_movement)
                
                if len(new_pattern) == len(last_pattern) and len(last_pattern) > 0:
                    # 计算模式相似度
                    pattern_diff = np.mean(np.abs(np.array(last_pattern) - np.array(new_pattern)))
                    
                    rospy.loginfo(f"  模式比较(起点{start_idx}): 差异={pattern_diff:.8f}")
                    
                    # 如果模式很相似，可能是重复
                    if pattern_diff < 0.001:
                        rospy.loginfo(f"  检测到相似移动模式，跳过前{start_idx}个点")
                        skip_count = start_idx
                        break
        
        # 应用结果
        if skip_count > 0:
            # 确保截断后还有足够的轨迹
            if skip_count < len(new_trajectory) - 8:
                result = new_trajectory[skip_count:]
                print(f"🎉 SUCCESS! Truncated {skip_count} points, kept {len(result)} points")
                rospy.loginfo(f"✅ 成功截断重复部分: 跳过{skip_count}个点，保留{len(result)}个点")
                
                # 记录截断统计
                if not hasattr(self, 'truncation_stats'):
                    self.truncation_stats = {'count': 0, 'total_skipped': 0}
                self.truncation_stats['count'] += 1
                self.truncation_stats['total_skipped'] += skip_count
                
                return result
            else:
                print(f"⚠️ Would truncate too much, keeping original")
                rospy.logwarn(f"⚠️ 跳过{skip_count}个点会导致轨迹过短(剩余{len(new_trajectory)-skip_count}个点)，保持原轨迹")
                return new_trajectory
        else:
            print(f"📝 No truncation needed")
            rospy.loginfo("✅ 未检测到需要截断的重复模式")
            return new_trajectory
    
    def _calculate_sequence_similarity(self, seq1, seq2):
        """计算两个序列的相似度"""
        if len(seq1) == 0 or len(seq2) == 0:
            return 0.0
        
        # 确保两个序列长度一致
        min_len = min(len(seq1), len(seq2))
        seq1_adj = seq1[:min_len]
        seq2_adj = seq2[:min_len]
        
        # 计算每个对应点的距离
        distances = []
        for i in range(min_len):
            dist = np.linalg.norm(seq1[i][:6] - seq2[i][:6])
            distances.append(dist)
        
        # 将距离转换为相似度（距离越小相似度越高）
        avg_distance = np.mean(distances)
        similarity = np.exp(-avg_distance * 50)  # 调整参数控制敏感度
        
        return similarity
    
    def _smooth_trajectory_speed(self, trajectory):
        """平滑轨迹速度，减少相邻点之间的突变"""
        if len(trajectory) < 3:
            return trajectory
        
        # 计算原始轨迹中相邻点的距离（速度）
        original_distances = []
        for i in range(1, len(trajectory)):
            dist = np.linalg.norm(trajectory[i][:6] - trajectory[i-1][:6])
            original_distances.append(dist)
        
        if len(original_distances) == 0:
            return trajectory
        
        # 计算平均速度作为目标
        avg_distance = np.mean(original_distances)
        max_distance = np.max(original_distances)
        
        rospy.loginfo(f"轨迹速度统计: 平均={avg_distance:.6f}, 最大={max_distance:.6f}")
        
        # 如果最大速度超过平均速度的3倍，进行平滑
        if max_distance > avg_distance * 3:
            rospy.loginfo("检测到速度突变，应用轨迹平滑")
            
            # 使用移动平均进行平滑
            smoothed_trajectory = [trajectory[0]]  # 保持起点不变
            
            for i in range(1, len(trajectory) - 1):
                # 对当前位置进行平滑处理
                prev_point = trajectory[i-1]
                current_point = trajectory[i]
                next_point = trajectory[i+1]
                
                # 计算平滑后的位置
                alpha = 0.3  # 平滑系数
                smoothed_point = alpha * current_point + (1 - alpha) * 0.5 * (prev_point + next_point)
                smoothed_trajectory.append(smoothed_point)
            
            smoothed_trajectory.append(trajectory[-1])  # 保持终点不变
            
            return np.array(smoothed_trajectory)
        else:
            rospy.loginfo("轨迹速度正常，无需平滑")
            return trajectory
        
    def _publish_arm_command(self, action: np.ndarray):
        """发布手臂控制命令"""
        if not HAS_ARM_MSGS:
            return
            
        try:
            # 提取手臂关节数据（左臂12-18，右臂19-25）
            left_arm_joints = action[12:19]  # 左臂7个关节
            right_arm_joints = action[19:26]  # 右臂7个关节
            
            # 转换为度数
            left_arm_degrees = [math.degrees(joint) for joint in left_arm_joints]
            right_arm_degrees = [math.degrees(joint) for joint in right_arm_joints]
            
            # 组合关节数据
            arm_joints = left_arm_degrees + right_arm_degrees
            
            # 创建消息
            arm_msg = armTargetPoses()
            arm_msg.times = [0.0]  # 立即执行
            arm_msg.values = arm_joints
            arm_msg.frame = 2  # local frame
            
            # 发布
            self.arm_target_pub.publish(arm_msg)
            
            rospy.logdebug(f"手臂命令发布: 左臂前3个={left_arm_degrees[:3]}, 右臂前3个={right_arm_degrees[:3]}")
            
        except Exception as e:
            rospy.logerr(f"发布手臂命令失败: {e}")
    
    def _start_callback(self, req):
        """开始推理回调"""
        if self.is_running:
            return TriggerResponse(False, "推理已在进行中")
        
        self.is_running = True
        self.trajectory_buffer = []
        self.next_trajectory_buffer = []
        self.current_trajectory_step = 0
        self.is_generating = False
        self.action_history_buffer = []
        self.motion_velocity_history = []
        self.motion_acceleration_history = []
        self.is_action_completed = False
        self.action_state = "ready"
        
        # 重置状态变量
        self.initial_position = None
        self.initial_left_hand_pos = None
        self.initial_right_hand_pos = None
        self.initial_tf_time = None
        self.action_start_time = None
        self.max_distance_reached = 0.0
        
        # 在推理开始时立即记录初始位置
        rospy.loginfo("推理开始，记录初始TF位置...")
        # 给TF系统一些时间来稳定
        rospy.sleep(0.5)
        
        # 尝试记录初始TF位置 - 使用身体中心作为参考
        try:
            # 使用最新的可用TF变换，避免时间戳问题
            try:
                # 获取身体中心位置（使用torso或base_link的原点）
                if hasattr(self, 'body_center_frame'):
                    body_transform = self.tf_buffer.lookup_transform(
                        self.base_frame, self.body_center_frame, rospy.Time(0), rospy.Duration(0.1))
                    self.initial_body_center = np.array([
                        body_transform.transform.translation.x,
                        body_transform.transform.translation.y,
                        body_transform.transform.translation.z
                    ])
                else:
                    self.initial_body_center = np.array([0.0, 0.0, 0.0])
                
                # 获取手部相对于身体的位置
                left_transform = self.tf_buffer.lookup_transform(
                    self.base_frame, self.left_hand_frame, rospy.Time(0), rospy.Duration(0.1))
                right_transform = self.tf_buffer.lookup_transform(
                    self.base_frame, self.right_hand_frame, rospy.Time(0), rospy.Duration(0.1))
                
                current_time = left_transform.header.stamp
            except Exception as e:
                rospy.logwarn(f"初始化TF变换获取失败: {e}")
                raise e
            
            # 计算手部相对于身体中心的位置
            self.initial_left_hand_pos = np.array([
                left_transform.transform.translation.x - self.initial_body_center[0],
                left_transform.transform.translation.y - self.initial_body_center[1],
                left_transform.transform.translation.z - self.initial_body_center[2]
            ])
            self.initial_right_hand_pos = np.array([
                right_transform.transform.translation.x - self.initial_body_center[0],
                right_transform.transform.translation.y - self.initial_body_center[1],
                right_transform.transform.translation.z - self.initial_body_center[2]
            ])
            self.initial_tf_time = current_time
            
            rospy.loginfo(f"推理开始时记录相对位置:")
            rospy.loginfo(f"  身体中心: [{self.initial_body_center[0]:.3f}, {self.initial_body_center[1]:.3f}, {self.initial_body_center[2]:.3f}]")
            rospy.loginfo(f"  左手相对位置: [{self.initial_left_hand_pos[0]:.3f}, {self.initial_left_hand_pos[1]:.3f}, {self.initial_left_hand_pos[2]:.3f}]")
            rospy.loginfo(f"  右手相对位置: [{self.initial_right_hand_pos[0]:.3f}, {self.initial_right_hand_pos[1]:.3f}, {self.initial_right_hand_pos[2]:.3f}]")
            
        except Exception as e:
            rospy.logwarn(f"推理开始时记录初始TF位置失败: {e}")
            rospy.loginfo("将在动作检测时重新记录初始位置")
            self.initial_body_center = np.array([0.0, 0.0, 0.0])
                              
        rospy.loginfo("开始连续轨迹ACT推理（已改进执行周期检测）")
        return TriggerResponse(True, "推理已开始")
    
    def _stop_callback(self, req):
        """停止推理回调"""
        if not self.is_running:
            return TriggerResponse(False, "推理未在进行中")
        
        self.is_running = False
        self.trajectory_buffer = []
        self.next_trajectory_buffer = []
        self.current_trajectory_step = 0
        self.is_generating = False
        self.action_history_buffer = []
        self.motion_velocity_history = []
        self.motion_acceleration_history = []
        self.is_action_completed = False
        self.action_state = "ready"
        
        # 重置状态变量
        self.initial_position = None
        self.initial_left_hand_pos = None
        self.initial_right_hand_pos = None
        self.action_start_time = None
        self.max_distance_reached = 0.0
        
        rospy.loginfo("停止连续轨迹ACT推理")
        return TriggerResponse(True, "推理已停止")
    
    def start_inference(self):
        """开始推理"""
        rospy.loginfo("连续轨迹ACT推理节点已启动（预生成模式，实现无缝衔接）...")
        rospy.loginfo("改进特性:")
        rospy.loginfo("  ✅ 预生成模式 - 当前轨迹执行到75%时预生成下一段")
        rospy.loginfo("  ✅ 无缝衔接 - 消除轨迹间的停顿，实现流畅运动")
        rospy.loginfo("  ✅ 智能预测 - 基于执行进度预测轨迹结束位置")
        rospy.loginfo("  ✅ 平滑切换 - 预生成轨迹就绪，即时切换")
        rospy.loginfo("  ✅ 重复检测 - 智能避免轨迹重复执行")
        
        if self.instruction_source == 'topic':
            rospy.loginfo("  ✅ 话题模式 - 通过 /vla_control/command 话题获取指令")
            rospy.loginfo("  📢 发布指令: rostopic pub /vla_control/command ros_vla_language/VLACommand '{instruction: \"wave\"}'")
            rospy.loginfo("  📢 发布指令: rostopic pub /vla_control/command ros_vla_language/VLACommand '{instruction: \"welcome\"}'")
            rospy.loginfo("  📢 发布指令: rostopic pub /vla_control/command ros_vla_language/VLACommand '{instruction: \"none\"}'")
        else:
            rospy.loginfo("  ✅ 手动模式 - 使用启动参数指定的固定指令")
            rospy.loginfo(f"  📝 当前指令: {self.instruction}")
        
        rospy.loginfo("使用以下命令控制推理:")
        rospy.loginfo("  开始推理: rosservice call /smooth_act_inference/start")
        rospy.loginfo("  停止推理: rosservice call /smooth_act_inference/stop")
        rospy.loginfo("按Ctrl+C退出")
        
        # 保持节点运行，但不立即开始推理
        rate = rospy.Rate(self.inference_frequency)
        step_count = 0
        
        rospy.loginfo("推理节点已启动，等待开始推理服务调用...")
        
        while not rospy.is_shutdown():
            if self.is_running:
                step_count += 1
                
                if self.current_joint_positions is not None:
                    # 每500步输出一次调试信息
                    if step_count % 500 == 0:
                        progress_ratio = self.current_trajectory_step / len(self.trajectory_buffer) if len(self.trajectory_buffer) > 0 else 0
                        rospy.loginfo(f"进度: 步数={step_count}, 主轨迹={len(self.trajectory_buffer)}, 当前步={self.current_trajectory_step}, 下一段={len(self.next_trajectory_buffer)}")
                                            
                    # 检查是否需要重新生成轨迹
                    regenerate_condition = self._should_regenerate_trajectory()
                    if step_count % 100 == 0:  # 每100步输出一次状态
                        print(f"📊 STATUS: Step={step_count}, Buffer={len(self.trajectory_buffer)}, CurrentStep={self.current_trajectory_step}, Regen={regenerate_condition}")
                    
                    if regenerate_condition:
                        print(f"🔄 DEBUG: About to regenerate trajectory!")
                        print(f"🔄 DEBUG: Current buffer length: {len(self.trajectory_buffer)}")
                        print(f"🔄 DEBUG: Current step: {self.current_trajectory_step}")
                        self._generate_new_trajectory()
                    
                    # 执行当前步
                    if len(self.trajectory_buffer) > 0 and self.current_trajectory_step < len(self.trajectory_buffer):
                        # 暂时禁用动作完成检测
                        if False:  # 暂时禁用
                            rospy.loginfo("动作已完成，跳过轨迹执行")
                            self.current_trajectory_step = len(self.trajectory_buffer)  # 跳过剩余轨迹
                            continue
                            
                        current_action = np.array(self.trajectory_buffer[self.current_trajectory_step])
                        
                        # 发布控制命令
                        if self.publish_commands:
                            self._publish_arm_command(current_action)
                        
                        rospy.logdebug(f"执行步 {self.current_trajectory_step}/{len(self.trajectory_buffer)}")
                        
                        # 更新性能统计和距离跟踪
                        self._update_performance_stats(current_action)
                        
                        # 先计算运动分析（用当前位置和历史中最后一个位置）
                        self._update_motion_analysis(current_action)
                        # 然后更新动作历史（把当前位置添加到历史）
                        self._update_action_history(current_action)
                        
                        # 简单的动作完成检测
                        if len(self.action_history_buffer) >= 2:
                            recent_trajectory = self.action_history_buffer
                            if self._intelligent_action_completion(recent_trajectory, current_action):
                                print(f"🛑 ACTION COMPLETION DETECTED at step {self.current_trajectory_step}!")
                                print(f"🛑 Action completed normally - back to start position")
                                rospy.loginfo("动作完成，停止发送控制消息")
                                self.is_action_completed = True
                                self.action_state = "completed"
                                self.trajectory_buffer = []
                                self.next_trajectory_buffer = []
                                self.current_trajectory_step = 0
                                # 直接停止发送任何控制消息
                                continue
                        
                        print(f"➡️ Incrementing step from {self.current_trajectory_step} to {self.current_trajectory_step + 1}")
                        self.current_trajectory_step += 1
                    else:
                        # 没有有效轨迹时 - 如果动作已完成，不发送任何控制消息
                        if self.is_action_completed:
                            rospy.logdebug("动作已完成，不发送控制消息")
                        else:
                            # 等待轨迹生成，保持当前位置
                            if self.publish_commands and self.current_joint_positions is not None:
                                self._publish_arm_command(self.current_joint_positions)
                                rospy.logdebug("等待轨迹生成，使用当前位置")
                else:
                    rospy.logdebug("等待关节位置数据...")
              
            rate.sleep()

    def _is_action_completed(self):
        """检测动作是否已完成 - 基于回到初始位置"""
        if self.instruction in ['none']:
            return True

        # 检查是否有当前关节位置和初始位置
        if self.current_joint_positions is None or self.initial_position is None:
            return False

        # 检查是否回到初始位置附近
        current_position = self.current_joint_positions
        joint_changes = np.abs(current_position - self.initial_position)

        # 手臂关节阈值
        arm_joints = [17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30]
        threshold = 0.1

        # 检查手臂关节是否都回到初始位置附近
        for joint in arm_joints:
            if joint < len(joint_changes) and joint_changes[joint] > threshold:
                return False

        rospy.loginfo(f"动作 '{self.instruction}' 已回到初始位置")
        return True

    def _stop_current_action(self):
        """停止当前动作"""
        rospy.loginfo(f"停止当前动作: {self.instruction}")

        # 清空轨迹缓冲区
        self.trajectory_buffer = []
        self.next_trajectory_buffer = []
        self.current_trajectory_step = 0

        # 设置完成状态
        self.is_action_completed = True
        self.action_state = "completed"

        # 将指令重置为none
        self.instruction = 'none'

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='单段轨迹ACT推理节点')
    parser.add_argument('--model_path', required=True, help='模型文件路径')
    parser.add_argument('--instruction', default='wave', help='指令类型 (wave/welcome/sayhi/thumbsup/none)')
    parser.add_argument('--instruction_source', choices=['manual', 'topic'], default='manual',
                       help='指令来源 (manual=手动输入, topic=ROS话题获取)')
    parser.add_argument('--control_mode', choices=['arm', 'base', 'none'], default='arm',
                       help='控制模式 (arm/base/none)')
    parser.add_argument('--frequency', type=float, default=30.0, help='推理频率Hz')
    parser.add_argument('--no_publish', action='store_true', help='不发布控制命令（仅测试）')
    parser.add_argument('--disable_truncation', action='store_true', help='禁用轨迹截断功能')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        sys.exit(1)
    
    # 构建配置
    config = {
        'instruction': args.instruction,
        'instruction_source': args.instruction_source,
        'control_mode': args.control_mode,
        'inference_frequency': args.frequency,
        'publish_commands': not args.no_publish,
        'enable_truncation': not args.disable_truncation  # 默认启用截断，添加 --disable_truncation 参数禁用
    }
    
    # 初始化ROS节点
    rospy.init_node('smooth_act_inference_node', anonymous=True)
    
    try:
        # 创建推理节点
        inference_node = SmoothACTInferenceNode(args.model_path, config)
        
        # 开始推理
        inference_node.start_inference()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS节点被中断")
    except Exception as e:
        rospy.logerr(f"推理节点运行出错: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()