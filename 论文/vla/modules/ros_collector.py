#!/usr/bin/env python3
"""
ROS1 机器人轨迹数据采集器
用于采集固定长度的primitive skill风格轨迹数据

支持功能：
- 每条episode固定长度帧数
- 可配置采集数量
- 可配置采样频率
- 支持关节角度、躯干位置、姿态、可选RGB图像
- ROS话题订阅和数据处理
- 安全退出机制
"""

import rospy
import json
import time
import numpy as np
import os
import signal
import sys
import re
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from std_msgs.msg import Header, Empty, Float64MultiArray
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Quaternion, Point
from message_filters import Subscriber, ApproximateTimeSynchronizer
from std_srvs.srv import Trigger, TriggerResponse
import cv2
from cv_bridge import CvBridge
import math


class CollectorNode:
    """ROS1轨迹数据采集节点"""
    
    def __init__(self):
        """初始化采集器节点"""
        # 获取ROS参数
        self._get_parameters()
        
        # 初始化状态变量
        self.episode_data = []
        self.current_frame = 0
        self.is_collecting = False
        self.shutdown_requested = False
        self.episode_start_number = 1  # 将在_create_save_directory中设置
        
        # 数据缓存
        self.joint_state_cache = None
        self.odom_cache = None
        self.orientation_cache = None
        self.image_cache = None
        self.last_sync_time = 0
        
        # 坐标系转换相关
        self.initial_position = None  # 初始位置 [x, y, z]
        self.initial_quat = None      # 初始四元数 [x, y, z, w]
        self.initial_yaw = None       # 初始偏航角
        self.coordinate_transform_initialized = False
        
        # 动作检测相关 - 简化逻辑：只检测开始，不检测停止
        self.motion_threshold = rospy.get_param('~motion_threshold', 0.01)  # 动作检测阈值
        self.min_frames_before_start = rospy.get_param('~min_frames_before_start', 3)  # 开始前需要的动作帧数
        self.motion_frames_count = 0  # 连续检测到动作的帧数
        self.collection_mode = rospy.get_param('~collection_mode', 'auto')  # 'manual' 或 'auto'
        
        # 手动控制服务
        self.manual_start_service = None
        self.manual_stop_service = None
        
        # CV Bridge用于图像转换
        self.cv_bridge = CvBridge()
        
        # 初始化技能接口（这里需要根据实际技能系统进行配置）
        self.skill_interface = self._setup_skill_interface()
        
        # 创建保存目录
        self._create_save_directory()
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # 初始化ROS订阅器
        self._setup_ros_subscribers()
        
        # 设置手动控制服务
        self._setup_manual_control_services()
        
        rospy.loginfo("CollectorNode初始化完成")
        rospy.loginfo(f"技能ID: {self.skill_id}")
        rospy.loginfo(f"指令: {self.instruction}")
        rospy.loginfo(f"采集参数: 每条{self.frames_per_episode}帧")
        rospy.loginfo(f"采集模式: {self.collection_mode}")
        rospy.loginfo(f"动作检测阈值: {self.motion_threshold}")
        rospy.loginfo(f"开始前动作帧数: {self.min_frames_before_start}")
        rospy.loginfo(f"保存路径: {self.save_dir}")
        rospy.loginfo(f"视觉采集: {'启用' if self.with_vision else '禁用'}")
        rospy.loginfo("重要：一旦开始采集，将持续到400帧完成，不会中途停止")

    def _get_parameters(self):
        """获取ROS参数"""
        # 基础参数
        self.skill_id = rospy.get_param('~skill_id', 'walk_back')  # 默认技能ID为walk_forward
        self.instruction = rospy.get_param('~instruction', '向后走')  # 默认指令为向前走
        self.frames_per_episode = rospy.get_param('~frames_per_episode', 400)  # 修改默认值为400
        self.frequency = rospy.get_param('~frequency', 100.0)
        self.with_vision = rospy.get_param('~with_vision', False)
        
        # 话题参数
        self.joint_state_topic = rospy.get_param('~joint_state_topic', '/humanoid_controller/optimizedState_mrt/joint_pos')
        self.odom_topic = rospy.get_param('~baselink_topic', '/humanoid_controller/optimizedState_mrt/base/pos_xyz')
        self.orientation_topic = rospy.get_param('~orientation_topic', '/humanoid_controller/optimizedState_mrt/base/angular_zyx')
        self.image_topic = rospy.get_param('~image_topic', '/camera/image_raw')
        
        # 保存路径
        self.save_dir = rospy.get_param('~save_dir', './trajectories')
        
        # 机器人参数
        self.num_joints = rospy.get_param('~num_joints', 26)
        
        # 同步参数
        self.sync_slop = rospy.get_param('~sync_slop', 0.1)  # 时间同步容差（秒）
        
        # 技能参数
        self.skill_params = rospy.get_param('~skill_params', {})
        
        rospy.loginfo("参数加载完成")
    
    def _setup_skill_interface(self):
        """设置技能接口（需要根据实际技能系统实现）"""
        # 这里是一个示例接口，实际使用时需要根据您的技能系统进行修改
        class MockSkillInterface:
            def __init__(self, skill_id, skill_params):
                self.skill_id = skill_id
                self.skill_params = skill_params
                self.step_count = 0
                self.joint_pos_buffer = []  # 存储关节位置缓冲
                self.current_action = [0.0] * 26  # 当前动作
            
            def step(self, observation):
                """模拟技能step方法，返回动作"""
                self.step_count += 1
                
                if "joint_pos" in observation:
                    current_pos = observation["joint_pos"]
                    
                    # 将当前关节位置添加到缓冲区
                    self.joint_pos_buffer.append(current_pos.copy())
                    
                    if len(self.joint_pos_buffer) >= 2:
                        # 返回上一帧的关节位置作为当前帧的action
                        # 即：第n帧的action = 第n+1帧的joint_pos
                        self.current_action = self.joint_pos_buffer[-1].copy()
                    else:
                        # 第一帧，暂时返回0，等第二帧时会更新
                        self.current_action = [0.0] * 26
                        
                    # 限制缓冲区大小，避免内存问题
                    if len(self.joint_pos_buffer) > 1000:
                        self.joint_pos_buffer = self.joint_pos_buffer[-500:]
                else:
                    # 如果没有关节数据，使用当前action或默认值
                    if not hasattr(self, 'current_action'):
                        self.current_action = [0.0] * 26
                
                return self.current_action
        
        return MockSkillInterface(self.skill_id, self.skill_params)
    
    def _setup_manual_control_services(self):
        """设置手动控制服务"""
        try:
            # 开始采集服务
            self.manual_start_service = rospy.Service(
                '/collector/start_collection', 
                Trigger, 
                self._manual_start_callback
            )
            
            # 停止采集服务
            self.manual_stop_service = rospy.Service(
                '/collector/stop_collection', 
                Trigger, 
                self._manual_stop_callback
            )
            
            rospy.loginfo("手动控制服务已设置:")
            rospy.loginfo("  开始采集: rosservice call /collector/start_collection")
            rospy.loginfo("  停止采集: rosservice call /collector/stop_collection")
            
        except Exception as e:
            rospy.logerr(f"设置手动控制服务失败: {e}")
    
    def _manual_start_callback(self, req):
        """手动开始采集回调"""
        if self.is_collecting:
            return TriggerResponse(False, "采集已在进行中")
        
        # 重置坐标系转换状态
        self._reset_coordinate_transform()
        
        self.is_collecting = True
        rospy.loginfo("手动触发开始采集")
        return TriggerResponse(True, "采集已开始")
    
    def _manual_stop_callback(self, req):
        """手动停止采集回调"""
        if not self.is_collecting:
            return TriggerResponse(False, "采集未在进行中")
        
        self.is_collecting = False
        rospy.loginfo("手动触发停止采集")
        
        # 保存当前episode
        if self.episode_data:
            self._finish_episode()
        
        return TriggerResponse(True, "采集已停止")
    
    def _detect_motion(self, joint_msg: Float64MultiArray, odom_msg: Float64MultiArray, orientation_msg: Float64MultiArray) -> bool:
        """检测是否有动作发生 - 简化版本，只检测关节运动"""
        try:
            # 主要检测关节运动 - 只要有任何一个关节的运动超过阈值就代表正在运动
            joint_motion = False
            moving_joints = []
            
            if hasattr(self, 'last_joint_positions'):
                joint_diffs = [abs(curr - last) for curr, last in zip(joint_msg.data, self.last_joint_positions)]
                
                # 检查哪些关节在运动
                for i, diff in enumerate(joint_diffs):
                    if diff > self.motion_threshold:
                        moving_joints.append(i)
                        joint_motion = True
                
                # 如果有关节在运动，输出详细信息
                if joint_motion:
                    rospy.loginfo(f"检测到关节运动: {len(moving_joints)}个关节在运动, "
                                f"关节索引: {moving_joints}, "
                                f"最大变化: {max(joint_diffs):.6f}, "
                                f"阈值: {self.motion_threshold}")
            else:
                rospy.loginfo("初始化关节位置基准值...")
            
            # 更新关节位置基准值
            self.last_joint_positions = list(joint_msg.data)
            
            # 简化逻辑：只要有关节运动就认为在运动
            has_motion = joint_motion
            
            if has_motion:
                rospy.loginfo(f"动作检测: 检测到运动! (基于关节运动)")
                rospy.loginfo(f"动作帧计数: {self.motion_frames_count}/{self.min_frames_before_start}, 采集状态: {self.is_collecting}")
            else:
                rospy.logdebug("动作检测: 无运动")
            
            return has_motion
            
        except Exception as e:
            rospy.logerr(f"动作检测失败: {e}")
            return False
    
    def _create_save_directory(self):
        """创建保存目录"""
        self.save_path = Path(self.save_dir)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # 如果启用视觉，创建图像子目录
        if self.with_vision:
            self.image_save_path = self.save_path / "images"
            self.image_save_path.mkdir(parents=True, exist_ok=True)
        
        rospy.loginfo(f"创建保存目录: {self.save_path}")
        if self.with_vision:
            rospy.loginfo(f"创建图像目录: {self.image_save_path}")
        
        # 查找现有的数据文件，确定下一个可用的编号
        self._find_next_episode_number()
    
    def _find_next_episode_number(self):
        """查找下一个可用的episode编号"""
        try:
            # 查找所有{skill_id}_*.json文件
            trajectory_files = list(self.save_path.glob(f"{self.skill_id}_*.json"))
            
            if trajectory_files:
                # 提取文件编号
                numbers = []
                for file in trajectory_files:
                    # 提取文件名中的数字部分
                    match = re.search(r'_(\d+)\.json$', str(file.name))
                    if match:
                        numbers.append(int(match.group(1)))
                
                if numbers:
                    # 找到最大的编号，下一个编号就是最大值+1
                    max_number = max(numbers)
                    self.episode_start_number = max_number + 1
                    rospy.loginfo(f"找到现有数据文件，下一个编号从 {self.episode_start_number} 开始")
                else:
                    self.episode_start_number = 1
                    rospy.loginfo("未找到有效的数据文件编号，从 1 开始")
            else:
                self.episode_start_number = 1
                rospy.loginfo("未找到现有数据文件，从 1 开始")
                
        except Exception as e:
            rospy.logerr(f"查找现有数据文件失败: {e}")
            self.episode_start_number = 1
            rospy.loginfo("使用默认编号从 1 开始")
    
    def _check_visual_consistency(self):
        """检查视觉采集一致性"""
        if self.with_vision:
            # 检查是否有图像数据
            if not hasattr(self, 'image_count'):
                self.image_count = 0
            # 检查图像采集是否正常
            if self.image_count == 0 and self.current_frame > 0:
                rospy.logwarn("启用了视觉采集，但没有接收到图像数据")
        else:
            rospy.loginfo("视觉采集已禁用")
    
    def _validate_data_integrity(self, episode_data):
        """验证数据完整性"""
        # 检查必要字段
        required_fields = ["instruction", "skill_id", "observations", "actions", "collect_visual"]
        for field in required_fields:
            if field not in episode_data:
                raise ValueError(f"缺少必要字段: {field}")
        
        # 检查observations和actions长度一致
        if len(episode_data["observations"]) != len(episode_data["actions"]):
            raise ValueError("observations和actions长度不一致")
        
        # 检查collect_visual标记一致性
        if episode_data["collect_visual"]:
            # 检查每个observation是否有image字段
            for i, obs in enumerate(episode_data["observations"]):
                if "image" not in obs:
                    rospy.logwarn(f"Frame {i} 缺少image字段，但collect_visual为True")
                elif obs["image"] is None:
                    rospy.logwarn(f"Frame {i} 的image字段为None，但collect_visual为True")
        else:
            # 检查是否没有图像数据
            for i, obs in enumerate(episode_data["observations"]):
                if "image" in obs and obs["image"] is not None:
                    rospy.logwarn(f"Frame {i} 包含图像数据，但collect_visual为False")
        
        # 检查每个observation的必要字段
        required_obs_fields = ["joint_pos", "joint_vel", "root_pos", "root_orien"]
        for i, obs in enumerate(episode_data["observations"]):
            for field in required_obs_fields:
                if field not in obs:
                    raise ValueError(f"Frame {i} 的observation缺少必要字段: {field}")
        
        rospy.loginfo("数据完整性检查通过")
        return True
    
    def _setup_ros_subscribers(self):
        """设置ROS话题订阅器"""
        try:
            # 订阅关节状态
            self.joint_state_sub = rospy.Subscriber(
                self.joint_state_topic, 
                Float64MultiArray, 
                self._joint_state_callback
            )
            rospy.loginfo(f"订阅关节状态话题: {self.joint_state_topic}")
            
            # 订阅里程计
            self.odom_sub = rospy.Subscriber(
                self.odom_topic, 
                Float64MultiArray, 
                self._odom_callback
            )
            rospy.loginfo(f"订阅里程计话题: {self.odom_topic}")
            
            # 订阅姿态
            self.orientation_sub = rospy.Subscriber(
                self.orientation_topic,
                Float64MultiArray,
                self._orientation_callback
            )
            rospy.loginfo(f"订阅姿态话题: {self.orientation_topic}")
            
            # 如果启用视觉，订阅图像话题
            if self.with_vision:
                self.image_sub = rospy.Subscriber(
                    self.image_topic,
                    Image,
                    self._image_callback
                )
                rospy.loginfo(f"订阅图像话题: {self.image_topic}")
            
            # 设置消息同步器
            self._setup_message_synchronizer()
            
        except Exception as e:
            rospy.logerr(f"设置ROS订阅器失败: {e}")
            raise
    
    def _setup_message_synchronizer(self):
        """设置消息同步器"""
        try:
            # 创建消息订阅器 (Float64MultiArray没有header，需要allow_headerless)
            joint_sub = Subscriber(self.joint_state_topic, Float64MultiArray, allow_headerless=True)
            odom_sub = Subscriber(self.odom_topic, Float64MultiArray, allow_headerless=True)
            orientation_sub = Subscriber(self.orientation_topic, Float64MultiArray, allow_headerless=True)
            
            # 根据视觉设置决定是否同步图像
            if self.with_vision:
                image_sub = Subscriber(self.image_topic, Image)
                self.sync = ApproximateTimeSynchronizer(
                    [joint_sub, odom_sub, orientation_sub, image_sub],
                    queue_size=10,
                    slop=self.sync_slop
                )
                self.sync.registerCallback(self._synchronized_callback_with_image)
                rospy.loginfo("设置同步器: 关节状态 + 里程计 + 姿态 + 图像")
            else:
                self.sync = ApproximateTimeSynchronizer(
                    [joint_sub, odom_sub, orientation_sub],
                    queue_size=10,
                    slop=self.sync_slop
                )
                self.sync.registerCallback(self._synchronized_callback)
                rospy.loginfo("设置同步器: 关节状态 + 里程计 + 姿态")
                
        except Exception as e:
            rospy.logerr(f"设置消息同步器失败: {e}")
            # 降级到定时处理模式 - 设置一个定时器来处理缓存的数据
            rospy.logwarn("降级到定时处理模式")
            self._setup_timer_callback()
    
    def _setup_timer_callback(self):
        """设置定时器回调"""
        # 创建一个定时器，以指定频率处理缓存的数据
        self.timer = rospy.Timer(rospy.Duration(1.0/self.frequency), self._timer_callback)
        rospy.loginfo(f"设置定时器处理模式，频率: {self.frequency}Hz")
    
    def _timer_callback(self, event):
        """定时器回调 - 处理缓存的数据"""
        if self.shutdown_requested:
            return
            
        # 检查是否有所有需要的数据
        if (self.joint_state_cache is not None and 
            self.odom_cache is not None and 
            self.orientation_cache is not None):
            
            # 只有在未开始采集时才进行动作检测
            if not self.is_collecting:
                has_motion = self._detect_motion(self.joint_state_cache, self.odom_cache, self.orientation_cache)
                
                if has_motion:
                    # 检测到动作，增加动作帧计数
                    self.motion_frames_count += 1
                    
                    # 如果连续检测到足够多的动作帧，开始采集
                    if self.motion_frames_count >= self.min_frames_before_start:
                        # 重置坐标系转换状态
                        self._reset_coordinate_transform()
                        self.is_collecting = True
                        rospy.loginfo(f"连续{self.min_frames_before_start}帧检测到动作，开始自动采集")
                        rospy.loginfo("重要：一旦开始采集，将持续到200帧完成，不会中途停止")
                else:
                    # 没有动作，重置动作帧计数
                    self.motion_frames_count = 0
            
            # 只有在采集状态时才处理数据
            if self.is_collecting:
                # 处理数据
                self._process_synchronized_data(
                    self.joint_state_cache, 
                    self.odom_cache, 
                    self.orientation_cache, 
                    None
                )
    
    def _joint_state_callback(self, msg: Float64MultiArray):
        """关节状态回调"""
        self.joint_state_cache = msg
    
    def _odom_callback(self, msg: Float64MultiArray):
        """里程计回调"""
        self.odom_cache = msg
    
    def _orientation_callback(self, msg: Float64MultiArray):
        """姿态回调"""
        self.orientation_cache = msg
    
    def _image_callback(self, msg: Image):
        """图像回调"""
        self.image_cache = msg
    
    def _synchronized_callback(self, joint_msg: Float64MultiArray, odom_msg: Float64MultiArray, orientation_msg: Float64MultiArray):
        """同步回调（无图像）"""
        current_time = time.time()
        if current_time - self.last_sync_time < 1.0 / self.frequency:
            return
        
        self.last_sync_time = current_time
        self._process_synchronized_data(joint_msg, odom_msg, orientation_msg, None)
    
    def _synchronized_callback_with_image(self, joint_msg: Float64MultiArray, odom_msg: Float64MultiArray, orientation_msg: Float64MultiArray, image_msg: Image):
        """同步回调（带图像）"""
        current_time = time.time()
        if current_time - self.last_sync_time < 1.0 / self.frequency:
            return
        
        self.last_sync_time = current_time
        self._process_synchronized_data(joint_msg, odom_msg, orientation_msg, image_msg)
    
    def _process_synchronized_data(self, joint_msg: Float64MultiArray, odom_msg: Float64MultiArray, orientation_msg: Float64MultiArray, image_msg: Optional[Image]):
        """处理同步数据"""
        if self.shutdown_requested:
            return
        
        try:
            # 根据采集模式处理数据
            if self.collection_mode == 'manual':
                # 手动模式：只有手动触发时才采集
                if not self.is_collecting:
                    return
            else:
                # 自动模式：只有在未开始采集时才检测动作
                if not self.is_collecting:
                    has_motion = self._detect_motion(joint_msg, odom_msg, orientation_msg)
                    
                    if has_motion:
                        # 检测到动作，增加动作帧计数
                        self.motion_frames_count += 1
                        
                        # 如果连续检测到足够多的动作帧，开始采集
                        if self.motion_frames_count >= self.min_frames_before_start:
                            # 重置坐标系转换状态
                            self._reset_coordinate_transform()
                            self.is_collecting = True
                            rospy.loginfo(f"连续{self.min_frames_before_start}帧检测到动作，开始自动采集")
                            rospy.loginfo("重要：一旦开始采集，将持续到200帧完成，不会中途停止")
                    else:
                        # 没有动作，重置动作帧计数
                        self.motion_frames_count = 0
            
            # 只有在采集状态时才处理数据
            if self.is_collecting:
                # 构建观测数据
                observation = self._build_observation(joint_msg, odom_msg, orientation_msg, image_msg)
                
                # 获取技能动作
                action = self.skill_interface.step(observation)
                
                # 添加到当前episode
                frame_data = {
                    "episode_index": 0,
                    "frame_index": self.current_frame,
                    "timestamp": time.time(),
                    "skill_id": self.skill_id,
                    "instruction": self.instruction,
                    "params": self.skill_params,
                    "observation": observation,
                    "action": action
                }
                
                self.episode_data.append(frame_data)
                self.current_frame += 1
                
                # 输出进度
                progress = (self.current_frame / self.frames_per_episode) * 100
                rospy.loginfo("Episode 1 - "
                             f"Frame {self.current_frame}/{self.frames_per_episode} "
                             f"({progress:.1f}%)")
                
                # 检查是否完成当前episode
                if self.current_frame >= self.frames_per_episode:
                    self._finish_episode()
                
        except Exception as e:
            rospy.logerr(f"处理同步数据时出错: {e}")
    
    def _build_observation(self, joint_msg: Float64MultiArray, odom_msg: Float64MultiArray, orientation_msg: Float64MultiArray, image_msg: Optional[Image]) -> Dict[str, Any]:
        """构建观测数据"""
        # 处理关节数据
        joint_positions = list(joint_msg.data)
        
        # 计算关节速度（使用位置差分）
        joint_velocities = [0.0] * len(joint_positions)
        if hasattr(self, 'last_joint_positions_for_vel'):
            joint_velocities = [(curr - last) * self.frequency for curr, last in zip(joint_positions, self.last_joint_positions_for_vel)]
        self.last_joint_positions_for_vel = joint_positions
        
        # 处理baselink位置数据（世界坐标系）
        if len(odom_msg.data) >= 3:
            world_pos = [odom_msg.data[0], odom_msg.data[1], odom_msg.data[2]]
        elif len(odom_msg.data) >= 2:
            world_pos = [odom_msg.data[0], odom_msg.data[1], 0.0]  # 假设z=0
        else:
            world_pos = [0.0, 0.0, 0.0]  # 默认值
        
        # 处理姿态数据（欧拉角转四元数）
        if len(orientation_msg.data) >= 3:
            # 假设欧拉角顺序为 ZYX (yaw, pitch, roll)
            yaw, pitch, roll = orientation_msg.data[0], orientation_msg.data[1], orientation_msg.data[2]
            world_quat = self._euler_to_quaternion(roll, pitch, yaw)
        else:
            # 默认姿态（单位四元数）
            world_quat = [0.0, 0.0, 0.0, 1.0]
        
        # 如果是第一次采集，初始化坐标系转换
        if self.is_collecting and not self.coordinate_transform_initialized:
            self._initialize_coordinate_transform(world_pos, world_quat)
        
        # 修复后的坐标转换逻辑：保存相对位移而不是绝对位置
        if self.coordinate_transform_initialized:
            # 1. 计算相对于初始位置的位移
            rel_pos = [
                world_pos[0] - self.initial_position[0],
                world_pos[1] - self.initial_position[1],
                world_pos[2] - self.initial_position[2]
            ]
            
            # 2. 将位移转换到初始朝向的坐标系（去除初始朝向影响）
            # 使用正确的旋转矩阵将世界坐标系的位移转换到机器人坐标系
            cos_yaw = math.cos(self.initial_yaw)
            sin_yaw = math.sin(self.initial_yaw)
            
            # 正确的旋转矩阵：将世界坐标系的位移转换到机器人坐标系
            # 机器人向前为+X方向，向左为+Y方向
            body_pos = [
                cos_yaw * rel_pos[0] + sin_yaw * rel_pos[1],   # x (前向)
                -sin_yaw * rel_pos[0] + cos_yaw * rel_pos[1],  # y (左向)
                rel_pos[2]  # z保持不变
            ]
            
            # 3. 计算相对四元数（相对于初始姿态）
            initial_quat_inv = [
                -self.initial_quat[0],  # x
                -self.initial_quat[1],  # y
                -self.initial_quat[2],  # z
                self.initial_quat[3]    # w (共轭)
            ]
            body_quat = self._quaternion_multiply(world_quat, initial_quat_inv)
            
        else:
            # 如果未初始化，使用原始世界坐标
            body_pos, body_quat = world_pos, world_quat
        
        observation = {
            "joint_pos": joint_positions,
            "joint_vel": joint_velocities,
            "root_pos": body_pos,        # 保存为相对于初始位置和朝向的位移
            "root_orien": body_quat      # 保存为相对于初始姿态的四元数
        }
        
        # 如果有图像数据，添加图像数据
        if image_msg is not None and self.with_vision:
            # 保存图像并返回路径
            image_path = self._save_image(image_msg)
            observation["image"] = image_path
        else:
            observation["image"] = None
        
        return observation
    
    def _quaternion_to_euler(self, x, y, z, w):
        """将四元数转换为欧拉角
        Args:
            x, y, z, w: 四元数分量
        Returns:
            [roll, pitch, yaw] 欧拉角 (rad)
        """
        # 欧拉角计算
        # roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # use 90° if out of range
        else:
            pitch = math.asin(sinp)
        
        # yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def _reset_coordinate_transform(self):
        """重置坐标系转换状态"""
        self.initial_position = None
        self.initial_quat = None
        self.initial_yaw = None
        self.coordinate_transform_initialized = False
        rospy.loginfo("坐标系转换状态已重置")
    
    def _initialize_coordinate_transform(self, world_pos, world_quat):
        """初始化坐标系转换参数
        Args:
            world_pos: 世界坐标系下的位置 [x, y, z]
            world_quat: 世界坐标系下的四元数 [x, y, z, w]
        """
        if not self.coordinate_transform_initialized:
            self.initial_position = world_pos.copy()
            self.initial_quat = world_quat.copy()
            
            # 从四元数提取初始偏航角
            _, _, initial_yaw = self._quaternion_to_euler(*world_quat)
            self.initial_yaw = initial_yaw
            
            self.coordinate_transform_initialized = True
            rospy.loginfo(f"坐标系转换初始化完成:")
            rospy.loginfo(f"  初始位置: [{self.initial_position[0]:.3f}, {self.initial_position[1]:.3f}, {self.initial_position[2]:.3f}]")
            rospy.loginfo(f"  初始偏航角: {math.degrees(self.initial_yaw):.1f}°")
    
      
    def _quaternion_multiply(self, q1, q2):
        """四元数乘法
        Args:
            q1: 第一个四元数 [x1, y1, z1, w1]
            q2: 第二个四元数 [x2, y2, z2, w2]
        Returns:
            [x, y, z, w] 乘积四元数
        """
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        
        return [x, y, z, w]
    
    def _euler_to_quaternion(self, roll, pitch, yaw):
        """将欧拉角转换为四元数
        Args:
            roll: 绕x轴旋转 (rad)
            pitch: 绕y轴旋转 (rad)  
            yaw: 绕z轴旋转 (rad)
        Returns:
            [x, y, z, w] 四元数
        """
        # 计算半角
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        # 计算四元数
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return [x, y, z, w]
    
    def _save_image(self, image_msg: Image) -> str:
        """保存图像并返回路径"""
        try:
            # 转换ROS图像为OpenCV格式
            cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
            
            # 生成图像文件名
            episode_number = self.episode_start_number
            image_filename = f"trajectory_{episode_number:03d}_frame{self.current_frame:03d}.jpg"
            image_path = self.image_save_path / image_filename
            
            # 保存图像
            cv2.imwrite(str(image_path), cv_image)
            
            # 更新图像计数
            if not hasattr(self, 'image_count'):
                self.image_count = 0
            self.image_count += 1
            
            return str(image_path)
            
        except Exception as e:
            rospy.logerr(f"保存图像失败: {e}")
            return ""
    
    def _finish_episode(self):
        """完成当前episode并保存"""
        try:
            # 检查视觉一致性
            self._check_visual_consistency()
            
            # 提取observations和actions
            observations = []
            actions = []
            
            for frame_data in self.episode_data:
                observations.append(frame_data["observation"])
                actions.append(frame_data["action"])
            
            # 重新计算actions：第n帧的action = 第n+1帧的joint_pos
            if len(observations) > 1:
                rospy.loginfo("重新计算actions，使第n帧的action = 第n+1帧的joint_pos")
                
                # 从第一帧到倒数第二帧，action = 下一帧的joint_pos
                for i in range(len(observations) - 1):
                    actions[i] = observations[i + 1]["joint_pos"].copy()
                
                # 最后一帧的action = 最后一帧的joint_pos
                actions[-1] = observations[-1]["joint_pos"].copy()
                
                rospy.loginfo(f"Actions重新计算完成，共{len(actions)}个actions")
            elif len(observations) == 1:
                # 只有一帧的情况
                actions[0] = observations[0]["joint_pos"].copy()
                rospy.loginfo("只有一帧数据，action设置为当前joint_pos")
            
            # 生成保存文件名
            episode_number = self.episode_start_number
            filename = f"{self.skill_id}_{episode_number:03d}.json"
            save_path = self.save_path / filename
            
            # 构建episode数据结构
            episode_data = {
                "instruction": self.instruction,
                "skill_id": self.skill_id,
                "observations": observations,
                "actions": actions,
                "collect_visual": self.with_vision,
                "episode_metadata": {
                    "episode_index": episode_number,
                    "total_frames": len(observations),
                    "skill_params": self.skill_params,
                    "collection_time": time.time()
                }
            }
            
            # 验证数据完整性
            self._validate_data_integrity(episode_data)
            
            # 保存episode数据
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(episode_data, f, indent=2, ensure_ascii=False)
            
            rospy.loginfo(f"Episode {episode_number} 保存完成: {save_path}")
            
            # 重置episode数据
            self.episode_data = []
            self.current_frame = 0
                
            # 重置采集状态，准备下一次采集
            self.is_collecting = False
            self.motion_frames_count = 0
            
            # 重置坐标系转换状态
            self._reset_coordinate_transform()
            
            # 采集完成，关闭系统
            self._shutdown()
                
        except Exception as e:
            rospy.logerr(f"保存episode失败: {e}")
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        rospy.logwarn("接收到中断信号，准备安全退出...")
        self.shutdown_requested = True
        
        # 保存当前采集的数据
        if self.episode_data:
            rospy.loginfo("保存当前采集的数据...")
            self._finish_episode()
        
        rospy.loginfo("安全退出完成")
        sys.exit(0)
    
    def start_collection(self):
        """开始数据采集"""
        rospy.loginfo("数据采集器已启动...")
        rospy.loginfo(f"采集模式: {self.collection_mode}")
        
        if self.collection_mode == 'manual':
            rospy.loginfo("手动模式 - 使用以下命令控制采集:")
            rospy.loginfo("  开始: rosservice call /collector/start_collection")
            rospy.loginfo("  停止: rosservice call /collector/stop_collection")
        else:
            rospy.loginfo("自动模式 - 检测到动作时自动开始采集")
            rospy.loginfo(f"动作检测阈值: {self.motion_threshold}")
            rospy.loginfo(f"开始前动作帧数: {self.min_frames_before_start}")
            rospy.loginfo("重要：一旦开始采集，将持续到400帧完成，不会中途停止")
        
        rospy.loginfo("按Ctrl+C安全停止采集")
        rospy.loginfo("请开始执行技能动作...")
        
        # 保持节点运行
        rate = rospy.Rate(10)  # 10Hz检查频率
        while not rospy.is_shutdown() and not self.shutdown_requested:
            rate.sleep()
    
    def _shutdown(self):
        """关闭节点"""
        rospy.loginfo("关闭采集器节点...")
        self.is_collecting = False
        self.shutdown_requested = True
        
        # 显示统计信息
        self._print_statistics()
    
    def _print_statistics(self):
        """打印统计信息"""
        total_frames = self.current_frame
        rospy.loginfo("=" * 50)
        rospy.loginfo("采集统计信息:")
        rospy.loginfo("完成Episode: 1")
        rospy.loginfo(f"总帧数: {total_frames}")
        rospy.loginfo(f"采样频率: {self.frequency}Hz")
        rospy.loginfo(f"技能ID: {self.skill_id}")
        rospy.loginfo(f"保存目录: {self.save_path}")
        rospy.loginfo("=" * 50)


def main():
    """主函数"""
    rospy.init_node('trajectory_collector', anonymous=True)
    
    try:
        # 创建采集器节点
        collector = CollectorNode()
        
        # 开始采集
        collector.start_collection()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS节点被中断")
    except Exception as e:
        rospy.logerr(f"采集器运行出错: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
