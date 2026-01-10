#!/usr/bin/env python3
"""
简化的机器人轨迹回播器
发布速度命令来控制机器人运动
"""

import rospy
import json
import time
import math
import numpy as np
import os
import sys
import argparse
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger, TriggerResponse
from kuavo_msgs.msg import armTargetPoses
from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest


class SimpleRobotPlayer:
    """简化的机器人回播器"""
    
    def __init__(self, trajectory_file: str, config: dict):
        """初始化回播器"""
        self.trajectory_file = trajectory_file
        self.config = config
        
        # 回播状态
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0
        self.trajectory_data = None
        self.loop_count = 0
        self.max_loops = config.get('max_loops', 1)
        
        # 回播控制
        self.playback_speed = config.get('playback_speed', 1.0)
        self.frequency = config.get('frequency', 50.0)
        self.frame_interval = 1.0 / self.frequency
        self.last_frame_time = 0
        
        # 机器人控制参数
        self.linear_scale = config.get('linear_scale', 1.0)  # 线性速度缩放
        self.angular_scale = config.get('angular_scale', 2.0)  # 角速度缩放
        
        # 位置变化阈值（用于过滤小幅度身体晃动）
        self.position_threshold = config.get('position_threshold', 0.005)  # 5mm阈值
        
        # 坐标系转换参数
        self.current_robot_yaw = 0.0  # 当前机器人的实际朝向
        self.use_coordinate_transform = config.get('use_coordinate_transform', True)
        
        # 手臂控制参数
        self.arm_control_mode = config.get('arm_control_mode', 2)  # 2: external_control
        self.arm_joint_names = config.get('arm_joint_names', 
            ['zarm_l1_joint', 'zarm_l2_joint', 'zarm_l3_joint', 'zarm_l4_joint', 'zarm_l5_joint', 'zarm_l6_joint', 'zarm_l7_joint',
             'zarm_r1_joint', 'zarm_r2_joint', 'zarm_r3_joint', 'zarm_r4_joint', 'zarm_r5_joint', 'zarm_r6_joint', 'zarm_r7_joint'])
        
        # 技能ID到控制方式的映射
        self.skill_control_mapping = config.get('skill_control_mapping', {
            'wave': 'arm',           # 挥手动作使用手臂控制
            'hand_wave': 'arm',      # 手部挥动使用手臂控制
            'arm_motion': 'arm',     # 手臂动作使用手臂控制
            'walk': 'base',          # 行走使用底盘控制
            'walk_forward': 'base',
            'walk_backward': 'base', 
            'walk_back': 'base',
            'turn': 'base',          # 转向使用底盘控制
            'turn_left': 'base',
            'turn_right': 'base',
            'stand': 'none',         # 站立不发送控制命令
            'idle': 'none'
        })
        
        # 加载轨迹数据
        self._load_trajectory_data()
        
        # 确定当前技能的控制方式
        self.control_mode = self._determine_control_mode()
        
        # 初始化ROS发布器
        self._setup_ros_publishers()
        
        # 设置控制服务
        self._setup_control_services()
        
        # 如果是手臂控制，设置手臂控制模式
        if self.control_mode == 'arm':
            self._setup_arm_control()
        
        rospy.loginfo("简单机器人回播器初始化完成")
        rospy.loginfo(f"轨迹文件: {self.trajectory_file}")
        rospy.loginfo(f"总帧数: {self.total_frames}")
        rospy.loginfo(f"回播速度: {self.playback_speed}x")
        rospy.loginfo(f"控制频率: {self.frequency}Hz")
        rospy.loginfo(f"技能ID: {self.trajectory_data.get('skill_id', 'unknown')}")
        rospy.loginfo(f"控制方式: {self.control_mode}")
        rospy.loginfo(f"位置变化阈值: {self.position_threshold}m")
    
    def _load_trajectory_data(self):
        """加载轨迹数据"""
        try:
            with open(self.trajectory_file, 'r', encoding='utf-8') as f:
                self.trajectory_data = json.load(f)
            
            # 验证数据格式
            if "observations" not in self.trajectory_data:
                raise ValueError("轨迹数据缺少observations字段")
            
            self.total_frames = len(self.trajectory_data["observations"])
            
            if self.total_frames == 0:
                raise ValueError("轨迹数据为空")
            
            rospy.loginfo(f"成功加载轨迹数据: {self.total_frames} 帧")
            
        except Exception as e:
            rospy.logerr(f"加载轨迹数据失败: {e}")
            raise
    
    def _determine_control_mode(self):
        """根据技能ID确定控制方式"""
        # 如果强制指定了控制模式，直接使用
        force_mode = self.config.get('force_control_mode', 'auto')
        if force_mode != 'auto':
            rospy.loginfo(f"使用强制指定的控制模式: {force_mode}")
            return force_mode
        
        # 否则根据技能ID自动确定控制方式
        skill_id = self.trajectory_data.get('skill_id', 'unknown')
        
        # 检查技能ID是否在映射中
        for key, mode in self.skill_control_mapping.items():
            if key in skill_id.lower():
                rospy.loginfo(f"根据技能ID '{skill_id}' 匹配到控制模式: {mode}")
                return mode
        
        # 默认使用底盘控制
        rospy.logwarn(f"技能ID '{skill_id}' 未在映射中找到，使用默认底盘控制")
        return 'base'
    
    def _setup_arm_control(self):
        """设置手臂控制模式"""
        try:
            rospy.wait_for_service('/arm_traj_change_mode', timeout=5.0)
            change_mode = rospy.ServiceProxy('/arm_traj_change_mode', changeArmCtrlMode)
            req = changeArmCtrlModeRequest()
            req.control_mode = self.arm_control_mode
            res = change_mode(req)
            
            if res.result:
                rospy.loginfo(f"手臂控制模式已设置为: {self.arm_control_mode}")
            else:
                rospy.logerr(f"设置手臂控制模式失败: {res.message}")
        except Exception as e:
            rospy.logerr(f"设置手臂控制模式时出错: {e}")
    
    def _setup_ros_publishers(self):
        """设置ROS发布器"""
        try:
            # 速度命令发布器（底盘控制）
            self.cmd_vel_pub = rospy.Publisher(
                '/cmd_vel',
                Twist,
                queue_size=10
            )
            
            # 手臂目标姿态发布器（手臂控制）
            self.arm_target_pub = rospy.Publisher(
                '/kuavo_arm_target_poses',
                armTargetPoses,
                queue_size=10
            )
            
            # 机器人腰部运动数据发布器
            self.waist_pub = rospy.Publisher(
                '/robot_waist_motion_data',
                Float64MultiArray,
                queue_size=10
            )
            
            rospy.loginfo("ROS发布器设置完成")
            
        except Exception as e:
            rospy.logerr(f"设置ROS发布器失败: {e}")
            raise
    
    def _setup_control_services(self):
        """设置控制服务"""
        try:
            # 开始回播服务
            self.start_service = rospy.Service(
                '/simple_player/start',
                Trigger,
                self._start_callback
            )
            
            # 停止回播服务
            self.stop_service = rospy.Service(
                '/simple_player/stop',
                Trigger,
                self._stop_callback
            )
            
            # 暂停/继续回播服务
            self.pause_service = rospy.Service(
                '/simple_player/pause',
                Trigger,
                self._pause_callback
            )
            
            # 重置回播服务
            self.reset_service = rospy.Service(
                '/simple_player/reset',
                Trigger,
                self._reset_callback
            )
            
            # 设置机器人朝向服务
            self.set_yaw_service = rospy.Service(
                '/simple_player/set_robot_yaw',
                Trigger,
                self._set_robot_yaw_callback
            )
            
            rospy.loginfo("控制服务设置完成:")
            rospy.loginfo("  开始: rosservice call /simple_player/start")
            rospy.loginfo("  停止: rosservice call /simple_player/stop")
            rospy.loginfo("  暂停/继续: rosservice call /simple_player/pause")
            rospy.loginfo("  重置: rosservice call /simple_player/reset")
            rospy.loginfo("  设置机器人朝向: rosservice call /simple_player/set_robot_yaw")
            rospy.loginfo("  设置机器人朝向参数: rosparam set /simple_player/robot_yaw_deg 90")
            
        except Exception as e:
            rospy.logerr(f"设置控制服务失败: {e}")
    
    def _start_callback(self, req):
        """开始回播回调"""
        if self.is_playing:
            return TriggerResponse(False, "回播已在进行中")
        
        self.is_playing = True
        self.last_frame_time = time.time()
        rospy.loginfo("开始回播轨迹数据")
        return TriggerResponse(True, "回播已开始")
    
    def _stop_callback(self, req):
        """停止回播回调"""
        if not self.is_playing:
            return TriggerResponse(False, "回播未在进行中")
        
        self.is_playing = False
        self._publish_zero_velocity()  # 停止机器人
        rospy.loginfo("停止回播轨迹数据")
        return TriggerResponse(True, "回播已停止")
    
    def _pause_callback(self, req):
        """暂停/继续回播回调"""
        if not self.is_playing:
            return TriggerResponse(False, "回播未在进行中")
        
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.last_frame_time = time.time()
            rospy.loginfo("继续回播轨迹数据")
        else:
            self._publish_zero_velocity()  # 暂停时停止机器人
            rospy.loginfo("暂停回播轨迹数据")
        
        return TriggerResponse(True, "回播状态已切换")
    
    def _reset_callback(self, req):
        """重置回播回调"""
        self.is_playing = False
        self.current_frame = 0
        self.loop_count = 0
        self.last_frame_time = 0
        self._publish_zero_velocity()  # 重置时停止机器人
        rospy.loginfo("重置回播到开始位置")
        return TriggerResponse(True, "回播已重置")
    
    def _set_robot_yaw_callback(self, req):
        """设置机器人朝向回调"""
        try:
            # 从请求中获取机器人朝向（这里简化处理，实际可能需要从参数中获取）
            # 暂时使用一个固定的值或者从ROS参数服务器获取
            robot_yaw_deg = rospy.get_param('/simple_player/robot_yaw_deg', 0.0)
            self.current_robot_yaw = math.radians(robot_yaw_deg)
            rospy.loginfo(f"设置机器人朝向: {robot_yaw_deg}° ({self.current_robot_yaw:.3f} rad)")
            return TriggerResponse(True, f"机器人朝向已设置为 {robot_yaw_deg}°")
        except Exception as e:
            rospy.logerr(f"设置机器人朝向失败: {e}")
            return TriggerResponse(False, f"设置机器人朝向失败: {e}")
    
    def _publish_zero_velocity(self):
        """发布零速度命令停止机器人"""
        try:
            twist_msg = Twist()
            twist_msg.linear.x = 0.0
            twist_msg.linear.y = 0.0
            twist_msg.linear.z = 0.0
            twist_msg.angular.x = 0.0
            twist_msg.angular.y = 0.0
            twist_msg.angular.z = 0.0
            self.cmd_vel_pub.publish(twist_msg)
        except Exception as e:
            rospy.logerr(f"发布零速度命令失败: {e}")
    
    def _calculate_velocity_command(self, current_obs, next_obs):
        """根据观测数据计算速度命令"""
        try:
            # 获取位置数据（现在已经是相对于初始位置的位移）
            if "root_pos" in current_obs:
                current_pos = np.array(current_obs["root_pos"])
                next_pos = np.array(next_obs["root_pos"])
                
                # 计算位置变化（已经是相对于初始位置的位移变化）
                delta_pos = next_pos - current_pos
                
                # 如果启用坐标系转换，将位移转换为世界坐标
                if self.use_coordinate_transform:
                    # 修复后的逻辑：根据机器人当前朝向转换位移方向
                    cos_yaw = math.cos(self.current_robot_yaw)
                    sin_yaw = math.sin(self.current_robot_yaw)
                    
                    # 旋转矩阵：从本体坐标系到世界坐标系
                    # 这是采集端转换的逆向操作
                    world_delta_x = cos_yaw * delta_pos[0] - sin_yaw * delta_pos[1]
                    world_delta_y = sin_yaw * delta_pos[0] + cos_yaw * delta_pos[1]
                    
                    rospy.loginfo(f"坐标系转换: 本体位移[{delta_pos[0]:.3f}, {delta_pos[1]:.3f}] -> "
                                f"世界位移[{world_delta_x:.3f}, {world_delta_y:.3f}], "
                                f"机器人朝向: {math.degrees(self.current_robot_yaw):.1f}°")
                    
                    delta_pos[0] = world_delta_x
                    delta_pos[1] = world_delta_y
                
                # 添加位置变化阈值过滤
                # 如果位移小于阈值，认为是身体晃动，不产生运动
                if abs(delta_pos[0]) < self.position_threshold and abs(delta_pos[1]) < self.position_threshold:
                    linear_x = 0.0
                    linear_y = 0.0
                    rospy.logdebug(f"位置变化[{delta_pos[0]:.3f}, {delta_pos[1]:.3f}]小于阈值{self.position_threshold}，过滤为0")
                else:
                    # 转换为速度 (位置变化 * 频率)
                    linear_x = delta_pos[0] * self.frequency * self.linear_scale
                    linear_y = delta_pos[1] * self.frequency * self.linear_scale
                
                # 如果有姿态数据，计算角速度
                if "root_orien" in current_obs and "root_orien" in next_obs:
                    current_quat = np.array(current_obs["root_orien"])
                    next_quat = np.array(next_obs["root_orien"])
                    
                    # 正确的角速度计算：从四元数提取偏航角变化
                    # 四元数格式: [x, y, z, w]
                    def quaternion_to_yaw(quat):
                        """从四元数提取偏航角"""
                        x, y, z, w = quat
                        # 偏航角 (绕Z轴旋转)
                        siny_cosp = 2 * (w * z + x * y)
                        cosy_cosp = 1 - 2 * (y * y + z * z)
                        return math.atan2(siny_cosp, cosy_cosp)
                    
                    current_yaw = quaternion_to_yaw(current_quat)
                    next_yaw = quaternion_to_yaw(next_quat)
                    
                    # 计算偏航角变化，并归一化到[-π, π]
                    yaw_diff = next_yaw - current_yaw
                    # 归一化角度差
                    while yaw_diff > math.pi:
                        yaw_diff -= 2 * math.pi
                    while yaw_diff < -math.pi:
                        yaw_diff += 2 * math.pi
                    
                    angular_z = yaw_diff * self.frequency * self.angular_scale
                else:
                    angular_z = 0.0
            else:
                linear_x = 0.0
                linear_y = 0.0
                angular_z = 0.0
            
            return linear_x, linear_y, angular_z
            
        except Exception as e:
            rospy.logerr(f"计算速度命令失败: {e}")
            return 0.0, 0.0, 0.0
    
    def _publish_frame_data(self, frame_index: int):
        """发布单帧数据"""
        if frame_index >= self.total_frames - 1:  # 需要下一帧来计算速度
            return False
        
        try:
            current_obs = self.trajectory_data["observations"][frame_index]
            next_obs = self.trajectory_data["observations"][frame_index + 1]
            
            # 根据控制方式选择不同的发布策略
            if self.control_mode == 'arm':
                # 手臂控制：发布手臂关节目标位置
                success = self._publish_arm_control(frame_index, current_obs)
            elif self.control_mode == 'base':
                # 底盘控制：发布速度命令
                success = self._publish_base_control(frame_index, current_obs, next_obs)
            else:  # 'none' 或其他
                # 不发布控制命令，只记录进度
                success = self._publish_no_control(frame_index)
            
            return success
            
        except Exception as e:
            rospy.logerr(f"发布第{frame_index}帧数据失败: {e}")
            return False
    
    def _publish_arm_control(self, frame_index: int, current_obs):
        """发布手臂控制命令"""
        try:
            if "joint_pos" not in current_obs:
                rospy.logwarn(f"第{frame_index}帧缺少关节数据，跳过手臂控制")
                return False
            
            # 提取手臂关节数据（根据实际机器人配置）
            joint_positions = current_obs["joint_pos"]
            if len(joint_positions) < 26:
                rospy.logwarn(f"第{frame_index}帧关节数据不足，跳过手臂控制")
                return False
            
            # 根据机器人配置提取手臂关节：左臂(12-18)，右臂(19-25)
            left_arm_joints = joint_positions[12:19]  # 左臂7个关节
            right_arm_joints = joint_positions[19:26]  # 右臂7个关节
            
            # 将弧度转换为度（手臂控制系统期望度数）
            left_arm_degrees = [math.degrees(joint) for joint in left_arm_joints]
            right_arm_degrees = [math.degrees(joint) for joint in right_arm_joints]
            
            # 组合左右臂关节数据（14个关节）
            arm_joints = left_arm_degrees + right_arm_degrees
            
            # 创建手臂目标姿态消息
            arm_msg = armTargetPoses()
            arm_msg.times = [0.0]  # 立即执行
            arm_msg.values = arm_joints
            arm_msg.frame = 2  # local frame
            
            # 发布手臂目标姿态
            self.arm_target_pub.publish(arm_msg)
            
            # 输出调试信息
            progress = (frame_index / self.total_frames) * 100
            rospy.loginfo(f"帧 {frame_index + 1}/{self.total_frames} ({progress:.1f}%) - "
                         f"手臂控制: 左臂={left_arm_degrees[:3]}..., 右臂={right_arm_degrees[:3]}...")
            
            return True
            
        except Exception as e:
            rospy.logerr(f"发布手臂控制失败: {e}")
            return False
    
    def _publish_base_control(self, frame_index: int, current_obs, next_obs):
        """发布底盘控制命令"""
        try:
            # 计算速度命令
            linear_x, linear_y, angular_z = self._calculate_velocity_command(current_obs, next_obs)
            
            # 发布速度命令
            twist_msg = Twist()
            twist_msg.linear.x = linear_x
            twist_msg.linear.y = linear_y
            twist_msg.linear.z = 0.0
            twist_msg.angular.x = 0.0
            twist_msg.angular.y = 0.0
            twist_msg.angular.z = angular_z
            
            self.cmd_vel_pub.publish(twist_msg)
            
            # 发布腰部运动数据（如果有）
            if "joint_pos" in current_obs and len(current_obs["joint_pos"]) >= 26:
                waist_msg = Float64MultiArray()
                # 假设腰部关节是第25和26个关节（根据实际机器人配置调整）
                waist_data = [
                    current_obs["joint_pos"][24] if len(current_obs["joint_pos"]) > 24 else 0.0,
                    current_obs["joint_pos"][25] if len(current_obs["joint_pos"]) > 25 else 0.0
                ]
                waist_msg.data = waist_data
                self.waist_pub.publish(waist_msg)
            
            # 输出调试信息
            progress = (frame_index / self.total_frames) * 100
            rospy.loginfo(f"帧 {frame_index + 1}/{self.total_frames} ({progress:.1f}%) - "
                         f"底盘控制: x={linear_x:.3f}, y={linear_y:.3f}, z={angular_z:.3f}")
            
            return True
            
        except Exception as e:
            rospy.logerr(f"发布底盘控制失败: {e}")
            return False
    
    def _publish_no_control(self, frame_index: int):
        """不发布控制命令，只记录进度"""
        try:
            # 输出调试信息
            progress = (frame_index / self.total_frames) * 100
            rospy.loginfo(f"帧 {frame_index + 1}/{self.total_frames} ({progress:.1f}%) - "
                         f"无控制模式，跳过发布")
            
            return True
            
        except Exception as e:
            rospy.logerr(f"无控制模式处理失败: {e}")
            return False
    
    def start_playback(self):
        """开始回播"""
        rospy.loginfo("简单机器人回播器已启动...")
        rospy.loginfo("使用以下命令控制回播:")
        rospy.loginfo("  开始: rosservice call /simple_player/start")
        rospy.loginfo("  停止: rosservice call /simple_player/stop")
        rospy.loginfo("  暂停/继续: rosservice call /simple_player/pause")
        rospy.loginfo("  重置: rosservice call /simple_player/reset")
        rospy.loginfo("按Ctrl+C退出")
        
        # 保持节点运行
        rate = rospy.Rate(100)  # 100Hz检查频率
        
        while not rospy.is_shutdown():
            if self.is_playing:
                current_time = time.time()
                
                # 检查是否该发布下一帧
                if current_time - self.last_frame_time >= self.frame_interval / self.playback_speed:
                    # 发布当前帧
                    success = self._publish_frame_data(self.current_frame)
                    
                    if success:
                        # 移动到下一帧
                        self.current_frame += 1
                        self.last_frame_time = current_time
                        
                        # 检查是否完成当前回播
                        if self.current_frame >= self.total_frames - 1:
                            self._handle_playback_complete()
                    else:
                        # 发布失败，停止回播
                        self.is_playing = False
                        rospy.logerr("发布数据失败，停止回播")
            
            rate.sleep()
    
    def _handle_playback_complete(self):
        """处理回播完成"""
        self.loop_count += 1
        
        if self.max_loops > 0 and self.loop_count >= self.max_loops:
            # 达到最大循环次数，停止回播
            self.is_playing = False
            self._publish_zero_velocity()
            rospy.loginfo(f"回播完成! 共循环 {self.loop_count} 次")
        else:
            # 继续循环
            self.current_frame = 0
            rospy.loginfo(f"完成第 {self.loop_count} 次循环，重新开始...")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='简单机器人轨迹回播器')
    parser.add_argument('--trajectory_file', help='轨迹数据文件路径')
    parser.add_argument('--speed', type=float, default=1.0, help='回播速度倍数 (默认: 1.0)')
    parser.add_argument('--frequency', type=float, default=50.0, help='控制频率Hz (默认: 50.0)')
    parser.add_argument('--loops', type=int, default=1, help='循环次数 (默认: 1, 0表示无限循环)')
    parser.add_argument('--linear-scale', type=float, default=1.0, help='线性速度缩放 (默认: 1.0)')
    parser.add_argument('--angular-scale', type=float, default=2.0, help='角速度缩放 (默认: 2.0)')
    parser.add_argument('--robot-yaw', type=float, default=0.0, help='机器人当前朝向角度 (默认: 0.0)')
    parser.add_argument('--no-transform', action='store_true', help='禁用坐标系转换')
    parser.add_argument('--position-threshold', type=float, default=0.005, help='位置变化阈值(米) (默认: 0.005)')
    parser.add_argument('--arm-control-mode', type=int, default=2, help='手臂控制模式 (默认: 2)')
    parser.add_argument('--control-mode', type=str, choices=['auto', 'base', 'arm', 'none'], 
                       default='auto', help='强制指定控制模式 (默认: auto，根据skill_id自动选择)')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.trajectory_file):
        print(f"错误: 轨迹文件不存在: {args.trajectory_file}")
        sys.exit(1)
    
    # 构建配置字典
    config = {
        'playback_speed': args.speed,
        'frequency': args.frequency,
        'max_loops': args.loops,
        'linear_scale': args.linear_scale,
        'angular_scale': args.angular_scale,
        'use_coordinate_transform': not args.no_transform,
        'position_threshold': args.position_threshold,
        'arm_control_mode': args.arm_control_mode,
        'force_control_mode': args.control_mode
    }
    
    # 初始化ROS节点
    rospy.init_node('simple_robot_player', anonymous=True)
    
    try:
        # 创建回播器
        player = SimpleRobotPlayer(args.trajectory_file, config)
        
        # 设置机器人初始朝向
        player.current_robot_yaw = math.radians(args.robot_yaw)
        rospy.loginfo(f"设置机器人初始朝向: {args.robot_yaw}°")
        
        # 开始回播
        player.start_playback()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS节点被中断")
    except Exception as e:
        rospy.logerr(f"回播器运行出错: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()