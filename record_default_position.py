#!/usr/bin/env python3
"""
记录机器人默认初始位置脚本
运行此脚本记录机器人当前姿态作为默认初始位置
"""

import rospy
import numpy as np
import json
import os
from std_msgs.msg import Float64MultiArray
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import TransformStamped

class DefaultPositionRecorder:
    def __init__(self):
        rospy.init_node('default_position_recorder')
        
        # TF设置
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.base_frame = "base_link"
        self.left_hand_frame = "zarm_l7_end_effector"
        self.right_hand_frame = "zarm_r7_end_effector"
        
        # 关节位置
        self.current_joint_positions = None
        
        # 订阅关节状态
        rospy.Subscriber('/humanoid_controller/optimizedState_mrt/joint_pos', 
                        Float64MultiArray, self._joint_state_callback)
        
        rospy.loginfo("默认位置记录器已启动")
        rospy.loginfo("请将机器人调整到标准的初始姿态")
        rospy.loginfo("然后按回车键记录当前位置...")
        
    def _joint_state_callback(self, msg):
        """关节状态回调"""
        self.current_joint_positions = np.array(msg.data[:26])
        
    def record_positions(self):
        """记录当前位置"""
        # 等待用户按下回车键
        input("按回车键记录当前位置...")
        
        # 检查关节数据
        if self.current_joint_positions is None:
            rospy.logerr("未接收到关节数据")
            return False
            
        # 记录关节位置
        joint_positions = self.current_joint_positions.tolist()
        
        # 尝试记录TF位置
        tf_positions = {}
        try:
            rospy.loginfo("获取TF变换...")
            
            # 获取左手末端位置
            left_transform = self.tf_buffer.lookup_transform(
                self.base_frame, self.left_hand_frame, rospy.Time(0), rospy.Duration(1.0))
            tf_positions['left_hand'] = [
                left_transform.transform.translation.x,
                left_transform.transform.translation.y,
                left_transform.transform.translation.z
            ]
            
            # 获取右手末端位置
            right_transform = self.tf_buffer.lookup_transform(
                self.base_frame, self.right_hand_frame, rospy.Time(0), rospy.Duration(1.0))
            tf_positions['right_hand'] = [
                right_transform.transform.translation.x,
                right_transform.transform.translation.y,
                right_transform.transform.translation.z
            ]
            
            rospy.loginfo("TF位置获取成功")
            
        except Exception as e:
            rospy.logwarn(f"TF位置获取失败: {e}")
            tf_positions = {}
        
        # 组合数据
        default_data = {
            'joint_positions': joint_positions,
            'tf_positions': tf_positions,
            'timestamp': rospy.Time.now().to_sec(),
            'description': '机器人默认初始位置'
        }
        
        # 保存到文件
        config_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(config_dir, 'default_initial_position.json')
        
        try:
            with open(config_file, 'w') as f:
                json.dump(default_data, f, indent=2)
            
            rospy.loginfo(f"默认位置已保存到: {config_file}")
            rospy.loginfo("关节位置: [%.3f, %.3f, %.3f, ...] (前3个关节)" % tuple(joint_positions[:3]))
            
            if tf_positions:
                rospy.loginfo("左手TF位置: [%.3f, %.3f, %.3f]" % tuple(tf_positions['left_hand']))
                rospy.loginfo("右手TF位置: [%.3f, %.3f, %.3f]" % tuple(tf_positions['right_hand']))
            
            return True
            
        except Exception as e:
            rospy.logerr(f"保存文件失败: {e}")
            return False
    
    def run(self):
        """运行记录器"""
        # 等待关节数据
        rospy.loginfo("等待关节数据...")
        while self.current_joint_positions is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
            
        if rospy.is_shutdown():
            return
            
        rospy.loginfo("关节数据已接收")
        
        # 记录位置
        if self.record_positions():
            rospy.loginfo("默认位置记录完成！")
        else:
            rospy.logerr("默认位置记录失败！")

def main():
    """主函数"""
    try:
        recorder = DefaultPositionRecorder()
        recorder.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("节点被中断")
    except Exception as e:
        rospy.logerr(f"运行出错: {e}")

if __name__ == '__main__':
    main()