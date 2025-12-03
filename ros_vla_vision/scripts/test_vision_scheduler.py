#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VLA Vision Scheduler Test Script
视觉调度器测试脚本
"""

import rospy
import json
import time
from std_msgs.msg import String
from ros_vla_vision.scripts.vla_vision_scheduler import VLAVisionScheduler

class VisionSchedulerTester:
    def __init__(self):
        rospy.init_node('vision_scheduler_tester', anonymous=True)
        
        # 发布器 - 发送测试命令
        self.command_pub = rospy.Publisher(
            '/vla/vision_scheduler/command', 
            String, 
            queue_size=10
        )
        
        # 订阅器 - 接收调度结果
        self.result_sub = rospy.Subscriber(
            '/vla/vision_scheduler/result', 
            String, 
            self.result_callback
        )
        
        # 测试结果存储
        self.test_results = []
        self.current_test = None
        
        rospy.loginfo("Vision Scheduler Tester initialized")
    
    def result_callback(self, msg):
        """结果回调"""
        try:
            result = json.loads(msg.data)
            
            if self.current_test:
                result['test_name'] = self.current_test['name']
                result['test_instruction'] = self.current_test['instruction']
                self.test_results.append(result)
                
                rospy.loginfo(f"测试结果: {self.current_test['name']}")
                rospy.loginfo(f"成功: {result.get('success', False)}")
                
                if result.get('success'):
                    rospy.loginfo("测试通过!")
                else:
                    rospy.loginfo(f"测试失败: {result.get('error', 'Unknown error')}")
                
                self.current_test = None
                
        except Exception as e:
            rospy.logerr(f"Error processing result: {str(e)}")
    
    def send_test_command(self, action, modules, params, test_name, instruction):
        """发送测试命令"""
        try:
            command = {
                'action': action,
                'modules': modules,
                'params': params
            }
            
            self.current_test = {
                'name': test_name,
                'instruction': instruction
            }
            
            command_msg = String()
            command_msg.data = json.dumps(command, ensure_ascii=False)
            self.command_pub.publish(command_msg)
            
            rospy.loginfo(f"发送测试命令: {test_name}")
            rospy.loginfo(f"指令: {instruction}")
            
        except Exception as e:
            rospy.logerr(f"Error sending test command: {str(e)}")
    
    def run_tests(self):
        """运行测试"""
        rospy.loginfo("开始运行视觉调度器测试...")
        
        # 等待调度器启动
        rospy.sleep(3)
        
        # 测试用例
        test_cases = [
            {
                'name': 'VLM场景理解测试',
                'instruction': '分析当前场景，描述你看到的内容',
                'modules': ['VLM'],
                'params': {
                    'task': '分析当前场景，描述你看到的内容',
                    'image_source': 'camera_1'
                }
            },
            {
                'name': 'YOLO物体检测测试',
                'instruction': '检测并定位场景中的所有物体',
                'modules': ['YOLO'],
                'params': {
                    'task': '检测并定位场景中的所有物体',
                    'image_source': 'camera_1'
                }
            },
            {
                'name': '深度信息测试',
                'instruction': '测量场景中物体的距离和3D位置',
                'modules': ['Depth'],
                'params': {
                    'task': '测量场景中物体的距离和3D位置',
                    'image_source': 'camera_1',
                    'depth_source': 'depth_cam_1'
                }
            },
            {
                'name': '多模态融合测试',
                'instruction': '找到桌子上的红色杯子并获取其3D位置',
                'modules': ['VLM', 'YOLO', 'Depth'],
                'params': {
                    'task': '找到桌子上的红色杯子并获取其3D位置',
                    'image_source': 'camera_1',
                    'depth_source': 'depth_cam_1'
                }
            },
            {
                'name': 'VLM+YOLO融合测试',
                'instruction': '识别并分类场景中的所有物体',
                'modules': ['VLM', 'YOLO'],
                'params': {
                    'task': '识别并分类场景中的所有物体',
                    'image_source': 'camera_1'
                }
            },
            {
                'name': 'YOLO+Depth融合测试',
                'instruction': '检测物体并获取其精确的3D坐标',
                'modules': ['YOLO', 'Depth'],
                'params': {
                    'task': '检测物体并获取其精确的3D坐标',
                    'image_source': 'camera_1',
                    'depth_source': 'depth_cam_1'
                }
            }
        ]
        
        # 运行测试
        for i, test_case in enumerate(test_cases):
            rospy.loginfo(f"\n{'='*50}")
            rospy.loginfo(f"运行测试 {i+1}/{len(test_cases)}: {test_case['name']}")
            rospy.loginfo(f"{'='*50}")
            
            self.send_test_command(
                action='call_perception',
                modules=test_case['modules'],
                params=test_case['params'],
                test_name=test_case['name'],
                instruction=test_case['instruction']
            )
            
            # 等待结果
            rospy.sleep(5)
        
        # 等待最后一个结果
        rospy.sleep(2)
        
        # 生成测试报告
        self.generate_test_report()
    
    def generate_test_report(self):
        """生成测试报告"""
        try:
            rospy.loginfo(f"\n{'='*60}")
            rospy.loginfo("视觉调度器测试报告")
            rospy.loginfo(f"{'='*60}")
            
            total_tests = len(self.test_results)
            passed_tests = sum(1 for result in self.test_results if result.get('success', False))
            failed_tests = total_tests - passed_tests
            
            rospy.loginfo(f"总测试数: {total_tests}")
            rospy.loginfo(f"通过: {passed_tests}")
            rospy.loginfo(f"失败: {failed_tests}")
            rospy.loginfo(f"成功率: {passed_tests/total_tests*100:.1f}%")
            
            rospy.loginfo(f"\n详细结果:")
            rospy.loginfo(f"{'-'*60}")
            
            for i, result in enumerate(self.test_results, 1):
                test_name = result.get('test_name', f'Test_{i}')
                instruction = result.get('test_instruction', 'N/A')
                success = result.get('success', False)
                error = result.get('error', 'N/A')
                
                status = "✓ 通过" if success else "✗ 失败"
                rospy.loginfo(f"{i}. {test_name}")
                rospy.loginfo(f"   指令: {instruction}")
                rospy.loginfo(f"   状态: {status}")
                
                if not success:
                    rospy.loginfo(f"   错误: {error}")
                
                # 显示模块使用情况
                if 'modules_used' in result:
                    rospy.loginfo(f"   使用模块: {', '.join(result['modules_used'])}")
                
                # 显示处理时间
                if 'timestamp' in result:
                    rospy.loginfo(f"   时间戳: {result['timestamp']}")
                
                rospy.loginfo(f"   {'-'*40}")
            
            # 保存测试报告到文件
            report_file = '/tmp/vision_scheduler_test_report.json'
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, ensure_ascii=False, indent=2)
            
            rospy.loginfo(f"\n测试报告已保存到: {report_file}")
            
        except Exception as e:
            rospy.logerr(f"Error generating test report: {str(e)}")
    
    def run_direct_tests(self):
        """直接测试调度器的指令分析功能"""
        try:
            rospy.loginfo("开始直接测试调度器的指令分析功能...")
            
            # 创建调度器实例
            scheduler = VLAVisionScheduler()
            
            # 测试指令
            test_instructions = [
                "找到桌子上的红色杯子",
                "检测房间里的人",
                "分析当前场景",
                "测量物体到相机的距离",
                "识别并定位所有椅子",
                "阅读屏幕上的文字",
                "描述这个房间的布局",
                "计算那个物体的大小",
                "看看周围有什么危险物品",
                "帮我找到最近的出口"
            ]
            
            for instruction in test_instructions:
                rospy.loginfo(f"\n测试指令: {instruction}")
                
                # 分析指令
                analysis_result = scheduler.analyze_task_and_call_modules(instruction)
                
                rospy.loginfo("分析结果:")
                rospy.loginfo(f"  动作: {analysis_result.get('action', 'Unknown')}")
                rospy.loginfo(f"  原因: {analysis_result.get('reason', 'Unknown')}")
                rospy.loginfo(f"  模块: {analysis_result.get('modules', [])}")
                rospy.loginfo(f"  任务: {analysis_result.get('params', {}).get('task', 'Unknown')}")
                
                time.sleep(1)
            
            rospy.loginfo("直接测试完成!")
            
        except Exception as e:
            rospy.logerr(f"Error in direct tests: {str(e)}")
    
    def run(self):
        """运行测试器"""
        try:
            rospy.loginfo("Vision Scheduler Tester is running...")
            
            # 等待ROS系统准备就绪
            rospy.sleep(2)
            
            # 运行直接测试
            self.run_direct_tests()
            
            # 等待一下
            rospy.sleep(3)
            
            # 运行完整的集成测试
            self.run_tests()
            
            rospy.loginfo("所有测试完成!")
            
        except rospy.ROSInterruptException:
            rospy.loginfo("Test interrupted")

if __name__ == '__main__':
    try:
        tester = VisionSchedulerTester()
        tester.run()
    except rospy.ROSInterruptException:
        pass
