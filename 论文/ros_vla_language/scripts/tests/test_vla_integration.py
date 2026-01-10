#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VLA语言系统测试脚本
VLA Language System Test Script

测试修改后的语言系统与VLA控制系统的集成
"""

import os
import sys
import time
import json
import logging
from typing import Dict, Any

# ROS相关导入
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from vla_language.msg import VLACommand
from vla_language.srv import ProcessText

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VLAIntegrationTester(Node):
    """VLA集成测试节点"""
    
    def __init__(self):
        super().__init__('vla_integration_tester')
        
        # 创建服务客户端
        self.process_text_client = self.create_client(ProcessText, '/vla_language/process_text')
        
        # 创建订阅者，监听VLA指令
        self.command_subscriber = self.create_subscription(
            VLACommand,
            '/vla_control/command',
            self.command_callback,
            10
        )
        
        # 监听语言服务的意图发布
        self.intent_subscriber = self.create_subscription(
            String,
            '/vla_language/latest_intent',
            self.intent_callback,
            10
        )
        
        # 测试数据
        self.test_cases = [
            "挥手",
            "挥挥手",
            "请挥挥手",
            "抱拳",
            "请抱拳",
            "做一个抱拳礼",
            "停止",
            "停止动作",
            "你好",
            "未知指令"
        ]
        
        # 测试结果
        self.test_results = []
        self.received_commands = []
        self.received_intents = []
        
        logger.info("✅ VLA集成测试节点已启动")
    
    def command_callback(self, msg):
        """监听VLA指令回调"""
        command_data = {
            'instruction': msg.instruction,
            'confidence': msg.confidence,
            'response_text': msg.response_text,
            'original_text': msg.command_text,
            'timestamp': time.time()
        }
        self.received_commands.append(command_data)
        logger.info(f"📡 收到VLA指令: {msg.instruction} (置信度: {msg.confidence:.2f})")
    
    def intent_callback(self, msg):
        """监听意图回调"""
        try:
            intent_data = json.loads(msg.data)
            self.received_intents.append(intent_data)
            logger.info(f"🎯 收到意图: {intent_data.get('intent')} (指令: {intent_data.get('instruction')})")
        except Exception as e:
            logger.error(f"解析意图数据失败: {str(e)}")
    
    def test_text_processing(self):
        """测试文本处理"""
        logger.info("🧪 开始测试文本处理...")
        
        # 等待服务可用
        if not self.process_text_client.wait_for_service(timeout_sec=5.0):
            logger.error("❌ 语言处理服务不可用")
            return False
        
        success_count = 0
        
        for i, test_text in enumerate(self.test_cases):
            logger.info(f"测试 {i+1}/{len(self.test_cases)}: '{test_text}'")
            
            try:
                # 调用服务
                request = ProcessText.Request()
                request.text = test_text
                
                future = self.process_text_client.call_async(request)
                rclpy.spin_until_future_complete(self, future)
                
                if future.result() is not None and future.result().success:
                    response = future.result()
                    logger.info(f"✅ 处理成功: {response.intent} (置信度: {response.confidence:.2f})")
                    success_count += 1
                else:
                    logger.warning(f"❌ 处理失败: {test_text}")
                
                # 等待处理
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"测试失败: {str(e)}")
        
        logger.info(f"📊 文本处理测试结果: {success_count}/{len(self.test_cases)} 成功")
        return success_count >= len(self.test_cases) * 0.8
    
    def test_command_publishing(self):
        """测试指令发布"""
        logger.info("🧪 开始测试指令发布...")
        
        # 等待接收指令
        start_time = time.time()
        timeout = 10  # 10秒超时
        
        while time.time() - start_time < timeout:
            if len(self.received_commands) > 0:
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        
        if len(self.received_commands) > 0:
            logger.info(f"✅ 成功收到 {len(self.received_commands)} 个VLA指令")
            
            # 显示收到的指令
            for i, cmd in enumerate(self.received_commands):
                logger.info(f"  指令 {i+1}: {cmd['instruction']} - {cmd['original_text']}")
            
            return True
        else:
            logger.warning("❌ 没有收到VLA指令")
            return False
    
    def test_intent_mapping(self):
        """测试意图映射"""
        logger.info("🧪 开始测试意图映射...")
        
        expected_mappings = {
            'wave': 'wave',
            'welcome': 'welcome',
            'stop': 'none',
            'unknown': 'none'
        }
        
        correct_mappings = 0
        
        for intent_data in self.received_intents:
            intent = intent_data.get('intent')
            instruction = intent_data.get('instruction')
            
            if intent in expected_mappings:
                expected_instruction = expected_mappings[intent]
                if instruction == expected_instruction:
                    correct_mappings += 1
                    logger.info(f"✅ 正确映射: {intent} -> {instruction}")
                else:
                    logger.warning(f"❌ 错误映射: {intent} -> {instruction} (期望: {expected_instruction})")
        
        total_mappings = len([i for i in self.received_intents if i.get('intent') in expected_mappings])
        
        if total_mappings > 0:
            accuracy = correct_mappings / total_mappings
            logger.info(f"📊 意图映射准确率: {accuracy:.2%} ({correct_mappings}/{total_mappings})")
            return accuracy >= 0.8
        else:
            logger.warning("❌ 没有找到有效的意图映射")
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("🚀 开始运行VLA集成测试...")
        
        test_results = []
        
        # 1. 测试文本处理
        test_results.append(("文本处理", self.test_text_processing()))
        
        # 2. 测试指令发布
        test_results.append(("指令发布", self.test_command_publishing()))
        
        # 3. 测试意图映射
        test_results.append(("意图映射", self.test_intent_mapping()))
        
        # 汇总结果
        logger.info("="*50)
        logger.info("📋 测试结果汇总:")
        logger.info("="*50)
        
        passed_tests = 0
        for test_name, result in test_results:
            status = "✅ 通过" if result else "❌ 失败"
            logger.info(f"{test_name}: {status}")
            if result:
                passed_tests += 1
        
        logger.info("="*50)
        logger.info(f"🎯 总体结果: {passed_tests}/{len(test_results)} 测试通过")
        
        if passed_tests == len(test_results):
            logger.info("🎉 所有测试通过！VLA语言系统集成成功！")
        else:
            logger.warning("⚠️ 部分测试失败，请检查系统配置")
        
        return passed_tests == len(test_results)

def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    # 创建测试节点
    tester = VLAIntegrationTester()
    
    try:
        # 运行测试
        success = tester.run_all_tests()
        
        if success:
            logger.info("✅ VLA语言系统集成测试完成，系统正常工作")
        else:
            logger.error("❌ VLA语言系统集成测试失败")
            
    except KeyboardInterrupt:
        logger.info("用户中断，正在关闭测试...")
    finally:
        tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()