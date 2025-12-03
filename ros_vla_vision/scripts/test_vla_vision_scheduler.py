#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VLA Vision Scheduler Test Version
测试版本的视觉调度器，支持终端选择图片输入或摄像头输入
"""

import json
import time
import threading
import requests
import base64
import cv2
import numpy as np
import yaml
import os
import sys
from typing import Dict, List, Any, Optional, Union

# 禁用OpenCV的GUI功能，避免在无显示环境下崩溃
os.environ['OPENCV_HEADLESS'] = '1'
try:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(1)
except:
    pass

class TestVLAVisionScheduler:
    def __init__(self):
        # 初始化基础配置（不依赖ROS）
        self.config = self.load_test_config()
        
        # 获取VLM引擎配置
        vlm_engine = self.config.get('vlm_engine', {})
        engine_type = vlm_engine.get('engine_type', 'zhipuai')
        
        # 获取通用配置
        common_config = vlm_engine.get('common', {})
        
        # 参数初始化
        self.enable_vlm = True  # 测试版本默认启用
        self.enable_yolo = False  # 测试版本暂时禁用
        self.enable_depth = False  # 测试版本暂时禁用
        self.confidence_threshold = 0.5
        
        # 根据引擎类型获取相应配置
        if engine_type == 'zhipuai':
            zhipuai_config = self.config.get('zhipuai', {})
            request_config = zhipuai_config.get('request', {})
            self.vlm_api_url = zhipuai_config.get('api_base', 'https://open.bigmodel.cn/api/paas/v4')
            self.vlm_api_key = zhipuai_config.get('api_key', '')
            self.vlm_model_name = zhipuai_config.get('model', 'GLM-4.1V-Thinking-FlashX')
            self.vlm_timeout = common_config.get('timeout', 30.0)
            self.max_tokens = request_config.get('max_tokens', 500)
            self.temperature = request_config.get('temperature', 0.1)
        elif engine_type == 'openai':
            openai_config = self.config.get('openai', {})
            request_config = openai_config.get('request', {})
            self.vlm_api_url = openai_config.get('api_base', 'https://api.openai.com/v1')
            self.vlm_api_key = openai_config.get('api_key', '')
            self.vlm_model_name = openai_config.get('model', 'gpt-4-vision-preview')
            self.vlm_timeout = common_config.get('timeout', 30.0)
            self.max_tokens = request_config.get('max_tokens', 500)
            self.temperature = request_config.get('temperature', 0.1)
        else:
            # 默认配置
            self.vlm_api_url = 'http://localhost:8000/vlm/analyze'
            self.vlm_api_key = ''
            self.vlm_model_name = 'gpt-4-vision-preview'
            self.vlm_timeout = 30.0
            self.max_tokens = 500
            self.temperature = 0.1
        
        # 模块状态
        self.module_status = {
            'VLM': self.enable_vlm,
            'YOLO': self.enable_yolo,
            'Depth': self.enable_depth
        }
        
        # 任务类型配置
        self.task_types = self.config['task_types']
        
        # 对象类别映射
        self.object_class_mapping = self.config['object_class_mapping']
        
        # 数据缓存
        self.current_image = None
        self.current_camera_info = None
        self.current_point_cloud = None
        self.current_detections = []
        self.current_object_info = {}
        
        # 摄像头相关
        self.camera = None
        self.camera_active = False
        
        print("VLA Vision Scheduler Test Version initialized")
        print(f"Config loaded with {len(self.task_types)} task types")
    
    def load_test_config(self) -> Dict[str, Any]:
        """加载测试配置文件"""
        try:
            # 尝试加载原始配置文件
            config_path = '/root/kuavo_ws/src/ros_vla_vision/config/vlm_api_params.yaml'
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                print(f"Configuration loaded from {config_path}")
                return config
            else:
                print("Config file not found, using default configuration")
                return self.get_default_config()
        except Exception as e:
            print(f"Failed to load config: {str(e)}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'basic_config': {
                'enable_vlm': True,
                'enable_yolo': True,
                'enable_depth': True,
                'confidence_threshold': 0.5
            },
            'vlm_api': {
                'url': 'http://localhost:8000/vlm/analyze',
                'api_key': '',
                'model_name': 'gpt-4-vision-preview',
                'timeout': 30.0
            },
            'task_types': {
                'vision_only': {
                    'keywords': ["什么", "哪些", "描述", "分析", "看看", "观察"],
                    'include_pose': False,
                    'include_depth': False,
                    'response_format': 'text_only'
                },
                'grasping': {
                    'keywords': ["抓取", "拿", "取", "给我", "递给我", "拿起"],
                    'include_pose': True,
                    'include_depth': True,
                    'include_grasp_points': True,
                    'response_format': 'full_data'
                },
                'location_query': {
                    'keywords': ["在哪里", "位置", "坐标", "多远", "距离"],
                    'include_pose': True,
                    'include_depth': True,
                    'include_grasp_points': False,
                    'response_format': 'pose_depth_only'
                }
            },
            'object_class_mapping': {
                '水': ['water', 'bottle', 'cup', 'glass'],
                '苹果': ['apple', 'fruit'],
                '杯子': ['cup', 'glass', 'mug'],
                '瓶子': ['bottle'],
                '书': ['book'],
                '手机': ['phone', 'cell phone'],
                '电脑': ['laptop', 'computer'],
                '椅子': ['chair'],
                '桌子': ['table'],
                '人': ['person'],
                '车': ['car', 'vehicle'],
                '水果': ['apple', 'banana', 'orange', 'fruit'],
                '食物': ['food', 'pizza', 'sandwich']
            }
        }
    
    def load_image_from_file(self, image_path: str) -> Optional[np.ndarray]:
        """从文件加载图像"""
        try:
            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                return None
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                return None
            
            print(f"Image loaded successfully: {image_path}")
            return image
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            return None
    
    def load_video_from_file(self, video_path: str) -> Optional[cv2.VideoCapture]:
        """从文件加载视频"""
        try:
            if not os.path.exists(video_path):
                print(f"Video file not found: {video_path}")
                return None
            
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                print(f"Failed to open video: {video_path}")
                return None
            
            # 获取视频信息
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            print(f"Video loaded successfully: {video_path}")
            print(f"  - FPS: {fps:.2f}")
            print(f"  - Frame count: {frame_count}")
            print(f"  - Duration: {duration:.2f} seconds")
            
            return video
        except Exception as e:
            print(f"Error loading video: {str(e)}")
            return None
    
    def extract_frames_from_video(self, video_path: str, max_frames: int = None) -> List[Dict[str, Any]]:
        """从视频中提取帧 - 性能优化版本，智能选择帧数"""
        try:
            video = self.load_video_from_file(video_path)
            if video is None:
                return []
            
            frames = []
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            
            print(f"视频信息: {frame_count}帧, {fps:.1f}FPS, {duration:.2f}秒")
            
            # 智能帧数选择策略 - 基于关键帧处理
            if max_frames is None:
                # 采用关键帧处理策略，类似官网的处理方式
                if duration <= 2.0:  # 短视频 - 提取关键帧以捕捉动作变化
                    max_frames = min(frame_count, 8)   # 最多8个关键帧
                    print(f"短视频关键帧模式：提取最多 {max_frames} 个关键帧")
                elif duration <= 5.0:  # 中等长度视频 - 增加关键帧数量
                    max_frames = min(frame_count, 10)  # 最多10个关键帧
                    print(f"中等长度视频关键帧模式：提取最多 {max_frames} 个关键帧")
                elif duration <= 10.0:  # 较长视频 - 进一步优化关键帧选择
                    max_frames = min(frame_count, 12)  # 最多12个关键帧
                    print(f"较长视频关键帧模式：提取最多 {max_frames} 个关键帧")
                else:  # 长视频 - 智能选择最具代表性的关键帧
                    max_frames = min(frame_count, 15)  # 最多15个关键帧
                    print(f"长视频关键帧模式：提取最多 {max_frames} 个关键帧")
            
            # 计算采样策略
            if frame_count <= max_frames:
                # 如果帧数少于最大帧数，提取所有帧
                step = 1
                print(f"提取所有帧：step=1, 总帧数={frame_count}")
            else:
                # 否则使用智能采样策略
                step = max(1, frame_count // max_frames)
                print(f"智能采样：step={step}, 预期提取帧数≈{frame_count//step}")
            
            frame_index = 0
            extracted_count = 0
            
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                
                if frame_index % step == 0:
                    # 优化：对视频帧进行预处理
                    processed_frame = self.preprocess_video_frame(frame, frame_index, frame_count)
                    frames.append({
                        'frame': processed_frame,
                        'index': frame_index,
                        'timestamp': frame_index / fps if fps > 0 else 0,
                        'is_keyframe': self._is_key_frame(frame, frame_index, frame_count)
                    })
                    extracted_count += 1
                    
                    # 限制最大帧数
                    if len(frames) >= max_frames:
                        break
                
                frame_index += 1
            
            video.release()
            print(f"成功提取 {len(frames)} 帧 (共{frame_count}帧)，采样率：{len(frames)/frame_count*100:.1f}%")
            
            # 显示帧提取统计
            if frames:
                keyframes = sum(1 for f in frames if f.get('is_keyframe', False))
                keyframe_percentage = (keyframes / len(frames)) * 100
                print(f"关键帧检测：{keyframes}/{len(frames)} 帧 ({keyframe_percentage:.1f}%)")
                
                # 显示关键帧索引
                keyframe_indices = [f['index'] for f in frames if f.get('is_keyframe', False)]
                print(f"关键帧位置：{keyframe_indices}")
            
            return frames
            
        except Exception as e:
            print(f"Error extracting frames from video: {str(e)}")
            return []
    
    def _is_key_frame(self, frame: np.ndarray, frame_index: int, total_frames: int) -> bool:
        """检测是否为关键帧 - 优化版本，专注于动作分析"""
        try:
            # 短视频策略：所有帧都视为关键帧以获得最佳动作分析
            if total_frames <= 8:
                return True
            
            # 边界帧总是关键帧（动作的开始和结束）
            if frame_index == 0 or frame_index == total_frames - 1:
                return True
            
            # 智能关键帧选择策略
            # 根据视频长度动态调整关键帧密度
            if total_frames <= 15:
                # 短视频：较高密度的关键帧
                key_frame_interval = max(1, total_frames // 6)
            elif total_frames <= 30:
                # 中等视频：中等密度的关键帧
                key_frame_interval = max(1, total_frames // 8)
            else:
                # 长视频：较低密度但更智能的关键帧选择
                key_frame_interval = max(1, total_frames // 10)
            
            # 均匀分布的基础关键帧
            if frame_index % key_frame_interval == 0:
                return True
            
            # 额外的关键帧：四分位点（用于捕捉动作变化）
            quarters = [total_frames // 4, total_frames // 2, 3 * total_frames // 4]
            if frame_index in quarters:
                return True
            
            return False
            
        except Exception:
            return False
    
    def preprocess_video_frame(self, frame: np.ndarray, frame_index: int = 0, total_frames: int = 1) -> np.ndarray:
        """预处理视频帧以优化性能 - 智能质量控制版本"""
        try:
            # 获取原始尺寸
            h, w = frame.shape[:2]
            original_size = f"{w}x{h}"
            
            # 优化1：智能尺寸控制
            # 根据视频长度和帧重要性动态调整尺寸
            if total_frames <= 10:  # 短视频保持高质量
                max_dimension = 1200  # 高质量
                quality_mode = "高质量"
            elif total_frames <= 30:  # 中等视频
                max_dimension = 1000  # 中等质量
                quality_mode = "中等质量"
            else:  # 长视频
                max_dimension = 800   # 标准质量
                quality_mode = "标准质量"
            
            # 关键帧使用更高质量
            if self._is_key_frame(frame, frame_index, total_frames):
                max_dimension = min(max_dimension * 1.2, 1600)  # 关键帧提高20%质量
                quality_mode += "+关键帧"
            
            if max(h, w) > max_dimension:
                scale = max_dimension / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                print(f"帧[{frame_index}]预处理[{quality_mode}]: {original_size} -> {new_w}x{new_h}")
            
            # 优化2：增强图像质量
            # 轻微的对比度和亮度调整
            if frame_index % 5 == 0:  # 每5帧显示一次状态
                print(f"帧[{frame_index}]质量控制：使用{quality_mode}模式")
            
            # 优化3：格式转换
            if frame.dtype == np.uint16:
                frame = frame.astype(np.uint8)
            
            return frame
            
        except Exception as e:
            print(f"Error preprocessing video frame: {str(e)}")
            return frame
    
    def process_video_analysis(self, video_path: str, instruction: str, use_batch: bool = True) -> Dict[str, Any]:
        """处理视频分析任务 - 支持批量API请求优化"""
        try:
            print(f"开始处理视频分析: {video_path}")
            print(f"分析指令: {instruction}")
            print(f"批量处理模式: {'启用' if use_batch else '禁用'}")
            
            # 记录开始时间
            start_time = time.time()
            
            # 提取视频帧 - 使用智能帧选择（不指定max_frames，让算法自动选择）
            frames = self.extract_frames_from_video(video_path)
            if not frames:
                return {'success': False, 'error': '无法提取视频帧'}
            
            print(f"成功提取 {len(frames)} 帧进行分析")
            
            if use_batch and len(frames) > 1:
                # 使用批量API处理
                print(f"使用批量API处理 {len(frames)} 帧...")
                
                # 提取所有帧的图像数据
                frame_images = [frame_data['frame'] for frame_data in frames]
                
                # 执行批量API调用
                batch_start = time.time()
                batch_results = self.call_vlm_api_batch(frame_images, instruction)
                batch_time = time.time() - batch_start
                
                print(f"批量API调用完成，耗时: {batch_time:.3f}秒")
                
                # 构建帧结果
                frame_results = []
                successful_results = []
                
                for i, (frame_data, vlm_result) in enumerate(zip(frames, batch_results)):
                    if vlm_result is not None:
                        # 将批量VLM结果转换为标准格式
                        vlm_formatted_result = {
                            'scene_description': vlm_result.get('description', ''),
                            'scene_type': vlm_result.get('scene_type', 'unknown'),
                            'objects': vlm_result.get('objects', []),
                            'scene_tags': vlm_result.get('tags', []),
                            'confidence': vlm_result.get('confidence', 0.0),
                            'processing_time': vlm_result.get('processing_time', 0.0),
                            'api_response': vlm_result
                        }
                        
                        result = {
                            'success': True,
                            'task_type': self.analyze_task_type(instruction),
                            'target_object': self.extract_target_object(instruction),
                            'instruction': instruction,
                            'vlm_result': vlm_formatted_result,
                            'timestamp': time.time(),
                            'batch_optimization': {
                                'batch_mode': True,
                                'batch_processing_time': batch_time,
                                'estimated_single_frame_time': batch_time / len(frames),
                                'time_saved': (batch_time * (len(frames) - 1)) - batch_time,
                                'efficiency_gain': f"{((len(frames) - 1) / len(frames) * 100):.1f}%"
                            }
                        }
                        
                        successful_results.append({
                            'frame_index': frame_data['index'],
                            'timestamp': frame_data['timestamp'],
                            'result': result
                        })
                    else:
                        # 处理失败的帧
                        result = {
                            'success': False,
                            'error': '批量API调用失败',
                            'instruction': instruction
                        }
                    
                    frame_results.append({
                        'frame_index': frame_data['index'],
                        'timestamp': frame_data['timestamp'],
                        'result': result
                    })
                    
                    # 释放帧内存
                    del frame_data['frame']
                
                print(f"批量处理成功: {len(successful_results)}/{len(frames)} 帧")
                
                # 显示批量处理性能统计
                if successful_results:
                    avg_single_frame_time = batch_time / len(frames)
                    estimated_single_api_time = avg_single_frame_time * len(frames)
                    time_saved = estimated_single_api_time - batch_time
                    efficiency_gain = (time_saved / estimated_single_api_time) * 100
                    
                    print(f"批量处理性能统计:")
                    print(f"  - 总处理时间: {batch_time:.3f}秒")
                    print(f"  - 平均每帧时间: {avg_single_frame_time:.3f}秒")
                    print(f"  - 预估单帧处理总时间: {estimated_single_api_time:.3f}秒")
                    print(f"  - 节省时间: {time_saved:.3f}秒")
                    print(f"  - 效率提升: {efficiency_gain:.1f}%")
                
            else:
                # 使用传统的单帧处理方式
                print("使用传统单帧处理模式...")
                
                frame_results = []
                for i, frame_data in enumerate(frames):
                    print(f"正在分析第 {i+1}/{len(frames)} 帧...")
                    
                    result = self.process_vision_task(instruction, frame_data['frame'])
                    frame_results.append({
                        'frame_index': frame_data['index'],
                        'timestamp': frame_data['timestamp'],
                        'result': result
                    })
                    
                    # 释放帧内存
                    del frame_data['frame']
                
                # 综合分析结果
                successful_results = [r for r in frame_results if r['result']['success']]
            
            if not successful_results:
                return {'success': False, 'error': '所有帧分析都失败了'}
            
            # 构建综合结果
            synthesis = self.synthesize_video_analysis_results(successful_results, instruction)
            
            # 记录结束时间
            end_time = time.time()
            total_processing_time = end_time - start_time
            
            return {
                'success': True,
                'video_path': video_path,
                'instruction': instruction,
                'frame_count': len(frames),
                'analyzed_frames': len(successful_results),
                'synthesis': synthesis,
                'frame_results': frame_results,
                'start_time': start_time,
                'end_time': end_time,
                'total_processing_time': total_processing_time,
                'processing_mode': 'batch' if use_batch and len(frames) > 1 else 'single_frame',
                'performance_stats': {
                    'frames_per_second': len(frames) / total_processing_time if total_processing_time > 0 else 0,
                    'average_time_per_frame': total_processing_time / len(frames) if len(frames) > 0 else 0
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def synthesize_video_analysis_results(self, frame_results: List[Dict], instruction: str) -> Dict[str, Any]:
        """综合视频分析结果"""
        try:
            # 提取所有成功的描述
            descriptions = []
            confidences = []
            
            for frame_result in frame_results:
                result = frame_result['result']
                if 'vlm_result' in result:
                    vlm_result = result['vlm_result']
                    descriptions.append(vlm_result.get('scene_description', ''))
                    confidences.append(vlm_result.get('confidence', 0))
            
            # 计算平均置信度
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # 分析指令类型以确定综合策略
            task_type = self.analyze_task_type(instruction)
            
            if task_type == 'vision_only':
                # 视觉理解任务：智能综合描述
                if len(descriptions) == 1:
                    synthesis = descriptions[0]
                else:
                    # 收集所有有效描述
                    valid_descriptions = [desc.strip() for desc in descriptions if desc.strip()]
                    
                    if not valid_descriptions:
                        synthesis = "无法从视频帧中提取有效信息"
                    else:
                        # 使用VLM进行智能总结
                        synthesis = self.create_intelligent_video_summary(valid_descriptions, instruction)
                
            elif task_type == 'grasping':
                # 抓取任务：检测目标物体
                target_object = self.extract_target_object(instruction)
                if target_object:
                    synthesis = f"在视频的{len(descriptions)}帧分析中，持续检测到目标物体：{target_object}"
                else:
                    synthesis = f"视频分析完成，分析了{len(descriptions)}帧内容"
                
            elif task_type == 'location_query':
                # 位置查询：分析物体位置变化
                target_object = self.extract_target_object(instruction)
                if target_object:
                    synthesis = f"目标物体'{target_object}'在视频中的位置分析（基于{len(descriptions)}帧）"
                else:
                    synthesis = f"视频位置分析完成，分析了{len(descriptions)}帧内容"
                
            else:
                # 默认综合
                synthesis = f"视频分析完成，处理了{len(descriptions)}帧数据"
            
            return {
                'description': synthesis,
                'confidence': avg_confidence,
                'task_type': task_type,
                'frame_count': len(descriptions),
                'raw_descriptions': descriptions  # 保留原始描述用于调试
            }
            
        except Exception as e:
            return {
                'description': f'视频分析综合失败: {str(e)}',
                'confidence': 0,
                'task_type': 'unknown',
                'frame_count': 0
            }
    
    
    def create_intelligent_video_summary(self, descriptions: List[str], instruction: str) -> str:
        """创建智能视频总结"""
        try:
            # 构建总结提示
            summary_prompt = f"""请根据以下多个视频帧的分析内容，生成一个连贯、具体的视频内容总结。

用户指令：{instruction}

各帧分析内容：
"""
            for i, desc in enumerate(descriptions):
                summary_prompt += f"帧{i+1}: {desc}\n"
            
            summary_prompt += """
要求：
1. 生成一个具体的、有意义的视频内容描述，不要说"动态场景"或"不同帧之间存在变化"这样的笼统描述
2. 如果视频中有人物，描述人物的动作和活动，必须基于实际观察到的内容
3. 如果视频中有物体，描述物体的状态和变化，客观准确
4. 如果视频是特定场景（如音乐会、会议、运动等），明确指出场景类型
5. 对于手部动作，必须根据实际轨迹描述，不要预设任何特定形状
6. 总结长度在2-3句话，要具体、生动、客观准确

请直接给出总结，不要添加其他解释："""

            # 调用VLM进行智能总结
            temp_result = self.execute_vlm_task(summary_prompt, np.zeros((10, 10, 3), dtype=np.uint8))
            
            if 'error' not in temp_result:
                summary = temp_result.get('scene_description', '').strip()
                if summary and len(summary) > 10:
                    return summary
            
            # 如果VLM总结失败，使用简单的关键词匹配总结
            return self.create_fallback_summary(descriptions)
            
        except Exception as e:
            print(f"智能总结生成失败: {str(e)}")
            return self.create_fallback_summary(descriptions)
    
    def create_fallback_summary(self, descriptions: List[str]) -> str:
        """创建备用总结"""
        try:
            # 合并所有描述
            all_text = ' '.join(descriptions).lower()
            
            # 常见场景关键词检测
            scenes = {
                '音乐会': ['音乐', '演奏', '乐队', '舞台', '表演', '歌手', '乐器'],
                '会议': ['会议', '演讲', '演示', '听众', '发言', '讨论'],
                '运动': ['运动', '比赛', '跑步', '跳跃', '球场', '运动员'],
                '烹饪': ['烹饪', '厨房', '食物', '厨师', '锅', '切菜'],
                '教学': ['教学', '老师', '学生', '黑板', '讲解', '学习'],
                '办公': ['办公', '电脑', '工作', '文档', '会议', '同事'],
                '户外': ['户外', '公园', '街道', '建筑', '天空', '树木']
            }
            
            # 检测场景
            detected_scene = '未知场景'
            for scene, keywords in scenes.items():
                if any(keyword in all_text for keyword in keywords):
                    detected_scene = scene
                    break
            
            # 检测人物动作
            actions = []
            if '人' in all_text or '人物' in all_text:
                if any(word in all_text for word in ['走', '跑', '移动']):
                    actions.append('移动')
                if any(word in all_text for word in ['坐', '站', '躺']):
                    actions.append('姿势变化')
                if any(word in all_text for word in ['说话', '演讲', '交流']):
                    actions.append('交流')
                if any(word in all_text for word in ['工作', '操作', '使用']):
                    actions.append('操作')
            
            # 生成总结
            if actions:
                action_str = '、'.join(actions)
                summary = f"视频显示了一个{detected_scene}，人物正在进行{action_str}等活动。"
            else:
                summary = f"视频显示了一个{detected_scene}，画面中有各种元素和活动。"
            
            return summary
            
        except Exception as e:
            print(f"备用总结生成失败: {str(e)}")
            return "视频内容分析失败，无法生成有效总结。"
    
    def display_video_result(self, result: Dict[str, Any]):
        """显示视频分析结果"""
        if result['success']:
            # 获取视频基本信息
            video_path = result['video_path']
            try:
                video = cv2.VideoCapture(video_path)
                if video.isOpened():
                    fps = video.get(cv2.CAP_PROP_FPS)
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0
                    video_info = f"{duration:.1f}秒, {frame_count}帧, {fps:.1f}FPS"
                    video.release()
                else:
                    video_info = "无法获取视频信息"
            except:
                video_info = "视频信息获取失败"
            
            # 计算分析耗时
            start_time = result.get('start_time', 0)
            end_time = result.get('end_time', 0)
            if start_time > 0 and end_time > 0:
                analysis_time_ms = int((end_time - start_time) * 1000)
                analyzed_frames = result.get('analyzed_frames', 0)
                time_info = f"视频分析耗时[分析了 {analyzed_frames} 帧，响应时间: {analysis_time_ms}ms]"
            else:
                time_info = "分析耗时信息不可用"
            
            synthesis = result.get('synthesis', {})
            
            print(f"\n{'='*60}")
            print(f"视频分析结果")
            print(f"{'='*60}")
            print(f"视频信息: {video_info}")
            print(f"{time_info}")
            print(f"分析指令: {result['instruction']}")
            
            if synthesis:
                print(f"\n总结:")
                print(f"{synthesis.get('description', '无结果')}")
                print(f"置信度: {synthesis.get('confidence', 0):.1f}%")
            
            print(f"{'='*60}")
        else:
            print(f"\n视频分析失败: {result.get('error', '未知错误')}")
    
    def start_realtime_video_analysis(self):
        """启动实时视频分析模式"""
        print("\n--- 实时视频分析模式 ---")
        
        # 使用固定视频路径
        video_path = "/root/test_video.mp4"  # 固定视频路径
        print(f"使用固定视频路径: {video_path}")
        
        # 检查视频文件是否存在
        if not os.path.exists(video_path):
            print(f"视频文件不存在: {video_path}")
            print("请将视频文件放置在指定路径")
            return
        
        try:
            # 加载视频
            video = self.load_video_from_file(video_path)
            if video is None:
                return
            
            print("\n视频已加载，可以开始分析")
            print("输入指令时将自动提取视频帧进行分析")
            print("输入 'quit' 或 'exit' 退出此模式")
            
            while True:
                # 获取用户指令
                instruction = input("\n请输入视频分析指令: ").strip()
                
                # 检查是否要退出
                if instruction.lower() in ['quit', 'exit', '退出']:
                    print("退出实时视频分析模式")
                    break
                
                if not instruction:
                    print("指令不能为空")
                    continue
                
                print("正在处理视频分析...")
                
                # 处理视频分析
                start_time = time.time()
                result = self.process_video_analysis(video_path, instruction)
                processing_time = time.time() - start_time
                
                # 显示结果
                self.display_video_result(result)
                
                # 显示处理耗时
                if result['success']:
                    print(f"\n视频处理耗时: {processing_time:.3f}秒")
                
        except Exception as e:
            print(f"实时视频分析模式发生错误: {str(e)}")
        finally:
            if 'video' in locals():
                video.release()
    
    def setup_camera(self, camera_id: int = 0) -> bool:
        """设置摄像头"""
        try:
            self.camera = cv2.VideoCapture(camera_id)
            if not self.camera.isOpened():
                print(f"Failed to open camera {camera_id}")
                return False
            
            self.camera_active = True
            print(f"Camera {camera_id} initialized successfully")
            return True
        except Exception as e:
            print(f"Error setting up camera: {str(e)}")
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """捕获一帧图像"""
        try:
            if not self.camera_active or self.camera is None:
                print("Camera not active")
                return None
            
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to capture frame")
                return None
            
            print("Frame captured successfully")
            return frame
        except Exception as e:
            print(f"Error capturing frame: {str(e)}")
            return None
    
    def release_camera(self):
        """释放摄像头"""
        try:
            if self.camera is not None:
                self.camera.release()
                self.camera_active = False
                print("Camera released")
        except Exception as e:
            print(f"Error releasing camera: {str(e)}")
    
    def call_vlm_api(self, image: np.ndarray, task: str) -> Optional[Dict[str, Any]]:
        """调用VLM API分析图像 - 性能优化版本"""
        return self.call_vlm_api_batch([image], task)[0]
    
    def call_vlm_api_batch(self, images: List[np.ndarray], task: str) -> List[Optional[Dict[str, Any]]]:
        """批量调用VLM API分析多帧图像 - 性能优化版本"""
        try:
            start_time = time.time()
            
            if not images:
                return []
            
            # 批量预处理所有图像
            preprocessed_images = []
            total_original_size = 0
            total_compressed_size = 0
            
            print(f"开始批量预处理 {len(images)} 帧图像...")
            preprocess_start = time.time()
            
            # 根据关键帧数量动态调整处理策略
            if len(images) <= 5:
                # 少量关键帧：高质量处理
                max_dimension = 1200
                jpeg_quality = 95
                quality_mode = "高质量"
            elif len(images) <= 10:
                # 中等数量关键帧：平衡处理
                max_dimension = 1000
                jpeg_quality = 90
                quality_mode = "平衡质量"
            else:
                # 较多关键帧：性能优先
                max_dimension = 800
                jpeg_quality = 85
                quality_mode = "性能优先"
            
            print(f"使用{quality_mode}模式：最大尺寸={max_dimension}px, JPEG质量={jpeg_quality}")
            
            for i, image in enumerate(images):
                h, w = image.shape[:2]
                total_original_size += h * w * 3  # 估算原始大小
                
                # 智能尺寸调整
                if max(h, w) > max_dimension:
                    scale = max_dimension / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # 智能压缩质量控制
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                _, buffer = cv2.imencode('.jpg', image, encode_params)
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                total_compressed_size += len(image_base64)
                
                preprocessed_images.append({
                    'base64': image_base64,
                    'original_size': f"{w}x{h}",
                    'index': i,
                    'quality_mode': quality_mode
                })
                
                if (i + 1) % 5 == 0 or (i + 1) == len(images):  # 每5帧或最后显示进度
                    print(f"已预处理 {i + 1}/{len(images)} 帧")
            
            preprocess_time = time.time() - preprocess_start
            print(f"批量预处理完成，耗时: {preprocess_time:.3f}秒")
            print(f"压缩率: {total_original_size/1024/1024:.1f}MB -> {total_compressed_size/1024:.1f}KB")
            print(f"平均压缩比: {total_original_size/len(image_base64):.1f}:1")
            
            # 优化策略3: 连接池和会话复用
            if not hasattr(self, 'session'):
                self.session = requests.Session()
                from requests.adapters import HTTPAdapter
                adapter = HTTPAdapter(
                    pool_connections=10,
                    pool_maxsize=10,
                    max_retries=3
                )
                self.session.mount('http://', adapter)
                self.session.mount('https://', adapter)
            
            # 准备API请求
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'VLA-Vision-Scheduler/1.0'
            }
            
            # 根据引擎类型准备不同的payload和headers
            vlm_engine = self.config.get('vlm_engine', {})
            engine_type = vlm_engine.get('engine_type', 'zhipuai')
            
            if engine_type in ['zhipuai', 'openai']:
                # 智谱AI和OpenAI格式 - 构建批量请求
                if self.vlm_api_key:
                    headers['Authorization'] = f'Bearer {self.vlm_api_key}'
                
                # 构建批量请求的消息内容
                messages = []
                for img_data in preprocessed_images:
                    messages.append({
                        'role': 'user',
                        'content': [
                            {
                                'type': 'text',
                                'text': f"{task} (帧{img_data['index'] + 1})"
                            },
                            {
                                'type': 'image_url',
                                'image_url': {
                                    'url': f'data:image/jpeg;base64,{img_data["base64"]}'
                                }
                            }
                        ]
                    })
                
                # 改进的批量处理策略：使用更有效的视频分析指令
                instruction = f"""请仔细分析这个视频的{len(images)}个关键帧。这是一个动作分析任务，请：

1. 识别视频中的人物、手部动作和手势
2. 分析手部运动轨迹和动作
3. 客观描述手部实际画出的形状或形成的轨迹
4. 注意手指的精确运动路径和方向
5. 提供具体、准确的动作描述

请特别注意：
- 这是基于智能关键帧选择的分析，每个关键帧都代表动作的重要阶段
- 必须根据实际视频内容分析，不要预设或假设任何特定形状
- 如果手在画形状，请根据实际轨迹准确描述是什么形状（可能是圆形、方形、三角形、直线、曲线或其他形状）
- 如果没有画出明确的形状，请如实描述手的运动状态
- 描述动作的起始、过程和结束
- 提供具体的细节，不要笼统描述
- 综合所有关键帧信息，给出客观、准确的动作描述

分析任务：{task}"""
                
                payload = {
                    'model': self.vlm_model_name,
                    'messages': messages,
                    'max_tokens': self.max_tokens * len(images),  # 增加token限制
                    'temperature': self.temperature,
                    'instruction': instruction  # 添加instruction字段
                }
                api_url = f"{self.vlm_api_url}/chat/completions"
                
            else:
                # 默认格式（兼容旧版本）- 构建批量请求
                if self.vlm_api_key:
                    headers['Authorization'] = f'Bearer {self.vlm_api_key}'
                
                # 添加instruction字段，说明这些是同一个视频中连续采样的帧
                instruction = "这些是同一个视频中连续采样的 10 帧，请综合分析这些帧"
                
                payload = {
                    'model': self.vlm_model_name,
                    'images': [img_data['base64'] for img_data in preprocessed_images],
                    'tasks': [f"{task} (帧{img_data['index'] + 1})" for img_data in preprocessed_images],
                    'max_tokens': self.max_tokens * len(images),
                    'temperature': self.temperature,
                    'batch_mode': True,  # 标识为批量模式
                    'instruction': instruction  # 添加instruction字段
                }
                api_url = self.vlm_api_url
            
            # 发送批量API请求
            print(f"发送批量API请求到: {api_url}")
            print(f"批量处理 {len(images)} 帧，预计节省网络开销: {len(images)-1} 次请求")
            request_start = time.time()
            
            # 动态调整超时时间
            dynamic_timeout = max(self.vlm_timeout * 1.5, self.vlm_timeout + len(images) * 2)
            print(f"使用动态超时: {dynamic_timeout:.1f}秒 (基础: {self.vlm_timeout}s + 批量额外: {len(images)*2}s)")
            
            try:
                response = self.session.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=dynamic_timeout
                )
                request_time = time.time() - request_start
                print(f"批量API请求耗时: {request_time:.3f}秒, 状态码: {response.status_code}")
                
                # 性能统计
                if response.status_code == 200:
                    avg_time_per_frame = request_time / len(images)
                    print(f"平均每帧处理时间: {avg_time_per_frame:.3f}秒")
                    if avg_time_per_frame < 1.0:
                        print("✓ 批量处理性能优秀")
                    elif avg_time_per_frame < 2.0:
                        print("✓ 批量处理性能良好")
                    else:
                        print("⚠ 批量处理性能较慢，建议优化")
                
            except requests.exceptions.Timeout:
                print(f"批量API请求超时 (>{dynamic_timeout:.1f}秒)")
                # 超时后智能降级策略
                if len(images) > 5:
                    print("检测到大量图像，尝试分批处理...")
                    return self._smart_batch_fallback(images, task)
                else:
                    print("降级为单帧处理模式...")
                    return self._fallback_to_single_frame_processing(images, task)
                
            except requests.exceptions.RequestException as e:
                print(f"批量API请求失败: {str(e)}")
                # 请求失败后智能降级
                if len(images) > 5:
                    print("检测到大量图像，尝试分批处理...")
                    return self._smart_batch_fallback(images, task)
                else:
                    print("降级为单帧处理模式...")
                    return self._fallback_to_single_frame_processing(images, task)
            
            if response.status_code == 200:
                parse_start = time.time()
                api_result = response.json()
                parse_time = time.time() - parse_start
                print(f"批量响应解析耗时: {parse_time:.3f}秒")
                
                # 解析批量API响应
                results = self._parse_batch_api_response(api_result, engine_type, preprocessed_images, start_time, 
                                                       preprocess_time, request_time, parse_time)
                
                print(f"批量处理完成，成功处理 {len([r for r in results if r is not None])}/{len(images)} 帧")
                return results
            else:
                print(f"批量API请求失败，状态码: {response.status_code}")
                print(f"响应内容: {response.text}")
                # 批量请求失败后降级为单帧处理
                print("降级为单帧处理模式...")
                return self._fallback_to_single_frame_processing(images, task)
                
        except Exception as e:
            print(f"批量VLM API调用发生异常: {str(e)}")
            # 异常情况下降级为单帧处理
            print("降级为单帧处理模式...")
            return self._fallback_to_single_frame_processing(images, task)
    
    def _smart_batch_fallback(self, images: List[np.ndarray], task: str) -> List[Optional[Dict[str, Any]]]:
        """智能分批处理降级方案"""
        try:
            total_images = len(images)
            print(f"开始智能分批处理，共 {total_images} 帧...")
            
            # 根据图像数量确定分批策略
            if total_images <= 10:
                batch_size = 5
            elif total_images <= 20:
                batch_size = 8
            else:
                batch_size = 10
            
            print(f"分批策略：每批 {batch_size} 帧")
            
            results = [None] * total_images
            
            # 分批处理
            for batch_start in range(0, total_images, batch_size):
                batch_end = min(batch_start + batch_size, total_images)
                batch_images = images[batch_start:batch_end]
                
                print(f"处理批次 {batch_start//batch_size + 1}: 帧 {batch_start+1}-{batch_end} ({len(batch_images)} 帧)")
                
                # 尝试批量处理当前批次
                batch_start_time = time.time()
                
                try:
                    # 使用较小的超时时间进行分批处理
                    original_timeout = self.vlm_timeout
                    self.vlm_timeout = min(original_timeout, 20.0)  # 分批使用较短超时
                    
                    batch_results = self.call_vlm_api_batch(batch_images, task)
                    
                    # 恢复原始超时时间
                    self.vlm_timeout = original_timeout
                    
                    batch_time = time.time() - batch_start_time
                    print(f"批次 {batch_start//batch_size + 1} 处理完成，耗时: {batch_time:.3f}秒")
                    
                    # 将结果存入对应位置
                    for i, result in enumerate(batch_results):
                        results[batch_start + i] = result
                        
                except Exception as e:
                    print(f"批次 {batch_start//batch_size + 1} 批量处理失败: {str(e)}")
                    print(f"批次 {batch_start//batch_size + 1} 降级为单帧处理...")
                    
                    # 恢复原始超时时间
                    self.vlm_timeout = original_timeout
                    
                    # 当前批次降级为单帧处理
                    for i, image in enumerate(batch_images):
                        print(f"处理批次 {batch_start//batch_size + 1} 的第 {i+1}/{len(batch_images)} 帧...")
                        result = self._call_vlm_api_single_frame(image, task)
                        results[batch_start + i] = result
                        
                        # 添加小延迟避免API频率限制
                        if i < len(batch_images) - 1:
                            time.sleep(0.2)
            
            # 统计最终结果
            successful_count = sum(1 for r in results if r is not None)
            print(f"智能分批处理完成：{successful_count}/{total_images} 帧成功")
            
            return results
            
        except Exception as e:
            print(f"智能分批处理失败: {str(e)}")
            print("最终降级为单帧处理...")
            return self._fallback_to_single_frame_processing(images, task)
    
    def _fallback_to_single_frame_processing(self, images: List[np.ndarray], task: str) -> List[Optional[Dict[str, Any]]]:
        """降级为单帧处理的备用方案"""
        results = []
        print(f"开始单帧处理 {len(images)} 帧...")
        
        for i, image in enumerate(images):
            print(f"处理第 {i+1}/{len(images)} 帧...")
            result = self._call_vlm_api_single_frame(image, task)
            results.append(result)
            
            # 添加小延迟避免API频率限制
            if i < len(images) - 1:
                time.sleep(0.1)
        
        return results
    
    def _call_vlm_api_single_frame(self, image: np.ndarray, task: str) -> Optional[Dict[str, Any]]:
        """单帧API调用的内部实现"""
        try:
            start_time = time.time()
            
            # 图像预处理
            max_dimension = 800
            h, w = image.shape[:2]
            if max(h, w) > max_dimension:
                scale = max_dimension / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]  # 提高质量以保证分析准确性
            _, buffer = cv2.imencode('.jpg', image, encode_params)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # 准备API请求
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'VLA-Vision-Scheduler/1.0'
            }
            
            vlm_engine = self.config.get('vlm_engine', {})
            engine_type = vlm_engine.get('engine_type', 'zhipuai')
            
            if engine_type == 'zhipuai':
                if self.vlm_api_key:
                    headers['Authorization'] = f'Bearer {self.vlm_api_key}'
                
                payload = {
                    'model': self.vlm_model_name,
                    'messages': [
                        {
                            'role': 'user',
                            'content': [
                                {'type': 'text', 'text': task},
                                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_base64}'}}
                            ]
                        }
                    ],
                    'max_tokens': self.max_tokens,
                    'temperature': self.temperature
                }
                api_url = f"{self.vlm_api_url}/chat/completions"
                
            elif engine_type == 'openai':
                if self.vlm_api_key:
                    headers['Authorization'] = f'Bearer {self.vlm_api_key}'
                
                payload = {
                    'model': self.vlm_model_name,
                    'messages': [
                        {
                            'role': 'user',
                            'content': [
                                {'type': 'text', 'text': task},
                                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_base64}'}}
                            ]
                        }
                    ],
                    'max_tokens': self.max_tokens,
                    'temperature': self.temperature
                }
                api_url = f"{self.vlm_api_url}/chat/completions"
                
            else:
                if self.vlm_api_key:
                    headers['Authorization'] = f'Bearer {self.vlm_api_key}'
                
                payload = {
                    'model': self.vlm_model_name,
                    'image': image_base64,
                    'task': task,
                    'max_tokens': self.max_tokens,
                    'temperature': self.temperature
                }
                api_url = self.vlm_api_url
            
            # 发送请求
            try:
                response = self.session.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.vlm_timeout
                )
                
                if response.status_code == 200:
                    api_result = response.json()
                    
                    if engine_type in ['zhipuai', 'openai']:
                        choices = api_result.get('choices', [])
                        if choices:
                            message = choices[0].get('message', {})
                            content = message.get('content', '')
                            answer_content = self.extract_answer_content(content)
                            
                            # 尝试从API响应中提取置信度
                            confidence = 0.8
                            if 'usage' in api_result:
                                # 某些API在usage中包含质量信息
                                usage = api_result['usage']
                                if 'completion_tokens' in usage and 'total_tokens' in usage:
                                    # 基于token使用情况估算置信度
                                    token_ratio = usage['completion_tokens'] / max(usage['total_tokens'], 1)
                                    confidence = min(0.95, max(0.1, token_ratio))
                            
                            return {
                                'description': answer_content,
                                'scene_type': 'unknown',
                                'objects': [],
                                'tags': [],
                                'confidence': confidence,
                                'processing_time': time.time() - start_time,
                                'raw_response': api_result
                            }
                    else:
                        description = api_result.get('description', '')
                        answer_content = self.extract_answer_content(description)
                        
                        return {
                            'description': answer_content,
                            'scene_type': api_result.get('scene_type', 'unknown'),
                            'objects': api_result.get('objects', []),
                            'tags': api_result.get('tags', []),
                            'confidence': api_result.get('confidence', 0.0),
                            'processing_time': time.time() - start_time,
                            'raw_response': api_result
                        }
                
            except Exception as e:
                print(f"单帧API调用失败: {str(e)}")
                return None
                
        except Exception as e:
            print(f"单帧处理异常: {str(e)}")
            return None
        
        return None
    
    def _parse_batch_api_response(self, api_result: Dict, engine_type: str, preprocessed_images: List[Dict], 
                                 start_time: float, preprocess_time: float, request_time: float, parse_time: float) -> List[Optional[Dict[str, Any]]]:
        """解析批量API响应"""
        results = []
        
        try:
            if engine_type in ['zhipuai', 'openai']:
                # GPT格式响应 - 处理批量结果
                choices = api_result.get('choices', [])
                
                if len(choices) == len(preprocessed_images):
                    # 理想情况：每个图像对应一个选择
                    for i, choice in enumerate(choices):
                        message = choice.get('message', {})
                        content = message.get('content', '')
                        answer_content = self.extract_answer_content(content)
                        
                        img_data = preprocessed_images[i]
                        results.append({
                            'description': answer_content,
                            'scene_type': 'unknown',
                            'objects': [],
                            'tags': [],
                            'confidence': 0.8,
                            'processing_time': time.time() - start_time,
                            'raw_response': api_result,
                            'batch_stats': {
                                'total_images': len(preprocessed_images),
                                'image_index': i,
                                'preprocess_time': preprocess_time,
                                'api_request_time': request_time,
                                'response_parse_time': parse_time,
                                'original_image_size': img_data['original_size'],
                                'batch_compression_ratio': f"{len(preprocessed_images)}:1"
                            }
                        })
                elif len(choices) == 1:
                    # 单个响应包含所有图像的分析结果
                    message = choices[0].get('message', {})
                    content = message.get('content', '')
                    
                    # 尝试解析批量结果
                    batch_results = self._parse_batch_content(content, len(preprocessed_images))
                    
                    for i, img_data in enumerate(preprocessed_images):
                        if i < len(batch_results):
                            answer_content = batch_results[i]
                        else:
                            answer_content = f"帧{i+1}分析结果不可用"
                        
                        results.append({
                            'description': answer_content,
                            'scene_type': 'unknown',
                            'objects': [],
                            'tags': [],
                            'confidence': 0.8,
                            'processing_time': time.time() - start_time,
                            'raw_response': api_result,
                            'batch_stats': {
                                'total_images': len(preprocessed_images),
                                'image_index': i,
                                'preprocess_time': preprocess_time,
                                'api_request_time': request_time,
                                'response_parse_time': parse_time,
                                'original_image_size': img_data['original_size'],
                                'batch_compression_ratio': f"{len(preprocessed_images)}:1"
                            }
                        })
                else:
                    # 不匹配的情况，为每个图像创建默认结果
                    for i, img_data in enumerate(preprocessed_images):
                        results.append({
                            'description': f"帧{i+1}: 批量响应格式不匹配",
                            'scene_type': 'unknown',
                            'objects': [],
                            'tags': [],
                            'confidence': 0.0,
                            'processing_time': time.time() - start_time,
                            'raw_response': api_result,
                            'batch_stats': {
                                'total_images': len(preprocessed_images),
                                'image_index': i,
                                'preprocess_time': preprocess_time,
                                'api_request_time': request_time,
                                'response_parse_time': parse_time,
                                'original_image_size': img_data['original_size'],
                                'batch_compression_ratio': f"{len(preprocessed_images)}:1"
                            }
                        })
            else:
                # 默认格式响应 - 处理批量结果
                if 'batch_results' in api_result:
                    # 支持批量响应的格式
                    batch_results = api_result.get('batch_results', [])
                    
                    for i, img_data in enumerate(preprocessed_images):
                        if i < len(batch_results):
                            batch_result = batch_results[i]
                            description = batch_result.get('description', '')
                            answer_content = self.extract_answer_content(description)
                            
                            results.append({
                                'description': answer_content,
                                'scene_type': batch_result.get('scene_type', 'unknown'),
                                'objects': batch_result.get('objects', []),
                                'tags': batch_result.get('tags', []),
                                'confidence': batch_result.get('confidence', 0.0),
                                'processing_time': time.time() - start_time,
                                'raw_response': api_result,
                                'batch_stats': {
                                    'total_images': len(preprocessed_images),
                                    'image_index': i,
                                    'preprocess_time': preprocess_time,
                                    'api_request_time': request_time,
                                    'response_parse_time': parse_time,
                                    'original_image_size': img_data['original_size'],
                                    'batch_compression_ratio': f"{len(preprocessed_images)}:1"
                                }
                            })
                        else:
                            results.append(None)
                else:
                    # 不支持批量响应，降级处理
                    for i, img_data in enumerate(preprocessed_images):
                        results.append({
                            'description': f"帧{i+1}: API不支持批量处理",
                            'scene_type': 'unknown',
                            'objects': [],
                            'tags': [],
                            'confidence': 0.0,
                            'processing_time': time.time() - start_time,
                            'raw_response': api_result,
                            'batch_stats': {
                                'total_images': len(preprocessed_images),
                                'image_index': i,
                                'preprocess_time': preprocess_time,
                                'api_request_time': request_time,
                                'response_parse_time': parse_time,
                                'original_image_size': img_data['original_size'],
                                'batch_compression_ratio': f"{len(preprocessed_images)}:1"
                            }
                        })
        
        except Exception as e:
            print(f"批量响应解析异常: {str(e)}")
            # 解析失败，返回None列表
            results = [None] * len(preprocessed_images)
        
        return results
    
    def _parse_batch_content(self, content: str, expected_count: int) -> List[str]:
        """解析批量内容，将单个响应拆分为多个结果"""
        try:
            # 尝试不同的解析策略
            
            # 策略1: 按帧标识符分割
            import re
            frame_pattern = r'帧(\d+)[：:]\s*(.*?)(?=\s*帧\d+[：:]|$)'
            matches = re.findall(frame_pattern, content, re.DOTALL)
            
            if matches:
                results = [''] * expected_count
                for frame_num, description in matches:
                    index = int(frame_num) - 1
                    if 0 <= index < expected_count:
                        results[index] = description.strip()
                return results
            
            # 策略2: 按序号分割
            number_pattern = r'(\d+)[\.、]\s*(.*?)(?=\s*\d+[\.、]|$)'
            matches = re.findall(number_pattern, content, re.DOTALL)
            
            if matches:
                results = [''] * expected_count
                for num, description in matches:
                    index = int(num) - 1
                    if 0 <= index < expected_count:
                        results[index] = description.strip()
                return results
            
            # 策略3: 按行分割
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            if len(lines) >= expected_count:
                return lines[:expected_count]
            
            # 策略4: 按段落分割
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            if len(paragraphs) >= expected_count:
                return paragraphs[:expected_count]
            
            # 策略5: 平均分割内容
            if len(content) > expected_count * 10:  # 内容足够长
                chunk_size = len(content) // expected_count
                results = []
                for i in range(expected_count):
                    start = i * chunk_size
                    end = (i + 1) * chunk_size if i < expected_count - 1 else len(content)
                    results.append(content[start:end].strip())
                return results
            
            # 策略6: 无法分割，返回重复内容
            return [content] * expected_count
            
        except Exception as e:
            print(f"批量内容解析失败: {str(e)}")
            return [f"解析失败: {str(e)}"] * expected_count
    
    def analyze_task_type(self, instruction: str) -> str:
        """分析指令类型"""
        instruction_lower = instruction.lower()
        
        for task_type, config in self.task_types.items():
            for keyword in config['keywords']:
                if keyword in instruction_lower:
                    return task_type
        
        # 默认返回视觉理解类型
        return 'vision_only'
    
    def extract_target_object(self, instruction: str) -> Optional[str]:
        """从指令中提取目标物体"""
        for object_name, class_list in self.object_class_mapping.items():
            if object_name in instruction:
                return object_name
        return None
    
    def extract_answer_content(self, content: str) -> str:
        """从API响应中提取<answer>标签内的内容"""
        try:
            # 查找<answer>标签
            start_tag = "<answer>"
            end_tag = "</answer>"
            
            start_index = content.find(start_tag)
            if start_index == -1:
                # 如果没有找到<answer>标签，返回原始内容
                return content
            
            end_index = content.find(end_tag, start_index)
            if end_index == -1:
                # 如果没有找到结束标签，返回开始标签之后的内容
                return content[start_index + len(start_tag):]
            
            # 提取标签内的内容
            answer_content = content[start_index + len(start_tag):end_index]
            
            # 去除首尾空白字符
            answer_content = answer_content.strip()
            
            return answer_content if answer_content else content
            
        except Exception:
            # 如果解析失败，返回原始内容
            return content
    
    def execute_vlm_task(self, task: str, image: np.ndarray) -> Dict[str, Any]:
        """执行VLM任务 - 调用VLM API分析图像"""
        try:
            if image is None:
                return {'error': 'No image provided'}
            
            # 调用VLM API
            vlm_result = self.call_vlm_api(image, task)
            
            if vlm_result:
                return {
                    'scene_description': vlm_result.get('description', ''),
                    'scene_type': vlm_result.get('scene_type', 'unknown'),
                    'objects': vlm_result.get('objects', []),
                    'scene_tags': vlm_result.get('tags', []),
                    'confidence': vlm_result.get('confidence', 0.0),
                    'processing_time': vlm_result.get('processing_time', 0.0),
                    'api_response': vlm_result
                }
            else:
                return {'error': 'VLM API call failed'}
                
        except Exception as e:
            return {'error': str(e)}
    
    def process_vision_task(self, instruction: str, image: np.ndarray) -> Dict[str, Any]:
        """处理视觉任务的主入口"""
        try:
            # 分析指令类型
            task_type = self.analyze_task_type(instruction)
            target_object = self.extract_target_object(instruction)
            
            # 执行VLM任务
            vlm_result = self.execute_vlm_task(instruction, image)
            
            if 'error' in vlm_result:
                return {
                    'success': False,
                    'error': vlm_result['error'],
                    'task_type': task_type,
                    'target_object': target_object
                }
            
            # 根据任务类型处理结果
            result = {
                'success': True,
                'task_type': task_type,
                'target_object': target_object,
                'instruction': instruction,
                'vlm_result': vlm_result,
                'timestamp': time.time()
            }
            
            # 添加任务特定的信息
            if task_type == 'vision_only':
                result['response'] = vlm_result.get('scene_description', '')
            elif task_type == 'grasping':
                result['response'] = f"检测到目标物体: {target_object}" if target_object else "未指定目标物体"
            elif task_type == 'location_query':
                result['response'] = f"查询物体位置: {target_object}" if target_object else "未指定查询物体"
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'instruction': instruction
            }
    
    def display_result(self, result: Dict[str, Any]):
        """显示处理结果"""
        if result['success']:
            # 直接显示VLM分析结果
            if 'vlm_result' in result:
                vlm_result = result['vlm_result']
                description = vlm_result.get('scene_description', '')
                if description:
                    print(f"\n{description}")
                else:
                    print("\n无法获取有效描述")
            else:
                print("\n处理失败")
        else:
            print(f"\n处理失败: {result.get('error', '未知错误')}")
    
    def start_realtime_stream_analysis(self):
        """启动实时流分析模式"""
        if not self.camera_active or self.camera is None:
            print("摄像头未激活")
            return
        
        print("\n--- 实时流分析模式 ---")
        print("说明：输入指令时将自动捕获当前视频帧进行分析")
        print("输入 'quit' 或 'exit' 退出此模式")
        
        try:
            while True:
                # 获取用户指令
                instruction = input("\n请输入视觉分析指令: ").strip()
                
                # 检查是否要退出
                if instruction.lower() in ['quit', 'exit', '退出']:
                    print("退出实时流分析模式")
                    break
                
                if not instruction:
                    print("指令不能为空")
                    continue
                
                print("正在捕获当前视频帧...")
                
                # 捕获当前帧（在内存中进行）
                start_time = time.time()
                image = self.capture_frame()
                capture_time = time.time() - start_time
                
                if image is None:
                    print("图像捕获失败")
                    continue
                
                print(f"图像捕获成功，耗时: {capture_time:.3f}秒")
                
                # 处理视觉任务
                process_start = time.time()
                result = self.process_vision_task(instruction, image)
                process_time = time.time() - process_start
                
                # 显示结果
                self.display_result(result)
                
                # 显示整体耗时
                if result['success']:
                    total_time = capture_time + process_time
                    print(f"\n整体耗时: {total_time:.3f}秒")
                
                # 释放图像内存（Python会自动处理，但我们可以显式删除引用）
                del image
                
        except Exception as e:
            print(f"实时流分析模式发生错误: {str(e)}")
    
    def start_realtime_preview(self):
        """启动实时摄像头预览"""
        if not self.camera_active or self.camera is None:
            print("摄像头未激活")
            return
        
        # 检查是否有显示环境
        if not (os.environ.get('DISPLAY') and not os.environ.get('SSH_CLIENT')):
            print("无显示环境，无法启动实时预览")
            print("请使用以下方法之一启用显示环境：")
            print("1. SSH X11转发: ssh -X username@server_ip")
            print("2. VNC服务器")
            print("3. 本地图形界面")
            return
        
        print("\n启动实时摄像头预览...")
        print("操作说明：")
        print("- 按空格键：捕获当前帧进行分析")
        print("- 按 'q' 键：退出预览模式")
        print("- 按 's' 键：保存当前帧到文件")
        
        try:
            window_name = "实时摄像头预览 - 按空格键进行分析"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("无法读取摄像头帧")
                    break
                
                # 在图像上显示操作提示
                display_frame = frame.copy()
                cv2.putText(display_frame, "Space: Analyze | Q: Quit | S: Save", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 显示帧
                cv2.imshow(window_name, display_frame)
                
                # 等待按键
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    # 退出预览
                    print("退出实时预览模式")
                    break
                    
                elif key == ord(' '):
                    # 捕获当前帧进行分析
                    print("\n捕获当前帧进行分析...")
                    captured_image = frame.copy()
                    
                    # 获取用户指令
                    instruction = input("请输入视觉分析指令: ").strip()
                    if instruction:
                        # 处理视觉任务
                        result = self.process_vision_task(instruction, captured_image)
                        
                        # 显示结果
                        self.display_result(result)
                    else:
                        print("指令为空，取消分析")
                    
                    # 重新显示预览窗口
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    
                elif key == ord('s'):
                    # 保存当前帧
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"captured_frame_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"当前帧已保存到: {filename}")
                    
        except Exception as e:
            print(f"实时预览模式发生错误: {str(e)}")
        finally:
            try:
                cv2.destroyAllWindows()
            except:
                pass
    
    def run_interactive(self):
        """运行交互式测试"""
        print("VLA Vision Scheduler Test Version")
        print("="*40)
        print("请选择输入方式:")
        print("1. 图片输入")
        print("2. 摄像头输入")
        print("3. 视频输入")
        print("4. 退出")
        
        while True:
            try:
                choice = input("\n请输入选择 (1/2/3/4): ").strip()
                
                if choice == '1':
                    self.handle_image_input()
                elif choice == '2':
                    self.handle_camera_input()
                elif choice == '3':
                    self.handle_video_input()
                elif choice == '4':
                    print("退出程序...")
                    self.release_camera()
                    break
                else:
                    print("无效选择，请重新输入")
                    
            except KeyboardInterrupt:
                print("\n程序被用户中断")
                self.release_camera()
                break
            except Exception as e:
                print(f"发生错误: {str(e)}")
    
    def handle_image_input(self):
        """处理图片输入"""
        print("\n--- 图片输入模式 ---")
        
        # 使用固定图片路径
        image_path = "/root/kuavo_ws/src/ros_vla_vision/img/4.png"
        print(f"使用固定图片路径: {image_path}")
        
        # 加载图片
        image = self.load_image_from_file(image_path)
        if image is None:
            print("图片加载失败")
            return
        
        # 检查是否有显示环境，如果没有则跳过显示
        if os.environ.get('DISPLAY') and not os.environ.get('SSH_CLIENT'):
            try:
                cv2.imshow("Input Image", image)
                cv2.waitKey(1000)  # 显示1秒
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"无法显示图片: {str(e)}")
        else:
            print("无显示环境，跳过图片显示")
        
        # 获取用户指令
        instruction = input("请输入视觉分析指令: ").strip()
        if not instruction:
            print("指令不能为空")
            return
        
        # 处理视觉任务
        result = self.process_vision_task(instruction, image)
        
        # 显示结果
        self.display_result(result)
    
    def start_continuous_video_analysis(self):
        """启动连续视频流分析模式 - 用于动作识别和实时交互"""
        if not self.camera_active or self.camera is None:
            print("摄像头未激活")
            return
        
        print("\n--- 连续视频流分析模式 ---")
        print("说明：持续分析视频流，支持动作识别和快速响应")
        print("特点：")
        print("- 多帧缓存分析，提高动作识别准确性")
        print("- 预测性分析，减少响应延迟")
        print("- 自适应质量控制，平衡速度和精度")
        print("输入 'quit' 或 'exit' 退出此模式")
        
        try:
            # 视频帧缓存
            frame_buffer = []
            max_buffer_size = 10  # 缓存最近10帧
            analysis_interval = 0.5  # 分析间隔（秒）
            last_analysis_time = 0
            
            # 性能优化参数
            adaptive_quality = True
            current_quality = 0.8  # 初始质量系数
            
            print("开始连续视频流分析...")
            
            while True:
                # 捕获当前帧
                capture_start = time.time()
                image = self.capture_frame()
                capture_time = time.time() - capture_start
                
                if image is None:
                    print("图像捕获失败，跳过此帧")
                    time.sleep(0.1)
                    continue
                
                # 添加到帧缓存
                frame_buffer.append({
                    'image': image,
                    'timestamp': time.time(),
                    'capture_time': capture_time
                })
                
                # 保持缓存大小，并确保释放旧帧内存
                if len(frame_buffer) > max_buffer_size:
                    old_frame = frame_buffer.pop(0)
                    # 显式释放旧帧的图像内存
                    if 'image' in old_frame:
                        del old_frame['image']
                        print(f"已释放旧帧内存，缓存大小: {len(frame_buffer)}")
                
                # 检查是否需要进行分析
                current_time = time.time()
                if current_time - last_analysis_time >= analysis_interval:
                    print(f"\n执行周期性分析 (缓存帧数: {len(frame_buffer)})")
                    
                    # 自适应质量控制
                    if adaptive_quality:
                        # 根据最近处理时间调整质量
                        if hasattr(self, 'last_processing_time'):
                            if self.last_processing_time > 2.0:  # 如果上次处理太慢
                                current_quality = max(0.5, current_quality - 0.1)
                                print(f"降低质量以提高速度: {current_quality:.1f}")
                            elif self.last_processing_time < 1.0:  # 如果上次处理很快
                                current_quality = min(1.0, current_quality + 0.05)
                                print(f"提高质量以改善精度: {current_quality:.1f}")
                    
                    # 使用最新的帧进行分析
                    latest_frame = frame_buffer[-1]
                    
                    # 快速分析指令（预设的常用指令）
                    quick_instructions = [
                        "描述当前画面",
                        "检测画面中的主要物体",
                        "识别画面中的人物动作"
                    ]
                    
                    # 执行快速分析
                    analysis_start = time.time()
                    quick_result = self.process_vision_task(
                        quick_instructions[len(frame_buffer) % len(quick_instructions)], 
                        latest_frame['image']
                    )
                    analysis_time = time.time() - analysis_start
                    
                    # 记录处理时间
                    self.last_processing_time = analysis_time
                    
                    # 显示快速结果
                    if quick_result['success']:
                        print(f"快速分析结果: {quick_result.get('response', '无结果')}")
                        print(f"分析耗时: {analysis_time:.3f}秒")
                        
                        # 如果分析时间足够快，显示更多细节
                        if analysis_time < 1.0:
                            print("✓ 响应时间优秀，适合实时交互")
                        elif analysis_time < 2.0:
                            print("✓ 响应时间良好")
                        else:
                            print("⚠ 响应时间较慢，建议优化")
                    
                    last_analysis_time = current_time
                
                # 检查用户输入（非阻塞）
                import select
                import sys
                
                if select.select([sys.stdin], [], [], 0)[0]:
                    user_input = sys.stdin.readline().strip()
                    
                    if user_input.lower() in ['quit', 'exit', '退出']:
                        print("退出连续视频流分析模式")
                        break
                    
                    if user_input:
                        print(f"\n用户指令: {user_input}")
                        print("使用帧缓存进行增强分析...")
                        
                        # 使用多帧缓存进行分析
                        enhanced_start = time.time()
                        
                        # 选择最佳帧（最新的帧）
                        best_frame = frame_buffer[-1]
                        
                        # 执行用户指定的分析
                        enhanced_result = self.process_vision_task(user_input, best_frame['image'])
                        enhanced_time = time.time() - enhanced_start
                        
                        # 显示结果
                        self.display_result(enhanced_result)
                        
                        if enhanced_result['success']:
                            print(f"\n增强分析完成")
                            print(f"- 使用缓存帧数: {len(frame_buffer)}")
                            print(f"- 分析耗时: {enhanced_time:.3f}秒")
                            print(f"- 帧缓存延迟: {best_frame['capture_time']:.3f}秒")
                            
                            # 评估实时性
                            total_delay = enhanced_time + best_frame['capture_time']
                            if total_delay < 1.0:
                                print("✓ 满足实时交互要求 (< 1秒)")
                            elif total_delay < 2.0:
                                print("✓ 基本满足实时交互要求 (< 2秒)")
                            else:
                                print("⚠ 不满足实时交互要求 (> 2秒)")
                
                # 控制帧率
                time.sleep(0.033)  # 约30 FPS
                
        except Exception as e:
            print(f"连续视频流分析模式发生错误: {str(e)}")
        finally:
            # 清理帧缓存
            for frame_data in frame_buffer:
                del frame_data['image']
            frame_buffer.clear()
    
    
    
    
    def handle_camera_input(self):
        """处理摄像头输入"""
        print("\n--- 摄像头输入模式 ---")
        
        # 设置摄像头
        camera_id = input("请输入摄像头ID (默认为0): ").strip()
        camera_id = int(camera_id) if camera_id.isdigit() else 0
        
        if not self.setup_camera(camera_id):
            print("摄像头设置失败")
            return
        
        try:
            while True:
                print("\n摄像头已就绪，请选择操作:")
                print("1. 实时流分析模式 (输入指令时自动捕获当前帧)")
                print("2. 实时预览模式 (按空格键捕获图像进行分析)")
                print("3. 单次捕获并分析")
                print("4. 连续视频流分析模式")
                print("5. 重新设置摄像头")
                print("6. 返回主菜单")
                
                sub_choice = input("请输入选择 (1/2/3/4/5/6): ").strip()
                
                if sub_choice == '1':
                    # 实时流分析模式
                    self.start_realtime_stream_analysis()
                    
                elif sub_choice == '2':
                    # 实时预览模式
                    self.start_realtime_preview()
                    
                elif sub_choice == '3':
                    # 单次捕获模式
                    instruction = input("请输入视觉分析指令: ").strip()
                    if not instruction:
                        print("指令不能为空")
                        continue
                    
                    print("正在捕获图像...")
                    image = self.capture_frame()
                    if image is None:
                        print("图像捕获失败")
                        continue
                    
                    # 检查是否有显示环境，如果没有则跳过显示
                    if os.environ.get('DISPLAY') and not os.environ.get('SSH_CLIENT'):
                        try:
                            cv2.imshow("Captured Image", image)
                            cv2.waitKey(1000)  # 显示1秒
                            cv2.destroyAllWindows()
                        except Exception as e:
                            print(f"无法显示图片: {str(e)}")
                    else:
                        print("无显示环境，跳过图片显示")
                    
                    # 处理视觉任务
                    result = self.process_vision_task(instruction, image)
                    
                    # 显示结果
                    self.display_result(result)
                    
                elif sub_choice == '4':
                    # 连续视频流分析模式
                    self.start_continuous_video_analysis()
                    
                elif sub_choice == '5':
                    # 重新设置摄像头
                    self.release_camera()
                    new_camera_id = input("请输入新的摄像头ID: ").strip()
                    new_camera_id = int(new_camera_id) if new_camera_id.isdigit() else 0
                    
                    if not self.setup_camera(new_camera_id):
                        print("摄像头设置失败")
                        break
                        
                elif sub_choice == '6':
                    # 返回主菜单
                    self.release_camera()
                    break
                    
                else:
                    print("无效选择，请重新输入")
                    
        except Exception as e:
            print(f"摄像头模式发生错误: {str(e)}")
        finally:
            self.release_camera()
    
    def handle_video_input(self):
        """处理视频输入"""
        print("\n--- 视频输入模式 ---")
        
        # 使用固定视频路径
        video_path = "/root/kuavo_ws/src/ros_vla_vision/img/8.mp4"  # 固定视频路径
        print(f"使用固定视频路径: {video_path}")
        
        # 检查视频文件是否存在
        if not os.path.exists(video_path):
            print(f"视频文件不存在: {video_path}")
            print("请将视频文件放置在指定路径")
            return
        
        try:
            while True:
                print("\n视频已就绪，请选择操作:")
                print("1. 视频帧提取分析（提取关键帧进行分析）")
                print("2. 实时视频分析模式（输入指令时自动分析）")
                print("3. 视频预览模式（播放视频并选择帧进行分析）")
                print("4. 返回主菜单")
                
                sub_choice = input("请输入选择 (1/2/3/4): ").strip()
                
                if sub_choice == '1':
                    # 视频帧提取分析
                    instruction = input("请输入视频分析指令: ").strip()
                    if not instruction:
                        print("指令不能为空")
                        continue
                    
                    print("正在提取视频帧并进行分析...")
                    result = self.process_video_analysis(video_path, instruction)
                    self.display_video_result(result)
                    
                elif sub_choice == '2':
                    # 实时视频分析模式
                    self.start_realtime_video_analysis()
                    
                elif sub_choice == '3':
                    # 视频预览模式
                    self.start_video_preview_mode(video_path)
                    
                elif sub_choice == '4':
                    # 返回主菜单
                    break
                    
                else:
                    print("无效选择，请重新输入")
                    
        except Exception as e:
            print(f"视频模式发生错误: {str(e)}")
    
    def start_video_preview_mode(self, video_path: str):
        """启动视频预览模式"""
        print("\n--- 视频预览模式 ---")
        
        try:
            # 加载视频
            video = self.load_video_from_file(video_path)
            if video is None:
                return
            
            # 检查是否有显示环境
            if not (os.environ.get('DISPLAY') and not os.environ.get('SSH_CLIENT')):
                print("无显示环境，无法启动视频预览")
                print("请使用以下方法之一启用显示环境：")
                print("1. SSH X11转发: ssh -X username@server_ip")
                print("2. VNC服务器")
                print("3. 本地图形界面")
                video.release()
                return
            
            print("视频预览控制说明：")
            print("- 空格键：暂停/播放")
            print("- 's' 键：保存当前帧")
            print("- 'a' 键：分析当前帧")
            print("- 'q' 键：退出预览")
            
            window_name = "视频预览"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            is_paused = False
            current_frame = None
            
            while True:
                if not is_paused:
                    ret, frame = video.read()
                    if not ret:
                        # 视频结束，重新开始
                        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = video.read()
                        if not ret:
                            break
                    
                    current_frame = frame.copy()
                
                if current_frame is not None:
                    # 在帧上显示控制提示
                    display_frame = current_frame.copy()
                    status = "暂停" if is_paused else "播放"
                    cv2.putText(display_frame, f"状态: {status} | Space:暂停/播放 | S:保存 | A:分析 | Q:退出", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # 显示当前帧信息
                    current_pos = video.get(cv2.CAP_PROP_POS_FRAMES)
                    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
                    cv2.putText(display_frame, f"帧: {int(current_pos)}/{int(total_frames)}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # 显示帧
                    cv2.imshow(window_name, display_frame)
                    
                    # 等待按键
                    key = cv2.waitKey(30) & 0xFF
                    
                    if key == ord('q'):
                        # 退出预览
                        print("退出视频预览模式")
                        break
                        
                    elif key == ord(' '):
                        # 暂停/播放
                        is_paused = not is_paused
                        print(f"视频{'暂停' if is_paused else '播放'}")
                        
                    elif key == ord('s'):
                        # 保存当前帧
                        if current_frame is not None:
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            filename = f"video_frame_{timestamp}.jpg"
                            cv2.imwrite(filename, current_frame)
                            print(f"当前帧已保存到: {filename}")
                        
                    elif key == ord('a'):
                        # 分析当前帧
                        if current_frame is not None:
                            print("\n分析当前视频帧...")
                            
                            # 获取用户指令
                            instruction = input("请输入视觉分析指令: ").strip()
                            if instruction:
                                # 处理视觉任务
                                result = self.process_vision_task(instruction, current_frame)
                                
                                # 显示结果
                                self.display_result(result)
                                
                                # 重新显示预览窗口
                                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                            else:
                                print("指令为空，取消分析")
                
                # 控制播放速度
                time.sleep(0.03)  # 约30 FPS
                
        except Exception as e:  
            print(f"视频预览模式发生错误: {str(e)}")
        finally:
            try:
                cv2.destroyAllWindows()
                if 'video' in locals():
                    video.release()
            except:
                pass

def main():
    """主函数"""
    try:
        # 创建测试调度器
        scheduler = TestVLAVisionScheduler()
        
        # 运行交互式测试
        scheduler.run_interactive()
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
    finally:
        try:
            cv2.destroyAllWindows()
        except:
            pass

if __name__ == '__main__':
    main()
