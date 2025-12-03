#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VLA Vision Scheduler
视觉调度核心，负责智能调用多种感知模块来辅助任务执行
"""

import rospy
import json
import time
import threading
import requests
import base64
import cv2
import numpy as np
import yaml
from typing import Dict, List, Any, Optional, Union
from std_msgs.msg import Header, String
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import Pose, Pose2D, Pose3D, Point, Vector3, Quaternion
from ros_vla_language.msg import VLACommand
from ros_vla_vision.msg import (
    VLAObjectPose, VLAObjectDepth, VLAVLMResult, VLAGraspPoint
)
from ros_vla_vision.srv import (
    DetectObjects, EstimatePose, ExtractFeatures, 
    UnderstandScene, GetObjectInfo
)

class VLAVisionScheduler:
    def __init__(self):
        rospy.init_node('vla_vision_scheduler', anonymous=True)
        
        # 加载配置文件
        self.config = self.load_config()
        
        # 参数初始化
        self.enable_vlm = self.config['basic_config']['enable_vlm']
        self.enable_yolo = self.config['basic_config']['enable_yolo']
        self.enable_depth = self.config['basic_config']['enable_depth']
        self.default_camera = self.config['basic_config']['default_camera']
        self.default_depth_camera = self.config['basic_config']['default_depth_camera']
        self.confidence_threshold = self.config['basic_config']['confidence_threshold']
        
        # VLM API配置
        self.vlm_api_url = self.config['vlm_api']['url']
        self.vlm_api_key = self.config['vlm_api']['api_key']
        self.vlm_model_name = self.config['vlm_api']['model_name']
        self.vlm_timeout = self.config['vlm_api']['timeout']
        self.vlm_max_retries = self.config['vlm_api']['max_retries']
        
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
        
        # 对象跟踪
        self.tracked_objects = {}
        self.object_tracking_enabled = self.config['object_tracking']['enable_tracking']
        self.tracking_timeout = self.config['object_tracking']['tracking_timeout']
        
        # 发布状态
        self.publishing_enabled = False
        self.current_target_object = None
        
        # ROS服务客户端
        self.init_service_clients()
        
        # 订阅器
        topics = self.config['ros_topics']['subscriptions']
        
        # 订阅语言指令
        self.language_command_sub = rospy.Subscriber(
            topics['language_command_topic'],
            VLACommand,
            self.language_command_callback
        )
        
        # 图像订阅
        self.image_sub = rospy.Subscriber(
            topics['camera_image'], 
            Image, 
            self.image_callback
        )
        self.camera_info_sub = rospy.Subscriber(
            topics['camera_info'], 
            CameraInfo, 
            self.camera_info_callback
        )
        self.point_cloud_sub = rospy.Subscriber(
            topics['depth_points'], 
            PointCloud2, 
            self.point_cloud_callback
        )
        
        # 发布器
        publications = self.config['ros_topics']['publications']
        self.scheduler_result_pub = rospy.Publisher(
            publications['scheduler_result_topic'], 
            String, 
            queue_size=10
        )
        self.object_pose_pub = rospy.Publisher(
            publications['object_pose_topic'],
            VLAObjectPose,
            queue_size=10
        )
        self.object_depth_pub = rospy.Publisher(
            publications['object_depth_topic'],
            VLAObjectDepth,
            queue_size=10
        )
        self.vlm_result_pub = rospy.Publisher(
            publications['vlm_result_topic'],
            VLAVLMResult,
            queue_size=10
        )
        self.target_object_info_pub = rospy.Publisher(
            publications['target_object_info_topic'],
            String,
            queue_size=10
        )
        
        rospy.loginfo("VLA Vision Scheduler initialized")
        rospy.loginfo(f"Config loaded with {len(self.task_types)} task types")
    
    def init_service_clients(self):
        """初始化ROS服务客户端"""
        try:
            # 等待服务可用
            rospy.loginfo("Waiting for vision services...")
            
            # 物体检测服务
            rospy.wait_for_service('/vla/detect_objects', timeout=10.0)
            self.detect_objects_client = rospy.ServiceProxy(
                '/vla/detect_objects', 
                DetectObjects
            )
            
            # 位姿估计服务
            rospy.wait_for_service('/vla/estimate_pose', timeout=10.0)
            self.estimate_pose_client = rospy.ServiceProxy(
                '/vla/estimate_pose', 
                EstimatePose
            )
            
            # 特征提取服务
            rospy.wait_for_service('/vla/extract_features', timeout=10.0)
            self.extract_features_client = rospy.ServiceProxy(
                '/vla/extract_features', 
                ExtractFeatures
            )
            
            # 物体信息服务
            rospy.wait_for_service('/vla/get_object_info', timeout=10.0)
            self.get_object_info_client = rospy.ServiceProxy(
                '/vla/get_object_info', 
                GetObjectInfo
            )
            
            rospy.loginfo("All vision services connected")
            
        except Exception as e:
            rospy.logerr(f"Failed to connect to vision services: {str(e)}")
            self.services_available = False
        else:
            self.services_available = True
    
    def image_callback(self, msg):
        """图像回调 - 只保存当前帧"""
        self.current_image = msg
    
    def camera_info_callback(self, msg):
        """相机信息回调"""
        self.current_camera_info = msg
    
    def point_cloud_callback(self, msg):
        """点云回调"""
        self.current_point_cloud = msg
        # 如果启用了持续发布，处理点云数据
        if self.publishing_enabled:
            self.process_continuous_data()
    
    
    def language_command_callback(self, msg):
        """语言指令回调"""
        try:
            command_text = msg.command_text
            rospy.loginfo(f"Received language command: {command_text}")
            
            # 分析指令类型
            task_type = self.analyze_task_type(command_text)
            target_object = self.extract_target_object(command_text)
            
            # 设置当前目标对象
            self.current_target_object = {
                'name': target_object,
                'type': task_type,
                'command': command_text,
                'timestamp': time.time()
            }
            
            # 根据任务类型处理
            if task_type == 'vision_only':
                # 纯视觉理解任务
                self.handle_vision_only_task(command_text)
            elif task_type == 'grasping':
                # 抓取任务
                self.handle_grasping_task(command_text, target_object)
            elif task_type == 'location_query':
                # 位置查询任务
                self.handle_location_query_task(command_text, target_object)
            else:
                rospy.logwarn(f"Unknown task type for command: {command_text}")
                
        except Exception as e:
            rospy.logerr(f"Error processing language command: {str(e)}")
    
    def handle_perception_call(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理感知调用请求"""
        try:
            modules = command_data.get('modules', [])
            params = command_data.get('params', {})
            task = params.get('task', '')
            image_source = params.get('image_source', self.default_camera)
            depth_source = params.get('depth_source', self.default_depth_camera)
            
            rospy.loginfo(f"Processing perception call: {task}")
            rospy.loginfo(f"Required modules: {modules}")
            
            # 检查模块可用性
            unavailable_modules = [m for m in modules if not self.module_status.get(m, False)]
            if unavailable_modules:
                return {
                    'success': False,
                    'error': f"Modules not available: {unavailable_modules}",
                    'timestamp': time.time()
                }
            
            # 执行感知任务
            result = self.execute_perception_task(modules, task, image_source, depth_source)
            
            return {
                'success': True,
                'result': result,
                'modules_used': modules,
                'task': task,
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def execute_perception_task(self, modules: List[str], task: str, 
                               image_source: str, depth_source: str) -> Dict[str, Any]:
        """执行感知任务"""
        try:
            if not self.services_available:
                raise Exception("Vision services not available")
            
            if not self.current_image:
                raise Exception("No current image available")
            
            result = {}
            
            # 根据任务类型和所需模块执行相应的感知操作
            if 'VLM' in modules:
                # VLM分析当前帧图像
                result['vlm_result'] = self.execute_vlm_task(task, self.current_image)
            
            if 'YOLO' in modules:
                result['yolo_result'] = self.execute_yolo_task(task)
            
            if 'Depth' in modules:
                result['depth_result'] = self.execute_depth_task(task)
            
            # 如果需要多模块融合
            if len(modules) > 1:
                result['fusion_result'] = self.fuse_multi_modal_results(result, modules)
            
            return result
            
        except Exception as e:
            rospy.logerr(f"Error executing perception task: {str(e)}")
            raise
    
    def execute_vlm_task(self, task: str, image: Image) -> Dict[str, Any]:
        """执行VLM任务 - 调用VLM API分析当前帧"""
        try:
            rospy.loginfo(f"Executing VLM task: {task}")
            
            # 将ROS图像转换为OpenCV格式
            cv_image = self.ros_to_cv2(image)
            if cv_image is None:
                return {'error': 'Failed to convert image'}
            
            # 调用VLM API
            vlm_result = self.call_vlm_api(cv_image, task)
            
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
            rospy.logerr(f"VLM task execution failed: {str(e)}")
            return {'error': str(e)}
    
    def call_vlm_api(self, image: np.ndarray, task: str) -> Optional[Dict[str, Any]]:
        """调用VLM API分析图像"""
        try:
            start_time = time.time()
            
            # 将图像编码为base64
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # 准备API请求
            headers = {
                'Content-Type': 'application/json',
            }
            
            if self.vlm_api_key:
                headers['Authorization'] = f'Bearer {self.vlm_api_key}'
            
            payload = {
                'model': self.vlm_model_name,
                'image': image_base64,
                'task': task,
                'max_tokens': 500,
                'temperature': 0.1
            }
            
            # 发送API请求
            response = requests.post(
                self.vlm_api_url,
                headers=headers,
                json=payload,
                timeout=30.0
            )
            
            if response.status_code == 200:
                api_result = response.json()
                
                # 解析API响应
                result = {
                    'description': api_result.get('description', ''),
                    'scene_type': api_result.get('scene_type', 'unknown'),
                    'objects': api_result.get('objects', []),
                    'tags': api_result.get('tags', []),
                    'confidence': api_result.get('confidence', 0.0),
                    'processing_time': time.time() - start_time,
                    'raw_response': api_result
                }
                
                rospy.loginfo(f"VLM API call successful, processing time: {result['processing_time']:.2f}s")
                return result
                
            else:
                rospy.logerr(f"VLM API call failed: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            rospy.logerr("VLM API call timeout")
            return None
        except requests.exceptions.RequestException as e:
            rospy.logerr(f"VLM API request failed: {str(e)}")
            return None
        except Exception as e:
            rospy.logerr(f"Error calling VLM API: {str(e)}")
            return None
    
    def ros_to_cv2(self, ros_image: Image) -> Optional[np.ndarray]:
        """将ROS图像转换为OpenCV格式"""
        try:
            import cv_bridge
            bridge = cv_bridge.CvBridge()
            return bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except ImportError:
            rospy.logerr("cv_bridge not installed")
            return None
        except Exception as e:
            rospy.logerr(f"Error converting image: {str(e)}")
            return None
    
    def execute_yolo_task(self, task: str) -> Dict[str, Any]:
        """执行YOLO任务"""
        try:
            rospy.loginfo(f"Executing YOLO task: {task}")
            
            # 从任务描述中提取目标类别
            target_classes = self.extract_target_classes_from_task(task)
            
            # 调用物体检测服务
            response = self.detect_objects_client(
                image=self.current_image,
                camera_info=self.current_camera_info,
                target_classes=target_classes,
                confidence_threshold=self.confidence_threshold
            )
            
            if response.success:
                detections = []
                for detection in response.detections:
                    detections.append({
                        'object_id': detection.object_id,
                        'object_class': detection.object_class,
                        'confidence': detection.confidence,
                        'bbox': {
                            'x': detection.bbox.x,
                            'y': detection.bbox.y,
                            'width': detection.bbox.theta,  # 注意：这里需要修正
                            'height': detection.bbox.theta   # 注意：这里需要修正
                        },
                        'pose': {
                            'x': detection.pose.position.x,
                            'y': detection.pose.position.y,
                            'z': detection.pose.position.z
                        }
                    })
                
                return {
                    'detections': detections,
                    'total_objects': response.total_objects,
                    'detection_method': response.detection_method,
                    'processing_time': response.processing_time
                }
            else:
                raise Exception(response.error_message)
                
        except Exception as e:
            rospy.logerr(f"YOLO task execution failed: {str(e)}")
            return {'error': str(e)}
    
    def execute_depth_task(self, task: str) -> Dict[str, Any]:
        """执行深度任务"""
        try:
            rospy.loginfo(f"Executing Depth task: {task}")
            
            if not self.current_point_cloud:
                return {'error': 'No point cloud data available'}
            
            # 首先进行物体检测
            detection_response = self.detect_objects_client(
                image=self.current_image,
                camera_info=self.current_camera_info,
                confidence_threshold=self.confidence_threshold
            )
            
            if not detection_response.success:
                raise Exception(detection_response.error_message)
            
            # 对检测到的物体进行深度信息获取
            depth_results = []
            for detection in detection_response.detections:
                # 调用物体信息服务获取深度信息
                object_info_response = self.get_object_info_client(
                    object_id=detection.object_id,
                    object_name=detection.object_class,
                    detection=detection,
                    image=self.current_image,
                    point_cloud=self.current_point_cloud,
                    info_level='detailed'
                )
                
                if object_info_response.success:
                    depth_results.append({
                        'object_id': detection.object_id,
                        'object_class': detection.object_class,
                        '3d_position': {
                            'x': object_info_response.object_info.pose.position.x,
                            'y': object_info_response.object_info.pose.position.y,
                            'z': object_info_response.object_info.pose.position.z
                        },
                        '3d_size': {
                            'x': object_info_response.object_info.size.x,
                            'y': object_info_response.object_info.size.y,
                            'z': object_info_response.object_info.size.z
                        },
                        'distance': self.calculate_distance(
                            object_info_response.object_info.pose.position
                        )
                    })
            
            return {
                'depth_results': depth_results,
                'total_objects': len(depth_results),
                'processing_time': sum(r.get('processing_time', 0) for r in depth_results)
            }
            
        except Exception as e:
            rospy.logerr(f"Depth task execution failed: {str(e)}")
            return {'error': str(e)}
    
    def extract_target_classes_from_task(self, task: str) -> List[str]:
        """从任务描述中提取目标类别"""
        try:
            # 简单的关键词匹配
            task_lower = task.lower()
            
            # 常见物体类别映射
            class_keywords = {
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
            
            target_classes = []
            for chinese_class, english_classes in class_keywords.items():
                if chinese_class in task_lower:
                    target_classes.extend(english_classes)
            
            # 如果没有匹配到特定类别，返回空列表表示检测所有类别
            return target_classes if target_classes else []
            
        except Exception as e:
            rospy.logerr(f"Error extracting target classes: {str(e)}")
            return []
    
    def calculate_distance(self, position: Point) -> float:
        """计算3D距离"""
        try:
            # 计算到原点的距离
            distance = (position.x**2 + position.y**2 + position.z**2)**0.5
            return distance
        except Exception as e:
            rospy.logerr(f"Error calculating distance: {str(e)}")
            return 0.0
    
    def fuse_multi_modal_results(self, results: Dict[str, Any], 
                                modules: List[str]) -> Dict[str, Any]:
        """融合多模态结果"""
        try:
            fused_result = {
                'fusion_method': 'late_fusion',
                'combined_objects': [],
                'scene_summary': '',
                'confidence_scores': {}
            }
            
            # 融合物体信息
            if 'yolo_result' in results and 'depth_result' in results:
                yolo_detections = results['yolo_result'].get('detections', [])
                depth_results = results['depth_result'].get('depth_results', [])
                
                # 基于物体ID匹配融合结果
                for yolo_det in yolo_detections:
                    matching_depth = None
                    for depth_res in depth_results:
                        if yolo_det['object_id'] == depth_res['object_id']:
                            matching_depth = depth_res
                            break
                    
                    if matching_depth:
                        fused_object = {
                            'object_id': yolo_det['object_id'],
                            'object_class': yolo_det['object_class'],
                            '2d_bbox': yolo_det['bbox'],
                            '3d_position': matching_depth['3d_position'],
                            '3d_size': matching_depth['3d_size'],
                            'distance': matching_depth['distance'],
                            'confidence': yolo_det['confidence']
                        }
                        fused_result['combined_objects'].append(fused_object)
            
            # 融合场景理解
            if 'vlm_result' in results:
                vlm_result = results['vlm_result']
                fused_result['scene_summary'] = vlm_result.get('scene_description', '')
                fused_result['scene_type'] = vlm_result.get('scene_type', '')
                fused_result['scene_tags'] = vlm_result.get('scene_tags', [])
            
            # 计算综合置信度
            for module in modules:
                if module in results:
                    if 'error' not in results[module]:
                        fused_result['confidence_scores'][module] = 0.8
                    else:
                        fused_result['confidence_scores'][module] = 0.0
            
            return fused_result
            
        except Exception as e:
            rospy.logerr(f"Error fusing multi-modal results: {str(e)}")
            return {'error': str(e)}
    
    def analyze_task_and_call_modules(self, user_instruction: str) -> Dict[str, Any]:
        """分析用户指令并调用相应模块"""
        try:
            rospy.loginfo(f"Analyzing user instruction: {user_instruction}")
            
            # 分析指令是否需要视觉信息
            needs_vision = self.needs_vision_analysis(user_instruction)
            
            if not needs_vision:
                return {
                    'action': 'no_vision_needed',
                    'reason': '指令不涉及视觉相关内容',
                    'modules': [],
                    'params': {}
                }
            
            # 决定需要调用的模块
            modules_to_call = self.determine_required_modules(user_instruction)
            
            # 生成感知任务描述
            perception_task = self.generate_perception_task_description(user_instruction)
            
            return {
                'action': 'call_perception',
                'reason': self.generate_reason_for_modules(user_instruction, modules_to_call),
                'modules': modules_to_call,
                'params': {
                    'task': perception_task,
                    'image_source': self.default_camera,
                    'depth_source': self.default_depth_camera if 'Depth' in modules_to_call else None
                }
            }
            
        except Exception as e:
            rospy.logerr(f"Error analyzing task: {str(e)}")
            return {
                'action': 'error',
                'error': str(e)
            }
    
    def needs_vision_analysis(self, instruction: str) -> bool:
        """判断指令是否需要视觉分析"""
        try:
            instruction_lower = instruction.lower()
            
            # 视觉相关关键词
            vision_keywords = [
                '看', '看见', '找到', '寻找', '检测', '识别', '定位',
                '颜色', '位置', '形状', '大小', '距离', '多远',
                '在哪里', '什么颜色', '什么样', '几个', '多少',
                '场景', '环境', '周围', '附近', '前面', '后面',
                '左边', '右边', '上面', '下面'
            ]
            
            # 检查是否包含视觉相关关键词
            for keyword in vision_keywords:
                if keyword in instruction_lower:
                    return True
            
            return False
            
        except Exception as e:
            rospy.logerr(f"Error determining if vision analysis is needed: {str(e)}")
            return False
    
    def determine_required_modules(self, instruction: str) -> List[str]:
        """确定需要调用的模块"""
        try:
            instruction_lower = instruction.lower()
            modules = []
            
            # VLM相关任务
            vlm_keywords = [
                '理解场景', '描述场景', '分析环境', '识别文字',
                '阅读', '理解', '解释', '总结', '分析'
            ]
            
            # YOLO相关任务
            yolo_keywords = [
                '检测', '识别', '找到', '定位', '边界框',
                '精确位置', '坐标', '检测物体'
            ]
            
            # 深度相关任务
            depth_keywords = [
                '距离', '多远', '深度', '3d', '三维',
                '位置', '坐标', '远近', '高度'
            ]
            
            # 检查需要哪些模块
            for keyword in vlm_keywords:
                if keyword in instruction_lower:
                    modules.append('VLM')
                    break
            
            for keyword in yolo_keywords:
                if keyword in instruction_lower:
                    modules.append('YOLO')
                    break
            
            for keyword in depth_keywords:
                if keyword in instruction_lower:
                    modules.append('Depth')
                    break
            
            # 如果没有明确指定，但需要视觉分析，默认使用VLM
            if not modules and self.needs_vision_analysis(instruction):
                modules.append('VLM')
            
            # 去重
            modules = list(set(modules))
            
            return modules
            
        except Exception as e:
            rospy.logerr(f"Error determining required modules: {str(e)}")
            return ['VLM']  # 默认返回VLM
    
    def generate_perception_task_description(self, instruction: str) -> str:
        """生成感知任务描述"""
        try:
            # 简单的任务描述生成
            task_descriptions = {
                '找': '检测并定位',
                '检测': '检测并识别',
                '识别': '识别并分类',
                '看': '观察并分析',
                '寻找': '搜索并定位',
                '分析': '分析并理解',
                '描述': '描述并解释'
            }
            
            # 找到匹配的动作
            for action, description in task_descriptions.items():
                if action in instruction:
                    return f"{description}{instruction}"
            
            # 如果没有匹配，返回原始指令
            return instruction
            
        except Exception as e:
            rospy.logerr(f"Error generating perception task description: {str(e)}")
            return instruction
    
    def generate_reason_for_modules(self, instruction: str, modules: List[str]) -> str:
        """生成调用模块的原因说明"""
        try:
            reasons = []
            
            if 'VLM' in modules:
                reasons.append("需要理解场景、识别物体类别和属性")
            
            if 'YOLO' in modules:
                reasons.append("需要高精度物体检测和边界框定位")
            
            if 'Depth' in modules:
                reasons.append("需要获取物体的距离和三维位置信息")
            
            if len(modules) > 1:
                reasons.append("需要多模态信息融合以提高准确性")
            
            return "；".join(reasons) if reasons else "需要进行视觉分析"
            
        except Exception as e:
            rospy.logerr(f"Error generating reason for modules: {str(e)}")
            return "需要进行视觉分析"
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            config_path = rospy.get_param('~config_file', 
                '/root/kuavo_ws/src/ros_vla_vision/config/vlm_api_params.yaml')
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            rospy.loginfo(f"Configuration loaded from {config_path}")
            return config
            
        except Exception as e:
            rospy.logerr(f"Failed to load config: {str(e)}")
            # 返回默认配置
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'basic_config': {
                'enable_vlm': True,
                'enable_yolo': True,
                'enable_depth': True,
                'default_camera': 'camera_1',
                'default_depth_camera': 'depth_cam_1',
                'confidence_threshold': 0.5
            },
            'vlm_api': {
                'url': 'http://localhost:8000/vlm/analyze',
                'api_key': '',
                'model_name': 'gpt-4-vision-preview',
                'timeout': 30.0,
                'max_retries': 3
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
                '杯子': ['cup', 'glass', 'mug']
            },
            'object_tracking': {
                'enable_tracking': True,
                'tracking_timeout': 10.0,
                'max_objects': 20,
                'position_tolerance': 0.1
            }
        }
    
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
    
    def handle_vision_only_task(self, instruction: str):
        """处理纯视觉理解任务"""
        try:
            rospy.loginfo(f"Handling vision-only task: {instruction}")
            
            # 调用VLM分析
            vlm_result = self.execute_vlm_task(instruction, self.current_image)
            
            # 创建VLM结果消息
            vlm_msg = VLAVLMResult()
            vlm_msg.header.stamp = rospy.Time.now()
            vlm_msg.task_id = str(int(time.time()))
            vlm_msg.instruction = instruction
            vlm_msg.task_type = 'vision_only'
            vlm_msg.scene_description = vlm_result.get('scene_description', '')
            vlm_msg.scene_type = vlm_result.get('scene_type', 'unknown')
            vlm_msg.objects = vlm_result.get('objects', [])
            vlm_msg.scene_tags = vlm_result.get('scene_tags', [])
            vlm_msg.confidence = vlm_result.get('confidence', 0.0)
            vlm_msg.processing_time = vlm_result.get('processing_time', 0.0)
            vlm_msg.include_pose = False
            vlm_msg.include_depth = False
            vlm_msg.include_grasp_points = False
            vlm_msg.is_grasping_task = False
            vlm_msg.is_location_query = False
            vlm_msg.vlm_model = self.vlm_model_name
            vlm_msg.api_call_success = 'error' not in vlm_result
            
            # 发布VLM结果
            self.vlm_result_pub.publish(vlm_msg)
            
            rospy.loginfo(f"Vision-only task completed, published VLM result")
            
        except Exception as e:
            rospy.logerr(f"Error handling vision-only task: {str(e)}")
    
    def handle_grasping_task(self, instruction: str, target_object: str):
        """处理抓取任务"""
        try:
            rospy.loginfo(f"Handling grasping task: {instruction}, target: {target_object}")
            
            # 检测目标物体
            target_classes = self.object_class_mapping.get(target_object, [target_object])
            
            # 调用物体检测
            detection_response = self.detect_objects_client(
                image=self.current_image,
                camera_info=self.current_camera_info,
                target_classes=target_classes,
                confidence_threshold=self.confidence_threshold
            )
            
            if not detection_response.success:
                rospy.logerr(f"Object detection failed: {detection_response.error_message}")
                return
            
            # 获取物体位姿和深度信息
            object_poses = []
            object_depths = []
            
            for detection in detection_response.detections:
                # 获取物体详细信息
                object_info_response = self.get_object_info_client(
                    object_id=detection.object_id,
                    object_name=detection.object_class,
                    detection=detection,
                    image=self.current_image,
                    point_cloud=self.current_point_cloud,
                    info_level='detailed'
                )
                
                if object_info_response.success:
                    # 创建位姿消息
                    pose_msg = VLAObjectPose()
                    pose_msg.header.stamp = rospy.Time.now()
                    pose_msg.object_id = detection.object_id
                    pose_msg.object_class = detection.object_class
                    pose_msg.object_name = target_object
                    pose_msg.pose = object_info_response.object_info.pose
                    pose_msg.confidence = detection.confidence
                    pose_msg.detection_method = detection_response.detection_method
                    pose_msg.is_tracked = True
                    pose_msg.tracking_quality = 1.0
                    pose_msg.reference_frame = "camera_link"
                    
                    # 创建深度消息
                    depth_msg = VLAObjectDepth()
                    depth_msg.header.stamp = rospy.Time.now()
                    depth_msg.object_id = detection.object_id
                    depth_msg.object_class = detection.object_class
                    depth_msg.object_name = target_object
                    depth_msg.position = object_info_response.object_info.pose.position
                    depth_msg.size = object_info_response.object_info.size
                    depth_msg.distance = self.calculate_distance(object_info_response.object_info.pose.position)
                    depth_msg.depth_confidence = 0.8
                    depth_msg.depth_source = "point_cloud"
                    depth_msg.is_valid = True
                    depth_msg.point_density = 1.0
                    depth_msg.reference_frame = "camera_link"
                    
                    object_poses.append(pose_msg)
                    object_depths.append(depth_msg)
            
            # 创建VLM结果消息
            vlm_msg = VLAVLMResult()
            vlm_msg.header.stamp = rospy.Time.now()
            vlm_msg.task_id = str(int(time.time()))
            vlm_msg.instruction = instruction
            vlm_msg.task_type = 'grasping'
            vlm_msg.scene_description = f"检测到{len(object_poses)}个{target_object}物体"
            vlm_msg.scene_type = 'grasping_scene'
            vlm_msg.objects = [target_object]
            vlm_msg.confidence = 0.8 if object_poses else 0.0
            vlm_msg.processing_time = 0.0
            vlm_msg.object_poses = object_poses
            vlm_msg.object_depths = object_depths
            vlm_msg.include_pose = True
            vlm_msg.include_depth = True
            vlm_msg.include_grasp_points = True
            vlm_msg.is_grasping_task = True
            vlm_msg.is_location_query = False
            vlm_msg.target_object = target_object
            vlm_msg.target_object_class = target_object
            vlm_msg.target_object_found = len(object_poses) > 0
            vlm_msg.vlm_model = self.vlm_model_name
            vlm_msg.api_call_success = True
            
            # 发布结果
            self.vlm_result_pub.publish(vlm_msg)
            
            # 分别发布位姿和深度信息
            for pose_msg in object_poses:
                self.object_pose_pub.publish(pose_msg)
            
            for depth_msg in object_depths:
                self.object_depth_pub.publish(depth_msg)
            
            # 发布目标物体信息
            target_info = {
                'target_object': target_object,
                'found_objects': len(object_poses),
                'poses': [{'object_id': p.object_id, 'confidence': p.confidence} for p in object_poses],
                'timestamp': time.time()
            }
            
            target_info_msg = String()
            target_info_msg.data = json.dumps(target_info, ensure_ascii=False)
            self.target_object_info_pub.publish(target_info_msg)
            
            rospy.loginfo(f"Grasping task completed, found {len(object_poses)} {target_object} objects")
            
        except Exception as e:
            rospy.logerr(f"Error handling grasping task: {str(e)}")
    
    def handle_location_query_task(self, instruction: str, target_object: str):
        """处理位置查询任务"""
        try:
            rospy.loginfo(f"Handling location query task: {instruction}, target: {target_object}")
            
            # 类似于抓取任务，但只返回位置信息
            target_classes = self.object_class_mapping.get(target_object, [target_object])
            
            # 调用物体检测
            detection_response = self.detect_objects_client(
                image=self.current_image,
                camera_info=self.current_camera_info,
                target_classes=target_classes,
                confidence_threshold=self.confidence_threshold
            )
            
            if not detection_response.success:
                rospy.logerr(f"Object detection failed: {detection_response.error_message}")
                return
            
            # 获取物体位置信息
            object_poses = []
            object_depths = []
            
            for detection in detection_response.detections:
                object_info_response = self.get_object_info_client(
                    object_id=detection.object_id,
                    object_name=detection.object_class,
                    detection=detection,
                    image=self.current_image,
                    point_cloud=self.current_point_cloud,
                    info_level='position_only'
                )
                
                if object_info_response.success:
                    # 创建位姿消息
                    pose_msg = VLAObjectPose()
                    pose_msg.header.stamp = rospy.Time.now()
                    pose_msg.object_id = detection.object_id
                    pose_msg.object_class = detection.object_class
                    pose_msg.object_name = target_object
                    pose_msg.pose = object_info_response.object_info.pose
                    pose_msg.confidence = detection.confidence
                    pose_msg.detection_method = detection_response.detection_method
                    pose_msg.is_tracked = True
                    pose_msg.tracking_quality = 1.0
                    pose_msg.reference_frame = "camera_link"
                    
                    # 创建深度消息
                    depth_msg = VLAObjectDepth()
                    depth_msg.header.stamp = rospy.Time.now()
                    depth_msg.object_id = detection.object_id
                    depth_msg.object_class = detection.object_class
                    depth_msg.object_name = target_object
                    depth_msg.position = object_info_response.object_info.pose.position
                    depth_msg.size = object_info_response.object_info.size
                    depth_msg.distance = self.calculate_distance(object_info_response.object_info.pose.position)
                    depth_msg.depth_confidence = 0.8
                    depth_msg.depth_source = "point_cloud"
                    depth_msg.is_valid = True
                    depth_msg.point_density = 1.0
                    depth_msg.reference_frame = "camera_link"
                    
                    object_poses.append(pose_msg)
                    object_depths.append(depth_msg)
            
            # 创建VLM结果消息
            vlm_msg = VLAVLMResult()
            vlm_msg.header.stamp = rospy.Time.now()
            vlm_msg.task_id = str(int(time.time()))
            vlm_msg.instruction = instruction
            vlm_msg.task_type = 'location_query'
            vlm_msg.scene_description = f"找到{len(object_poses)}个{target_object}物体"
            vlm_msg.scene_type = 'location_query'
            vlm_msg.objects = [target_object]
            vlm_msg.confidence = 0.8 if object_poses else 0.0
            vlm_msg.processing_time = 0.0
            vlm_msg.object_poses = object_poses
            vlm_msg.object_depths = object_depths
            vlm_msg.include_pose = True
            vlm_msg.include_depth = True
            vlm_msg.include_grasp_points = False
            vlm_msg.is_grasping_task = False
            vlm_msg.is_location_query = True
            vlm_msg.target_object = target_object
            vlm_msg.target_object_class = target_object
            vlm_msg.target_object_found = len(object_poses) > 0
            vlm_msg.vlm_model = self.vlm_model_name
            vlm_msg.api_call_success = True
            
            # 发布结果
            self.vlm_result_pub.publish(vlm_msg)
            
            # 分别发布位姿和深度信息
            for pose_msg in object_poses:
                self.object_pose_pub.publish(pose_msg)
            
            for depth_msg in object_depths:
                self.object_depth_pub.publish(depth_msg)
            
            rospy.loginfo(f"Location query task completed, found {len(object_poses)} {target_object} objects")
            
        except Exception as e:
            rospy.logerr(f"Error handling location query task: {str(e)}")
    
    def process_continuous_data(self):
        """处理连续数据（当启用持续发布时）"""
        try:
            if not self.current_image or not self.current_point_cloud:
                return
            
            # 持续检测物体
            detection_response = self.detect_objects_client(
                image=self.current_image,
                camera_info=self.current_camera_info,
                confidence_threshold=self.confidence_threshold
            )
            
            if detection_response.success:
                for detection in detection_response.detections:
                    # 更新跟踪信息
                    self.update_object_tracking(detection)
                    
                    # 如果是当前目标物体，发布信息
                    if (self.current_target_object and 
                        detection.object_class in self.object_class_mapping.get(
                            self.current_target_object['name'], [])):
                        
                        self.publish_target_object_info(detection)
                        
        except Exception as e:
            rospy.logerr(f"Error processing continuous data: {str(e)}")
    
    def update_object_tracking(self, detection):
        """更新对象跟踪信息"""
        try:
            object_id = detection.object_id
            current_time = time.time()
            
            if object_id not in self.tracked_objects:
                # 新物体
                self.tracked_objects[object_id] = {
                    'object_class': detection.object_class,
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'detection_count': 1,
                    'pose_history': []
                }
            else:
                # 更新现有物体
                self.tracked_objects[object_id]['last_seen'] = current_time
                self.tracked_objects[object_id]['detection_count'] += 1
            
            # 清理超时的跟踪对象
            self.cleanup_old_tracks()
            
        except Exception as e:
            rospy.logerr(f"Error updating object tracking: {str(e)}")
    
    def cleanup_old_tracks(self):
        """清理旧的跟踪记录"""
        try:
            current_time = time.time()
            expired_objects = []
            
            for object_id, track_info in self.tracked_objects.items():
                if current_time - track_info['last_seen'] > self.tracking_timeout:
                    expired_objects.append(object_id)
            
            for object_id in expired_objects:
                del self.tracked_objects[object_id]
                rospy.logdebug(f"Removed expired track: {object_id}")
                
        except Exception as e:
            rospy.logerr(f"Error cleaning up old tracks: {str(e)}")
    
    def publish_target_object_info(self, detection):
        """发布目标物体信息"""
        try:
            # 获取物体详细信息
            object_info_response = self.get_object_info_client(
                object_id=detection.object_id,
                object_name=detection.object_class,
                detection=detection,
                image=self.current_image,
                point_cloud=self.current_point_cloud,
                info_level='detailed'
            )
            
            if object_info_response.success:
                # 发布位姿信息
                pose_msg = VLAObjectPose()
                pose_msg.header.stamp = rospy.Time.now()
                pose_msg.object_id = detection.object_id
                pose_msg.object_class = detection.object_class
                pose_msg.object_name = self.current_target_object['name']
                pose_msg.pose = object_info_response.object_info.pose
                pose_msg.confidence = detection.confidence
                pose_msg.detection_method = "continuous_detection"
                pose_msg.is_tracked = True
                pose_msg.tracking_quality = 1.0
                pose_msg.reference_frame = "camera_link"
                
                # 发布深度信息
                depth_msg = VLAObjectDepth()
                depth_msg.header.stamp = rospy.Time.now()
                depth_msg.object_id = detection.object_id
                depth_msg.object_class = detection.object_class
                depth_msg.object_name = self.current_target_object['name']
                depth_msg.position = object_info_response.object_info.pose.position
                depth_msg.size = object_info_response.object_info.size
                depth_msg.distance = self.calculate_distance(object_info_response.object_info.pose.position)
                depth_msg.depth_confidence = 0.8
                depth_msg.depth_source = "point_cloud"
                depth_msg.is_valid = True
                depth_msg.point_density = 1.0
                depth_msg.reference_frame = "camera_link"
                
                self.object_pose_pub.publish(pose_msg)
                self.object_depth_pub.publish(depth_msg)
                
        except Exception as e:
            rospy.logerr(f"Error publishing target object info: {str(e)}")
    
    def process_user_instruction(self, instruction: str) -> Dict[str, Any]:
        """处理用户指令的主入口"""
        try:
            rospy.loginfo(f"Processing user instruction: {instruction}")
            
            # 分析指令并决定调用策略
            analysis_result = self.analyze_task_and_call_modules(instruction)
            
            # 如果不需要视觉模块，直接返回
            if analysis_result['action'] == 'no_vision_needed':
                return {
                    'success': True,
                    'message': '指令不涉及视觉内容，可直接基于语言信息处理',
                    'action': 'proceed_with_language_only'
                }
            
            # 如果需要调用感知模块
            if analysis_result['action'] == 'call_perception':
                # 执行感知调用
                perception_result = self.handle_perception_call(analysis_result)
                
                if perception_result['success']:
                    return {
                        'success': True,
                        'message': '感知任务执行成功',
                        'perception_result': perception_result['result'],
                        'modules_used': perception_result['modules_used'],
                        'action': 'perception_completed'
                    }
                else:
                    return {
                        'success': False,
                        'error': perception_result['error'],
                        'action': 'perception_failed'
                    }
            
            # 错误情况
            if analysis_result['action'] == 'error':
                return {
                    'success': False,
                    'error': analysis_result['error'],
                    'action': 'analysis_failed'
                }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'action': 'processing_failed'
            }
    
    def run(self):
        """运行视觉调度器"""
        rospy.loginfo("VLA Vision Scheduler is running...")
        rospy.loginfo("Waiting for user instructions...")
        
        # 示例：处理一些测试指令
        test_instructions = [
            "找到桌子上的红色杯子",
            "检测房间里的人",
            "分析当前场景",
            "测量物体到相机的距离",
            "识别并定位所有椅子"
        ]
        
        for instruction in test_instructions:
            rospy.loginfo(f"\n=== 测试指令: {instruction} ===")
            result = self.process_user_instruction(instruction)
            rospy.loginfo(f"处理结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
            time.sleep(2)  # 等待处理完成
        
        rospy.spin()

if __name__ == '__main__':
    try:
        scheduler = VLAVisionScheduler()
        scheduler.run()
    except rospy.ROSInterruptException:
        pass
