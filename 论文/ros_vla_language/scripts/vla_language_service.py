#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VLA语言服务 - ROS服务版本
VLA Language Service - ROS Service Version

长时间对话控制流程服务
Long-term Dialogue Control Service

实时语音录制 -> LLM分析 -> JSON生成 -> TTS反馈
"""

import os
import sys
import time
import asyncio
import logging
import json
import numpy as np
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import queue
import signal
import wave
import select
import termios
import tty

# ROS相关导入
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger
from vla_language.msg import VLAIntent, VLAAction, VLACommand
from vla_language.srv import ProcessText, GetIntent, GenerateAction

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 尝试导入必要的包
try:
    import whisper
    WHISPER_AVAILABLE = True
    logger.info("✅ Whisper已加载")
except ImportError:
    WHISPER_AVAILABLE = False
    logger.error("❌ Whisper未安装，请运行: pip install openai-whisper")

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
    logger.info("✅ Edge TTS已加载")
except ImportError:
    EDGE_TTS_AVAILABLE = False
    logger.error("❌ Edge TTS未安装，请运行: pip install edge-tts")

try:
    import sounddevice as sd
    SOUND_DEVICE_AVAILABLE = True
    logger.info("✅ SoundDevice已加载")
except ImportError:
    SOUND_DEVICE_AVAILABLE = False
    logger.error("❌ SoundDevice未安装，请运行: pip install sounddevice")

try:
    import soundfile as sf
    SOUND_FILE_AVAILABLE = True
    logger.info("✅ SoundFile已加载")
except ImportError:
    SOUND_FILE_AVAILABLE = False
    logger.error("❌ SoundFile未安装，请运行: pip install soundfile")

try:
    import pygame
    PYGAME_AVAILABLE = True
    logger.info("✅ PyGame已加载")
except ImportError:
    PYGAME_AVAILABLE = False
    logger.error("❌ PyGame未安装，请运行: pip install pygame")

# 数据类定义
@dataclass
class RecognitionResult:
    """语音识别结果"""
    text: str
    confidence: float
    duration: float
    engine: str = "whisper"

@dataclass
class IntentResult:
    """意图识别结果"""
    intent: str
    confidence: float
    action: Dict[str, Any]
    processing_time: float

@dataclass
class TTSResult:
    """TTS生成结果"""
    audio_path: str
    duration: float
    text: str
    engine: str = "edge_tts"

class AudioRecorder:
    """音频录制器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sample_rate = config.get('sample_rate', 16000)
        self.channels = config.get('channels', 1)
        self.chunk_duration = config.get('chunk_duration', 0.5)
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        self.silence_threshold = config.get('silence_threshold', 0.01)
        self.silence_duration = config.get('silence_duration', 2.0)
        self.min_recording_duration = config.get('min_recording_duration', 1.0)
        
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.silence_counter = 0
        self.recording_start_time = 0
        self.last_sound_time = 0
        
        # 音频缓冲区
        self.audio_buffer = []
        self.silence_buffer = []
        
        logger.info(f"音频录制器初始化: {self.sample_rate}Hz, {self.channels}ch")
    
    def _detect_silence(self, audio_data: np.ndarray) -> bool:
        """检测静音"""
        if len(audio_data) == 0:
            return True
        
        # 计算音频能量
        energy = np.mean(np.abs(audio_data))
        return energy < self.silence_threshold
    
    def _audio_callback(self, indata, frames, time_info, status):
        """音频回调函数"""
        if status:
            logger.warning(f"音频回调状态: {status}")
        
        # 将音频数据放入队列
        self.audio_queue.put(indata.copy())
        
        # 检测静音
        if self._detect_silence(indata):
            self.silence_counter += 1
            self.silence_buffer.append(indata.copy())
        else:
            # 有声音，重置静音计数器
            if self.silence_counter > 0:
                self.audio_buffer.extend(self.silence_buffer)
                self.silence_buffer = []
            self.silence_counter = 0
            self.audio_buffer.append(indata.copy())
            self.last_sound_time = time.time()
    
    def start_recording(self):
        """开始录制"""
        if self.is_recording:
            logger.warning("已经在录制中")
            return
        
        self.is_recording = True
        self.audio_buffer = []
        self.silence_buffer = []
        self.silence_counter = 0
        self.recording_start_time = time.time()
        self.last_sound_time = time.time()
        
        logger.info("🎤 开始录制音频...")
        
        # 启动音频流
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self._audio_callback,
                blocksize=self.chunk_size,
                dtype=np.float32
            )
            self.stream.start()
        except Exception as e:
            logger.error(f"启动音频流失败: {str(e)}")
            self.is_recording = False
    
    def stop_recording(self) -> Optional[np.ndarray]:
        """停止录制并返回音频数据"""
        if not self.is_recording:
            logger.warning("没有在录制中")
            return None
        
        logger.info("🛑 停止录制音频")
        
        # 停止音频流
        try:
            self.stream.stop()
            self.stream.close()
        except Exception as e:
            logger.error(f"停止音频流失败: {str(e)}")
        
        self.is_recording = False
        
        # 等待剩余音频数据处理
        time.sleep(0.1)
        
        # 合并所有音频数据
        if self.audio_buffer:
            full_audio = np.concatenate(self.audio_buffer, axis=0)
            recording_duration = time.time() - self.recording_start_time
            logger.info(f"录制完成，时长: {recording_duration:.2f}秒，音频长度: {len(full_audio)}")
            return full_audio
        
        return None
    
    def should_stop_recording(self) -> bool:
        """判断是否应该停止录制"""
        if not self.is_recording:
            return False
        
        # 检查最小录制时间
        current_duration = time.time() - self.recording_start_time
        if current_duration < self.min_recording_duration:
            return False
        
        # 检查静音持续时间
        silence_duration = self.silence_counter * self.chunk_duration
        if silence_duration >= self.silence_duration:
            logger.info(f"检测到静音 {silence_duration:.1f}秒，停止录制")
            return True
        
        return False
    
    def get_recording_status(self) -> Dict[str, Any]:
        """获取录制状态"""
        return {
            'is_recording': self.is_recording,
            'duration': time.time() - self.recording_start_time if self.is_recording else 0,
            'silence_counter': self.silence_counter,
            'buffer_size': len(self.audio_buffer)
        }

class ProductionSpeechRecognizer:
    """正式版语音识别器 - 使用Whisper"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.sample_rate = config.get('sample_rate', 16000)
        self.model_name = config.get('model_name', 'base')
        
        if WHISPER_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """加载Whisper模型"""
        try:
            logger.info(f"加载Whisper模型: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            logger.info("✅ Whisper模型加载成功")
        except Exception as e:
            logger.error(f"❌ Whisper模型加载失败: {str(e)}")
            self.model = None
    
    def recognize_audio(self, audio_data: np.ndarray) -> Optional[RecognitionResult]:
        """识别音频"""
        if not WHISPER_AVAILABLE or self.model is None:
            logger.error("Whisper不可用")
            return None
        
        try:
            start_time = time.time()
            
            # 保存临时音频文件
            temp_path = "/tmp/temp_recording.wav"
            sf.write(temp_path, audio_data, self.sample_rate)
            
            logger.info("开始识别音频...")
            result = self.model.transcribe(
                temp_path,
                language='zh',
                fp16=False,
                verbose=False
            )
            
            recognized_text = result['text'].strip()
            duration = time.time() - start_time
            
            # 计算置信度
            confidence = min(1.0, len(recognized_text) / 20 * (1.0 / max(duration, 0.1)))
            
            logger.info(f"识别结果: '{recognized_text}'")
            logger.info(f"识别耗时: {duration:.2f}秒")
            
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return RecognitionResult(
                text=recognized_text,
                confidence=confidence,
                duration=duration,
                engine="whisper"
            )
            
        except Exception as e:
            logger.error(f"音频识别失败: {str(e)}")
            return None

class ProductionTTSGenerator:
    """正式版TTS生成器 - 使用Edge TTS"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.voice = config.get('voice', 'zh-CN-XiaoxiaoNeural')
        self.rate = config.get('rate', '+0%')
        self.volume = config.get('volume', '+0%')
        
        if not EDGE_TTS_AVAILABLE:
            logger.error("Edge TTS不可用")
    
    async def generate_audio(self, text: str, output_path: str) -> Optional[TTSResult]:
        """生成音频"""
        if not EDGE_TTS_AVAILABLE:
            logger.error("Edge TTS不可用")
            return None
        
        try:
            start_time = time.time()
            
            logger.info(f"开始生成TTS音频: '{text[:50]}...'")
            
            # 创建Edge TTS通信对象
            communicate = edge_tts.Communicate(
                text=text,
                voice=self.voice,
                rate=self.rate,
                volume=self.volume
            )
            
            # 生成音频
            await communicate.save(output_path)
            
            duration = time.time() - start_time
            
            # 验证文件
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                logger.info(f"✅ TTS生成成功: {output_path} ({file_size} bytes)")
                logger.info(f"生成耗时: {duration:.2f}秒")
                
                return TTSResult(
                    audio_path=output_path,
                    duration=duration,
                    text=text,
                    engine="edge_tts"
                )
            else:
                raise Exception("音频文件生成失败")
                
        except Exception as e:
            logger.error(f"TTS生成失败: {str(e)}")
            return None
    
    def generate_audio_sync(self, text: str, output_path: str) -> Optional[TTSResult]:
        """同步生成音频"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.generate_audio(text, output_path))
            loop.close()
            return result
        except Exception as e:
            logger.error(f"同步TTS生成失败: {str(e)}")
            return None

class ProductionLLMProcessor:
    """正式版LLM处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engine_type = config.get('engine_type', 'zhipuai')
        self.max_tokens = config.get('max_tokens', 100)
        self.temperature = config.get('temperature', 0.1)
        self.timeout = config.get('timeout', 5.0)
        
        # 智能缓存
        self.cache = {}
        self.cache_timeout = config.get('cache_timeout', 300)  # 5分钟
        
        logger.info(f"初始化LLM处理器: {self.engine_type}")
    
    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """检查缓存是否有效"""
        return (time.time() - cache_entry['timestamp']) < self.cache_timeout
    
    async def process_text_async(self, text: str) -> IntentResult:
        """异步处理文本"""
        start_time = time.time()
        
        # 检查缓存
        cache_key = self._get_cache_key(text)
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if self._is_cache_valid(cache_entry):
                logger.info(f"🎯 缓存命中: '{text[:30]}...'")
                self.cache_hits = getattr(self, 'cache_hits', 0) + 1
                return IntentResult(
                    intent=cache_entry['intent'],
                    confidence=cache_entry['confidence'],
                    action=cache_entry['action'],
                    processing_time=time.time() - start_time
                )
        
        try:
            # 根据文本内容确定意图
            intent, confidence, action = self._classify_intent(text)
            
            # 更新缓存
            self.cache[cache_key] = {
                'intent': intent,
                'confidence': confidence,
                'action': action,
                'timestamp': time.time()
            }
            
            processing_time = time.time() - start_time
            
            logger.info(f"LLM处理结果: {intent}, 置信度: {confidence:.2f}, 耗时: {processing_time:.2f}秒")
            
            return IntentResult(
                intent=intent,
                confidence=confidence,
                action=action,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"LLM处理失败: {str(e)}")
            return IntentResult(
                intent="unknown",
                confidence=0.0,
                action={},
                processing_time=time.time() - start_time
            )
    
    def _classify_intent(self, text: str) -> Tuple[str, float, Dict[str, Any]]:
        """分类意图 - 使用JSON配置文件"""
        text = text.lower().strip()
        
        # 加载意图配置
        intent_config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'intent_patterns.json')
        intent_config = {}
        
        try:
            if os.path.exists(intent_config_path):
                with open(intent_config_path, 'r', encoding='utf-8') as f:
                    intent_config = json.load(f)
        except Exception as e:
            logger.error(f"加载意图配置文件失败: {str(e)}")
        
        best_intent = "unknown"
        best_confidence = 0.0
        best_action = {}
        
        # 遍历所有意图
        for intent_key, intent_data in intent_config.get('intents', {}).items():
            patterns = intent_data.get('patterns', [])
            confidence_threshold = intent_data.get('confidence_threshold', 0.0)
            
            # 计算匹配度
            match_count = sum(1 for pattern in patterns if pattern in text)
            confidence = match_count / len(patterns) if patterns else 0.0
            
            # 检查是否超过阈值
            if confidence > best_confidence and confidence >= confidence_threshold:
                best_confidence = confidence
                best_intent = intent_key
                
                # 获取意图配置
                response_text = intent_data.get('response', '')
                action_config = intent_data.get('action', {})
                json_response_config = intent_data.get('json_response', {})
                
                # 更新JSON响应中的时间戳
                json_response_config['timestamp'] = time.time()
                json_response_config['confidence'] = confidence
                
                # 构建动作
                best_action = {
                    'type': action_config.get('type', 'response'),
                    'text': response_text,
                    'json_response': json_response_config
                }
                
                # 添加其他动作参数
                for key, value in action_config.items():
                    if key != 'type':
                        best_action[key] = value
        
        # 如果没有匹配到任何意图，使用默认的unknown意图
        if best_confidence == 0.0:
            unknown_intent = intent_config.get('intents', {}).get('unknown', {})
            best_action = {
                'type': 'response',
                'text': unknown_intent.get('response', '抱歉，我没有理解您的请求。请尝试用更清晰的语言表达。'),
                'json_response': unknown_intent.get('json_response', {})
            }
            best_action['json_response']['timestamp'] = time.time()
            best_action['json_response']['confidence'] = 0.0
        
        return best_intent, best_confidence, best_action

class VLALanguageService(Node):
    """VLA语言服务节点"""
    
    def __init__(self):
        super().__init__('vla_language_service')
        
        # 配置参数
        self.declare_parameter('audio.sample_rate', 16000)
        self.declare_parameter('audio.channels', 1)
        self.declare_parameter('audio.chunk_duration', 0.5)
        self.declare_parameter('audio.silence_threshold', 0.01)
        self.declare_parameter('audio.silence_duration', 2.0)
        self.declare_parameter('audio.min_recording_duration', 1.0)
        self.declare_parameter('speech.model_name', 'base')
        self.declare_parameter('tts.voice', 'zh-CN-XiaoxiaoNeural')
        self.declare_parameter('tts.rate', '+0%')
        self.declare_parameter('tts.volume', '+0%')
        self.declare_parameter('llm.engine_type', 'zhipuai')
        self.declare_parameter('llm.max_tokens', 100)
        self.declare_parameter('llm.temperature', 0.1)
        self.declare_parameter('llm.timeout', 5.0)
        self.declare_parameter('llm.cache_timeout', 300)
        
        # 获取配置
        config = self._get_config()
        
        # 初始化组件
        self.recorder = AudioRecorder(config.get('audio', {}))
        self.speech_recognizer = ProductionSpeechRecognizer(config.get('speech', {}))
        self.tts_generator = ProductionTTSGenerator(config.get('tts', {}))
        self.llm_processor = ProductionLLMProcessor(config.get('llm', {}))
        
        # 创建音频目录
        self.audio_dir = os.path.join(os.path.dirname(__file__), 'audio')
        if not os.path.exists(self.audio_dir):
            os.makedirs(self.audio_dir)
        
        # 创建服务
        self.create_service(ProcessText, self.process_text_callback)
        self.create_service(GetIntent, self.get_intent_callback)
        self.create_service(GenerateAction, self.generate_action_callback)
        self.create_service(Trigger, self.start_recording_callback)
        self.create_service(Trigger, self.stop_recording_callback)
        
        # 创建发布者
        self.intent_publisher = self.create_publisher(VLAIntent, 'vla_intent', 10)
        self.action_publisher = self.create_publisher(VLAAction, 'vla_action', 10)
        self.command_publisher = self.create_publisher(VLACommand, 'vla_command', 10)
        self.latest_intent_publisher = self.create_publisher(String, '/vla_language/latest_intent', 10)
        
        # 创建计时器
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # 系统状态
        self.is_recording = False
        self.current_intent = None
        self.current_action = None
        
        logger.info("✅ VLA语言服务已启动")
    
    def _get_config(self) -> Dict[str, Any]:
        """获取配置"""
        return {
            'audio': {
                'sample_rate': self.get_parameter('audio.sample_rate').value,
                'channels': self.get_parameter('audio.channels').value,
                'chunk_duration': self.get_parameter('audio.chunk_duration').value,
                'silence_threshold': self.get_parameter('audio.silence_threshold').value,
                'silence_duration': self.get_parameter('audio.silence_duration').value,
                'min_recording_duration': self.get_parameter('audio.min_recording_duration').value
            },
            'speech': {
                'model_name': self.get_parameter('speech.model_name').value
            },
            'tts': {
                'voice': self.get_parameter('tts.voice').value,
                'rate': self.get_parameter('tts.rate').value,
                'volume': self.get_parameter('tts.volume').value
            },
            'llm': {
                'engine_type': self.get_parameter('llm.engine_type').value,
                'max_tokens': self.get_parameter('llm.max_tokens').value,
                'temperature': self.get_parameter('llm.temperature').value,
                'timeout': self.get_parameter('llm.timeout').value,
                'cache_timeout': self.get_parameter('llm.cache_timeout').value
            }
        }
    
    def process_text_callback(self, request, response):
        """处理文本服务回调"""
        try:
            start_time = time.time()
            
            # 处理文本
            intent_result = asyncio.run(self.llm_processor.process_text_async(request.text))
            
            # 创建响应
            response.success = True
            response.intent = intent_result.intent
            response.confidence = intent_result.confidence
            response.processing_time = intent_result.processing_time
            
            # 生成JSON响应
            json_response = intent_result.action.get('json_response', {})
            json_response['timestamp'] = time.time()
            json_response['confidence'] = intent_result.confidence
            json_response['original_text'] = request.text
            
            # 保存到文件
            json_path = os.path.join(self.audio_dir, "latest_response.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_response, f, ensure_ascii=False, indent=2)
            
            # 发布意图消息
            intent_msg = VLAIntent()
            intent_msg.intent = intent_result.intent
            intent_msg.confidence = intent_result.confidence
            intent_msg.timestamp = time.time()
            self.intent_publisher.publish(intent_msg)
            
            # 发布动作消息
            if intent_result.action:
                action_msg = VLAAction()
                action_msg.type = intent_result.action.get('type', 'response')
                action_msg.text = intent_result.action.get('text', '')
                action_msg.json_data = json.dumps(intent_result.action.get('json_response', {}))
                self.action_publisher.publish(action_msg)
            
            # 发布最新的意图数据到话题
            latest_intent_msg = String()
            latest_intent_data = {
                'intent': intent_result.intent,
                'confidence': intent_result.confidence,
                'response': json_response.get('response', ''),
                'instruction': json_response.get('instruction', 'none'),
                'original_text': request.text,
                'timestamp': time.time()
            }
            latest_intent_msg.data = json.dumps(latest_intent_data, ensure_ascii=False)
            self.latest_intent_publisher.publish(latest_intent_msg)
            
            logger.info(f"文本处理完成: {intent_result.intent} (置信度: {intent_result.confidence:.2f})")
            
        except Exception as e:
            self.get_logger().error(f"文本处理失败: {str(e)}")
            response.success = False
            response.message = str(e)
        
        return response
    
    def get_intent_callback(self, request, response):
        """获取意图服务回调"""
        try:
            # 从文件读取最新的意图
            json_path = os.path.join(self.audio_dir, "latest_response.json")
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                response.success = True
                response.intent = json_data.get('intent', 'unknown')
                response.confidence = json_data.get('confidence', 0.0)
                response.timestamp = json_data.get('timestamp', 0.0)
            else:
                response.success = False
                response.message = "没有找到意图数据"
                
        except Exception as e:
            self.get_logger().error(f"获取意图失败: {str(e)}")
            response.success = False
            response.message = str(e)
        
        return response
    
    def generate_action_callback(self, request, response):
        """生成动作服务回调"""
        try:
            # 处理文本生成动作
            intent_result = asyncio.run(self.llm_processor.process_text_async(request.text))
            
            response.success = True
            response.action_type = intent_result.action.get('type', 'response')
            response.action_text = intent_result.action.get('text', '')
            response.json_data = json.dumps(intent_result.action.get('json_response', {}))
            
            # 发布命令消息
            command_msg = VLACommand()
            command_msg.type = intent_result.action.get('type', 'response')
            command_msg.text = intent_result.action.get('text', '')
            command_msg.json_data = json.dumps(intent_result.action.get('json_response', {}))
            self.command_publisher.publish(command_msg)
            
            logger.info(f"动作生成完成: {intent_result.action.get('type', 'response')}")
            
        except Exception as e:
            self.get_logger().error(f"生成动作失败: {str(e)}")
            response.success = False
            response.message = str(e)
        
        return response
    
    def start_recording_callback(self, request, response):
        """开始录制服务回调"""
        try:
            if not self.is_recording:
                self.recorder.start_recording()
                self.is_recording = True
                response.success = True
                response.message = "开始录制音频"
                logger.info("🎤 开始录制音频")
            else:
                response.success = False
                response.message = "已经在录制中"
                
        except Exception as e:
            self.get_logger().error(f"开始录制失败: {str(e)}")
            response.success = False
            response.message = str(e)
        
        return response
    
    def stop_recording_callback(self, request, response):
        """停止录制服务回调"""
        try:
            if self.is_recording:
                audio_data = self.recorder.stop_recording()
                self.is_recording = False
                
                if audio_data is not None:
                    # 处理音频
                    self._process_audio(audio_data)
                    response.success = True
                    response.message = "录制完成并处理"
                else:
                    response.success = False
                    response.message = "录制失败"
            else:
                response.success = False
                response.message = "没有在录制中"
                
        except Exception as e:
            self.get_logger().error(f"停止录制失败: {str(e)}")
            response.success = False
            response.message = str(e)
        
        return response
    
    def _process_audio(self, audio_data: np.ndarray):
        """处理音频"""
        try:
            # 1. 语音识别
            recognition_result = self.speech_recognizer.recognize_audio(audio_data)
            if not recognition_result:
                logger.error("语音识别失败")
                return
            
            # 2. LLM处理
            llm_result = asyncio.run(self.llm_processor.process_text_async(recognition_result.text))
            
            # 3. 生成JSON响应
            json_response = llm_result.action.get('json_response', {})
            json_response['recognition'] = {
                'text': recognition_result.text,
                'confidence': recognition_result.confidence,
                'duration': recognition_result.duration,
                'engine': recognition_result.engine
            }
            json_response['original_text'] = recognition_result.text
            
            # 只保存一个最新的JSON文件
            json_path = os.path.join(self.audio_dir, "latest_response.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_response, f, ensure_ascii=False, indent=2)
            
            logger.info(f"JSON响应已保存: {json_path}")
            
            # 发布意图消息
            intent_msg = VLAIntent()
            intent_msg.intent = llm_result.intent
            intent_msg.confidence = llm_result.confidence
            intent_msg.timestamp = time.time()
            self.intent_publisher.publish(intent_msg)
            
            # 发布动作消息
            if llm_result.action:
                action_msg = VLAAction()
                action_msg.type = llm_result.action.get('type', 'response')
                action_msg.text = llm_result.action.get('text', '')
                action_msg.json_data = json.dumps(llm_result.action.get('json_response', {}))
                self.action_publisher.publish(action_msg)
            
            # 发布最新的意图数据到话题
            latest_intent_msg = String()
            latest_intent_data = {
                'intent': llm_result.intent,
                'confidence': llm_result.confidence,
                'response': json_response.get('response', ''),
                'instruction': json_response.get('instruction', 'none'),
                'original_text': recognition_result.text,
                'timestamp': time.time()
            }
            latest_intent_msg.data = json.dumps(latest_intent_data, ensure_ascii=False)
            self.latest_intent_publisher.publish(latest_intent_msg)
            
            # 4. 生成TTS响应
            if llm_result.action.get('type') == 'response':
                tts_output_path = os.path.join(self.audio_dir, f"tts_{int(time.time())}.wav")
                tts_result = self.tts_generator.generate_audio_sync(
                    llm_result.action['text'], 
                    tts_output_path
                )
                
                if tts_result:
                    logger.info(f"TTS音频已生成: {tts_output_path}")
                    # 播放音频
                    self._play_audio(tts_output_path)
            
            logger.info(f"处理完成: {llm_result.intent} (置信度: {llm_result.confidence:.2f})")
            
        except Exception as e:
            logger.error(f"音频处理失败: {str(e)}")
    
    def _play_audio(self, audio_path: str):
        """播放音频"""
        try:
            if not PYGAME_AVAILABLE:
                logger.warning("PyGame未安装，无法播放音频")
                return
            
            pygame.mixer.init()
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        except Exception as e:
            logger.error(f"播放音频失败: {str(e)}")
    
    def timer_callback(self):
        """定时器回调"""
        if self.is_recording:
            # 检查是否应该停止录制
            if self.recorder.should_stop_recording():
                audio_data = self.recorder.stop_recording()
                self.is_recording = False
                
                if audio_data is not None:
                    # 处理音频
                    self._process_audio(audio_data)

def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    # 检查依赖
    if not WHISPER_AVAILABLE:
        print("❌ Whisper不可用，请安装: pip install openai-whisper")
        return
    
    if not EDGE_TTS_AVAILABLE:
        print("❌ Edge TTS不可用，请安装: pip install edge-tts")
        return
    
    if not SOUND_DEVICE_AVAILABLE:
        print("❌ SoundDevice不可用，请安装: pip install sounddevice")
        return
    
    if not SOUND_FILE_AVAILABLE:
        print("❌ SoundFile不可用，请安装: pip install soundfile")
        return
    
    # 创建服务节点
    service_node = VLALanguageService()
    
    try:
        rclpy.spin(service_node)
    except KeyboardInterrupt:
        print("用户中断，正在关闭服务...")
    finally:
        service_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
