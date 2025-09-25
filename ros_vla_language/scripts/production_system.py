#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
正式使用版语音识别+LLM+TTS系统
Production Speech Recognition + LLM + TTS System

实时语音录制 -> LLM分析 -> JSON生成 -> TTS反馈
支持ROS话题发布、开机自启动、持续对话、内存TTS流式播放
"""

import os
import sys
import time
import asyncio
import logging
import json
import numpy as np
import threading
import io
import wave
import subprocess
import tempfile
import argparse
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import queue
import signal
import select
import termios
import tty
import rospy
import std_msgs.msg
from std_msgs.msg import String, Bool
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Twist
from std_msgs.msg import UInt8MultiArray
from ros_vla_language.msg import VLACommand
import sounddevice as sd
import soundfile as sf
import edge_tts
import whisper

# ROS节点初始化
print("🚀 正在初始化ROS节点...")
try:
    rospy.init_node('vla_language_system', anonymous=True)
    print("✅ ROS节点初始化成功")
except Exception as e:
    print(f"❌ ROS节点初始化失败: {e}")
    sys.exit(1)

# 配置日志
print("📝 正在配置日志...")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
print("✅ 日志配置完成")

# ROS话题发布器
class ROSPublisher:
    """ROS话题发布器"""
    
    def __init__(self):
        # 语音识别结果发布器
        self.recognition_pub = rospy.Publisher('/vla/recognition_result', String, queue_size=10)
        
        # VLA指令发布器
        self.command_pub = rospy.Publisher('/vla_control/command', VLACommand, queue_size=10)
        
        logger.info("✅ ROS话题发布器初始化完成（使用VLACommand消息类型）")
        
    def publish_recognition(self, text: str, confidence: float):
        """发布语音识别结果"""
        msg = String()
        # 确保JSON使用ensure_ascii=False来正确显示中文
        msg.data = json.dumps({
            'text': text,
            'confidence': confidence,
            'timestamp': time.time()
        }, ensure_ascii=False)
        self.recognition_pub.publish(msg)
        logger.info(f"📢 发布语音识别结果: {text}")
    
    def publish_command(self, intent: str, confidence: float, action: Dict):
        """发布VLA指令"""
        # 创建VLACommand消息
        msg = VLACommand()
        
        # 检查是否为指令类型（需要执行机器人动作）
        # 修复：使用action['type']字段来判断是否为命令类型
        action_type = action.get('type', 'response')
        response_type = action.get('response_type', 'command' if action_type in ['wave', 'welcome'] else 'conversation')
        
        # 使用action_type来判断是否为命令类型
        if action_type in ['wave', 'welcome'] and intent in ['wave', 'welcome', 'stop']:
            # 映射intent到instruction
            instruction_mapping = {
                'wave': 'wave',
                'welcome': 'welcome', 
                'stop': 'none',
                'unknown': 'none'
            }
            
            instruction = instruction_mapping.get(intent, 'none')
            
            # 设置消息字段
            msg.instruction = instruction
            
            # 发布消息
            self.command_pub.publish(msg)
            print(f"🤖 发布机器人指令: {intent} -> {instruction}")
        else:
            # 对话类型，不发布指令
            logger.debug(f"对话模式，不发布机器人指令: {intent}")
            # 仍然创建消息但不发布（保持接口一致性）
            msg.instruction = 'none'

# 内存TTS播放器
class MemoryTTSPlayer:
    """内存TTS播放器 - 流式播放音频在内存中生成和播放"""
    
    def __init__(self, publisher: ROSPublisher, tts_config: Dict[str, Any] = None):
        self.publisher = publisher
        self.is_playing = False
        self.tts_config = tts_config or {}
        self.output_device = self.tts_config.get('output_device', 'hw:1,0')  # 必须使用USB音频设备
        self.voice = self.tts_config.get('voice', 'zh-CN-XiaoxiaoNeural')
        self.rate = self.tts_config.get('rate', '+0%')
        self.volume = self.tts_config.get('volume', '+0%')
        
        # 多线程和中断控制
        self.playback_process = None
        self.interrupt_flag = False
        self.interrupt_lock = threading.Lock()
        self.asr_thread = None
        self.asr_active = False
        
        # 记录设备信息
        device_info = f"默认设备" if self.output_device == 'default' else f"设备 {self.output_device}"
        logger.info(f"✅ 内存TTS播放器初始化完成，输出设备: {device_info}")
    
    async def generate_and_play_streaming(self, text: str, voice: str = None):
        """异步版本的流式生成和播放TTS音频（性能优化版）"""
        # 移除文本长度限制，这可能导致性能问题
        # logger.info(f"🎵 TTS处理文本长度: {len(text)} 字符")
        
        # 在文本开头添加一个短停顿，防止第一个字被吞掉
        padded_text = "." + text if text else text
        logger.info(f"🎵 开始流式生成TTS音频: '{padded_text[:30]}...'")
        
        # 使用配置的语音或传入的语音
        tts_voice = voice or self.voice
        
        # 记录生成开始时间
        generation_start_time = time.time()
        
        # 创建Edge TTS通信对象 - 平衡质量和性能
        communicate = edge_tts.Communicate(
            text=padded_text,
            voice=tts_voice,
            rate=self.rate,  # 使用配置的正常语速
            volume=self.volume  # 使用配置的正常音量
        )
        
        # 真正的流式TTS - 边生成边播放
        first_chunk_time = None
        total_chunks = 0
        total_audio_size = 0
        
        # 创建音频播放队列
        import asyncio
        audio_queue = asyncio.Queue()
        playback_task = None
        
        async def audio_player():
            """音频播放器任务"""
            try:
                while True:
                    chunk = await audio_queue.get()
                    if chunk is None:  # 结束信号
                        break
                    # 播放这个音频块
                    await self._play_audio_chunk_async(chunk)
                    audio_queue.task_done()
            except Exception as e:
                logger.error(f"音频播放失败: {e}")
        
        # 启动播放任务
        playback_task = asyncio.create_task(audio_player())
        
        try:
            # 流式接收并立即播放音频数据
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    # 记录第一个chunk的时间
                    if first_chunk_time is None:
                        first_chunk_time = time.time() - generation_start_time
                        print(f"🎵 [流式TTS] 首个音频块生成耗时: {first_chunk_time:.3f}秒")
                    
                    # 立即发送到播放队列
                    await audio_queue.put(chunk["data"])
                    total_chunks += 1
                    total_audio_size += len(chunk["data"])
                    
                    print(f"🎵 [流式TTS] 已播放 {total_chunks} 个音频块")
            
            # 发送结束信号
            await audio_queue.put(None)
            
            # 等待播放完成
            if playback_task:
                await playback_task
                
            generation_time = time.time() - generation_start_time
            print(f"🎵 [流式TTS] 完成 - 总耗时: {generation_time:.3f}秒, 音频大小: {total_audio_size} 字节, 分块数: {total_chunks}")
            
            if first_chunk_time:
                print(f"🎵 [流式TTS] 用户等待时间: {first_chunk_time:.3f}秒 (而不是{generation_time:.3f}秒)")
            
            return generation_time
            
        except Exception as e:
            logger.error(f"流式TTS失败: {e}")
            # 确保播放任务被取消
            if playback_task:
                playback_task.cancel()
            return generation_time
    
    async def _play_audio_chunk_async(self, audio_chunk):
        """异步播放单个音频块"""
        # 使用线程池来同步播放音频块，避免阻塞事件循环
        import io
        import tempfile
        
        try:
            # 将音频块写入临时文件
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_chunk)
                temp_file_path = temp_file.name
            
            # 使用ffplay播放这个音频块
            subprocess.run([
                'ffplay', 
                '-nodisp',          # 不显示视频窗口
                '-autoexit',         # 播放完成后自动退出
                '-loglevel', 'quiet',  # 静音模式
                temp_file_path
            ], check=True, capture_output=True)
            
            # 删除临时文件
            os.unlink(temp_file_path)
            
        except Exception as e:
            logger.error(f"音频块播放失败: {e}")
    
    async def _play_audio_data_async(self, audio_data):
        """异步播放音频数据"""
        # 使用线程池来同步播放，避免阻塞事件循环
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._play_audio_data_sync, audio_data)
    
    def generate_and_play_streaming_sync(self, text: str, voice: str = None):
        """同步版本的流式生成和播放TTS音频 - 支持中断（极致性能版）"""
        import threading
        import sys
        import select
        
        # 移除文本长度限制，避免性能问题
        # print(f"🎵 [TTS调试] 处理文本长度: {len(text)} 字符")
        
        def run_async_in_thread():
            """在线程中运行异步函数"""
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._generate_audio_data(text, voice))
            finally:
                loop.close()
        
        # 记录总开始时间
        total_start_time = time.time()
        
        # 在线程中生成音频数据
        audio_data = run_async_in_thread()
        
        if not audio_data:
            logger.warning("⚠️ TTS生成失败，没有音频数据")
            return None
        
        # 计算生成时间
        generation_time = time.time() - total_start_time
        print(f"🎵 TTS生成耗时: {generation_time:.3f}秒")
        
        # 在单独线程中播放音频，支持键盘中断
        def play_with_keyboard_interrupt():
            try:
                self._play_audio_data_sync(audio_data)
            except KeyboardInterrupt:
                print("\n🛑 键盘中断检测到，停止播放")
                self.interrupt_playback()
        
        play_thread = threading.Thread(target=play_with_keyboard_interrupt)
        play_thread.daemon = True
        play_thread.start()
        
        print("💡 TTS播放中，按 Ctrl+C 或空格键可以中断播放...")
        
        # 等待播放完成或用户中断
        while play_thread.is_alive():
            # 检查键盘输入
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1)
                if key == ' ':  # 空格键中断
                    print("\n🛑 空格键中断检测到，停止播放")
                    self.interrupt_playback()
                    break
                elif key == '\x03':  # Ctrl+C
                    print("\n🛑 Ctrl+C中断检测到，停止播放")
                    self.interrupt_playback()
                    break
            
            play_thread.join(timeout=0.1)
        
        play_thread.join(timeout=1.0)
        
        # 返回生成时间（不包括播放时间）
        return generation_time
    
    async def _generate_audio_data(self, text: str, voice: str = None):
        """异步生成音频数据（极致性能版）"""
        # 在文本开头添加一个短停顿，防止第一个字被吞掉
        padded_text = "." + text if text else text
        logger.info(f"🎵 开始生成TTS音频: '{padded_text[:30]}...'")
        
        # 使用配置的语音或传入的语音
        tts_voice = voice or self.voice
        
        # 创建Edge TTS通信对象 - 平衡质量和性能
        communicate = edge_tts.Communicate(
            text=padded_text,
            voice=tts_voice,
            rate=self.rate,  # 使用配置的正常语速
            volume=self.volume  # 使用配置的正常音量
        )
        
        # 收集所有音频数据
        audio_data = bytearray()
        chunk_count = 0
        
        # 流式接收音频数据
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])
                chunk_count += 1
        
        logger.info(f"🎵 TTS音频生成完成，大小: {len(audio_data)} 字节，分块: {chunk_count}")
        return audio_data
    
    def _play_audio_data_sync(self, audio_data):
        """同步播放音频数据（支持中断）"""
        try:
            # 直接使用ffplay通过管道播放，不创建临时文件
            # 设置环境变量指定音频设备，尝试使用PulseAudio避免冲突
            env = os.environ.copy()
            
            # 优先尝试PulseAudio，如果没有则回退到ALSA
            if 'PULSE_SERVER' in os.environ:
                env['SDL_AUDIODRIVER'] = 'pulse'
                print("🔊 使用PulseAudio音频驱动")
            else:
                env['SDL_AUDIODRIVER'] = 'alsa'
                print("🔊 使用ALSA音频驱动")
            
            # 使用配置的输出设备
            if self.output_device != 'default':
                if env['SDL_AUDIODRIVER'] == 'alsa':
                    env['AUDIODEV'] = self.output_device
                logger.info(f"使用指定的输出设备: {self.output_device}")
            else:
                logger.info("使用默认输出设备")
            
            # 使用子进程和管道播放音频，带重试机制
            max_retries = 3
            retry_delay = 0.5
            
            for attempt in range(max_retries):
                try:
                    print(f"🔊 正在使用设备 {self.output_device} 播放音频 (尝试 {attempt + 1}/{max_retries})...")
                    self.is_playing = True
                    self.playback_process = subprocess.Popen(
                        ['ffplay', '-autoexit', '-nodisp', '-f', 'mp3', '-ar', '24000', '-ac', '1', '-i', '-'],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        env=env
                    )
                    
                    # 将音频数据写入管道
                    stdout, stderr = self.playback_process.communicate(input=audio_data)
                    
                    if self.playback_process.returncode == 0:
                        logger.info(f"✅ TTS音频播放完成 (使用输出设备: {self.output_device})")
                        break
                    else:
                        error_msg = stderr.decode('utf-8')
                        if "Device or resource busy" in error_msg and attempt < max_retries - 1:
                            print(f"⚠️ 音频设备忙，等待 {retry_delay} 秒后重试...")
                            time.sleep(retry_delay)
                            continue
                        else:
                            logger.error(f"❌ 音频设备播放失败: {error_msg}")
                            break
                            
                except subprocess.TimeoutExpired:
                    print(f"⚠️ 播放超时，尝试 {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        logger.error("❌ 音频播放超时")
                except Exception as e:
                    print(f"⚠️ 播放异常: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"❌ 音频播放异常: {str(e)}")
                finally:
                    # 重置播放状态
                    self.is_playing = False
                    self.playback_process = None
            
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.error(f"❌ 音频设备播放异常: {str(e)}")
            self.is_playing = False
            self.playback_process = None
            
        except Exception as e:
            logger.error(f"❌ TTS播放失败: {str(e)}")
            self.is_playing = False
            self.playback_process = None
            
            # 直接使用ffplay通过管道播放，不创建临时文件
            try:
                # 设置环境变量指定音频设备
                env = os.environ.copy()
                env['SDL_AUDIODRIVER'] = 'alsa'
                
                # 使用配置的输出设备
                if self.output_device != 'default':
                    env['AUDIODEV'] = self.output_device
                    logger.info(f"使用指定的输出设备: {self.output_device}")
                else:
                    logger.info("使用默认输出设备")
                
                # 使用子进程和管道播放音频，带重试机制
                max_retries = 3
                retry_delay = 0.5
                
                for attempt in range(max_retries):
                    try:
                        print(f"🔊 正在使用设备 {self.output_device} 播放音频 (尝试 {attempt + 1}/{max_retries})...")
                        self.is_playing = True
                        self.playback_process = subprocess.Popen(
                            ['ffplay', '-autoexit', '-nodisp', '-f', 'mp3', '-ar', '24000', '-ac', '1', '-i', '-'],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            env=env
                        )
                        
                        # 将音频数据写入管道（移除超时设置）
                        stdout, stderr = self.playback_process.communicate(input=audio_data)
                        
                        if self.playback_process.returncode == 0:
                            logger.info(f"✅ TTS音频播放完成 (使用输出设备: {self.output_device})")
                            break
                        else:
                            error_msg = stderr.decode('utf-8')
                            if "Device or resource busy" in error_msg and attempt < max_retries - 1:
                                print(f"⚠️ 音频设备忙，等待 {retry_delay} 秒后重试...")
                                time.sleep(retry_delay)
                                continue
                            else:
                                logger.error(f"❌ 音频设备播放失败: {error_msg}")
                                break
                                
                    except subprocess.TimeoutExpired:
                        print(f"⚠️ 播放超时，尝试 {attempt + 1}/{max_retries}")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                        else:
                            logger.error("❌ 音频播放超时")
                    except Exception as e:
                        print(f"⚠️ 播放异常: {str(e)}")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                        else:
                            logger.error(f"❌ 音频播放异常: {str(e)}")
                    finally:
                        # 重置播放状态
                        self.is_playing = False
                        self.playback_process = None
                    
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
                logger.error(f"❌ 音频设备播放异常: {str(e)}")
                self.is_playing = False
                self.playback_process = None
                
        except Exception as e:
            logger.error(f"❌ TTS流式生成和播放失败: {str(e)}")
            self.is_playing = False
            self.playback_process = None
    
            
    def _play_audio(self, audio_data: np.ndarray):
        """播放音频（保留原方法以兼容）"""
        return self._play_audio_directly(audio_data)
    
    def is_playing_audio(self) -> bool:
        """检查是否正在播放音频"""
        return self.is_playing
    
    def interrupt_playback(self):
        """中断音频播放"""
        with self.interrupt_lock:
            if self.is_playing and self.playback_process:
                print(f"🛑 [中断] 终止TTS播放...")
                try:
                    self.playback_process.terminate()
                    self.playback_process.wait(timeout=0.5)
                except:
                    try:
                        self.playback_process.kill()
                    except:
                        pass
                finally:
                    self.is_playing = False
                    self.playback_process = None
                    print(f"✅ [中断] TTS播放已终止")
    
    def start_asr_during_tts(self, speech_recognizer, callback):
        """在TTS播放期间启动ASR线程 - 使用独立音频设备"""
        if self.asr_thread and self.asr_thread.is_alive():
            print("⚠️ ASR监听线程已在运行")
            return
        
        self.asr_active = True
        self.asr_thread = threading.Thread(target=self._asr_during_tts_worker_simple, args=(speech_recognizer, callback))
        self.asr_thread.daemon = True
        self.asr_thread.start()
        print("🎤 启动TTS期间的ASR监听线程（使用独立设备）")
        logger.info("🎤 启动TTS期间的ASR监听线程")
        time.sleep(0.2)
    
    def stop_asr_during_tts(self):
        """停止TTS期间的ASR监听"""
        self.asr_active = False
        if self.asr_thread and self.asr_thread.is_alive():
            self.asr_thread.join(timeout=1.0)
        logger.info("🛑 停止TTS期间的ASR监听线程")
    
    def _asr_during_tts_worker_simple(self, speech_recognizer, callback):
        """ASR工作线程 - 简化版本，避免复杂的音频设备冲突"""
        import time
        
        try:
            print("🎤 [ASR监听] 启动简化版ASR监听（仅键盘检测）")
            print("💡 [ASR监听] TTS播放期间，按空格键可以中断播放")
            
            # 等待TTS开始播放
            while self.asr_active and not self.is_playing:
                time.sleep(0.1)
            
            print("🔍 [ASR监听] TTS播放中，等待用户中断...")
            
            # 简单的键盘检测，避免音频设备冲突
            import sys
            import select
            
            while self.asr_active and self.is_playing:
                try:
                    # 检查键盘输入
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1)
                        if key == ' ':  # 空格键中断
                            print("🎯 [ASR监听] 检测到空格键，模拟唤醒词中断")
                            callback("夸父 中断")
                            break
                    
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"❌ [ASR监听] 键盘检测异常: {e}")
                    break
                    
        except Exception as e:
            print(f"❌ [ASR监听] 简化监听异常: {e}")
        finally:
            print("🛑 [ASR监听] 线程结束")
    
    def _asr_during_tts_worker(self, speech_recognizer, callback):
        """ASR工作线程 - 在TTS播放期间监听唤醒词（简化版本）"""
        import sounddevice as sd
        import numpy as np
        import time
        
        # 简化配置
        sample_rate = 48000
        channels = 1
        chunk_duration = 1.0  # 1秒 chunks
        chunk_size = int(sample_rate * chunk_duration)
        
        try:
            print("🎤 [ASR监听] 启动简化版ASR监听线程")
            
            # 直接使用默认设备，避免设备选择问题
            with sd.InputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype=np.float32,
                blocksize=chunk_size
            ) as stream:
                print("✅ [ASR监听] 音频流打开成功")
                
                # 等待TTS开始播放
                while self.asr_active and not self.is_playing:
                    time.sleep(0.1)
                
                print("🔍 [ASR监听] 开始监听唤醒词...")
                
                while self.asr_active and self.is_playing:
                    try:
                        # 读取1秒音频数据
                        audio_data, overflowed = stream.read(chunk_size)
                        
                        if overflowed:
                            print("⚠️ [ASR监听] 音频缓冲区溢出")
                        
                        # 简单能量检查
                        audio_energy = np.mean(np.abs(audio_data))
                        if audio_energy > 0.001:  # 简单阈值
                            print(f"🔍 [ASR监听] 检测到语音，能量: {audio_energy:.4f}")
                            
                            # 快速识别
                            result = speech_recognizer._recognize_wake_word_only(audio_data)
                            if result and result.text:
                                cleaned_text = result.text.lower().replace("，", ",").replace("。", ".").replace("？", "?")
                                if "夸父" in cleaned_text:
                                    print(f"🎯 [ASR监听] 检测到唤醒词: {result.text}")
                                    callback(result.text)
                                    break
                        
                        # 短暂休眠，减少CPU占用
                        time.sleep(0.1)
                        
                    except Exception as e:
                        print(f"❌ [ASR监听] 处理异常: {e}")
                        break
                        
        except Exception as e:
            print(f"❌ [ASR监听] 线程异常: {e}")
        finally:
            print("🛑 [ASR监听] 线程结束")

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
    audio_data: np.ndarray
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
        self.silence_threshold = config.get('silence_threshold', 0.05)  # 提高静音阈值，减少误触发
        self.silence_duration = config.get('silence_duration', 5.0)
        self.min_recording_duration = config.get('min_recording_duration', 1.0)
        self.input_device = config.get('input_device', None)  # 从配置获取输入设备
        
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.silence_counter = 0
        self.recording_start_time = 0
        self.last_sound_time = 0
        
        # 音频缓冲区
        self.audio_buffer = []
        self.silence_buffer = []
        
        # 调试信息
        self.debug_audio_levels = []
        self.last_debug_time = time.time()
        self.debug_interval = 2.0  # 每2秒输出一次调试信息
        
        # 记录设备信息
        device_info = f"默认设备" if self.input_device is None else f"设备 {self.input_device}"
        logger.info(f"音频录制器初始化: {self.sample_rate}Hz, {self.channels}ch, 输入设备: {device_info}")
    
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
        
        # 计算音频能量用于调试
        audio_energy = np.mean(np.abs(indata))
        
        # 实时声音检测 - 当检测到明显声音时立即输出调试信息
        if audio_energy > 0.05:  # 高于此阈值认为有明显声音
            print(f"🎤 [实时检测] 检测到声音! 能量: {audio_energy:.4f} | 静音计数: {self.silence_counter}")
            
            # 根据能量级别给出具体提示
            if audio_energy > 0.2:
                print("🔊 [声音强度] 强声音输入!")
            elif audio_energy > 0.1:
                print("🔊 [声音强度] 中等声音输入")
            else:
                print("🔊 [声音强度] 轻微声音输入")
                
        elif audio_energy > 0.01:  # 低能量区间，减少输出频率
            # 只在每100次检测中输出一次，避免刷屏
            if not hasattr(self, '_low_energy_counter'):
                self._low_energy_counter = 0
            self._low_energy_counter += 1
            
            if self._low_energy_counter % 100 == 0:
                print(f"🔇 [背景音] 检测到低能量背景音: {audio_energy:.4f}")
                
        # 静音状态提示
        if self._detect_silence(indata) and hasattr(self, '_last_sound_time') and (time.time() - getattr(self, '_last_sound_time', 0)) > 3.0:
            if not hasattr(self, '_silence_reported'):
                self._silence_reported = True
                print("🔇 [静音状态] 当前为静音状态")
        else:
            self._silence_reported = False
            self._last_sound_time = time.time()
        
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
            return
        
        self.is_recording = True
        self.audio_buffer = []
        self.silence_buffer = []
        self.silence_counter = 0
        self.recording_start_time = time.time()
        self.last_sound_time = time.time()
        
        # 启动音频流
        try:
            # 构建流参数
            stream_params = {
                'samplerate': self.sample_rate,
                'channels': self.channels,
                'callback': self._audio_callback,
                'blocksize': self.chunk_size,
                'dtype': np.float32
            }
            
            # 如果指定了输入设备，添加设备参数
            if self.input_device is not None:
                stream_params['device'] = self.input_device
                logger.info(f"使用指定的输入设备: {self.input_device}")
            else:
                logger.info("使用默认输入设备")
            
            self.stream = sd.InputStream(**stream_params)
            self.stream.start()
        except Exception as e:
            self.is_recording = False
            logger.error(f"启动音频流失败: {str(e)}")
    
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
        
        # 检查静音持续时间 - 增加防误触机制
        silence_duration = self.silence_counter * self.chunk_duration
        
        # 只有当静音持续时间足够长时才停止
        if silence_duration >= self.silence_duration:
            # 额外检查：确保音频缓冲区不为空（有实际音频内容）
            if len(self.audio_buffer) > 0:
                logger.info(f"检测到静音 {silence_duration:.1f}秒，停止录制")
                return True
            else:
                # 如果没有音频内容，重置静音计数器
                self.silence_counter = 0
                return False
        
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
        self.model_name = config.get('model_name', 'small')
        self.vad_threshold = config.get('vad_threshold', 0.3)  # 语音活动检测阈值
        self.min_audio_length = config.get('min_audio_length', 1.0)  # 最小音频长度（秒）
        
        try:
            logger.info(f"加载Whisper模型: {self.model_name}")
            
            # 临时保存环境变量
            old_cc = os.environ.get('CC')
            old_cxx = os.environ.get('CXX')
            
            # 临时取消CC和CXX环境变量，避免Whisper编译错误
            if 'CC' in os.environ:
                del os.environ['CC']
            if 'CXX' in os.environ:
                del os.environ['CXX']
            
            try:
                self.model = whisper.load_model(self.model_name)
                logger.info("✅ Whisper模型加载成功")
            finally:
                # 恢复环境变量
                if old_cc:
                    os.environ['CC'] = old_cc
                if old_cxx:
                    os.environ['CXX'] = old_cxx
                    
        except Exception as e:
            logger.error(f"❌ Whisper模型加载失败: {str(e)}")
            self.model = None
    
    def _recognize_wake_word_only(self, audio_data: np.ndarray) -> Optional[RecognitionResult]:
        """简化的唤醒词检测 - 仅用于TTS打断，跳过VAD过滤以提高性能"""
        if self.model is None:
            return None
        
        # 简单的能量检查，只过滤明显的噪声
        audio_energy = np.mean(np.abs(audio_data))
        if audio_energy < 0.0005:  # 进一步降低能量阈值
            return None
        
        try:
            # 确保音频数据是一维的
            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()
            
            # Whisper需要浮点数音频数据
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # 重采样到16000Hz
            if self.sample_rate != 16000:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=self.sample_rate, target_sr=16000)
            
            # 使用Whisper进行快速识别
            result = self.model.transcribe(
                audio_data,
                language="zh",
                temperature=0.0,
                beam_size=1,  # 使用较小beam_size提高速度
                fp16=False,
                verbose=False
            )
            
            text = result.get("text", "").strip()
            if text:
                return RecognitionResult(text=text, confidence=0.8)
            
        except Exception as e:
            # 静默处理错误，避免大量调试输出
            pass
        
        return None
    
    def recognize_audio(self, audio_data: np.ndarray) -> Optional[RecognitionResult]:
        """识别音频 - 带语音活动检测（极致性能优化版）"""
        if self.model is None:
            logger.error("Whisper不可用")
            return None
        
        # 计算音频长度
        audio_duration = len(audio_data) / self.sample_rate
        
        # 快速语音活动检测
        if audio_duration < self.min_audio_length:
            return None
        
        # 快速能量检测
        audio_energy = np.mean(np.abs(audio_data))
        if audio_energy < self.vad_threshold:
            return None
        
        # 适度的音频长度限制以优化性能，保持识别精度
        max_audio_duration = 8.0  # 最多处理8秒音频
        if audio_duration > max_audio_duration:
            # 截取音频
            max_samples = int(max_audio_duration * self.sample_rate)
            audio_data = audio_data[:max_samples]
            audio_duration = max_audio_duration
            print(f"🔍 [ASR优化] 截取音频到 {max_audio_duration} 秒以优化性能")
        
        try:
            start_time = time.time()
            
            print(f"🔍 [ASR详情] 音频形状: {audio_data.shape}, 采样率: {self.sample_rate}")
            
            logger.info(f"开始识别音频 (长度: {audio_duration:.2f}秒, 能量: {audio_energy:.3f})...")
            
            # 临时保存和清除环境变量，避免Triton编译错误
            old_cc = os.environ.get('CC')
            old_cxx = os.environ.get('CXX')
            
            if 'CC' in os.environ:
                del os.environ['CC']
            if 'CXX' in os.environ:
                del os.environ['CXX']
            
            try:
                # 确保音频数据是一维的
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.flatten()
                
                print(f"🔍 [ASR详情] 音频形状: {audio_data.shape}, 采样率: {self.sample_rate}")
                
                # Whisper需要浮点数音频数据，确保数据类型正确
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                
                # Whisper默认使用16000Hz采样率，需要重采样
                if self.sample_rate != 16000:
                    import librosa
                    audio_data = librosa.resample(audio_data, orig_sr=self.sample_rate, target_sr=16000)
                    print(f"🔍 [ASR详情] 音频已重采样到16000Hz，新形状: {audio_data.shape}")
                
                # 真正能提升性能的参数组合
                result = self.model.transcribe(
                    audio_data,
                    language='zh',
                    fp16=False,
                    verbose=False,
                    # 性能优化参数 - 基于测试结果
                    temperature=0.0,  # 确定性输出
                    beam_size=1,  # 最小beam size提升速度
                    patience=0.0,  # 无耐心等待
                    best_of=1,  # 单个候选最快
                    # 精简参数
                    initial_prompt="夸父",  # 最简提示词
                    suppress_tokens=[],  # 不抑制任何token
                    # 禁用所有额外功能
                    word_timestamps=False,
                    # 宽松参数
                    compression_ratio_threshold=3.0,  # 宽松压缩率
                    logprob_threshold=-2.0,  # 宽松概率阈值
                    no_speech_threshold=0.8,  # 宽松语音检测
                    condition_on_previous_text=False,  # 不依赖前面的文本
                    task="transcribe"  # 明确转录任务
                )
            finally:
                # 恢复环境变量
                if old_cc:
                    os.environ['CC'] = old_cc
                if old_cxx:
                    os.environ['CXX'] = old_cxx
            
            recognized_text = result['text'].strip()
            duration = time.time() - start_time
            
            # 如果识别结果为空，返回None
            if not recognized_text:
                logger.info("识别结果为空")
                return None
            
            # 计算置信度（基于识别时长和音频长度）
            confidence = min(1.0, len(recognized_text) / max(audio_duration * 2, 1) * (1.0 / max(duration, 0.1)))
            
            logger.info(f"识别结果: '{recognized_text}'")
            logger.info(f"识别耗时: {duration:.2f}秒, 置信度: {confidence:.3f}")
            
            return RecognitionResult(
                text=recognized_text,
                confidence=confidence,
                duration=duration,
                engine="whisper_small"
            )
            
        except Exception as e:
            print(f"❌ [ASR错误] 识别失败: {str(e)}")
            import traceback
            print(f"❌ [ASR错误详情] {traceback.format_exc()}")
            logger.error(f"音频识别失败: {str(e)}")
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
    
    async def _call_llm_for_intent(self, text: str) -> Tuple[str, float, Dict[str, Any]]:
        """调用LLM进行意图分析"""
        if self.engine_type == 'zhipuai':
            return await self._call_zhipuai_llm(text)
        else:
            # 默认回退到关键词匹配
            return self._classify_intent(text)
    
    async def _call_zhipuai_llm(self, text: str) -> Tuple[str, float, Dict[str, Any]]:
        """调用智谱AI进行意图分析和对话"""
        print(f"🔍 [DEBUG] 开始调用智谱AI LLM API...")
        try:
            import httpx
            
            # 从配置文件获取API密钥
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'llm_params.yaml')
            api_key = None
            
            # 从配置文件读取API密钥
            try:
                import yaml
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)
                        if config_data and 'zhipuai' in config_data:
                            api_key = config_data['zhipuai'].get('api_key')
                            if not api_key:
                                logger.warning("配置文件中未找到API密钥")
                        else:
                            logger.warning("配置文件中未找到zhipuai配置")
                else:
                    logger.warning("配置文件不存在")
            except Exception as e:
                logger.error(f"读取配置文件失败: {e}")
            
            if not api_key:
                logger.error("未找到API密钥")
                print(f"❌ [DEBUG] 未找到API密钥，回退到关键词匹配")
                return self._classify_intent(text)
            
            # 极简系统提示词以获得最快速度
            system_prompt = """你是夸父机器人。支持：wave、welcome、stop。

规则：
- 动作请求：command类型
- 其他：conversation类型

JSON格式：
{
  "type": "command|conversation",
  "intent": "wave|welcome|stop|conversation", 
  "confidence": 0.8,
  "response": "回应",
  "instruction": "wave|welcome|stop|none"
}

只返回JSON。"""
            
            # 构建请求
            url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            # 从配置文件读取模型设置
            model_name = None
            try:
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)
                        if config_data and 'zhipuai' in config_data:
                            model_name = config_data['zhipuai'].get('model')
                            if not model_name:
                                logger.warning("配置文件中未找到模型设置")
                        else:
                            logger.warning("配置文件中未找到zhipuai配置")
                else:
                    logger.warning("配置文件不存在")
            except Exception as e:
                logger.error(f"读取模型配置失败: {e}")
            
            if not model_name:
                logger.error("未找到模型配置")
                print(f"❌ [DEBUG] 未找到模型配置，回退到关键词匹配")
                return self._classify_intent(text)
            
            data = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                "max_tokens": 300,  # 确保能生成完整的JSON响应
                "temperature": 0.3
            }
            
            # 极致减少超时时间以提高响应速度
            timeout = httpx.Timeout(
                connect=3.0,    # 连接超时3秒
                read=8.0,       # 读取超时8秒
                write=3.0,      # 写入超时3秒
                pool=5.0        # 连接池超时5秒
            )
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, headers=headers, json=data)
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    
                    # 添加调试信息，显示API返回的原始数据
                    print(f"🔍 [DEBUG] LLM API原始返回数据:")
                    print(f"   状态码: {response.status_code}")
                    print(f"   返回内容: {content}")
                    logger.info(f"[DEBUG] LLM API原始返回: {content}")
                    
                    # 解析JSON响应
                    try:
                        import json
                        llm_result = json.loads(content.strip())
                        
                        # 解析新的返回格式
                        response_type = llm_result.get('type', 'conversation')
                        intent = llm_result.get('intent', 'conversation')
                        confidence = llm_result.get('confidence', 0.5)
                        response_text = llm_result.get('response', '抱歉，我不太理解您的请求。')
                        instruction = llm_result.get('instruction', 'none')
                        
                        # 根据类型决定动作类型
                        if response_type == 'command' and intent in ['wave', 'welcome']:
                            action_type = intent  # 执行机器人动作
                        else:
                            action_type = 'response'  # 仅对话回应
                        
                        # 构建动作
                        action = {
                            'type': action_type,
                            'text': response_text,
                            'response_type': response_type,
                            'json_response': {
                                'intent': intent,
                                'confidence': confidence,
                                'instruction': instruction,
                                'response': response_text,
                                'action': action_type,
                                'timestamp': time.time()
                            }
                        }
                        
                        # 记录不同类型的处理
                        if response_type == 'command':
                            logger.info(f"🤖 指令执行: {intent} -> {instruction}")
                        else:
                            logger.info(f"💬 对话回应: {response_text[:50]}...")
                        
                        return intent, confidence, action
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"LLM返回的JSON解析失败: {content}")
                        print(f"❌ [DEBUG] JSON解析失败: {e}")
                        print(f"❌ [DEBUG] 原始内容: {content}")
                        
                        # 尝试修复被截断的JSON
                        try:
                            # 更智能的JSON修复
                            fixed_content = self._fix_json_content(content)
                            if fixed_content:
                                print(f"🔧 [DEBUG] 修复后的JSON: {fixed_content}")
                                llm_result = json.loads(fixed_content)
                                
                                # 解析修复后的JSON
                                response_type = llm_result.get('type', 'conversation')
                                intent = llm_result.get('intent', 'conversation')
                                confidence = llm_result.get('confidence', 0.5)
                                response_text = llm_result.get('response', '抱歉，我没有理解您的请求。请尝试用更清晰的语言表达。')
                                instruction = llm_result.get('instruction', 'none')
                                
                                # 如果response字段不完整，尝试从content中提取
                                if response_text.endswith('...') or len(response_text) < 10:
                                    response_text = "抱歉，我遇到了一些技术问题，请稍后再试。"
                                
                                # 根据类型决定动作类型
                                if response_type == 'command' and intent in ['wave', 'welcome']:
                                    action_type = intent  # 执行机器人动作
                                else:
                                    action_type = 'response'  # 仅对话回应
                                
                                # 构建动作
                                action = {
                                    'type': action_type,
                                    'text': response_text,
                                    'response_type': response_type,
                                    'json_response': {
                                        'intent': intent,
                                        'confidence': confidence,
                                        'instruction': instruction,
                                        'response': response_text,
                                        'action': action_type,
                                        'timestamp': time.time()
                                    }
                                }
                                
                                logger.info(f"🔧 JSON修复成功，使用修复后的结果")
                                return intent, confidence, action
                        except Exception as fix_e:
                            logger.error(f"JSON修复失败: {fix_e}")
                            print(f"❌ [DEBUG] JSON修复失败: {fix_e}")
                        
                        return self._classify_intent(text)
                        
                else:
                    error_msg = response.text if response.text else "未知错误"
                    logger.error(f"LLM API调用失败: {response.status_code}")
                    print(f"❌ [DEBUG] API调用失败:")
                    print(f"   状态码: {response.status_code}")
                    print(f"   错误信息: {error_msg}")
                    return self._classify_intent(text)
                    
        except Exception as e:
            logger.error(f"LLM调用异常: {str(e)}")
            print(f"❌ [DEBUG] LLM调用异常: {str(e)}")
            import traceback
            print(f"❌ [DEBUG] 异常详情: {traceback.format_exc()}")
            return self._classify_intent(text)
    
    async def process_text_async(self, text: str) -> IntentResult:
        """异步处理文本"""
        start_time = time.time()
        
        # 临时禁用缓存以确保使用新的系统提示词
        logger.info(f"🔄 跳过缓存，直接调用LLM: '{text[:30]}...'")
        
        # 生成缓存键用于后续缓存
        cache_key = self._get_cache_key(text)
        
        try:
            # 使用LLM进行意图分析
            intent, confidence, action = await self._call_llm_for_intent(text)
            
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
    
    def _fix_json_content(self, content: str) -> Optional[str]:
        """尝试修复不完整的JSON内容"""
        try:
            # 基础清理
            content = content.strip()
            
            # 如果已经是有效的JSON，直接返回
            json.loads(content)
            return content
        except:
            pass
        
        try:
            # 策略1: 查找最后一个完整的JSON对象
            last_brace = content.rfind('}')
            if last_brace != -1:
                first_brace = content.find('{')
                if first_brace != -1 and first_brace < last_brace:
                    candidate = content[first_brace:last_brace + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except:
                        pass
            
            # 策略2: 修复缺失的引号
            lines = content.split('\n')
            fixed_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.endswith(','):
                    # 检查是否是字符串字段但缺少结束引号
                    if ':' in line and line.count('"') == 1:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip()
                            if value and value[0] == '"' and value[-1] != '"':
                                value += '"'
                                line = f'{key}: {value}'
                fixed_lines.append(line)
            
            candidate = '\n'.join(fixed_lines)
            # 确保JSON以}结束
            if not candidate.endswith('}'):
                candidate += '}'
            
            try:
                json.loads(candidate)
                return candidate
            except:
                pass
            
            # 策略3: 创建最小化的JSON
            if '"response"' in content:
                # 提取response字段的内容
                response_start = content.find('"response"')
                if response_start != -1:
                    response_part = content[response_start:]
                    value_start = response_part.find(':')
                    if value_start != -1:
                        value_part = response_part[value_start + 1:].strip()
                        if value_part.startswith('"'):
                            # 找到字符串结束位置，如果没有找到则使用下一个引号
                            quote_end = value_part.find('"', 1)
                            if quote_end == -1:
                                # 如果没有找到结束引号，取到行尾或最后一个字符
                                line_end = value_part.find('\n')
                                if line_end != -1:
                                    quote_end = line_end
                                else:
                                    quote_end = len(value_part) - 1
                            
                            if quote_end > 1:
                                response_text = value_part[1:quote_end]
                                # 清理响应文本
                                response_text = response_text.strip().rstrip('，。！？,.!?')
                                if len(response_text) > 0:
                                    # 创建简单的JSON
                                    simple_json = f'{{"type": "conversation", "intent": "conversation", "confidence": 0.8, "response": "{response_text}", "instruction": "none"}}'
                                    try:
                                        json.loads(simple_json)
                                        print(f"🔧 [DEBUG] 成功创建简化JSON: {simple_json}")
                                        return simple_json
                                    except:
                                        pass
            
            return None
            
        except Exception as e:
            print(f"🔧 [DEBUG] JSON修复失败: {e}")
            return None
    
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
            
            # 如果匹配到了关键词，提高置信度
            if match_count > 0:
                confidence = max(confidence, 0.8)  # 至少0.8的置信度
            
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

class ProductionSystem:
    """正式版系统"""
    
    def __init__(self, config: Dict[str, Any], input_mode: str = 'voice'):
        self.config = config
        self.input_mode = input_mode  # voice 或 text
        self.audio_mode = config.get('audio_mode', 'microphone')  # microphone 或 preset
        
        # 初始化ROS组件
        self.publisher = ROSPublisher()
        
        # 初始化TTS播放器
        self.tts_player = MemoryTTSPlayer(self.publisher, config.get('tts', {}))
        
        # 初始化组件 - 确保sample_rate配置正确传递
        asr_config = config.get('asr', {})
        speech_config = config.get('speech', {})
        
        # 将asr配置中的sample_rate传递给speech配置
        if 'sample_rate' in asr_config and 'sample_rate' not in speech_config:
            speech_config['sample_rate'] = asr_config['sample_rate']
        
        self.recorder = AudioRecorder(asr_config)  # 使用asr配置而不是audio配置
        self.speech_recognizer = ProductionSpeechRecognizer(speech_config)
        self.llm_processor = ProductionLLMProcessor(config.get('llm', {}))
        
        # 中断处理状态
        self.pending_wake_word = None
        self.pending_command = None
        
        # 系统状态
        self.is_running = False
        self.recording_thread = None
        self.user_input_thread = None
        
        # 录音状态管理（仅在语音模式下使用）
        self.recording_state = 'IDLE'  # IDLE, WAITING_FOR_WAKE_WORD, LISTENING_FOR_COMMAND, PROCESSING, PLAYING_RESPONSE
        self.wake_word_detected = False
        self.last_processing_time = 0
        self.processing_cooldown = 2.0  # 处理冷却时间2秒
        
        # 性能统计
        self.performance_stats = {
            'total_requests': 0,
            'avg_asr_time': 0.0,
            'avg_llm_time': 0.0,
            'avg_tts_generation_time': 0.0,
            'avg_response_time': 0.0,  # 从输入到系统开始响应的时间
            'min_response_time': float('inf'),
            'max_response_time': 0.0
        }
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"✅ 正式版系统初始化完成 (输入模式: {self.input_mode})")
    
    def _wake_word_detected_during_tts(self, detected_text: str):
        """TTS期间检测到唤醒词的回调函数"""
        print(f"🎯 [中断] 检测到唤醒词，中断TTS: {detected_text}")
        
        # 中断当前TTS播放
        self.tts_player.interrupt_playback()
        
        # 保存检测到的唤醒词和命令
        self.pending_wake_word = detected_text
        
        # 提取命令部分（去除唤醒词）
        command_text = detected_text.replace("夸父夸父", "").strip()
        if command_text:
            self.pending_command = command_text
        else:
            self.pending_command = "wake_word_only"
        
        print(f"🎯 [中断] 中断完成，待处理: {self.pending_command}")
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info(f"收到信号 {signum}，正在关闭系统...")
        self._print_performance_summary()
        self.stop()
        sys.exit(0)
    
    def _update_performance_stats(self, asr_time: float, llm_time: float, tts_time: float, response_time: float):
        """更新性能统计"""
        stats = self.performance_stats
        stats['total_requests'] += 1
        
        # 计算移动平均
        alpha = 0.3  # 平滑因子
        stats['avg_asr_time'] = (1 - alpha) * stats['avg_asr_time'] + alpha * asr_time
        stats['avg_llm_time'] = (1 - alpha) * stats['avg_llm_time'] + alpha * llm_time
        stats['avg_tts_generation_time'] = (1 - alpha) * stats['avg_tts_generation_time'] + alpha * tts_time
        stats['avg_response_time'] = (1 - alpha) * stats['avg_response_time'] + alpha * response_time
        
        # 更新最值
        stats['min_response_time'] = min(stats['min_response_time'], response_time)
        stats['max_response_time'] = max(stats['max_response_time'], response_time)
    
    def _print_performance_summary(self):
        """打印性能统计摘要"""
        stats = self.performance_stats
        if stats['total_requests'] == 0:
            return
        
        print("\n" + "="*60)
        print("📊 性能统计摘要 - 输入到响应时间")
        print("="*60)
        print(f"📈 总请求数: {stats['total_requests']}")
        print(f"⏱️  平均响应时间: {stats['avg_response_time']:.3f}秒")
        print(f"⏱️  最快响应时间: {stats['min_response_time']:.3f}秒")
        print(f"⏱️  最慢响应时间: {stats['max_response_time']:.3f}秒")
        print(f"🎤  平均ASR时间: {stats['avg_asr_time']:.3f}秒")
        print(f"🧠  平均LLM时间: {stats['avg_llm_time']:.3f}秒")
        print(f"🔊  平均TTS生成时间: {stats['avg_tts_generation_time']:.3f}秒")
        
        # 计算各阶段占比
        if stats['avg_response_time'] > 0:
            asr_percent = (stats['avg_asr_time'] / stats['avg_response_time']) * 100
            llm_percent = (stats['avg_llm_time'] / stats['avg_response_time']) * 100
            tts_percent = (stats['avg_tts_generation_time'] / stats['avg_response_time']) * 100
            print(f"📊 时间占比 - ASR: {asr_percent:.1f}%, LLM: {llm_percent:.1f}%, TTS生成: {tts_percent:.1f}%")
        
        # 性能优化建议
        self._provide_performance_recommendations(stats)
        
        print("="*60)
    
    def _provide_performance_recommendations(self, stats: Dict[str, Any]):
        """提供性能优化建议"""
        print("\n🔧 性能优化建议:")
        
        # ASR优化建议
        if stats['avg_asr_time'] > 2.0:
            print("  🎤 ASR时间较长，建议:")
            print("    - 使用更小的Whisper模型（tiny或base）")
            print("    - 减少音频输入长度")
            print("    - 启用音频预过滤")
        
        # LLM优化建议
        if stats['avg_llm_time'] > 3.0:
            print("  🧠 LLM时间较长，建议:")
            print("    - 减少max_tokens数量")
            print("    - 使用更快的模型")
            print("    - 启用响应缓存")
        
        # TTS优化建议
        if stats['avg_tts_generation_time'] > 2.0:
            print("  🔊 TTS生成时间较长，建议:")
            print("    - 使用更快的TTS引擎")
            print("    - 减少响应文本长度")
            print("    - 预先生成常用回应")
        
        # 总体优化建议
        if stats['avg_response_time'] > 5.0:
            print("  ⚡ 总体优化建议:")
            print("    - 考虑使用更快的硬件")
            print("    - 启用并行处理")
            print("    - 实施流式处理")
        
        # 性能评级 - 目标2-3秒
        if stats['avg_response_time'] < 2.0:
            print("  🏆 性能评级: 优秀 (< 2秒)")
        elif stats['avg_response_time'] < 3.0:
            print("  🏆 性能评级: 良好 (2-3秒) ✓ 目标达成")
        elif stats['avg_response_time'] < 5.0:
            print("  🏆 性能评级: 一般 (3-5秒)")
        else:
            print("  🏆 性能评级: 需要优化 (> 5秒) ❌ 需要进一步优化")
        
        print()
    
    def _recording_loop(self):
        """录制循环 - 带状态管理和唤醒词检测"""
        logger.info("🔄 录制循环开始")
        last_check_time = time.time()
        audio_check_completed = False
        audio_started = False
        loop_count = 0
        
        while self.is_running:
            loop_count += 1
            current_time = time.time()
            
            # 每10秒输出一次状态信息
            if loop_count % 100 == 0:
                logger.info(f"🔄 录制循环运行中... 状态: {self.recording_state}")
            
            # 检查TTS是否正在播放，如果是则暂停录音
            if self.tts_player.is_playing_audio():
                if self.recorder.is_recording:
                    self.recorder.stop_recording()
                    self.recording_state = 'PLAYING_RESPONSE'
                    logger.info("🔇 TTS播放中，暂停录音")
                time.sleep(0.1)
                continue
            
            # 检查处理冷却时间
            if self.recording_state == 'PROCESSING':
                if current_time - self.last_processing_time < self.processing_cooldown:
                    time.sleep(0.1)
                    continue
                else:
                    self.recording_state = 'WAITING_FOR_WAKE_WORD'
                    logger.info("🔄 冷却时间结束，等待唤醒词")
                    print("🎤 等待唤醒词 '夸父'...")
            
            if self.recorder.is_recording:
                # 检查是否应该停止录制
                if self.recorder.should_stop_recording():
                    print("🛑 检测到语音输入，正在处理...")
                    logger.info("🛑 检测到应该停止录制")
                    audio_data = self.recorder.stop_recording()
                    
                    if audio_data is not None:
                        logger.info(f"🎵 获得音频数据，长度: {len(audio_data)}")
                        # 处理音频
                        self._process_audio_with_wake_word_detection(audio_data)
                    else:
                        logger.info("🔇 没有获得音频数据，处理空音频")
                        # 处理空音频
                        self._process_empty_audio()
                    # 重置启动标志
                    audio_started = False
            else:
                # 根据状态决定是否开始录音
                if self.recording_state in ['IDLE', 'WAITING_FOR_WAKE_WORD']:
                    # 只在启动时检查一次音频设备状态
                    if not audio_check_completed:
                        print("🔍 检查音频设备...")
                        logger.info("🔍 检查音频设备...")
                        try:
                            # 获取音频设备列表
                            devices = sd.query_devices()
                            logger.info(f"🎵 音频设备列表: {len(devices)} 个设备")
                            input_devices = [i for i, dev in enumerate(devices) if dev['max_input_channels'] > 0]
                            logger.info(f"🎤 输入设备: {input_devices}")
                            
                            if input_devices:
                                print("🎵 找到输入设备，等待语音输入...")
                                logger.info("🎵 找到输入设备，开始录制...")
                                # 静默启动录制，不打印信息
                                self.recorder.start_recording()
                                audio_started = True
                                self.recording_state = 'WAITING_FOR_WAKE_WORD'
                                print("🎤 等待唤醒词 '夸父'...")
                                logger.info("🎤 开始监听唤醒词...")
                                # 设置标志，避免重复检查
                                audio_check_completed = True
                            else:
                                print("🔇 没有找到输入设备，处理空音频...")
                                logger.info("🔇 没有找到输入设备，处理空音频...")
                                # 没有输入设备，直接处理空音频
                                self._process_empty_audio()
                                # 设置标志，避免重复检查
                                audio_check_completed = True
                        except Exception as e:
                            print(f"❌ 音频设备检查失败: {str(e)}")
                            logger.error(f"❌ 音频设备检查失败: {str(e)}")
                            # 检查失败，直接处理空音频
                            self._process_empty_audio()
                            audio_check_completed = True
                    else:
                        # 音频设备检查已完成，正常监听模式
                        if not self.recorder.is_recording and not audio_started:
                            # 如果录制停止了，重新开始
                            logger.info("🔄 重新开始录制...")
                            try:
                                self.recorder.start_recording()
                                audio_started = True
                                self.recording_state = 'WAITING_FOR_WAKE_WORD'
                            except Exception as e:
                                logger.error(f"重新开始录制失败: {e}")
                                time.sleep(1.0)
                        time.sleep(0.1)
                else:
                    # 其他状态下等待一段时间再检查
                    time.sleep(0.1)
        
        logger.info("🛑 录制循环结束")
    
    def _process_audio(self, audio_data: np.ndarray):
        """处理音频"""
        try:
            # 1. 语音识别
            recognition_result = self.speech_recognizer.recognize_audio(audio_data)
            if not recognition_result:
                logger.error("语音识别失败")
                return
            
            # 发布语音识别结果
            self.publisher.publish_recognition(recognition_result.text, recognition_result.confidence)
            
            # 2. LLM处理
            llm_result = asyncio.run(self.llm_processor.process_text_async(recognition_result.text))
            
            # 发布意图识别结果
            self.publisher.publish_command(llm_result.intent, llm_result.confidence, llm_result.action)
            
            # 3. 生成JSON响应
            json_response = llm_result.action.get('json_response', {})
            json_response['recognition'] = {
                'text': recognition_result.text,
                'confidence': recognition_result.confidence,
                'duration': recognition_result.duration,
                'engine': recognition_result.engine
            }
            
            # 只保存一个最新的JSON文件
            json_path = os.path.join(os.path.dirname(__file__), "latest_response.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_response, f, ensure_ascii=False, indent=2)
            
            logger.info(f"JSON响应已保存: {json_path}")
            
            # 4. 生成TTS响应
            # 只要action中有text字段，就播放TTS
            if 'text' in llm_result.action and llm_result.action['text']:
                tts_text = llm_result.action['text']
                print(f"🔊 系统回应: '{tts_text}'")
                print("🎵 正在播放语音回应...")
                
                # 启动ASR监听（支持唤醒词打断）
                self.tts_player.start_asr_during_tts(self.speech_recognizer, self._wake_word_detected_during_tts)
                
                try:
                    # 使用流式TTS播放
                    self.tts_player.generate_and_play_streaming_sync(tts_text)
                    print("✅ 语音回应播放完成")
                except:
                    print("🛑 系统回应被中断")
                
                # 停止ASR监听
                self.tts_player.stop_asr_during_tts()
                
                # 检查是否有待处理的唤醒词
                if self.pending_wake_word:
                    print(f"🎯 处理系统回应期间的唤醒词: {self.pending_wake_word}")
                    self._process_pending_wake_word()
            
            logger.info(f"处理完成: {llm_result.intent} (置信度: {llm_result.confidence:.2f})")
            
        except Exception as e:
            logger.error(f"音频处理失败: {str(e)}")
    
    def _process_audio_with_wake_word_detection(self, audio_data: np.ndarray):
        """处理音频并带唤醒词检测"""
        try:
            # 端到端时间测量开始
            pipeline_start_time = time.time()
            print(f"⏱️ [性能] 开始处理音频流程，音频长度: {len(audio_data)} 采样点")
            
            # 1. 语音识别
            asr_start_time = time.time()
            print("🎵 正在进行语音识别...")
            recognition_result = self.speech_recognizer.recognize_audio(audio_data)
            asr_time = time.time() - asr_start_time
            print(f"⏱️ [性能] 语音识别耗时: {asr_time:.3f}秒")
            if not recognition_result:
                print("❌ 语音识别失败")
                logger.error("语音识别失败")
                return
            
            # 发布语音识别结果
            self.publisher.publish_recognition(recognition_result.text, recognition_result.confidence)
            
            # 2. 唤醒词检测
            recognized_text = recognition_result.text.lower().strip()
            
            # 显示ASR结果
            print(f"🎤 ASR识别结果: '{recognition_result.text}'")
            
            # 清理识别文本，移除标点符号
            cleaned_text = recognized_text.replace("，", ",").replace("。", ".").replace("？", "?")
            
            # 检查是否包含唤醒词 "夸父"
            if "夸父" in cleaned_text:
                print(f"🎯 检测到唤醒词 '夸父'")
                logger.info(f"🎯 检测到唤醒词 '夸父': {recognized_text}")
                self.wake_word_detected = True
                self.recording_state = 'LISTENING_FOR_COMMAND'
                
                # 如果只有唤醒词没有其他内容，提示用户
                command_text = cleaned_text.replace("夸父", "").strip()
                # 移除标点符号后检查是否为空
                command_text = command_text.replace(",", "").replace(".", "").replace("?", "").strip()
                if len(command_text) == 0:
                    print("🎤 只检测到唤醒词，等待指令...")
                    logger.info("🎤 只检测到唤醒词，等待指令...")
                    # 处理只有唤醒词的情况
                    llm_result = asyncio.run(self.llm_processor.process_text_async("wake_word_only"))
                    # 只要action中有text字段，就播放TTS
                    if 'text' in llm_result.action and llm_result.action['text']:
                        print(f"🔊 TTS回应: '{llm_result.action['text']}'")
                        self.recording_state = 'PLAYING_RESPONSE'
                        self.last_processing_time = time.time()
                        self.tts_player.generate_and_play_streaming_sync(llm_result.action['text'])
                        print("✅ 语音回应播放完成")
                    # 重置状态，等待新的指令
                    self.wake_word_detected = False
                    self.recording_state = 'PROCESSING'
                    self.last_processing_time = time.time()
                    return
            
            # 3. 根据状态决定是否处理指令
            if self.recording_state == 'LISTENING_FOR_COMMAND' and self.wake_word_detected:
                # 移除唤醒词，只处理指令部分
                command_text = cleaned_text.replace("夸父", "").strip()
                
                if command_text:
                    print(f"🎤 处理指令: '{command_text}'")
                    logger.info(f"🎤 处理指令: {command_text}")
                    
                    # LLM处理
                    llm_start_time = time.time()
                    print("🧠 正在进行LLM分析...")
                    llm_result = asyncio.run(self.llm_processor.process_text_async(command_text))
                    llm_time = time.time() - llm_start_time
                    print(f"⏱️ [性能] LLM处理耗时: {llm_time:.3f}秒")
                    
                    # 发布意图识别结果
                    self.publisher.publish_command(llm_result.intent, llm_result.confidence, llm_result.action)
                    
                    # 生成JSON响应
                    json_response = llm_result.action.get('json_response', {})
                    json_response['recognition'] = {
                        'text': recognition_result.text,
                        'confidence': recognition_result.confidence,
                        'duration': recognition_result.duration,
                        'engine': recognition_result.engine,
                        'wake_word_detected': True,
                        'command_text': command_text
                    }
                    
                    # 只保存一个最新的JSON文件
                    json_path = os.path.join(os.path.dirname(__file__), "latest_response.json")
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(json_response, f, ensure_ascii=False, indent=2)
                    
                    logger.info(f"JSON响应已保存: {json_path}")
                    
                    # 生成TTS响应
                    # 只要action中有text字段，就播放TTS
                    if 'text' in llm_result.action and llm_result.action['text']:
                        tts_text = llm_result.action['text']
                        print(f"🔊 TTS回应: '{tts_text}'")
                        print("🎵 正在播放语音回应...")
                        self.recording_state = 'PLAYING_RESPONSE'
                        self.last_processing_time = time.time()
                        
                        # 启动ASR监听（在TTS播放期间监听唤醒词）
                        self.tts_player.start_asr_during_tts(self.speech_recognizer, self._wake_word_detected_during_tts)
                        
                        # 响应开始时间（在TTS开始生成前测量）
                        response_start_time = time.time() - pipeline_start_time
                        print(f"⏱️ [性能] 响应开始时间: {response_start_time:.3f}秒")
                        print(f"⏱️ [性能] 处理时间分布 - ASR: {asr_time:.3f}s, LLM: {llm_time:.3f}s")
                        
                        # 更新性能统计（响应开始时间）
                        self._update_performance_stats(asr_time, llm_time, 0.0, response_start_time)
                        
                        try:
                            # 使用流式TTS播放
                            print("🎵 正在生成TTS...")
                            
                            # 分离TTS生成和播放时间
                            tts_generation_time = self.tts_player.generate_and_play_streaming_sync(tts_text)
                            
                            print(f"✅ 语音回应播放完成")
                            print(f"⏱️ [性能] TTS生成耗时: {tts_generation_time:.3f}秒")
                            
                        except:
                            print("🛑 TTS播放被中断")
                        
                        # 停止ASR监听
                        self.tts_player.stop_asr_during_tts()
                        
                        # 检查是否有待处理的唤醒词
                        if self.pending_wake_word:
                            print(f"🎯 处理TTS期间检测到的唤醒词: {self.pending_wake_word}")
                            self._process_pending_wake_word()
                        else:
                            # 正常结束，重置状态
                            self.recording_state = 'PROCESSING'
                            self.last_processing_time = time.time()
                    
                    print(f"✅ 指令处理完成: {llm_result.intent} (置信度: {llm_result.confidence:.2f})")
                    logger.info(f"指令处理完成: {llm_result.intent} (置信度: {llm_result.confidence:.2f})")
                    
                    # 重置唤醒词状态，进入冷却时间
                    self.wake_word_detected = False
                    self.recording_state = 'PROCESSING'
                    self.last_processing_time = time.time()
                else:
                    print("🎤 检测到唤醒词但没有有效指令")
                    logger.info("🎤 检测到唤醒词但没有有效指令")
                    self.recording_state = 'WAITING_FOR_WAKE_WORD'
            else:
                # 没有检测到唤醒词，忽略这段音频
                print(f"🔇 未检测到唤醒词，忽略: '{recognized_text}'")
                logger.info(f"🔇 未检测到唤醒词，忽略: {recognized_text}")
                self.recording_state = 'WAITING_FOR_WAKE_WORD'
            
        except Exception as e:
            print(f"❌ 音频处理失败: {str(e)}")
            logger.error(f"音频处理失败: {str(e)}")
            self.recording_state = 'WAITING_FOR_WAKE_WORD'
    
    def _process_pending_wake_word(self):
        """处理TTS期间检测到的唤醒词"""
        if not self.pending_wake_word:
            return
        
        try:
            print(f"🎯 处理待处理的唤醒词: {self.pending_wake_word}")
            
            # 获取命令文本
            command_text = self.pending_command
            
            # 重置待处理状态
            self.pending_wake_word = None
            self.pending_command = None
            
            # 设置状态为处理中
            self.recording_state = 'PROCESSING'
            self.last_processing_time = time.time()
            
            if command_text == "wake_word_only":
                # 只有唤醒词，没有命令
                print("🎤 只检测到唤醒词，生成回应...")
                llm_result = asyncio.run(self.llm_processor.process_text_async("wake_word_only"))
                
                if 'text' in llm_result.action and llm_result.action['text']:
                    print(f"🔊 唤醒词回应: '{llm_result.action['text']}'")
                    
                    # 启动ASR监听（支持唤醒词打断）
                    self.tts_player.start_asr_during_tts(self.speech_recognizer, self._wake_word_detected_during_tts)
                    
                    try:
                        self.tts_player.generate_and_play_streaming_sync(llm_result.action['text'])
                        print("✅ 唤醒词回应播放完成")
                    except:
                        print("🛑 唤醒词回应被中断")
                    
                    # 停止ASR监听
                    self.tts_player.stop_asr_during_tts()
                    
                    # 检查是否有待处理的唤醒词
                    if self.pending_wake_word:
                        print(f"🎯 处理唤醒词回应期间的唤醒词: {self.pending_wake_word}")
                        self._process_pending_wake_word()
            else:
                # 有具体的命令
                print(f"🎤 处理TTS期间的命令: '{command_text}'")
                
                # LLM处理
                llm_result = asyncio.run(self.llm_processor.process_text_async(command_text))
                
                # 发布意图识别结果
                self.publisher.publish_command(llm_result.intent, llm_result.confidence, llm_result.action)
                
                # 生成JSON响应
                json_response = llm_result.action.get('json_response', {})
                json_response['recognition'] = {
                    'text': self.pending_wake_word,
                    'confidence': 0.9,
                    'duration': 0.0,
                    'engine': 'tts_interrupt',
                    'wake_word_detected': True,
                    'command_text': command_text,
                    'interrupted': True
                }
                
                # 保存JSON响应
                json_path = os.path.join(os.path.dirname(__file__), "latest_response.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_response, f, ensure_ascii=False, indent=2)
                
                logger.info(f"TTS中断JSON响应已保存: {json_path}")
                
                # 播放新的TTS回应
                if 'text' in llm_result.action and llm_result.action['text']:
                    tts_text = llm_result.action['text']
                    print(f"🔊 新的TTS回应: '{tts_text}'")
                    print("🎵 正在播放新的语音回应...")
                    
                    # 再次启动ASR监听（支持嵌套中断）
                    self.tts_player.start_asr_during_tts(self.speech_recognizer, self._wake_word_detected_during_tts)
                    
                    try:
                        self.tts_player.generate_and_play_streaming_sync(tts_text)
                        print("✅ 新的语音回应播放完成")
                    except:
                        print("🛑 新的TTS播放被中断")
                    
                    # 停止ASR监听
                    self.tts_player.stop_asr_during_tts()
                    
                    # 检查是否还有待处理的唤醒词
                    if self.pending_wake_word:
                        print(f"🎯 处理嵌套的唤醒词: {self.pending_wake_word}")
                        self._process_pending_wake_word()
                
                print(f"✅ TTS期间指令处理完成: {llm_result.intent} (置信度: {llm_result.confidence:.2f})")
            
            # 重置状态
            self.recording_state = 'WAITING_FOR_WAKE_WORD'
            
        except Exception as e:
            print(f"❌ 处理待处理唤醒词失败: {e}")
            logger.error(f"处理待处理唤醒词失败: {e}")
            self.recording_state = 'WAITING_FOR_WAKE_WORD'
    
    def _process_empty_audio(self):
        """处理空音频（没有麦克风输入的情况）"""
        try:
            print("🎤 处理空音频（没有麦克风输入）")
            logger.info("🎤 处理空音频（没有麦克风输入）")
            
            # 空音频不应该调用LLM或播放TTS，只是简单记录
            print("🔇 空音频，跳过LLM调用和TTS播放")
            logger.info("🔇 空音频，跳过LLM调用和TTS播放")
            
            # 创建空的识别结果
            recognition_result = RecognitionResult(
                text="",
                confidence=0.0,
                duration=0.0,
                engine="empty"
            )
            
            # 发布语音识别结果
            self.publisher.publish_recognition(recognition_result.text, recognition_result.confidence)
            
            # 显示ASR结果
            print("🎤 ASR识别结果: '' (无语音输入)")
            print("✅ 空音频处理完成：无语音输入，等待下一次录音")
            
        except Exception as e:
            print(f"❌ 空音频处理失败: {str(e)}")
            logger.error(f"空音频处理失败: {str(e)}")
            import traceback
            logger.error(f"空音频处理错误详情: {traceback.format_exc()}")
    
    def start(self):
        """启动系统"""
        if self.is_running:
            logger.warning("系统已经在运行中")
            return
        
        if self.input_mode == 'voice':
            self._start_voice_mode()
        else:
            self._start_text_mode()
    
    def _start_voice_mode(self):
        """启动语音模式"""
        print("🚀 启动正式版语音识别系统...")
        print("=" * 60)
        print("🎤 语音识别+LLM+TTS系统")
        print("实时语音录制 -> LLM分析 -> JSON生成 -> TTS反馈")
        print("支持唤醒词检测、ROS话题发布、内存TTS流式播放")
        print("=" * 60)
        print("💡 使用方法：说'夸父'唤醒，然后说'挥手'或'抱拳'")
        print("💡 提示：按Ctrl+C可以停止")
        print("💡 性能监控：在命令行输入 'stats' 查看性能统计")
        print("=" * 60)
        
        self.is_running = True
        
        # 启动录制线程
        print("🔄 启动录制线程...")
        self.recording_thread = threading.Thread(target=self._recording_loop)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        print("✅ 录制线程已启动")
        
        # 等待一下确保录制线程启动
        time.sleep(1.0)
        
        # 自动开始录制
        print("🎤 开始录制...")
        try:
            self.recorder.start_recording()
            print("✅ 录制已开始")
        except Exception as e:
            print(f"❌ 录制启动失败: {e}")
            import traceback
            traceback.print_exc()
            self.stop()
            return
        
        print("✅ 语音模式启动完成，正在监听...")
        
        # 保持主线程运行
        try:
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n🛑 用户中断，正在关闭系统...")
        finally:
            self.stop()
    
    def _start_text_mode(self):
        """启动文本模式"""
        print("🚀 启动正式版文本输入系统...")
        print("=" * 60)
        print("💬 文本输入+LLM+TTS系统")
        print("文本输入 -> LLM分析 -> JSON生成 -> TTS反馈")
        print("支持ROS话题发布、内存TTS流式播放")
        print("=" * 60)
        print("💡 支持的指令：")
        print("   - 挥手、招手、hello、wave -> 触发挥手动作")
        print("   - 抱拳、敬礼、welcome -> 触发抱拳动作")
        print("   - 停止、stop -> 停止当前动作")
        print("   - 退出、exit -> 退出程序")
        print("=" * 60)
        
        self.is_running = True
        
        # 保持主线程运行，等待用户输入
        try:
            while self.is_running:
                try:
                    # 获取用户输入
                    user_input = input("\n💬 请输入指令：").strip()
                    
                    if not user_input:
                        continue
                    
                    # 检查退出指令
                    if user_input.lower() in ['退出', 'exit', 'quit', 'q']:
                        print("👋 再见！")
                        break
                    
                    print(f"📝 处理指令: '{user_input}'")
                    
                    # 处理文本输入
                    self._process_text_input(user_input)
                    
                except KeyboardInterrupt:
                    print("\n🛑 用户中断，正在关闭系统...")
                    break
                except EOFError:
                    print("\n👋 输入结束，再见！")
                    break
                except Exception as e:
                    print(f"❌ 处理输入失败: {e}")
                    logger.error(f"处理文本输入失败: {e}")
                    
        finally:
            self.stop()
    
    def _process_text_input(self, text: str):
        """处理文本输入"""
        try:
            # 端到端时间测量开始
            pipeline_start_time = time.time()
            print(f"⏱️ [性能] 开始处理文本输入: '{text[:30]}...'")
            
            # 发布语音识别结果（模拟）
            self.publisher.publish_recognition(text, 1.0)
            
            # LLM处理
            llm_start_time = time.time()
            print("🧠 正在分析指令...")
            llm_result = asyncio.run(self.llm_processor.process_text_async(text))
            llm_time = time.time() - llm_start_time
            print(f"⏱️ [性能] LLM处理耗时: {llm_time:.3f}秒")
            
            # 发布VLA指令
            self.publisher.publish_command(llm_result.intent, llm_result.confidence, llm_result.action)
            
            # 生成JSON响应
            json_response = llm_result.action.get('json_response', {})
            json_response['recognition'] = {
                'text': text,
                'confidence': 1.0,
                'duration': 0.0,
                'engine': 'text_input'
            }
            
            # 保存JSON响应
            json_path = os.path.join(os.path.dirname(__file__), "latest_response.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_response, f, ensure_ascii=False, indent=2)
            
            logger.info(f"JSON响应已保存: {json_path}")
            
            # 生成TTS响应
            # 只要action中有text字段，就播放TTS
            if 'text' in llm_result.action and llm_result.action['text']:
                response_text = llm_result.action['text']
                print(f"🔊 系统回应: '{response_text}'")
                print("🎵 正在播放语音回应...")
                
                # 启动ASR监听（支持唤醒词打断）
                self.tts_player.start_asr_during_tts(self.speech_recognizer, self._wake_word_detected_during_tts)
                
                # TTS处理
                tts_start_time = time.time()
                try:
                    # 使用流式TTS播放
                    tts_generation_time = self.tts_player.generate_and_play_streaming_sync(response_text)
                    tts_time = tts_generation_time if tts_generation_time else 0.0
                    print("✅ 语音回应播放完成")
                    print(f"⏱️ [性能] TTS生成耗时: {tts_time:.3f}秒")
                except:
                    print("🛑 系统回应被中断")
                    tts_time = 0.0
                
                # 停止ASR监听
                self.tts_player.stop_asr_during_tts()
                
                # 检查是否有待处理的唤醒词
                if self.pending_wake_word:
                    print(f"🎯 处理文本输入回应期间的唤醒词: {self.pending_wake_word}")
                    self._process_pending_wake_word()
                
                # 响应开始时间统计
                response_start_time = time.time() - pipeline_start_time
                print(f"⏱️ [性能] 响应开始时间: {response_start_time:.3f}秒")
                print(f"⏱️ [性能] 处理时间分布 - LLM: {llm_time:.3f}s, TTS生成: {tts_time:.3f}s")
                
                # 更新性能统计
                self._update_performance_stats(0.0, llm_time, tts_time, response_start_time)  # ASR时间为0（文本输入）
            
            print(f"✅ 指令处理完成: {llm_result.intent} (置信度: {llm_result.confidence:.2f})")
            
        except Exception as e:
            print(f"❌ 处理文本输入失败: {e}")
            logger.error(f"处理文本输入失败: {e}")
            import traceback
            logger.error(f"文本输入错误详情: {traceback.format_exc()}")
    
    def stop(self):
        """停止系统"""
        if not self.is_running:
            return
        
        logger.info("🛑 正在停止系统...")
        self.is_running = False
        
        # 停止录制
        if self.recorder.is_recording:
            self.recorder.stop_recording()
        
        # 等待线程结束
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=1.0)
        
        logger.info("✅ 系统已停止")

def load_audio_config() -> Dict[str, Any]:
    """加载音频配置"""
    audio_config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'audio_config.yaml')
    
    if os.path.exists(audio_config_path):
        import yaml
        with open(audio_config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    # 默认音频配置
    return {
        'audio_mode': 'microphone',
        'asr': {
            'input_device': 'hw:3,0',  # USB Composite Device: Audio
            'sample_rate': 48000,  # USB设备支持的采样率
            'channels': 1,
            'chunk_duration': 0.05,
            'silence_threshold': 0.01,
            'silence_duration': 1.5,
            'min_recording_duration': 0.5,
            'vad_threshold': 0.005,
            'min_audio_length': 1.0
        },
        'tts': {
            'output_device': 'default',
            'voice': 'zh-CN-XiaoxiaoNeural',
            'rate': '+0%',
            'volume': '+0%'
        },
        'debug': {
            'enabled': False,
            'log_device_info': True
        }
    }

def load_config() -> Dict[str, Any]:
    """加载配置"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'llm_params.yaml')
    
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            llm_config = yaml.safe_load(f)
    else:
        llm_config = {}
    
    # 加载音频配置
    audio_config = load_audio_config()
    
    # 合并配置
    config = {
        'audio_mode': audio_config.get('audio_mode', 'microphone'),
        'asr': audio_config.get('asr', {}),
        'speech': {
            'model_name': 'small',
            'vad_threshold': audio_config.get('asr', {}).get('vad_threshold', 0.005),
            'min_audio_length': audio_config.get('asr', {}).get('min_audio_length', 1.0)
        },
        'llm': {
            'engine_type': 'zhipuai',
            'max_tokens': 100,
            'temperature': 0.1,
            'timeout': 5.0,
            'cache_timeout': 300
        },
        'tts': audio_config.get('tts', {}),
        'debug': audio_config.get('debug', {})
    }
    
    # 更新LLM配置（如果存在）
    if llm_config:
        config['llm'].update(llm_config.get('llm', {}))
    
    return config

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='VLA语言系统 - 支持语音和文本输入')
    parser.add_argument('--input_mode', choices=['voice', 'text'], default='voice',
                       help='输入模式 (voice=语音识别, text=文本输入)')
    parser.add_argument('--config', help='配置文件路径（可选）')
    
    args = parser.parse_args()
    
    print("🚀 开始启动VLA语言系统...")
    print(f"📝 输入模式: {args.input_mode}")
    
    try:
        # 加载配置
        print("📋 正在加载配置...")
        config = load_config()
        print("✅ 配置加载完成")
        
        # 初始化系统
        print("🔧 正在初始化系统...")
        system = ProductionSystem(config, args.input_mode)
        print("✅ 系统初始化完成")
        
        # 启动系统
        print("🚀 正在启动系统...")
        system.start()
        
    except Exception as e:
        print(f"❌ 系统启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
