#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
麦克风测试脚本
Microphone Test Script

用于测试麦克风设备是否正常工作，包括：
1. 检测可用的音频设备
2. 测试麦克风录制功能
3. 实时显示音频电平
4. 保存测试音频文件
"""

import os
import sys
import time
import numpy as np
import threading
import queue
import argparse
import logging
import wave
import tempfile
import subprocess
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import sounddevice as sd
    import soundfile as sf
except ImportError as e:
    print(f"❌ 缺少必要的音频库: {e}")
    print("请安装: pip install sounddevice soundfile")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MicrophoneTester:
    """麦克风测试器"""
    
    def __init__(self, config=None):
        self.config = config or self.get_default_config()
        self.sample_rate = self.config.get('sample_rate', 16000)
        self.channels = self.config.get('channels', 1)
        self.chunk_duration = self.config.get('chunk_duration', 0.1)
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        self.test_duration = self.config.get('test_duration', 10)
        
        # 强制指定使用hw:3,0 (USB Composite Device: Audio)
        self.default_device = 'hw:0,0'
        
        # 状态变量
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.audio_levels = []
        self.max_level = 0.0
        self.recording_start_time = 0
        
        # 音频缓冲区
        self.audio_buffer = []
        
        logger.info(f"麦克风测试器初始化: {self.sample_rate}Hz, {self.channels}ch")
    
    def get_default_config(self):
        """获取默认配置"""
        return {
            'sample_rate': 48000,  # USB设备支持48000Hz
            'channels': 1,
            'chunk_duration': 0.1,
            'test_duration': 10,
            'silence_threshold': 0.01,
            'output_dir': '/tmp'
        }
    
    def list_audio_devices(self, force_refresh=False):
        """列出所有音频设备"""
        print("🎵 音频设备列表:")
        print("=" * 60)
        
        try:
            # 强制刷新设备列表
            if force_refresh:
                print("🔄 强制刷新音频设备列表...")
                # 尝试重新初始化音频系统
                try:
                    sd._terminate()
                    sd._initialize()
                    time.sleep(0.5)  # 等待设备重新初始化
                except:
                    pass
            
            devices = sd.query_devices()
            input_devices = []
            output_devices = []
            
            for i, dev in enumerate(devices):
                device_info = f"[{i}] {dev['name']}"
                if dev['max_input_channels'] > 0:
                    device_info += f" (输入: {dev['max_input_channels']}ch)"
                    input_devices.append((i, dev))
                if dev['max_output_channels'] > 0:
                    device_info += f" (输出: {dev['max_output_channels']}ch)"
                    output_devices.append((i, dev))
                
                print(device_info)
            
            print("\n📊 设备统计:")
            print(f"  输入设备: {len(input_devices)} 个")
            print(f"  输出设备: {len(output_devices)} 个")
            
            # 推荐默认输入设备
            if input_devices:
                default_input = sd.default.device[0]
                print(f"  默认输入设备: [{default_input}] {devices[default_input]['name']}")
            
            # 显示系统音频设备信息
            print("\n🔧 系统音频设备信息:")
            print(f"  默认输入设备: {sd.default.device[0]}")
            print(f"  默认输出设备: {sd.default.device[1]}")
            print(f"  默认采样率: {sd.default.samplerate}Hz")
            
            # 检查ALSA设备
            self._check_alsa_devices()
            
            return input_devices, output_devices
            
        except Exception as e:
            print(f"❌ 获取音频设备列表失败: {e}")
            return [], []
    
    def _check_alsa_devices(self):
        """检查ALSA设备"""
        try:
            print("\n🔍 ALSA设备信息:")
            # 检查/proc/asound/devices
            if os.path.exists('/proc/asound/devices'):
                with open('/proc/asound/devices', 'r') as f:
                    alsa_devices = f.read()
                    print("  ALSA设备列表:")
                    for line in alsa_devices.split('\n'):
                        if 'audio' in line.lower() or 'capture' in line.lower():
                            print(f"    {line.strip()}")
            
            # 检查arecord命令
            try:
                result = subprocess.run(['arecord', '-l'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    print("  arecord检测到的录音设备:")
                    print(result.stdout)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
        except Exception as e:
            print(f"  ⚠️ ALSA设备检查失败: {e}")
    
    def refresh_devices(self):
        """刷新音频设备列表"""
        print("🔄 正在刷新音频设备...")
        
        # 方法1: 使用sounddevice的刷新
        try:
            sd._terminate()
            sd._initialize()
            time.sleep(1)
        except:
            pass
        
        # 方法2: 重新查询设备
        try:
            devices = sd.query_devices()
            print(f"✅ 设备刷新完成，发现 {len(devices)} 个设备")
            return True
        except Exception as e:
            print(f"❌ 设备刷新失败: {e}")
            return False
    
    def calculate_audio_level(self, audio_data):
        """计算音频电平"""
        if len(audio_data) == 0:
            return 0.0
        
        # 计算RMS (Root Mean Square)
        rms = np.sqrt(np.mean(np.square(audio_data)))
        
        # 转换为分贝 (dB)
        if rms > 0:
            db = 20 * np.log10(rms)
        else:
            db = -np.inf
        
        # 归一化到0-1范围
        normalized_level = min(1.0, max(0.0, (db + 60) / 60))
        
        return normalized_level
    
    def audio_callback(self, indata, frames, time_info, status):
        """音频回调函数"""
        if status:
            logger.warning(f"音频回调状态: {status}")
        
        # 将音频数据放入队列
        self.audio_queue.put(indata.copy())
        
        # 计算音频电平
        level = self.calculate_audio_level(indata)
        self.audio_levels.append(level)
        self.max_level = max(self.max_level, level)
        
        # 添加到音频缓冲区
        self.audio_buffer.append(indata.copy())
    
    def start_recording(self, device=None):
        """开始录制"""
        if self.is_recording:
            print("⚠️ 已经在录制中")
            return False
        
        # 强制使用设备hw:3,0
        device = device or self.default_device
        
        try:
            print(f"🎤 开始录制音频 (设备: {device} - USB Composite Device: Audio)")
            self.is_recording = True
            self.audio_buffer = []
            self.audio_levels = []
            self.max_level = 0.0
            self.recording_start_time = time.time()
            
            # 尝试启动音频流，如果采样率不支持则尝试其他采样率
            stream_config = {
                'samplerate': self.sample_rate,
                'channels': self.channels,
                'callback': self.audio_callback,
                'blocksize': self.chunk_size,
                'dtype': np.float32,
                'device': device
            }
            
            try:
                self.stream = sd.InputStream(**stream_config)
            except Exception as e:
                print(f"⚠️ 采样率 {self.sample_rate}Hz 不支持，尝试 48000Hz...")
                stream_config['samplerate'] = 48000
                self.sample_rate = 48000
                self.chunk_size = int(self.sample_rate * self.chunk_duration)
                try:
                    self.stream = sd.InputStream(**stream_config)
                except Exception as e2:
                    print(f"⚠️ 采样率 48000Hz 也不支持，尝试 44100Hz...")
                    stream_config['samplerate'] = 44100
                    self.sample_rate = 44100
                    self.chunk_size = int(self.sample_rate * self.chunk_duration)
                    self.stream = sd.InputStream(**stream_config)
            self.stream.start()
            
            return True
            
        except Exception as e:
            print(f"❌ 启动录制失败: {e}")
            self.is_recording = False
            return False
    
    def stop_recording(self):
        """停止录制"""
        if not self.is_recording:
            print("⚠️ 没有在录制中")
            return None
        
        try:
            print("🛑 停止录制")
            self.is_recording = False
            
            # 停止音频流
            self.stream.stop()
            self.stream.close()
            
            # 等待剩余音频数据处理
            time.sleep(0.1)
            
            # 合并所有音频数据
            if self.audio_buffer:
                full_audio = np.concatenate(self.audio_buffer, axis=0)
                recording_duration = time.time() - self.recording_start_time
                print(f"录制完成，时长: {recording_duration:.2f}秒")
                return full_audio
            
            return None
            
        except Exception as e:
            print(f"❌ 停止录制失败: {e}")
            return None
    
    def display_audio_level(self):
        """实时显示音频电平"""
        if not self.is_recording:
            return
        
        # 获取最近的音频电平
        recent_levels = self.audio_levels[-10:] if self.audio_levels else [0]
        current_level = recent_levels[-1] if recent_levels else 0
        
        # 创建电平条
        bar_length = 50
        filled_length = int(bar_length * current_level)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        # 计算统计信息
        avg_level = np.mean(recent_levels) if recent_levels else 0
        peak_level = self.max_level
        
        # 显示电平信息
        duration = time.time() - self.recording_start_time
        print(f"\r🎤 [{bar}] {current_level:.2f} | 平均: {avg_level:.2f} | 峰值: {peak_level:.2f} | 时长: {duration:.1f}s", end='', flush=True)
    
    def save_audio_file(self, audio_data, filename=None):
        """保存音频文件"""
        if audio_data is None or len(audio_data) == 0:
            print("❌ 没有音频数据可保存")
            return None
        
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"microphone_test_{timestamp}.wav"
            
            filepath = os.path.join(self.config['output_dir'], filename)
            
            # 确保输出目录存在
            os.makedirs(self.config['output_dir'], exist_ok=True)
            
            # 保存音频文件
            sf.write(filepath, audio_data, self.sample_rate)
            
            print(f"✅ 音频文件已保存: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"❌ 保存音频文件失败: {e}")
            return None
    
    def run_test(self, device=None, duration=None, save_recording=True):
        """运行麦克风测试"""
        print("🚀 开始麦克风测试")
        print("🎤 强制使用设备hw:3,0: USB Composite Device: Audio")
        print("=" * 60)
        
        # 列出音频设备
        input_devices, _ = self.list_audio_devices()
        
        if not input_devices:
            print("❌ 没有找到可用的输入设备")
            return False
        
        # 设置测试参数
        test_duration = duration or self.test_duration
        
        # 强制使用设备hw:3,0
        device = device or self.default_device
        
        # 开始录制
        if not self.start_recording(device):
            return False
        
        print(f"\n🎤 正在录制音频 (测试时长: {test_duration}秒)")
        print("💡 请对着麦克风说话或制造声音")
        print("按 Ctrl+C 可以提前停止测试")
        print("-" * 60)
        
        try:
            # 实时显示音频电平
            start_time = time.time()
            while self.is_recording and (time.time() - start_time) < test_duration:
                self.display_audio_level()
                time.sleep(0.1)
            
            # 停止录制
            audio_data = self.stop_recording()
            
            if audio_data is not None:
                print("\n" + "=" * 60)
                print("📊 测试结果:")
                
                # 计算统计信息
                recording_duration = time.time() - self.recording_start_time
                avg_level = np.mean(self.audio_levels) if self.audio_levels else 0
                peak_level = self.max_level
                
                print(f"  录制时长: {recording_duration:.2f}秒")
                print(f"  平均电平: {avg_level:.3f}")
                print(f"  峰值电平: {peak_level:.3f}")
                print(f"  音频数据大小: {len(audio_data)} 样本")
                print(f"  采样率: {self.sample_rate}Hz")
                print(f"  声道数: {self.channels}")
                
                # 评估麦克风状态
                if peak_level < 0.01:
                    print("  ⚠️  警告: 音频电平过低，可能麦克风未正常工作")
                elif peak_level < 0.1:
                    print("  💡 提示: 音频电平较低，请检查麦克风音量设置")
                else:
                    print("  ✅ 麦克风工作正常")
                
                # 保存录音
                if save_recording:
                    self.save_audio_file(audio_data)
                
                return True
            else:
                print("❌ 录制失败，没有获得音频数据")
                return False
                
        except KeyboardInterrupt:
            print("\n\n🛑 用户中断测试")
            audio_data = self.stop_recording()
            
            if audio_data is not None and save_recording:
                self.save_audio_file(audio_data)
            
            return True
        
        except Exception as e:
            print(f"❌ 测试过程中发生错误: {e}")
            return False
    
    def run_continuous_test(self, device=None):
        """运行连续测试模式"""
        print("🔄 连续测试模式")
        print("🎤 强制使用设备hw:3,0: USB Composite Device: Audio")
        print("=" * 60)
        print("💡 此模式将持续监控麦克风输入")
        print("按 Ctrl+C 停止测试")
        print("-" * 60)
        
        # 强制使用设备hw:3,0
        device = device or self.default_device
        
        # 开始录制
        if not self.start_recording(device):
            return False
        
        try:
            while self.is_recording:
                self.display_audio_level()
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\n🛑 停止连续测试")
            self.stop_recording()
            return True
        
        except Exception as e:
            print(f"❌ 连续测试过程中发生错误: {e}")
            return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='麦克风测试工具')
    parser.add_argument('--device', type=int, help='指定音频设备ID')
    parser.add_argument('--duration', type=int, default=10, help='测试时长(秒)')
    parser.add_argument('--sample-rate', type=int, default=16000, help='采样率')
    parser.add_argument('--channels', type=int, default=1, help='声道数')
    parser.add_argument('--output-dir', default='/tmp', help='输出目录')
    parser.add_argument('--continuous', action='store_true', help='连续测试模式')
    parser.add_argument('--no-save', action='store_true', help='不保存录音文件')
    parser.add_argument('--list-devices', action='store_true', help='仅列出音频设备')
    parser.add_argument('--refresh', action='store_true', help='强制刷新音频设备列表')
    
    args = parser.parse_args()
    
    # 创建测试器配置
    config = {
        'sample_rate': args.sample_rate,
        'channels': args.channels,
        'test_duration': args.duration,
        'output_dir': args.output_dir
    }
    
    # 创建测试器
    tester = MicrophoneTester(config)
    
    # 强制刷新设备列表
    if args.refresh:
        print("🔄 强制刷新音频设备...")
        tester.refresh_devices()
        tester.list_audio_devices(force_refresh=True)
        return
    
    # 仅列出设备
    if args.list_devices:
        tester.list_audio_devices(force_refresh=True)
        return
    
    print("🎤 麦克风测试工具")
    print("=" * 60)
    
    # 运行测试
    if args.continuous:
        success = tester.run_continuous_test(args.device)
    else:
        success = tester.run_test(
            device=args.device,
            duration=args.duration,
            save_recording=not args.no_save
        )
    
    if success:
        print("\n✅ 测试完成")
        sys.exit(0)
    else:
        print("\n❌ 测试失败")
        sys.exit(1)

if __name__ == '__main__':
    main()
