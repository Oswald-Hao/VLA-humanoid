#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
音频设备检测脚本
Audio Device Detection Script

用于检测系统中的音频输入和输出设备，帮助用户配置audio_config.yaml
"""

import os
import sys
import subprocess
import json
import sounddevice as sd
from typing import List, Dict, Any

def print_header(title: str):
    """打印标题"""
    print("=" * 60)
    print(f"🎵 {title}")
    print("=" * 60)

def print_section(title: str):
    """打印小节标题"""
    print(f"\n📋 {title}")
    print("-" * 40)

def detect_python_audio_devices():
    """使用sounddevice库检测音频设备"""
    print_section("使用Python sounddevice库检测音频设备")
    
    try:
        # 获取所有设备
        devices = sd.query_devices()
        print(f"📊 检测到 {len(devices)} 个音频设备:\n")
        
        input_devices = []
        output_devices = []
        
        for i, device in enumerate(devices):
            device_info = {
                'id': i,
                'name': device['name'],
                'hostapi': device['hostapi'],
                'max_input_channels': device['max_input_channels'],
                'max_output_channels': device['max_output_channels'],
                'default_sample_rate': device['default_sample_rate']
            }
            
            # 分类设备
            if device['max_input_channels'] > 0:
                input_devices.append(device_info)
            if device['max_output_channels'] > 0:
                output_devices.append(device_info)
            
            # 打印设备信息
            device_type = []
            if device['max_input_channels'] > 0:
                device_type.append(f"输入({device['max_input_channels']}ch)")
            if device['max_output_channels'] > 0:
                device_type.append(f"输出({device['max_output_channels']}ch)")
            
            print(f"设备 [{i}]: {device['name']}")
            print(f"  类型: {', '.join(device_type)}")
            print(f"  采样率: {device['default_sample_rate']}Hz")
            print(f"  Host API: {device['hostapi']}")
            print()
        
        # 打印输入设备推荐
        if input_devices:
            print("🎤 推荐的输入设备 (用于ASR语音识别):")
            for device in input_devices:
                if 'USB' in device['name'] or 'Microphone' in device['name']:
                    print(f"  ✓ 设备 [{device['id']}]: {device['name']} (推荐)")
                else:
                    print(f"  - 设备 [{device['id']}]: {device['name']}")
        
        # 打印输出设备推荐
        if output_devices:
            print("\n🔊 推荐的输出设备 (用于TTS语音播放):")
            for device in output_devices:
                if 'USB' in device['name'] or 'Speaker' in device['name'] or 'Headphone' in device['name']:
                    print(f"  ✓ 设备 [{device['id']}]: {device['name']} (推荐)")
                else:
                    print(f"  - 设备 [{device['id']}]: {device['name']}")
        
        return {
            'input_devices': input_devices,
            'output_devices': output_devices
        }
        
    except Exception as e:
        print(f"❌ Python sounddevice检测失败: {e}")
        return None

def detect_alsa_devices():
    """使用ALSA检测音频设备"""
    print_section("使用ALSA检测音频设备")
    
    try:
        # 检测ALSA设备
        result = subprocess.run(['aplay', '-l'], capture_output=True, text=True)
        if result.returncode == 0:
            print("🔊 ALSA输出设备:")
            print(result.stdout)
        
        result = subprocess.run(['arecord', '-l'], capture_output=True, text=True)
        if result.returncode == 0:
            print("\n🎤 ALSA输入设备:")
            print(result.stdout)
            
    except FileNotFoundError:
        print("⚠️ ALSA工具未安装，跳过ALSA设备检测")
    except Exception as e:
        print(f"❌ ALSA检测失败: {e}")

def detect_pulseaudio_devices():
    """检测PulseAudio设备"""
    print_section("使用PulseAudio检测音频设备")
    
    try:
        # 检测PulseAudio设备
        result = subprocess.run(['pactl', 'list', 'sources'], capture_output=True, text=True)
        if result.returncode == 0:
            print("🎤 PulseAudio输入设备:")
            lines = result.stdout.split('\n')
            current_device = ""
            for line in lines:
                if line.strip().startswith('Name:'):
                    current_device = line.split(':')[1].strip()
                    print(f"  设备: {current_device}")
                elif line.strip().startswith('device.description') and current_device:
                    description = line.split('=')[1].strip().strip('"')
                    print(f"    描述: {description}")
                    current_device = ""
        
        result = subprocess.run(['pactl', 'list', 'sinks'], capture_output=True, text=True)
        if result.returncode == 0:
            print("\n🔊 PulseAudio输出设备:")
            lines = result.stdout.split('\n')
            current_device = ""
            for line in lines:
                if line.strip().startswith('Name:'):
                    current_device = line.split(':')[1].strip()
                    print(f"  设备: {current_device}")
                elif line.strip().startswith('device.description') and current_device:
                    description = line.split('=')[1].strip().strip('"')
                    print(f"    描述: {description}")
                    current_device = ""
            
    except FileNotFoundError:
        print("⚠️ PulseAudio工具未安装，跳过PulseAudio设备检测")
    except Exception as e:
        print(f"❌ PulseAudio检测失败: {e}")

def generate_config_recommendation(devices_info: Dict[str, List[Dict[Any, Any]]]):
    """生成配置建议"""
    print_section("配置建议")
    
    if not devices_info:
        print("❌ 无法生成配置建议，设备检测失败")
        return
    
    input_devices = devices_info.get('input_devices', [])
    output_devices = devices_info.get('output_devices', [])
    
    print("📝 根据检测结果，建议的audio_config.yaml配置:")
    print()
    
    # ASR输入设备建议
    print("ASR (语音识别) 输入设备建议:")
    recommended_input = None
    for device in input_devices:
        if 'USB' in device['name'] or 'Microphone' in device['name']:
            recommended_input = device
            break
    
    if recommended_input:
        print(f"  推荐使用设备索引: {recommended_input['id']}")
        print(f"  设备名称: {recommended_input['name']}")
        print(f"  在配置文件中设置为: asr.input_device: {recommended_input['id']}")
    else:
        print("  未找到推荐的输入设备，建议使用默认设备")
        print("  在配置文件中设置为: asr.input_device: null")
    
    print()
    
    # TTS输出设备建议
    print("TTS (语音合成) 输出设备建议:")
    recommended_output = None
    for device in output_devices:
        if 'USB' in device['name'] or 'Speaker' in device['name'] or 'Headphone' in device['name']:
            recommended_output = device
            break
    
    if recommended_output:
        print(f"  推荐使用设备索引: {recommended_output['id']}")
        print(f"  设备名称: {recommended_output['name']}")
        print("  对于ALSA设备，可能需要使用hw:X,Y格式")
        print("  建议先尝试: tts.output_device: \"hw:2,0\"")
        print(f"  如果hw:2,0不工作，可以尝试设备索引: {recommended_output['id']}")
    else:
        print("  未找到推荐的输出设备，建议使用默认设备")
        print("  在配置文件中设置为: tts.output_device: \"default\"")
    
    print()
    print("💡 提示:")
    print("  1. 对于ASR，可以使用设备索引(数字)或设备名称")
    print("  2. 对于TTS，建议使用ALSA设备名称格式，如: hw:2,0, default, pulse")
    print("  3. 如果设备不工作，可以尝试其他设备或使用默认值")
    print("  4. USB设备通常是更好的选择")

def test_device(device_id=None, device_type="input"):
    """测试设备"""
    print(f"\n🧪 测试设备: {device_id} ({device_type})")
    
    try:
        if device_type == "input":
            # 测试输入设备
            duration = 3  # 3秒测试录音
            print(f"🎤 开始{duration}秒测试录音...")
            
            recording = sd.rec(
                int(duration * 16000),
                samplerate=16000,
                channels=1,
                dtype='float32',
                device=device_id
            )
            sd.wait()
            
            if len(recording) > 0:
                print("✅ 输入设备测试成功")
                print(f"   录制了 {len(recording)} 个采样点")
                return True
            else:
                print("❌ 输入设备测试失败 - 没有录制到数据")
                return False
                
        elif device_type == "output":
            # 测试输出设备
            print("🔊 播放测试音频...")
            
            # 生成测试音频
            duration = 2  # 2秒测试音频
            sample_rate = 22050
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            frequency = 440  # A4音符
            test_audio = 0.3 * np.sin(2 * np.pi * frequency * t)
            
            sd.play(test_audio, sample_rate, device=device_id)
            sd.wait()
            
            print("✅ 输出设备测试成功")
            return True
            
    except Exception as e:
        print(f"❌ 设备测试失败: {e}")
        return False

def main():
    """主函数"""
    print_header("音频设备检测工具")
    print("本工具将帮助您检测系统中的音频设备，为配置audio_config.yaml提供参考")
    print()
    
    # 检测Python音频设备
    devices_info = detect_python_audio_devices()
    
    # 检测ALSA设备
    detect_alsa_devices()
    
    # 检测PulseAudio设备
    detect_pulseaudio_devices()
    
    # 生成配置建议
    if devices_info:
        generate_config_recommendation(devices_info)
    
    # 询问是否测试设备
    print_section("设备测试")
    print("🧪 您可以测试特定的音频设备")
    
    try:
        choice = input("\n是否要测试特定设备? (y/n): ").strip().lower()
        if choice == 'y':
            # 测试输入设备
            input_choice = input("输入要测试的输入设备ID (直接回车跳过): ").strip()
            if input_choice:
                device_id = int(input_choice) if input_choice.isdigit() else input_choice
                test_device(device_id, "input")
            
            # 测试输出设备
            output_choice = input("输入要测试的输出设备ID (直接回车跳过): ").strip()
            if output_choice:
                device_id = int(output_choice) if output_choice.isdigit() else output_choice
                test_device(device_id, "output")
                
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断测试")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
    
    print("\n" + "=" * 60)
    print("✅ 音频设备检测完成")
    print("💡 请根据上述结果修改 audio_config.yaml 文件")
    print("=" * 60)

if __name__ == '__main__':
    # 检查必要的依赖
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError as e:
        print(f"❌ 缺少必要的依赖: {e}")
        print("请安装: pip install sounddevice numpy")
        sys.exit(1)
    
    main()
