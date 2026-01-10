#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
直接音频输出测试
"""

import os
import sys
import subprocess
import tempfile
import time
import signal

def generate_simple_tone():
    """生成一个简单的正弦波音频文件"""
    # 使用sox生成测试音频
    test_audio = "/tmp/test_sine.wav"
    
    try:
        # 生成1kHz正弦波，持续2秒
        result = subprocess.run([
            'sox', '-n', test_audio, 
            'synth', '2', 'sine', '1000', 
            'vol', '0.8'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            return test_audio
        else:
            print(f"❌ sox生成失败: {result.stderr}")
            return None
            
    except FileNotFoundError:
        print("❌ sox不可用")
        return None

def test_direct_playback():
    """直接测试音频播放"""
    
    print("🔍 直接音频输出测试")
    print("=" * 50)
    
    # 生成测试音频
    audio_file = generate_simple_tone()
    if not audio_file:
        print("❌ 无法生成测试音频")
        return
    
    print(f"📁 测试音频: {audio_file}")
    
    # 检查文件
    if os.path.exists(audio_file):
        file_size = os.path.getsize(audio_file)
        print(f"📊 文件大小: {file_size} 字节")
    else:
        print("❌ 音频文件不存在")
        return
    
    # 测试不同的播放命令
    test_commands = [
        ("USB设备hw:2,0", ['ffplay', '-autoexit', '-nodisp', '-i', audio_file], {'SDL_AUDIODRIVER': 'alsa', 'AUDIODEV': 'hw:2,0'}),
        ("USB设备plughw:2,0", ['ffplay', '-autoexit', '-nodisp', '-i', audio_file], {'SDL_AUDIODRIVER': 'alsa', 'AUDIODEV': 'plughw:2,0'}),
        ("默认设备", ['ffplay', '-autoexit', '-nodisp', '-i', audio_file], {}),
    ]
    
    for name, cmd, env_vars in test_commands:
        print(f"\n🔊 测试: {name}")
        print(f"   命令: {' '.join(cmd)}")
        
        try:
            env = os.environ.copy()
            env.update(env_vars)
            
            # 使用subprocess.Popen来获取实时输出
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 等待进程完成或超时
            try:
                stdout, stderr = proc.communicate(timeout=15)
                
                if proc.returncode == 0:
                    print(f"   ✅ {name} 成功")
                    if stderr:
                        print(f"   输出: {stderr[:200]}...")
                else:
                    print(f"   ❌ {name} 失败 (代码: {proc.returncode})")
                    if stderr:
                        print(f"   错误: {stderr[:500]}")
                    
            except subprocess.TimeoutExpired:
                proc.kill()
                print(f"   ❌ {name} 超时")
                
        except Exception as e:
            print(f"   ❌ {name} 异常: {e}")
    
    # 系统信息
    print(f"\n🔍 系统音频信息:")
    print("=" * 50)
    
    # 检查音频相关进程
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        audio_processes = [line for line in result.stdout.split('\n') if any(x in line.lower() for x in ['audio', 'sound', 'pulse', 'alsa'])]
        
        if audio_processes:
            print("音频相关进程:")
            for proc in audio_processes:
                if proc.strip():
                    print(f"   {proc}")
        else:
            print("   没有发现音频相关进程")
    except Exception as e:
        print(f"   进程检查失败: {e}")
    
    # 清理
    try:
        os.unlink(audio_file)
    except:
        pass

if __name__ == '__main__':
    test_direct_playback()