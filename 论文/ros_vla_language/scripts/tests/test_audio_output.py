#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
精确音频设备测试脚本
"""

import os
import sys
import subprocess
import tempfile
import asyncio
import edge_tts

async def test_audio_output():
    """测试所有可能的音频输出方式"""
    
    print("🔍 精确音频设备测试")
    print("=" * 50)
    
    # 生成TTS音频
    communicate = edge_tts.Communicate(
        text="你好，这是音频测试。",
        voice='zh-CN-XiaoxiaoNeural'
    )
    
    audio_data = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data.extend(chunk["data"])
    
    # 保存到临时文件
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_filename = f.name
        f.write(audio_data)
    
    print(f"📁 测试音频文件: {temp_filename}")
    print(f"📊 文件大小: {len(audio_data)} 字节")
    
    # 测试不同的播放方法
    playback_methods = [
        ("USB设备 hw:2,0", ['ffplay', '-autoexit', '-nodisp', '-i', temp_filename], {'SDL_AUDIODRIVER': 'alsa', 'AUDIODEV': 'hw:2,0'}),
        ("USB设备 plughw:2,0", ['ffplay', '-autoexit', '-nodisp', '-i', temp_filename], {'SDL_AUDIODRIVER': 'alsa', 'AUDIODEV': 'plughw:2,0'}),
        ("系统默认", ['ffplay', '-autoexit', '-nodisp', '-i', temp_filename], {}),
        ("直接ALSA", ['aplay', '-D', 'hw:2,0', temp_filename], {}),
        ("ALSA插件", ['aplay', '-D', 'plughw:2,0', temp_filename], {}),
    ]
    
    for method_name, command, env_vars in playback_methods:
        print(f"\n🔊 测试方法: {method_name}")
        
        try:
            env = os.environ.copy()
            env.update(env_vars)
            
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                timeout=10,
                env=env
            )
            
            if result.returncode == 0:
                print(f"   ✅ {method_name} 播放成功")
            else:
                print(f"   ❌ {method_name} 播放失败: {result.stderr}")
                
        except FileNotFoundError as e:
            print(f"   ❌ 命令不存在: {e}")
        except subprocess.TimeoutExpired:
            print(f"   ❌ {method_name} 播放超时")
        except Exception as e:
            print(f"   ❌ {method_name} 播放异常: {e}")
    
    # 检查USB设备状态
    print(f"\n🔍 USB设备状态检查:")
    print("=" * 50)
    
    try:
        # 检查USB音频设备详细信息
        result = subprocess.run(['cat', '/proc/asound/cards'], 
                              capture_output=True, text=True, timeout=5)
        print("音频设备列表:")
        print(result.stdout)
        
        # 检查设备是否真的存在
        if os.path.exists('/proc/asound/card2'):
            print("✅ USB音频设备card2存在")
        else:
            print("❌ USB音频设备card2不存在")
            
    except Exception as e:
        print(f"❌ 设备检查失败: {e}")
    
    # 保留文件供手动测试
    print(f"\n📁 手动测试命令:")
    print(f"   ffplay -i {temp_filename}")
    print(f"   aplay -D hw:2,0 {temp_filename}")
    print(f"   aplay -D plughw:2,0 {temp_filename}")
    
    # 清理
    try:
        os.unlink(temp_filename)
    except:
        pass

if __name__ == '__main__':
    asyncio.run(test_audio_output())