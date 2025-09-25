#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TTS功能测试脚本 - 流式播放音频到扬声器
"""

import os
import sys
import time
import asyncio
import subprocess
import tempfile
import io
import edge_tts
import pygame

async def test_tts_streaming():
    """测试TTS流式播放功能"""
    test_text = "你好，这是一个TTS流式播放测试。"
    
    print(f"🎵 测试TTS流式播放: {test_text}")
    
    try:
        # 创建Edge TTS通信对象
        communicate = edge_tts.Communicate(
            text=test_text,
            voice='zh-CN-XiaoxiaoNeural',
            rate='+0%',
            volume='+0%'
        )
        
        print("🎵 开始流式生成和播放TTS音频...")
        
        # 方法1: 优先使用子进程进行流式播放（更可靠）
        print("\n🎵 尝试子进程流式播放...")
        subprocess_success = await stream_with_subprocess(communicate)
        
        # 方法2: 如果子进程失败，尝试pygame流式播放
        if not subprocess_success:
            print("\n🎵 子进程失败，尝试pygame流式播放...")
            pygame_success = await stream_with_pygame(communicate)
        else:
            pygame_success = True
        
        # 方法3: 如果流式播放都失败，使用传统文件播放方式作为对比
        if not (pygame_success or subprocess_success):
            print("\n🎵 流式播放失败，测试传统文件播放方式作为对比...")
            await test_traditional_file_playback(test_text)
        else:
            print("\n🎵 流式播放成功，跳过传统文件播放测试")
        
        print("✅ TTS流式播放测试完成")
        
    except Exception as e:
        print(f"❌ TTS流式播放测试失败: {str(e)}")
        import traceback
        print(f"❌ 错误详情: {traceback.format_exc()}")

async def stream_with_pygame(communicate):
    """使用pygame进行流式播放"""
    print("🎵 尝试使用pygame进行流式播放...")
    
    try:
        # 设置环境变量避免X11问题
        os.environ['SDL_AUDIODRIVER'] = 'alsa'
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        
        # 初始化pygame音频，不初始化视频
        pygame.mixer.init(frequency=24000, size=-16, channels=2, buffer=4096)
        print("✅ pygame音频初始化成功")
        
        # 收集所有音频数据
        audio_data = bytearray()
        
        # 流式接收音频数据
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])
        
        if len(audio_data) == 0:
            raise Exception("未接收到音频数据")
        
        print(f"🎵 接收到音频数据大小: {len(audio_data)} 字节")
        
        # 创建音频对象并播放
        audio_buffer = io.BytesIO(audio_data)
        pygame.mixer.music.load(audio_buffer)
        
        print("🎵 开始播放音频...")
        pygame.mixer.music.play()
        
        # 等待播放完成
        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.05)
        
        print("✅ pygame流式播放完成")
        
    except Exception as e:
        print(f"❌ pygame流式播放失败: {str(e)}")
        return False
    finally:
        # 清理pygame资源
        try:
            pygame.mixer.quit()
            print("🎵 pygame资源已释放")
        except:
            pass
    
    return True

async def stream_with_subprocess(communicate):
    """使用子进程进行流式播放"""
    print("🎵 尝试使用子进程进行流式播放...")
    
    try:
        # 收集所有音频数据
        audio_data = bytearray()
        
        # 流式接收音频数据
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])
        
        if len(audio_data) == 0:
            raise Exception("未接收到音频数据")
        
        print(f"🎵 接收到音频数据大小: {len(audio_data)} 字节")
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_filename = temp_file.name
        
        # 写入音频数据
        with open(temp_filename, 'wb') as f:
            f.write(audio_data)
        
        print("🎵 开始播放音频...")
        
        # 方法1: 尝试使用USB音频设备直接播放（最可靠）
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"🎵 尝试使用USB音频设备直接播放 (尝试 {attempt + 1}/{max_retries})...")
                
                # 强制终止可能占用设备的进程
                if attempt > 0:
                    subprocess.run(['pkill', '-f', 'ffplay'], capture_output=True)
                    time.sleep(1)
                
                # 设置环境变量指定USB音频设备
                env = os.environ.copy()
                env['SDL_AUDIODRIVER'] = 'alsa'
                env['AUDIODEV'] = 'hw:1,0'  # USB音频设备
                
                # 添加更多ALSA环境变量
                env['ALSA_PCM_CARD'] = '2'
                env['ALSA_PCM_DEVICE'] = '0'
                
                result = subprocess.run([
                    'ffplay', 
                    '-autoexit', 
                    '-nodisp', 
                    '-i', temp_filename
                ], 
                capture_output=True, text=True, timeout=15, env=env)
                
                if result.returncode == 0:
                    print("✅ 子进程流式播放完成 (使用USB音频设备)")
                    return True
                else:
                    print(f"❌ USB音频设备播放失败: {result.stderr}")
                    if attempt < max_retries - 1:
                        print(f"⏳ 等待 2 秒后重试...")
                        time.sleep(2)
                        
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
                print(f"❌ USB音频设备播放异常: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"⏳ 等待 2 秒后重试...")
                    time.sleep(2)
        
        # 方法2: 尝试使用aplay（ALSA工具）
        try:
            print("🎵 尝试使用aplay播放...")
            result = subprocess.run(['aplay', '--device=default', temp_filename], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ 子进程流式播放完成 (使用aplay)")
                return True
            else:
                print(f"❌ aplay播放失败: {result.stderr}")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print("❌ aplay不可用，尝试其他方法...")
        
        # 方法3: 尝试使用ffplay标准方式
        try:
            print("🎵 尝试使用ffplay标准播放...")
            result = subprocess.run(['ffplay', '-autoexit', '-nodisp', temp_filename], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ 子进程流式播放完成 (使用ffplay)")
                return True
            else:
                print(f"❌ ffplay播放失败: {result.stderr}")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print("❌ ffplay不可用，尝试paplay...")
        
        # 方法4: 尝试使用paplay
        try:
            print("🎵 尝试使用paplay播放...")
            result = subprocess.run(['paplay', temp_filename], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ 子进程流式播放完成 (使用paplay)")
                return True
            else:
                print(f"❌ paplay播放失败: {result.stderr}")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print("❌ paplay不可用")
        
        print("❌ 所有播放方法都失败了")
        print("💡 调试信息：")
        print(f"   临时文件路径: {temp_filename}")
        print(f"   文件大小: {len(audio_data)} 字节")
        print("   请手动测试音频文件:")
        print(f"   ffplay {temp_filename}")
        print(f"   aplay {temp_filename}")
        
        return False
        
    except Exception as e:
        print(f"❌ 子进程流式播放失败: {str(e)}")
        import traceback
        print(f"❌ 错误详情: {traceback.format_exc()}")
        return False
    finally:
        # 清理临时文件
        try:
            if 'temp_filename' in locals():
                os.unlink(temp_filename)
                print("🎵 临时文件已清理")
        except:
            pass

async def test_traditional_file_playback(text):
    """测试传统文件播放方式作为对比"""
    try:
        # 创建临时音频文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_filename = temp_file.name
        
        # 同时保存一个固定文件名用于调试
        debug_filename = "/tmp/test_tts_output.wav"
        
        # 创建新的Edge TTS通信对象
        communicate = edge_tts.Communicate(
            text=text,
            voice='zh-CN-XiaoxiaoNeural',
            rate='+0%',
            volume='+0%'
        )
        
        print("🎵 使用传统方式生成TTS音频...")
        
        # 使用Edge TTS的标准保存方法
        await communicate.save(temp_filename)
        
        print("🎵 TTS音频生成完成")
        
        # 复制到调试文件
        import shutil
        shutil.copy2(temp_filename, debug_filename)
        print(f"🎵 音频文件已保存: {temp_filename}")
        print(f"🎵 调试音频文件已保存: {debug_filename}")
        
        # 验证文件
        if os.path.exists(temp_filename):
            file_size = os.path.getsize(temp_filename)
            print(f"🎵 音频文件大小: {file_size} 字节")
        else:
            raise Exception("音频文件生成失败")
        
        # 尝试播放音频
        print("🎵 开始播放音频到扬声器...")
        
        播放成功 = False
        
        # 方法1: 使用ffplay
        try:
            print("🎵 尝试使用ffplay播放...")
            result = subprocess.run(['ffplay', '-autoexit', '-nodisp', temp_filename], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("🎵 音频播放完成 (使用ffplay)")
                播放成功 = True
            else:
                print(f"❌ ffplay播放失败: {result.stderr}")
                
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print("❌ ffplay不可用，尝试其他方法...")
        
        if not 播放成功:
            # 方法2: 使用aplay
            try:
                print("🎵 尝试使用aplay播放...")
                result = subprocess.run(['aplay', temp_filename], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print("🎵 音频播放完成 (使用aplay)")
                    播放成功 = True
                else:
                    print(f"❌ aplay播放失败: {result.stderr}")
                    
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                print("❌ aplay不可用，尝试其他方法...")
        
        if not 播放成功:
            # 方法3: 使用paplay
            try:
                print("🎵 尝试使用paplay播放...")
                result = subprocess.run(['paplay', temp_filename], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print("🎵 音频播放完成 (使用paplay)")
                    播放成功 = True
                else:
                    print(f"❌ paplay播放失败: {result.stderr}")
                    
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                print("❌ paplay不可用")
        
        if not 播放成功:
            print("❌ 所有播放方法都失败了")
            print("💡 请手动测试音频文件:")
            print(f"   ffplay {debug_filename}")
            print(f"   aplay {debug_filename}")
            print(f"   paplay {debug_filename}")
        
        # 清理临时文件
        try:
            os.unlink(temp_filename)
            print("🎵 临时文件已清理")
        except:
            pass
        
    except Exception as e:
        print(f"❌ 传统文件播放测试失败: {str(e)}")
        import traceback
        print(f"❌ 错误详情: {traceback.format_exc()}")

async def test_tts():
    """测试TTS功能（保持向后兼容）"""
    await test_tts_streaming()

if __name__ == '__main__':
    print("🚀 开始TTS功能测试...")
    
    # 运行TTS测试
    asyncio.run(test_tts())
    
    print("🏁 TTS测试结束")
