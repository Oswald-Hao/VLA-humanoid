#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试 /proc/asound/cards 音频设备检测
Test /proc/asound/cards audio device detection
"""

def print_header(title: str):
    """打印标题"""
    print("=" * 60)
    print(f"🎵 {title}")
    print("=" * 60)

def print_section(title: str):
    """打印小节标题"""
    print(f"\n📋 {title}")
    print("-" * 40)

def detect_asound_cards():
    """使用 /proc/asound/cards 检测音频设备"""
    print_section("使用 /proc/asound/cards 检测音频设备")
    
    try:
        # 读取 /proc/asound/cards 文件
        with open('/proc/asound/cards', 'r') as f:
            content = f.read()
        
        print("📊 /proc/asound/cards 内容:")
        print(content)
        
        # 解析 cards 文件内容
        cards = []
        lines = content.strip().split('\n')
        
        for line in lines:
            # 跳过空行和缩进的行（详细信息）
            if not line.strip() or line.strip().startswith(' '):
                continue
            
            # 只处理包含数字ID的行（格式如：0 [PCH] : HDA-Intel - HDA Intel PCH）
            if line.strip()[0].isdigit():
                parts = line.split(':')
                if len(parts) >= 2:
                    # 提取数字ID（行首的数字）
                    card_part = parts[0].strip()
                    card_id = card_part.split(' ')[0]  # 获取数字ID
                    
                    card_info = parts[1].strip()
                    cards.append({
                        'id': card_id,
                        'info': card_info
                    })
        
        if cards:
            print(f"\n🎵 检测到 {len(cards)} 个音频卡:")
            for card in cards:
                print(f"  卡 [{card['id']}]: {card['info']}")
                
                # 为每个卡提供ALSA设备名称建议
                print(f"    ALSA设备名称建议:")
                print(f"      - hw:{card['id']},0 (默认设备)")
                print(f"      - plughw:{card['id']},0 (插件设备)")
                print(f"      - default:{card['id']} (默认设备)")
        
        return cards
        
    except FileNotFoundError:
        print("❌ /proc/asound/cards 文件不存在")
        return None
    except Exception as e:
        print(f"❌ 读取 /proc/asound/cards 失败: {e}")
        return None

def generate_config_recommendation(cards):
    """基于 /proc/asound/cards 生成配置建议"""
    print_section("基于 /proc/asound/cards 的配置建议")
    
    if not cards:
        print("❌ 无法生成配置建议，设备检测失败")
        return
    
    print("📝 根据检测结果，建议的audio_config.yaml配置:")
    print()
    
    # 查找USB设备（通常是最好的选择）
    usb_card = None
    builtin_card = None
    nvidia_card = None
    
    for card in cards:
        if 'USB' in card['info']:
            usb_card = card
        elif 'NVidia' in card['info']:
            nvidia_card = card
        else:
            builtin_card = card
    
    # ASR输入设备建议
    print("ASR (语音识别) 输入设备建议:")
    if usb_card:
        print(f"  ✅ 推荐使用USB设备: hw:{usb_card['id']},0")
        print(f"     设备信息: {usb_card['info']}")
    elif builtin_card:
        print(f"  ⚠️ 无USB设备，使用内置设备: hw:{builtin_card['id']},0")
        print(f"     设备信息: {builtin_card['info']}")
    else:
        print("  ❌ 未找到合适的输入设备")
    print(f"  在配置文件中设置为: asr.input_device: \"hw:{usb_card['id'] if usb_card else (builtin_card['id'] if builtin_card else 'default')},0\"")
    
    print()
    
    # TTS输出设备建议
    print("TTS (语音合成) 输出设备建议:")
    if usb_card:
        print(f"  ✅ 推荐使用USB设备: hw:{usb_card['id']},0")
        print(f"     设备信息: {usb_card['info']}")
    elif builtin_card:
        print(f"  ⚠️ 使用内置设备: hw:{builtin_card['id']},0")
        print(f"     设备信息: {builtin_card['info']}")
    elif nvidia_card:
        print(f"  ⚠️ 使用NVIDIA设备: hw:{nvidia_card['id']},0")
        print(f"     设备信息: {nvidia_card['info']}")
    else:
        print("  ❌ 未找到合适的输出设备")
    print(f"  在配置文件中设置为: tts.output_device: \"hw:{usb_card['id'] if usb_card else (builtin_card['id'] if builtin_card else (nvidia_card['id'] if nvidia_card else 'default'))},0\"")
    
    print()
    print("💡 提示:")
    print("  1. USB设备通常是最佳选择，音质更好且延迟更低")
    print("  2. 如果USB设备不工作，可以尝试内置设备")
    print("  3. ALSA设备名称格式: hw:card_id,device_id")
    print("  4. 也可以使用 'default' 作为设备名称")

def main():
    """主函数"""
    print_header("/proc/asound/cards 音频设备检测工具")
    print("本工具使用 /proc/asound/cards 检测系统中的音频设备")
    print("为配置 audio_config.yaml 提供参考")
    print()
    
    # 检测 /proc/asound/cards 设备
    cards = detect_asound_cards()
    
    # 生成配置建议
    if cards:
        generate_config_recommendation(cards)
    
    print("\n" + "=" * 60)
    print("✅ 音频设备检测完成")
    print("💡 请根据上述结果修改 audio_config.yaml 文件")
    print("=" * 60)

if __name__ == '__main__':
    main()
