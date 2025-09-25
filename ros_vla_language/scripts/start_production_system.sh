#!/bin/bash

# 正式版系统启动脚本
# Production System Startup Script

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_blue() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# 检查Python环境
check_python() {
    log_info "检查Python环境..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python3未安装"
        exit 1
    fi
    
    python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    log_info "Python版本: $python_version"
    
    # 检查关键依赖
    log_info "检查关键依赖..."
    
    # Whisper
    if python3 -c "import whisper" 2>/dev/null; then
        log_info "✅ Whisper已安装"
    else
        log_warn "⚠️ Whisper未安装，将安装: pip install openai-whisper"
        pip3 install openai-whisper
    fi
    
    # Edge TTS
    if python3 -c "import edge_tts" 2>/dev/null; then
        log_info "✅ Edge TTS已安装"
    else
        log_warn "⚠️ Edge TTS未安装，将安装: pip install edge-tts"
        pip3 install edge-tts
    fi
    
    # SoundDevice
    if python3 -c "import sounddevice" 2>/dev/null; then
        log_info "✅ SoundDevice已安装"
    else
        log_warn "⚠️ SoundDevice未安装，将安装: pip install sounddevice"
        pip3 install sounddevice
    fi
    
    # SoundFile
    if python3 -c "import soundfile" 2>/dev/null; then
        log_info "✅ SoundFile已安装"
    else
        log_warn "⚠️ SoundFile未安装，将安装: pip install soundfile"
        pip3 install soundfile
    fi
    
    # PyGame (用于音频播放)
    if python3 -c "import pygame" 2>/dev/null; then
        log_info "✅ PyGame已安装"
    else
        log_warn "⚠️ PyGame未安装，将安装: pip install pygame"
        pip3 install pygame
    fi
}

# 检查系统工具
check_system_tools() {
    log_info "检查系统工具..."
    
    # 检查FFmpeg
    if command -v ffmpeg &> /dev/null; then
        log_info "✅ FFmpeg已安装"
    else
        log_warn "⚠️ FFmpeg未安装，将安装..."
        apt-get update -qq
        apt-get install -y ffmpeg
    fi
    
    # 检查音频目录
    if [ ! -d "audio" ]; then
        log_info "创建音频目录..."
        mkdir -p audio
    fi
}


# 启动系统
start_system() {
    log_info "启动正式版系统..."
    log_blue "=============================================="
    log_blue "🎤 正式版语音识别+LLM+TTS系统"
    log_blue "实时语音录制 -> LLM分析 -> JSON生成 -> TTS反馈"
    log_blue "=============================================="
    log_blue ""
    
    # 启动系统
    python3 production_system.py
}

# 主函数
main() {
    log_blue "=============================================="
    log_blue "🚀 启动正式版语音识别+LLM+TTS系统"
    log_blue "=============================================="
    log_blue ""
    
    # 检查依赖
    check_python
    
    # 检查系统工具
    check_system_tools
    
  
    
    # 启动系统
    start_system
}

# 处理命令行参数
case "${1:-}" in
    "deps")
        log_info "安装依赖..."
        check_python
        ;;
    "help"|"--help"|"-h")
        echo "用法: $0 [选项]"
        echo ""
        echo "选项:"
        echo "  deps     安装依赖"
        echo "  help     显示此帮助信息"
        echo ""
        echo "默认: 启动完整系统"
        ;;
    *)
        main
        ;;
esac
