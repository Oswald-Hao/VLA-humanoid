#!/bin/bash

# VLA Vision Scheduler Start Script
# 视觉调度器启动脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查ROS环境
check_ros_environment() {
    log_info "检查ROS环境..."
    
    if [ -z "$ROS_DISTRO" ]; then
        log_error "ROS环境未设置，请先运行: source /opt/ros/<your_ros_version>/setup.bash"
        exit 1
    fi
    
    log_success "ROS环境检查通过: $ROS_DISTRO"
}

# 检查工作空间
check_workspace() {
    log_info "检查工作空间..."
    
    if [ ! -f "devel/setup.bash" ]; then
        log_error "工作空间未编译，请先运行: catkin_make"
        exit 1
    fi
    
    # 激活工作空间
    source devel/setup.bash
    log_success "工作空间检查通过"
}

# 检查依赖
check_dependencies() {
    log_info "检查依赖..."
    
    # 检查Python依赖
    python3 -c "import rospy" 2>/dev/null || {
        log_error "rospy未安装，请运行: pip install rospy"
        exit 1
    }
    
    python3 -c "import cv2" 2>/dev/null || {
        log_warning "OpenCV未安装，某些功能可能不可用"
    }
    
    python3 -c "import numpy" 2>/dev/null || {
        log_error "numpy未安装，请运行: pip install numpy"
        exit 1
    }
    
    log_success "依赖检查通过"
}

# 启动视觉调度器
start_vision_scheduler() {
    log_info "启动VLA视觉调度器..."
    
    # 设置参数
    ENABLE_VLM=${ENABLE_VLM:-true}
    ENABLE_YOLO=${ENABLE_YOLO:-true}
    ENABLE_DEPTH=${ENABLE_DEPTH:-true}
    DEFAULT_CAMERA=${DEFAULT_CAMERA:-camera_1}
    DEFAULT_DEPTH_CAMERA=${DEFAULT_DEPTH_CAMERA:-depth_cam_1}
    CONFIDENCE_THRESHOLD=${CONFIDENCE_THRESHOLD:-0.5}
    
    # 设置环境变量
    export ENABLE_VLM=$ENABLE_VLM
    export ENABLE_YOLO=$ENABLE_YOLO
    export ENABLE_DEPTH=$ENABLE_DEPTH
    export DEFAULT_CAMERA=$DEFAULT_CAMERA
    export DEFAULT_DEPTH_CAMERA=$DEFAULT_DEPTH_CAMERA
    export CONFIDENCE_THRESHOLD=$CONFIDENCE_THRESHOLD
    
    log_info "启动参数:"
    log_info "  - VLM模块: $ENABLE_VLM"
    log_info "  - YOLO模块: $ENABLE_YOLO"
    log_info "  - 深度模块: $ENABLE_DEPTH"
    log_info "  - 默认相机: $DEFAULT_CAMERA"
    log_info "  - 默认深度相机: $DEFAULT_DEPTH_CAMERA"
    log_info "  - 置信度阈值: $CONFIDENCE_THRESHOLD"
    
    # 启动调度器
    if [ "$1" = "test" ]; then
        log_info "启动测试模式..."
        roslaunch ros_vla_vision vla_vision_scheduler.launch &
        SCHEDULER_PID=$!
        
        # 等待调度器启动
        sleep 5
        
        # 运行测试
        log_info "运行测试脚本..."
        python3 $(rospack find ros_vla_vision)/scripts/test_vision_scheduler.py
        
        # 清理
        kill $SCHEDULER_PID 2>/dev/null || true
        wait $SCHEDULER_PID 2>/dev/null || true
        
    else
        log_info "启动生产模式..."
        roslaunch ros_vla_vision vla_vision_scheduler.launch
    fi
}

# 显示帮助信息
show_help() {
    echo "VLA视觉调度器启动脚本"
    echo ""
    echo "用法: $0 [选项] [模式]"
    echo ""
    echo "选项:"
    echo "  -h, --help          显示帮助信息"
    echo "  --enable-vlm        启用VLM模块 (默认: true)"
    echo "  --enable-yolo       启用YOLO模块 (默认: true)"
    echo "  --enable-depth      启用深度模块 (默认: true)"
    echo "  --camera            默认相机名称 (默认: camera_1)"
    echo "  --depth-camera      默认深度相机名称 (默认: depth_cam_1)"
    echo "  --confidence        置信度阈值 (默认: 0.5)"
    echo ""
    echo "模式:"
    echo "  test               运行测试模式"
    echo "  production         运行生产模式 (默认)"
    echo ""
    echo "示例:"
    echo "  $0                          # 启动生产模式"
    echo "  $0 test                     # 启动测试模式"
    echo "  $0 --enable-vlm false       # 禁用VLM模块"
    echo "  $0 --camera front_camera    # 使用front_camera作为默认相机"
    echo ""
}

# 主函数
main() {
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            --enable-vlm)
                ENABLE_VLM="$2"
                shift 2
                ;;
            --enable-yolo)
                ENABLE_YOLO="$2"
                shift 2
                ;;
            --enable-depth)
                ENABLE_DEPTH="$2"
                shift 2
                ;;
            --camera)
                DEFAULT_CAMERA="$2"
                shift 2
                ;;
            --depth-camera)
                DEFAULT_DEPTH_CAMERA="$2"
                shift 2
                ;;
            --confidence)
                CONFIDENCE_THRESHOLD="$2"
                shift 2
                ;;
            test|production)
                MODE="$1"
                shift
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 设置默认模式
    MODE=${MODE:-production}
    
    log_info "VLA视觉调度器启动脚本"
    log_info "模式: $MODE"
    
    # 检查环境
    check_ros_environment
    check_workspace
    check_dependencies
    
    # 启动调度器
    start_vision_scheduler "$MODE"
    
    if [ "$MODE" = "test" ]; then
        log_success "测试完成"
    else
        log_success "视觉调度器已启动"
    fi
}

# 捕获中断信号
trap 'log_warning "正在停止..."; exit 0' INT TERM

# 运行主函数
main "$@"
