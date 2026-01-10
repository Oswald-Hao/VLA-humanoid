# 轨迹数据回播器使用说明

## 概述

`replayer.py` 是一个ROS1轨迹数据回播器，用于将采集的轨迹数据回放到相应的ROS话题中。该工具与 `ros_collector.py` 数据采集器配对使用，实现数据的采集和回播功能。

## 功能特性

- **轨迹数据加载**: 从JSON文件加载采集的轨迹数据
- **多话题发布**: 支持关节数据、里程计、姿态、图像等多种话题的发布
- **灵活的回播控制**: 支持开始、停止、暂停、重置等控制操作
- **可配置参数**: 支持回播速度、频率、循环次数等参数配置
- **服务接口**: 提供ROS服务接口进行远程控制
- **完整的状态发布**: 除了原始数据格式，还发布标准的ROS消息格式

## 安装依赖

```bash
pip install rospy cv-bridge numpy opencv-python
```

## 使用方法

### 1. 基本使用

```bash
# 直接使用Python脚本
python3 modules/replayer.py trajectory_001.json

# 使用启动脚本
./playback.sh trajectory_001.json
```

### 2. 参数配置

```bash
# 调整回播速度
python3 modules/replayer.py trajectory_001.json --speed 0.5

# 调整回播频率
python3 modules/replayer.py trajectory_001.json --frequency 50

# 设置循环次数
python3 modules/replayer.py trajectory_001.json --loops 3

# 自定义话题
python3 modules/replayer.py trajectory_001.json \
    --joint-topic /custom/joint_pos \
    --odom-topic /custom/base_pos \
    --orientation-topic /custom/base_angular
```

### 3. 使用启动脚本

```bash
# 显示帮助
./playback.sh --help

# 基本回播
./playback.sh trajectory_001.json

# 完整参数示例
./playback.sh trajectory_001.json \
    -s 0.5 \
    -f 50 \
    -l 3 \
    --joint-topic /custom/joint_pos \
    --odom-topic /custom/base_pos
```

## 控制服务

回播器提供以下ROS服务进行控制：

```bash
# 开始回播
rosservice call /player/start_playback

# 停止回播
rosservice call /player/stop_playback

# 暂停/继续回播
rosservice call /player/pause_playback

# 重置回播
rosservice call /player/reset_playback
```

## 发布的话题

回播器会发布以下话题：

### 原始数据格式（与采集器对应）

- `/humanoid_controller/optimizedState_mrt/joint_pos` (Float64MultiArray)
- `/humanoid_controller/optimizedState_mrt/base/pos_xyz` (Float64MultiArray)  
- `/humanoid_controller/optimizedState_mrt/base/angular_zyx` (Float64MultiArray)
- `/camera/image_raw` (sensor_msgs/Image) - 如果轨迹包含图像数据

### 标准ROS消息格式

- `/joint_states` (sensor_msgs/JointState)
- `/odom` (nav_msgs/Odometry)

## 数据格式要求

轨迹数据文件必须包含以下字段：

```json
{
    "instruction": "向前走",
    "skill_id": "walk",
    "observations": [
        {
            "joint_pos": [0.1, 0.2, ...],
            "joint_vel": [0.0, 0.0, ...],
            "root_pos": [0.0, 0.0, 0.8],
            "root_orien": [0.0, 0.0, 0.0, 1.0],
            "image": "path/to/image.jpg"  // 可选
        }
    ],
    "actions": [...],
    "collect_visual": true/false
}
```

## 使用场景

### 1. 数据验证

回播采集的数据，验证数据的完整性和正确性：

```bash
python3 modules/replayer.py trajectory_001.json --loops 1
```

### 2. 算法测试

在仿真环境中回播数据，测试算法的性能：

```bash
python3 modules/replayer.py trajectory_001.json --speed 0.1
```

### 3. 机器人控制

将采集的演示数据用于控制真实机器人：

```bash
python3 modules/replayer.py trajectory_001.json \
    --joint-topic /robot/joint_commands \
    --frequency 100
```

## 注意事项

1. **ROS环境**: 确保ROS环境正确配置，roscore正在运行
2. **话题匹配**: 确保回播的话题与机器人控制器期望的话题匹配
3. **数据一致性**: 确保轨迹数据的格式与机器人控制器期望的格式一致
4. **图像路径**: 如果轨迹包含图像数据，确保图像文件路径正确
5. **回播速度**: 根据实际需求调整回播速度，避免过快导致控制问题

## 故障排除

### 常见问题

1. **无法连接到ROS Master**
   - 检查 `ROS_MASTER_URI` 环境变量
   - 确保roscore正在运行

2. **轨迹文件加载失败**
   - 检查文件路径是否正确
   - 验证JSON格式是否正确

3. **话题发布失败**
   - 检查话题名称是否正确
   - 确保消息类型匹配

4. **图像发布失败**
   - 检查图像文件是否存在
   - 确保OpenCV和cv_bridge正确安装

### 调试建议

1. 使用 `rostopic echo` 检查话题数据
2. 使用 `rosservice list` 检查服务是否正常
3. 查看回播器的日志输出了解详细状态

## 与采集器的配合

`replayer.py` 与 `ros_collector.py` 完全兼容：

1. 使用 `ros_collector.py` 采集数据
2. 使用 `replayer.py` 回播数据
3. 确保话题配置一致
4. 验证回播数据与原始数据的匹配性

通过这种配对使用，可以实现完整的数据采集和回播流程，用于机器人学习、测试和验证。