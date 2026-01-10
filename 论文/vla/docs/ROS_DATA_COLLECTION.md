# ROS 数据采集器使用说明

## 概述

ROS 数据采集器是为 VLA (Vision-Language-Action) 框架设计的机器人轨迹数据采集系统，专门用于采集26关节人形机器人的运动数据。该系统支持自动和手动两种采集模式，能够同步采集关节数据、基座位置、姿态和视觉信息。

## 主要功能

- **同步数据采集**: 支持关节数据、基座位置、姿态和图像的同步采集
- **灵活的采集模式**: 自动模式（动作检测）和手动模式
- **可配置参数**: 支持通过 ROS 参数配置所有关键设置
- **数据完整性检查**: 自动验证采集数据的完整性和一致性
- **安全退出机制**: 支持安全保存已采集的数据

## 系统要求

- ROS1 (Noetic 或 Melodic)
- Python 3.6+
- 必需的 ROS 包：
  - `std_msgs`
  - `sensor_msgs`
  - `nav_msgs`
  - `geometry_msgs`
  - `message_filters`
  - `cv_bridge`

## 安装依赖

```bash
# 安装 ROS 依赖
sudo apt-get install ros-noetic-cv-bridge ros-noetic-message-filters

# 安装 Python 依赖
pip install opencv-python numpy
```

## 配置说明

### 默认配置

采集器已针对26关节人形机器人进行了预配置，使用 `std_msgs/Float64MultiArray` 数据格式：

```bash
# 机器人配置
- 关节数量: 26
- 采样频率: 100Hz (从500Hz降采样)
- 数据格式: Float64MultiArray
- 采集模式: 自动模式 (动作检测)

# 默认话题
- 关节状态: /humanoid_controller/optimizedState_mrt/joint_pos
- 基座位置: /humanoid_controller/optimizedState_mrt/base/pos_xyz  
- 基座姿态: /humanoid_controller/optimizedState_mrt/base/angular_zyx
- 图像话题: /camera/image_raw (可选)
```

### 关键参数说明

```bash
# 采集控制参数
collection_mode: "auto"        # 采集模式: auto(默认)/manual
motion_threshold: 0.01         # 动作检测阈值 (rad)
min_motion_frames: 5           # 最小动作帧数，用于确定动作停止

# 数据采集参数
frames_per_episode: 35         # 每个episode的帧数
num_episodes: 1               # 采集的episode数量 (默认1次)
frequency: 100.0               # 采样频率 (Hz)
with_vision: false             # 是否采集视觉数据

# 技能描述参数
instruction: "向前走"           # 技能的自然语言描述
skill_id: "walk"               # 技能的唯一标识符
```

## 使用方法

### 1. 启动采集器

使用 launch 文件启动（推荐）：

```bash
# 基础启动（默认配置）
roslaunch vla collector.launch

# 自定义指令和参数
roslaunch vla collector.launch instruction:="向左转" skill_id:="turn_left" num_episodes:=10 #默认num_episodes=1

# 完整自定义参数
roslaunch vla collector.launch \
  instruction:="向前走" \
  skill_id:="walk_forward" \
  frames_per_episode:=400 \
  frequency:=100.0 \
  num_joints:=26 \
  collection_mode:="auto"
```

### 2. 参数配置

采集器通过 launch 文件参数进行配置，主要参数包括：

```bash
# 基础参数
instruction:="向前走"           # 采集指令描述
skill_id:="walk"               # 技能标识
num_episodes:=1                # 采集episode数量 (默认1次)
frames_per_episode:=35         # 每个episode的帧数
frequency:=100.0               # 采样频率(Hz)
with_vision:=false             # 是否采集视觉数据

# 机器人参数
num_joints:=26                 # 关节数量
joint_state_topic:="/humanoid_controller/optimizedState_mrt/joint_pos"
baselink_topic:="/humanoid_controller/optimizedState_mrt/base/pos_xyz"
orientation_topic:="/humanoid_controller/optimizedState_mrt/base/angular_zyx"

# 采集控制
collection_mode:="auto"        # 采集模式: auto/manual
motion_threshold:=0.01         # 动作检测阈值
min_motion_frames:=5           # 最小动作帧数
```

### 3. 采集模式

#### 自动模式（默认）
- 通过动作检测自动开始/停止采集
- 检测到关节运动时自动开始采集
- 连续多帧无动作时自动停止采集
- 适用于连续运动的技能采集

#### 手动模式
- 通过 ROS 服务手动控制采集
- 开始采集：`rosservice call /collector/start_collection`
- 停止采集：`rosservice call /collector/stop_collection`
- 适用于精确控制的采集场景

### 4. 数据格式

采集的数据保存为 JSON 格式，每个 episode 包含：

```json
{
  "instruction": "向前走",
  "skill_id": "walk",
  "observations": [
    {
      "joint_pos": [26个关节角度],
      "joint_vel": [26个关节速度],
      "root_pos": [x, y, z],
      "root_orien": [x, y, z, w],
      "image": "路径或null"
    }
  ],
  "actions": [
    [26个关节动作]
  ],
  "collect_visual": false,
  "episode_metadata": {
    "episode_index": 0,
    "total_frames": 35,
    "skill_params": {},
    "collection_time": 1234567890.0
  }
}
```

## 数据处理

### 1. 欧拉角转四元数

系统自动将基座姿态的欧拉角（ZYX顺序）转换为四元数：

```python
def euler_to_quaternion(roll, pitch, yaw):
    # 欧拉角转四元数实现
    return [x, y, z, w]
```

### 2. 关节速度计算

通过位置差分计算关节速度：

```python
joint_velocities = [(curr - last) * frequency for curr, last in zip(joint_positions, last_joint_positions)]
```

### 3. 数据同步

使用 `ApproximateTimeSynchronizer` 进行多话题数据同步：

- 同步容差：0.1秒（可配置）
- 队列大小：10
- 支持图像同步（可选）

## 故障排除

### 常见问题

1. **话题连接失败**
   - 检查话题名称是否正确
   - 确认话题正在发布
   - 使用 `rostopic list` 和 `rostopic echo` 检查

2. **数据同步问题**
   - 调整 `sync_slop` 参数
   - 检查各话题的发布频率
   - 确认时间戳是否同步

3. **动作检测不敏感**
   - 调整 `motion_threshold` 参数
   - 检查 `min_motion_frames` 设置
   - 确认关节数据变化范围

4. **图像采集失败**
   - 检查摄像头话题
   - 确认 `cv_bridge` 安装正确
   - 检查图像格式支持

### 调试命令

```bash
# 检查话题
rostopic list
rostopic echo /humanoid_controller/optimizedState_mrt/joint_pos -n1

# 检查服务
rosservice list
rosservice call /collector/start_collection

# 查看日志
rostopic echo /rosout
```

## 性能优化

### 1. 采样率优化
- 原始数据：500Hz
- 采样频率：100Hz（推荐）
- 平衡数据质量和存储空间

### 2. 存储优化
- 压缩图像数据
- 使用 NPZ 格式存储大量数据
- 定期清理旧数据

### 3. 内存管理
- 控制 episode 长度
- 及时保存已完成数据
- 监控内存使用情况

## 扩展开发

### 1. 添加新的数据源

```python
# 在 _setup_ros_subscribers 中添加新话题
self.new_sub = rospy.Subscriber(new_topic, MsgType, self._new_callback)

# 更新消息同步器
# 修改 _setup_message_synchronizer 方法
```

### 2. 自定义技能接口

继承 `MockSkillInterface` 类：

```python
class CustomSkillInterface(MockSkillInterface):
    def step(self, observation):
        # 实现自定义技能逻辑
        return action
```

### 3. 数据后处理

在 `_build_observation` 方法中添加自定义处理逻辑：

```python
def _build_observation(self, ...):
    # 现有处理逻辑
    
    # 添加自定义处理
    processed_data = self.custom_process(raw_data)
    
    return observation
```

## 安全注意事项

1. **数据备份**: 定期备份采集的重要数据
2. **紧急停止**: 使用 Ctrl+C 安全停止采集
3. **参数验证**: 修改参数前进行合理性检查
4. **存储空间**: 监控磁盘使用情况，避免空间不足

## 版本历史

- v1.0: 基础数据采集功能
- v1.1: 添加姿态数据处理
- v1.2: 支持手动采集模式
- v1.3: 增强数据完整性检查

## 联系支持

如有问题或建议，请通过以下方式联系：

- 提交 Issue 到项目仓库
- 发送邮件至技术支持
- 查看项目文档和 Wiki