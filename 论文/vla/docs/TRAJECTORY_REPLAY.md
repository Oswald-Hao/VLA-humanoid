# 数据回播使用说明

## 概述

本系统提供了MuJoCo数据回播功能，可以将采集的机器人轨迹数据在MuJoCo仿真环境中进行可视化回播。

## 文件结构

```
vla/
├── scripts/
│   ├── trajectory_replayer.py    # 主回播脚本
│   └── replay_demo.py           # 简单演示脚本
├── models/
│   ├── kuavo_s45.xml           # 机器人模型文件（需要用户提供）
│   └── README.md               # 模型文件说明
└── trajectories/               # 采集的轨迹数据
    ├── trajectory_001.json
    ├── trajectory_002.json
    └── ...
```

## 安装依赖

```bash
# 安装MuJoCo和mujoco_viewer
pip install mujoco mujoco_viewer
```

## 使用方法

### 1. 简单演示（推荐）

```bash
# 自动查找最新的轨迹文件并回播
python scripts/replay_demo.py
```

### 2. 直接使用回播脚本

```bash
# 回播指定轨迹文件
python scripts/trajectory_replayer.py --trajectory trajectories/trajectory_001.json

# 指定模型文件
python scripts/trajectory_replayer.py --trajectory trajectories/trajectory_001.json --model models/kuavo_s45.xml

# 循环播放
python scripts/trajectory_replayer.py --trajectory trajectories/trajectory_001.json --loop

# 调整播放速度
python scripts/trajectory_replayer.py --trajectory trajectories/trajectory_001.json --speed 2.0
```

## 控制键

回播器支持以下键盘控制：

| 键 | 功能 |
|---|---|
| 空格 | 播放/暂停 |
| r | 重置到第一帧 |
| l | 循环播放开关 |
| + | 加速播放 |
| - | 减速播放 |
| ← | 上一帧 |
| → | 下一帧 |
| q | 退出 |

## 机器人模型文件

### 要求

1. **格式**: MuJoCo XML模型文件
2. **位置**: `models/kuavo_s45.xml`
3. **关节数量**: 26个关节（与数据采集一致）
4. **兼容性**: MuJoCo 2.0+ 或 3.0+

### 模型配置

回播器需要以下配置：

```xml
<model>
    <!-- 26个关节定义 -->
    <joint name="joint_0" type="hinge"/>
    <joint name="joint_1" type="hinge"/>
    ...
    <joint name="joint_25" type="hinge"/>
    
    <!-- 传感器定义 -->
    <touch sensor="touch_sensor"/>
    <accelerometer sensor="accel_sensor"/>
    <gyro sensor="gyro_sensor"/>
</model>
```

### 关节名称映射

回播器会自动尝试映射关节名称。如果模型中的关节名称与数据采集时不一致，请修改`trajectory_replayer.py`中的`_get_joint_names()`方法。

## 数据格式要求

回播器支持以下数据格式：

```json
{
  "observations": [
    {
      "joint_pos": [26个关节角度],
      "joint_vel": [26个关节速度],
      "root_pos": [x, y, z],
      "root_orien": [x, y, z, w],
      "image": "图像路径或null"
    }
  ],
  "actions": [
    [26个关节动作]
  ],
  "instruction": "技能描述",
  "skill_id": "技能标识"
}
```

## 故障排除

### 常见问题

1. **模型文件不存在**
   ```
   错误: 模型文件不存在: models/kuavo_s45.xml
   解决: 请将机器人模型文件放到models/文件夹中
   ```

2. **关节数量不匹配**
   ```
   错误: 关节数量不匹配: 模型26, 数据25
   解决: 检查模型文件和数据采集的关节数量是否一致
   ```

3. **MuJoCo导入失败**
   ```
   错误: 请先安装MuJoCo和mujoco_viewer
   解决: pip install mujoco mujoco_viewer
   ```

4. **轨迹文件不存在**
   ```
   错误: 轨迹文件不存在: trajectories/trajectory_001.json
   解决: 请先使用数据采集器采集轨迹数据
   ```

### 调试模式

回播器包含详细的调试信息，可以通过以下方式启用：

```bash
# 启用详细输出
python scripts/trajectory_replayer.py --trajectory trajectories/trajectory_001.json --verbose
```

## 性能优化

### 1. 播放速度

- 默认速度：1.0x（实时）
- 调整范围：0.1x - 5.0x
- 建议：对于快速动作，使用0.5x速度观察细节

### 2. 图形设置

- 如果播放卡顿，可以降低图形质量
- 在mujoco_viewer中调整渲染选项

### 3. 内存使用

- 大型轨迹文件可能占用较多内存
- 建议单个轨迹文件不超过1000帧

## 扩展开发

### 自定义关节映射

如果需要自定义关节名称映射，请修改`_get_joint_names()`方法：

```python
def _get_joint_names(self):
    """自定义关节名称映射"""
    return [
        "left_hip_yaw_joint",
        "left_hip_roll_joint",
        "left_hip_pitch_joint",
        # ... 其他关节名称
    ]
```

### 添加新的数据格式

如果需要支持其他数据格式，请修改`_load_trajectory()`方法。

### 自定义渲染

可以通过继承`TrajectoryReplayer`类来自定义渲染效果：

```python
class CustomReplayer(TrajectoryReplayer):
    def render(self):
        # 自定义渲染逻辑
        super().render()
```

## 示例工作流程

1. **采集数据**
   ```bash
   roslaunch vla collector.launch
   ```

2. **回播数据**
   ```bash
   python scripts/replay_demo.py
   ```

3. **分析数据**
   - 观察机器人动作是否符合预期
   - 检查数据完整性和质量
   - 验证关节位置和动作的一致性

## 注意事项

1. 确保模型文件与数据采集时使用的机器人保持一致
2. 回播过程中保持窗口焦点以接收键盘输入
3. 大型轨迹文件可能需要较长的加载时间
4. 建议在回播前备份重要的轨迹数据