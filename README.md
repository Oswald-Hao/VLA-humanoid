# VLA (Visual-Language Action) 人形机器人控制项目

## 项目简介

这是一个基于视觉-语言动作（VLA）模型的人形机器人控制项目，专注于解决**26自由度人形机器人**的全身动作预测和执行中的关键问题。

虽然当前训练数据主要集中在挥手和抱拳等上肢动作，但模型架构支持全身26个关节的协同控制，包括腿部、腰部、手臂等完整的人形机器人动作。

### 🔥 新增功能：大模型语音集成
- **自然语言交互**：集成了大模型语音识别和生成能力
- **智能对话系统**：机器人可以与用户进行自然语言交流
- **动作指令理解**：通过对话理解用户意图，执行相应的动作库动作
- **实时语音响应**：支持实时语音交互，提供更好的用户体验

### 核心问题
- **中间状态预测衰减**：从中间状态预测时动作幅度衰减95%
- **轨迹连接卡顿**：相邻轨迹段连接不平滑，出现"一抽一抽"现象
- **动作理解不完整**：模型缺乏上下文理解，无法正确执行完整动作

### 机器人配置
- **关节数量**: 26个自由度（完整人形机器人）
- **关节分布**: 
  - 腿部关节: 0-11 (12个关节)
  - 左臂关节: 12-18 (7个关节) 
  - 右臂关节: 19-25 (7个关节)
- **关键关节**: 左臂的l_arm_pitch(12), l_arm_roll(13), l_arm_yaw(14)
- **动作指令**: 挥手、抱拳等全身动作

## 演示视频

以下是VLA人形机器人控制系统的演示视频：

<div align="center">
  <img src="video/wave.gif" alt="VLA人形机器人挥手演示" width="640">
</div>

*GIF展示人形机器人执行挥手动作，运动轨迹控制仍需改进*

## 项目结构

```
vla/
├── simple_act_train.py          # 主要训练脚本（改进模型）
├── smooth_act_inference_node.py # 推理节点
├── improved_act_model.py        # 改进的模型架构
├── ros_vla_language/            # 大模型语音交互模块 🔥
│   ├── scripts/
│   │   ├── production_system.py     # 正式版系统核心
│   │   ├── start_production_system.sh # 一键启动脚本
│   │   ├── vla_language_service.py  # VLA语言服务
│   │   └── tests/                   # 测试文件
│   ├── config/
│   │   ├── llm_params.yaml          # LLM参数配置
│   │   ├── audio_config.yaml        # 音频配置
│   │   └── intent_patterns.json     # 意图模式配置
│   ├── msg/                        # ROS消息定义
│   ├── srv/                        # ROS服务定义
│   └── CMakeLists.txt              # ROS编译文件
├── trajectories/                # 训练数据
│   ├── wave_001.json ~ wave_008.json    # 挥手动作数据
│   └── welcome_001.json ~ welcome_008.json # 抱拳动作数据
├── checkpoints/                 # 模型保存目录
│   ├── improved_act_model_v1/           # 改进模型
│   └── key_joint_act_model_v1/          # 原始模型
└── modules/                      # 核心模块
    ├── direct_action_predictor.py
    └── training_logic.py
```

## 核心技术

### 大模型语音集成系统 🔥

#### 1. 语音识别与处理
- **Whisper语音识别**：使用OpenAI Whisper进行高精度中文语音识别
- **实时音频录制**：智能静音检测，自动开始/停止录制
- **音频格式处理**：16kHz采样率，支持多种音频格式
- **噪声抑制**：在嘈杂环境中也能准确识别

#### 2. 大模型意图理解
- **智能意图识别**：支持自我介绍、抓取、移动、停止、搜索等5种主要意图
- **自然语言处理**：通过大模型理解用户真实意图
- **JSON响应生成**：自动生成结构化的JSON响应
- **智能缓存机制**：5分钟缓存有效期，优化重复指令处理

#### 3. 语音合成系统
- **Edge TTS**：使用Edge TTS生成高质量中文语音
- **自然语音生成**：生成接近人声的语音回应
- **实时音频播放**：自动播放TTS生成的音频
- **多语音支持**：支持zh-CN-XiaoxiaoNeural等多种语音

#### 4. ROS集成架构
- **ROS服务接口**：提供ProcessText、GetIntent、GenerateAction等服务
- **消息定义**：VLAIntent、VLAAction、VLACommand等ROS消息
- **参数配置**：支持LLM参数、音频配置、意图模式等配置
- **模块化设计**：易于扩展和维护的ROS包结构

### 改进的模型架构 (`ImprovedKeyJointACTGenerator`)

1. **动作强度调节器**
   - 解决中间状态预测衰减问题
   - 确保从任何状态预测都保持完整动作幅度

2. **状态感知增强器**
   - 让模型理解当前状态和上下文
   - 避免简单重复动作，理解动作的开始、继续、结束

3. **增强时间编码**
   - 多尺度时间编码（线性、正弦、相位、频率）
   - 更好地理解动作的时序特性

4. **多尺度预测**
   - 3个并行预测头确保稳定输出
   - 减少预测方差，提高轨迹平滑性

### 关键特性

- **两层架构**：指令分类 + 动作预测
- **关键关节专注**：自动识别每个指令的关键关节
- **时序平滑**：使用Transformer处理时序依赖
- **指令多样性**：确保不同指令产生完全不同的动作模式

## 快速开始

### 1. 训练模型

```bash
# 训练改进模型（推荐）
python3 simple_act_train.py

# 模型将保存到 checkpoints/improved_act_model_v1/
```

### 2. 推理测试

```bash
# 启动推理节点
python3 smooth_act_inference_node.py
```

### 3. 语音交互测试 🔥

```bash
# 进入ROS语音模块目录
cd ros_vla_language/scripts

# 启动正式版语音交互系统
./start_production_system.sh
```

### 4. 模型验证

```bash
# 验证模型性能
python3 model_vs_robot_comparison.py（该文件为数据与模型输出曲线拟合文件我已删除，如需请自己编写）
```

## 训练配置

### 模型参数
- **状态维度**: 26 (人形机器人全身关节数量)
- **动作维度**: 26 (人形机器人全身关节数量)
- **指令数量**: 2 (挥手、抱拳等全身动作)
- **隐藏维度**: 256
- **轨迹长度**: 32 (预测后32步)
- **关键关节数量**: 8 (每个指令重点关注8个关节)

### 训练参数
- **批次大小**: 64
- **学习率**: 1e-4
- **训练轮数**: 2000
- **优化器**: AdamW
- **损失函数**: 多目标组合损失

## 预期效果

| 问题 | 原模型 | 改进模型 |
|------|--------|----------|
| 抱拳卡住 | ❌ 卡在中间 | ✅ 完整执行 |
| 一抽一抽 | ❌ 连接卡顿 | ✅ 流畅连贯 |
| 动作理解 | ❌ 重复挥手 | ✅ 上下文感知 |
| 中间衰减 | ❌ 95%衰减 | ✅ 保持幅度 |

## 关键改进

### 1. 解决中间状态预测衰减
```python
# 动作强度调节器
action_intensity = self.action_intensity_regulator(intensity_input)
final_predictions = final_predictions * action_intensity.unsqueeze(-1).unsqueeze(-1)
```

### 2. 状态感知增强
```python
# 状态感知增强器
state_aware_input = torch.cat([state_features, states], dim=-1)
enhanced_features = self.state_aware_enhancer(state_aware_input)
```

### 3. 多尺度预测
```python
# 3个并行预测头
for head in self.prediction_heads:
    pred = head(temporal_features)
    multi_scale_predictions.append(pred)
```

## 数据格式

训练数据使用JSON格式，包含：
- `instruction`: 动作指令（"wave"或"welcome"）
- `observations`: 关节位置观测序列
- `joint_pos`: 26个关节的位置数据（完整人形机器人）

### 关节分布说明
```python
joint_indices = {
    '腿部关节': list(range(0, 12)),    # 0-11: 腿部12个关节
    '左臂关节': list(range(12, 19)),   # 12-18: 左臂7个关节
    '右臂关节': list(range(19, 26)),   # 19-25: 右臂7个关节
}
```

### 关键关节
- **l_arm_pitch** (索引12): 左臂俯仰
- **l_arm_roll** (索引13): 左臂滚动  
- **l_arm_yaw** (索引14): 左臂偏航

这些关节在挥手和抱拳动作中起关键作用。

示例：
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
    '挥手': 0,
    '抱拳': 1
}
```

## 模型文件

- **改进模型**: `checkpoints/improved_act_model_v1/best_model.pth`
- **原始模型**: `checkpoints/key_joint_act_model_v1/best_model.pth`

## 依赖环境

- Python 3.8+
- PyTorch 1.12+
- ROS (机器人操作系统)
- CUDA (可选，用于GPU加速)

## 训练技巧

1. **自动混合精度**: 启用GPU加速训练
2. **梯度裁剪**: 防止梯度爆炸，限制为5.0
3. **学习率调度**: 使用ReduceLROnPlateau
4. **早停机制**: 防止过拟合，耐心值为150轮
5. **质量监控**: 实时监控训练质量

## 常见问题

### VLA模型相关问题
#### Q: 训练时出现维度错误
A: 确保输入数据的维度正确，状态应为[batch_size, 26]，动作应为[batch_size, 32, 26]

#### Q: 模型预测幅度太小
A: 改进模型已包含动作强度调节器，应该能解决此问题

#### Q: 轨迹连接不平滑
A: 多尺度预测和时序编码应该能改善连接平滑性

### 语音交互相关问题 🔥
#### Q: 语音识别不准确
A: 确保麦克风设备正常，尽量在安静环境下使用，系统支持噪声抑制但过强噪声会影响识别率

#### Q: 大模型响应慢
A: 首次使用需要加载模型，后续响应会加快。网络状况也会影响响应速度。

#### Q: 动作执行失败
A: 确保VLA模型已正确加载，检查动作指令是否在支持的动作库中。

#### Q: 语音合成不自然
A: 系统使用最新的语音合成技术，如遇不自然情况可尝试重新启动服务。

## 开发计划

### 已完成功能 ✅
- [x] 基础VLA模型架构
- [x] 26自由度人形机器人控制
- [x] 大模型语音集成系统
- [x] 自然语言交互功能
- [x] 动作库执行能力

### 规划中功能 🚀
- [ ] 添加更多动作指令
- [ ] 集成视觉输入
- [ ] 优化推理速度
- [ ] 添加在线学习功能
- [ ] 支持实时调整
- [ ] 多轮对话记忆增强
- [ ] 情感识别与回应
- [ ] 复杂任务规划能力

## 贡献指南

1. Fork本项目
2. 创建功能分支
3. 提交代码变更
4. 发起Pull Request

## 许可证

本项目采用MIT许可证。

## 联系方式

如有问题或建议，请提交Issue或联系开发团队。