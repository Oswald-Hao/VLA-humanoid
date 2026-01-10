# 🎤 正式版语音识别+LLM+TTS系统

## 📋 系统概述

这是一个完全集成的正式版语音识别+LLM+TTS系统，专为实时语音交互设计。系统实现了完整的语音录制、LLM分析、JSON生成和TTS反馈流程。

### 🎯 核心特性
- **实时语音录制**: 智能静音检测，自动开始/停止录制
- **语音识别**: 使用Whisper进行高精度中文语音识别
- **意图识别**: 智能LLM处理，支持5种主要意图
- **JSON生成**: 自动生成结构化的JSON响应
- **语音合成**: 使用Edge TTS生成高质量中文语音
- **智能缓存**: 优化重复指令的处理速度
- **音频播放**: 自动播放TTS生成的音频

### 📊 性能目标
- **语音识别准确率**: >80%
- **意图识别准确率**: >90%
- **总响应时间**: <5秒
- **缓存命中率**: >80%

## 🚀 快速启动

### 第一步：启动系统
```bash
# 进入脚本目录
cd kuavo_ws/src/ros_vla_language/scripts

# 启动正式版系统
./start_production_system.sh
```

### 第二步：交互使用
系统启动后会显示：
```
🚀 启动正式版语音识别+LLM+TTS系统
==============================================
🎤 正式版语音识别+LLM+TTS系统
实时语音录制 -> LLM分析 -> JSON生成 -> TTS反馈
==============================================

✅ 正式版系统初始化完成
🚀 启动正式版系统...
==============================================
🎤 正式版语音识别+LLM+TTS系统
实时语音录制 -> LLM分析 -> JSON生成 -> TTS反馈
==============================================

✅ 系统启动完成
🎤 按空格键开始录制，再次按空格键停止录制
📝 按 Ctrl+C 退出系统
```

### 第三步：使用流程
1. **开始录制**: 按空格键开始录制音频
2. **停止录制**: 再次按空格键停止录制（检测到静音自动停止）
3. **自动处理**: 系统自动进行语音识别、LLM分析、JSON生成
4. **TTS反馈**: 自动生成并播放语音反馈
5. **查看结果**: JSON文件保存在audio目录下

## 📁 系统文件结构

### 核心文件
- `production_system.py` - 正式版系统核心
- `start_production_system.sh` - 一键启动脚本

### 配置文件
- `config/llm_params.yaml` - LLM参数配置
- `config/skills.json` - 技能配置

### 输出目录
- `audio/` - 保存生成的JSON响应和TTS音频文件

## 🔧 系统组件

### 1. 音频录制器 (AudioRecorder)
- **引擎**: SoundDevice
- **采样率**: 16kHz
- **通道数**: 1
- **静音检测**: 自动检测静音并停止录制
- **实时处理**: 实时音频流处理

### 2. 语音识别器 (ProductionSpeechRecognizer)
- **引擎**: Whisper
- **模型**: base (平衡速度和精度)
- **语言**: 中文
- **功能**: 音频文件识别

### 3. LLM处理器 (ProductionLLMProcessor)
- **功能**: 意图识别、动作生成、JSON响应
- **支持意图**:
  - 自我介绍
  - 抓取
  - 移动
  - 停止
  - 搜索
- **智能缓存**: 5分钟缓存有效期

### 4. TTS生成器 (ProductionTTSGenerator)
- **引擎**: Edge TTS
- **语音**: zh-CN-XiaoxiaoNeural
- **格式**: WAV (16kHz)
- **功能**: 同步音频生成和播放

## 📈 工作流程

```
用户按下空格键
    ↓
开始实时音频录制
    ↓
检测到静音自动停止
    ↓
Whisper语音识别
    ↓
LLM意图分析
    ↓
生成JSON响应
    ↓
Edge TTS语音合成
    ↓
自动播放音频反馈
    ↓
保存JSON和音频文件
```

## 🎯 支持的指令

### 自我介绍
- "你好，请介绍一下你自己"
- "你是谁"
- "介绍一下"
- "hi"
- "hello"

### 抓取
- "请帮我抓起桌子上的杯子"
- "拿一下那个杯子"
- "抓取物体"
- "grab"
- "take"

### 移动
- "移动到门口"
- "走到门口"
- "去门口位置"
- "move"
- "go"

### 停止
- "停止当前动作"
- "停下来"
- "取消"
- "stop"
- "cancel"

### 搜索
- "搜索红色的球"
- "找一下红色的球"
- "寻找球"
- "search"
- "find"

## 📋 JSON响应格式

系统会生成以下格式的JSON文件：

```json
{
  "intent": "self_introduction",
  "confidence": 0.95,
  "response": "你好！我是一个智能语音助手，很高兴为您服务。",
  "timestamp": 1625097600.123,
  "recognition": {
    "text": "你好，请介绍一下你自己",
    "confidence": 0.88,
    "duration": 2.34,
    "engine": "whisper"
  }
}
```

## 🛠️ 故障排除

### 依赖安装
如果遇到依赖问题，运行：
```bash
./start_production_system.sh deps
```

### 清理测试文件
清理无用的测试文件：
```bash
./start_production_system.sh cleanup
```

### 常见问题

#### 1. Whisper未安装
```bash
pip install openai-whisper
```

#### 2. Edge TTS未安装
```bash
pip install edge-tts
```

#### 3. SoundDevice未安装
```bash
pip install sounddevice
```

#### 4. 键盘检测不工作
```bash
pip install keyboard
```

#### 5. 音频播放不工作
```bash
pip install pygame
```

#### 6. FFmpeg未安装
```bash
sudo apt install ffmpeg
```

## 🔧 高级配置

### Whisper模型配置
在 `config/llm_params.yaml` 中修改：
```yaml
speech:
  model_name: "base"  # 可选: tiny, base, small, medium, large
```

### Edge TTS配置
```yaml
tts:
  voice: "zh-CN-XiaoxiaoNeural"  # 可选其他语音
  rate: "+0%"  # 语速
  volume: "+0%"  # 音量
```

### 音频录制配置
```yaml
audio:
  sample_rate: 16000
  channels: 1
  chunk_duration: 0.5
  silence_threshold: 0.01
  silence_duration: 1.0
```

### LLM缓存配置
```yaml
llm:
  cache_timeout: 300  # 缓存有效期(秒)
  max_tokens: 100     # 最大token数
  temperature: 0.1    # 温度参数
```

## 📊 性能监控

系统会自动处理并记录以下信息：
- 语音识别结果和置信度
- LLM处理时间和意图
- TTS生成时间和音频文件
- JSON响应结构
- 缓存命中率

## 🎉 使用示例

### 示例1：自我介绍
1. 按空格键开始录制
2. 说："你好，请介绍一下你自己"
3. 系统自动识别并生成响应
4. 播放："你好！我是一个智能语音助手，很高兴为您服务。"
5. 保存JSON文件到audio目录

### 示例2：抓取指令
1. 按空格键开始录制
2. 说："请帮我抓起桌子上的杯子"
3. 系统识别为抓取意图
4. 生成JSON响应包含抓取动作
5. 保存JSON文件到audio目录

## 📋 验证清单

完成以下步骤确认系统正常：

- [ ] Whisper语音识别正常工作
- [ ] Edge TTS语音合成正常工作
- [ ] LLM意图识别准确率>90%
- [ ] JSON生成格式正确
- [ ] 音频录制和播放正常
- [ ] 静音检测功能正常
- [ ] 缓存机制正常工作

## 💡 使用建议

### 1. 首次使用
```bash
./start_production_system.sh deps  # 安装依赖
./start_production_system.sh      # 启动系统
```

### 2. 日常使用
```bash
./start_production_system.sh  # 启动系统进行语音交互
```

### 3. 清理维护
```bash
./start_production_system.sh cleanup  # 清理测试文件
```

## 🎯 总结

这个正式版系统已经完全集成了实时语音录制、LLM分析、JSON生成和TTS反馈功能，为机器人提供了完整的语音交互解决方案。系统具有以下优势：

1. **实时性**: 实时音频录制和处理
2. **智能化**: 智能静音检测和自动停止
3. **准确性**: Whisper提供准确的中文语音识别
4. **高质量**: Edge TTS生成自然的中文语音
5. **结构化**: 自动生成标准化的JSON响应
6. **易使用**: 一键启动，空格键控制
7. **可扩展**: 模块化设计，易于扩展

现在您可以立即使用这个系统进行实时语音交互，系统会自动处理您的语音指令并生成相应的JSON响应和语音反馈。
