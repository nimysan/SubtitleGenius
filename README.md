# SubtitleGenius

基于GenAI的实时MP4音频流字幕生成工具

## Open Universal Arabic ASR Leaderboard

[Open Universal Arabic ASR Leaderboard](https://arxiv.org/html/2412.13788v1)

[arabicspeech](https://arabicspeech.org/)

[ASR的一些常见评估指标](https://huggingface.co/learn/audio-course/chapter5/evaluation)

[HuggingFace ASR课程](https://huggingface.co/learn/audio-course/chapter5/fine-tuning)

[CommonVoice数据集](https://commonvoice.mozilla.org/en/datasets)

## 功能特性

- 🎵 实时音频流提取和处理
- 🤖 集成多种GenAI模型 (OpenAI Whisper, GPT-4, Claude, Amazon Transcribe等)
- 🌊 **Amazon Transcribe 流式处理** - 支持实时语音识别
- 🇸🇦 **默认支持 Arabic 语言** - 专为阿拉伯语优化
- 📝 智能字幕生成和优化
- 🎬 支持多种字幕格式 (SRT, WebVTT)
- ⚡ 低延迟实时处理
- 🌐 多语言支持
- 🔧 可配置的处理参数

## 安装

### 使用 uv (推荐)

1. 安装 uv (如果还没有安装)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. 克隆项目
```bash
git clone <repository-url>
cd SubtitleGenius
```

3. 安装依赖
```bash
uv sync
```

4. 安装流式处理依赖
```bash
# 自动安装流式处理相关依赖
python install_streaming.py

# 或手动安装
pip install amazon-transcribe boto3 pydantic-settings
```

5. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env 文件，填入你的API密钥
```

### AWS 配置 (用于 Transcribe)

```bash
# 设置 AWS 凭证
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=us-east-1

# 或使用 AWS CLI 配置
aws configure
```

## 快速开始

### 🌊 流式处理 (推荐)

```bash
# Arabic 语音实时识别示例
python example_streaming_arabic.py

# 交互式流式处理测试
python test_streaming_arabic.py

# 实时麦克风输入
python -c "
import asyncio
from subtitle_genius.models.transcribe_model import TranscribeModel
from subtitle_genius.stream.processor import StreamProcessor

async def main():
    model = TranscribeModel(use_streaming=True)
    processor = StreamProcessor()
    audio_stream = processor.start_microphone_stream()
    
    async for subtitle in model.transcribe_stream(audio_stream, language='ar'):
        print(f'[{subtitle.start:.1f}s] {subtitle.text}')

asyncio.run(main())
"
```

### 启动 Web 界面

```bash
# 启动简化版界面（推荐）
uv run python launch.py --simple

# 或启动完整版界面
uv run python launch.py --full

# 指定端口
uv run python launch.py --simple --port 8080
```

访问 http://127.0.0.1:7860 使用 Web 界面。

### 基本用法

```bash
# 处理单个MP4文件 (默认 Arabic)
uv run subtitle-genius process video.mp4

# 指定语言
uv run subtitle-genius process video.mp4 --language ar

# 实时处理音频流
uv run subtitle-genius stream --input rtmp://example.com/live/stream

# 批量处理
uv run subtitle-genius batch --input-dir ./videos --output-dir ./subtitles
```

### Python API

```python
from subtitle_genius import SubtitleGenerator
from subtitle_genius.models.transcribe_model import TranscribeModel
from subtitle_genius.stream.processor import StreamProcessor

# 🌊 流式处理 (推荐)
async def streaming_example():
    model = TranscribeModel(use_streaming=True)
    processor = StreamProcessor()
    
    # 文件流式处理
    audio_stream = processor.process_file_stream("audio.wav")
    async for subtitle in model.transcribe_stream(audio_stream, language="ar"):
        print(f"[{subtitle.start}s] {subtitle.text}")
    
    # 实时麦克风
    mic_stream = processor.start_microphone_stream()
    async for subtitle in model.transcribe_stream(mic_stream, language="ar"):
        print(f"实时: {subtitle.text}")

# 📦 批处理模式
async def batch_example():
    model = TranscribeModel(use_streaming=False)
    subtitles = await model.transcribe("audio.wav", language="ar")
    
    for subtitle in subtitles:
        print(f"[{subtitle.start}] {subtitle.text}")
```

## 配置

主要配置选项在 `.env` 文件中：

- `OPENAI_API_KEY`: OpenAI API密钥
- `ANTHROPIC_API_KEY`: Anthropic API密钥
- `AWS_ACCESS_KEY_ID`: AWS访问密钥ID (用于Amazon Transcribe)
- `AWS_SECRET_ACCESS_KEY`: AWS秘密访问密钥
- `AWS_REGION`: AWS区域 (默认: us-east-1)
- `AWS_S3_BUCKET`: S3存储桶名称 (默认: subtitle-genius-temp)
- `SUBTITLE_LANGUAGE`: 字幕语言 (默认: ar - Arabic)
- `AUDIO_SAMPLE_RATE`: 音频采样率 (默认: 16000)

## 支持的语言

### Amazon Transcribe 支持的语言
- `ar` / `ar-SA`: Arabic (Saudi Arabia) - **默认**
- `ar-AE`: Arabic (UAE)
- `zh-CN`: Chinese (Simplified)
- `en-US`: English (US)
- `en-GB`: English (UK)
- `ja-JP`: Japanese
- `ko-KR`: Korean
- `fr-FR`: French
- `de-DE`: German
- `es-ES`: Spanish
- `ru-RU`: Russian

## 架构

```
subtitle_genius/
├── core/           # 核心处理逻辑
├── models/         # AI模型集成
│   └── transcribe_model.py  # 🌊 流式 + 批处理 Transcribe
├── audio/          # 音频处理
├── subtitle/       # 字幕生成和格式化
├── stream/         # 🌊 实时流处理
└── cli/            # 命令行界面
```

## 测试

```bash
# 安装开发依赖
uv sync --extra dev

# 基础功能测试
uv run pytest

# Amazon Transcribe 测试
python test_transcribe.py

# 🌊 流式处理测试
python test_streaming_arabic.py

# 简单示例
python example_streaming_arabic.py

# 代码格式化
uv run black .

# 类型检查
uv run mypy subtitle_genius

# 使用 Makefile
make dev    # 安装开发依赖
make test   # 运行测试
make format # 格式化代码
make lint   # 代码检查
```

## 🌊 流式处理特性

### 实时处理能力
- **麦克风输入**: 实时语音识别
- **文件流**: 模拟实时处理文件
- **RTMP流**: 处理直播流
- **低延迟**: 优化的流式处理管道

### Arabic 语言优化
- 默认使用 Arabic (ar-SA)
- 支持多种阿拉伯语方言
- 优化的语言模型配置
- 实时阿拉伯语语音识别

### 使用场景
- 🎤 实时会议记录
- 📺 直播字幕生成
- 🎬 视频内容转录
- 📞 电话会议记录
- 🎓 教育培训字幕

## 故障排除

### 流式处理问题
```bash
# 安装流式处理依赖
pip install amazon-transcribe

# 检查 AWS 凭证
aws sts get-caller-identity

# 测试麦克风权限 (macOS)
# 系统偏好设置 > 安全性与隐私 > 麦克风
```

### 常见错误
1. **ImportError: amazon-transcribe**
   ```bash
   pip install amazon-transcribe
   ```

2. **AWS 凭证错误**
   ```bash
   export AWS_ACCESS_KEY_ID=your_key
   export AWS_SECRET_ACCESS_KEY=your_secret
   ```

3. **PyAudio 安装失败**
   ```bash
   # macOS
   brew install portaudio
   pip install pyaudio
   
   # Ubuntu
   sudo apt-get install portaudio19-dev
   pip install pyaudio
   ```

## 下载视频 使用yt-dlp

```bash
yt-dlp --cookies-from-browser chrome –-merge-output-format mp4 https://youtu.be/0PggkKx9m54  
ffmpeg -i input.webm -c:v libx264 -crf 23 -c:a aac -b:a 128k output.mp4
```

## 许可证

MIT License
