# SubtitleGenius

基于GenAI的实时MP4音频流字幕生成工具

## 功能特性

- 🎵 实时音频流提取和处理
- 🤖 集成多种GenAI模型 (OpenAI Whisper, GPT-4, Claude等)
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

4. 安装开发依赖 (可选)
```bash
uv sync --extra dev
```

5. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env 文件，填入你的API密钥
```

### 传统方式

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 基本用法

```bash
# 处理单个MP4文件
uv run subtitle-genius process video.mp4

# 实时处理音频流
uv run subtitle-genius stream --input rtmp://example.com/live/stream

# 批量处理
uv run subtitle-genius batch --input-dir ./videos --output-dir ./subtitles
```

### Python API

```python
from subtitle_genius import SubtitleGenerator

# 初始化生成器
generator = SubtitleGenerator(
    model="openai-whisper",
    language="zh-CN"
)

# 处理音频文件
subtitles = await generator.generate_from_file("audio.wav")

# 实时处理
async for subtitle in generator.generate_realtime(audio_stream):
    print(f"[{subtitle.start}] {subtitle.text}")
```

## 配置

主要配置选项在 `.env` 文件中：

- `OPENAI_API_KEY`: OpenAI API密钥
- `ANTHROPIC_API_KEY`: Anthropic API密钥
- `SUBTITLE_LANGUAGE`: 字幕语言 (默认: zh-CN)
- `AUDIO_SAMPLE_RATE`: 音频采样率 (默认: 16000)

## 架构

```
subtitle_genius/
├── core/           # 核心处理逻辑
├── models/         # AI模型集成
├── audio/          # 音频处理
├── subtitle/       # 字幕生成和格式化
├── stream/         # 实时流处理
└── cli/            # 命令行界面
```

## 开发

```bash
# 安装开发依赖
uv sync --extra dev

# 运行测试
uv run pytest

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

## 许可证

MIT License
