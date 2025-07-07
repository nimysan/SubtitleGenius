# DASH流媒体字幕处理系统

这个系统可以将webm视频转换为带字幕的DASH流媒体，支持实时字幕生成和翻译。系统通过引入固定延迟（10-30秒），解决了实时字幕生成与视频同步的问题。

## 功能特点

- 接收webm格式的视频文件
- 实时提取音频并生成翻译字幕
- 将字幕与视频同步并封装为DASH格式
- 保持固定的延迟时间（10-30秒）
- 确保视频播放流畅，不出现额外停顿
- 提供内置的DASH流媒体服务器和播放器

## 系统要求

- Python 3.7+
- ffmpeg（用于音视频处理）
- SubtitleGenius库（用于字幕生成，可选）

## 安装依赖

```bash
# 安装Python依赖
pip install numpy asyncio

# 安装ffmpeg（Ubuntu/Debian）
sudo apt install ffmpeg

# 安装ffmpeg（macOS）
brew install ffmpeg

# 安装ffmpeg（Windows）
# 下载并安装: https://ffmpeg.org/download.html
```

## 文件说明

- `video_buffer.py`: 视频缓冲器，实现固定延迟
- `subtitle_generator.py`: 字幕生成器，从音频生成WebVTT字幕
- `dash_subtitle_pipeline.py`: 主控制脚本，整合所有组件
- `dash_server.py`: DASH静态服务器，用于分发DASH内容
- `index.html`: DASH播放器页面

## 使用方法

### 1. 处理视频并生成DASH流

```bash
uv run dash_subtitle_pipeline.py \
  --input ../chinese_a_b.webm \
  --output ./dash_output \
  --language zh \
  --target zh \
  --delay 20 \
  --segment 4 \
  --endpoint your-sagemaker-endpoint \
  --region us-east-1
```

参数说明：
- `--input`: 输入webm文件路径
- `--output`: 输出DASH目录
- `--language`: 源语言代码（默认：ar）
- `--target`: 目标翻译语言代码（默认：zh）
- `--delay`: 视频延迟秒数（默认：20）
- `--segment`: DASH分段时长（默认：4秒）
- `--endpoint`: SageMaker端点名称（可选）
- `--region`: AWS区域（默认：us-east-1）
- `--verbose`: 输出详细日志

### 2. 启动DASH服务器

```bash
uv run dash_server.py \
  --dir ./dash_output \
  --port 8002 \
  --create-player
```

参数说明：
- `--dir`: 要提供服务的目录（默认：当前目录）
- `--port`: 服务器端口（默认：8000）
- `--cors`: CORS策略（默认：*）
- `--create-player`: 创建HTML播放器页面
- `--manifest`: DASH清单文件名（用于播放器页面）

### 3. 播放DASH流

#### 浏览器播放

1. 打开浏览器访问 http://localhost:8000/
2. 确认DASH清单URL正确（默认为manifest.mpd）
3. 点击"加载"按钮加载视频
4. 使用视频播放器控件控制播放

#### VLC播放

1. 打开VLC媒体播放器
2. 选择"媒体" > "打开网络串流"
3. 输入网络URL: http://localhost:8000/manifest.mpd
4. 点击"播放"按钮

## 单独使用各组件

### 视频缓冲器

```bash
python video_buffer.py \
  --input input.webm \
  --output output.webm \
  --delay 20 \
  --fps 30 \
  --resolution 1280x720
```

### 字幕生成器

```bash
python subtitle_generator.py \
  --audio input.wav \
  --output subtitles.vtt \
  --language ar \
  --target zh \
  --endpoint your-sagemaker-endpoint \
  --region us-east-1
```

### DASH服务器

```bash
python dash_server.py \
  --dir ./dash_output \
  --port 8000 \
  --cors "*" \
  --create-player
```

## 模拟模式

如果SubtitleGenius库不可用，系统会自动切换到模拟模式，生成模拟字幕。这对于测试系统流程非常有用。

## 故障排除

### 常见问题

1. **命名管道创建失败**
   - 检查临时目录权限
   - 确保没有同名文件

2. **ffmpeg命令失败**
   - 确认ffmpeg已正确安装
   - 检查输入文件格式是否支持

3. **字幕生成失败**
   - 检查AWS凭证配置
   - 确认SageMaker端点可用

4. **播放器无法加载视频**
   - 检查DASH清单文件是否正确生成
   - 确认服务器正在运行
   - 检查浏览器控制台错误

### 调试技巧

使用`--verbose`参数获取详细日志：

```bash
python dash_subtitle_pipeline.py --input input.webm --output ./dash_output --verbose
```

## 许可证

MIT License
