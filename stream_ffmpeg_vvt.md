# DASH流媒体实时字幕处理系统设计

## 1. 系统概述

本文档描述了一个基于ffmpeg和SubtitleGenius的实时DASH流媒体字幕处理系统。该系统能够接收webm格式的视频文件，实时生成翻译字幕，并将字幕与视频一起封装为DASH格式进行直播。系统设计允许10-30秒的固定延迟，以确保字幕生成和视频同步。

### 1.1 系统目标

- 接收webm格式的视频文件
- 实时提取音频并生成翻译字幕
- 将字幕与视频同步并封装为DASH格式
- 保持固定的延迟时间（10-30秒）
- 确保视频播放流畅，不出现额外停顿

### 1.2 系统架构图

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│             │    │             │    │             │    │             │
│  视频输入   │───▶│  音频提取   │───▶│ 字幕生成器  │───▶│  字幕格式   │
│  (webm)    │    │             │    │(SubtitleGenius)│  │  转换器    │
│             │    └─────────────┘    └─────────────┘    └─────────────┘
│             │                                                 │
│             │    ┌─────────────┐    ┌─────────────┐           ▼
│             │───▶│  视频缓冲   │───▶│  DASH封装   │◀────┌─────────────┐
│             │    │  (延迟处理) │    │  (ffmpeg)   │     │  WebVTT    │
└─────────────┘    └─────────────┘    └─────────────┘     │  字幕文件   │
                                             │            └─────────────┘
                                             ▼
                                      ┌─────────────┐
                                      │  DASH直播   │
                                      │  输出       │
                                      └─────────────┘
```

## 2. 技术方案详细设计

### 2.1 系统组件

系统由以下几个主要组件组成：

1. **视频输入处理器**：接收webm格式的视频文件
2. **音频提取器**：从视频中提取音频流
3. **字幕生成器**：使用SubtitleGenius处理音频并生成字幕
4. **字幕格式转换器**：将生成的字幕转换为WebVTT格式
5. **视频缓冲器**：实现视频的固定延迟处理
6. **DASH封装器**：使用ffmpeg将视频和字幕封装为DASH格式

### 2.2 数据流

1. webm视频文件输入系统
2. 视频流进入缓冲区，同时提取音频流
3. 音频流发送到SubtitleGenius进行处理
4. SubtitleGenius生成字幕并转换为WebVTT格式
5. 缓冲区输出延迟后的视频流
6. ffmpeg将延迟后的视频流和WebVTT字幕一起封装为DASH格式
7. DASH流输出到指定目录或服务器

### 2.3 延迟机制

系统使用固定大小的缓冲区实现视频延迟：

1. 系统启动时，视频帧首先填充缓冲区，直到达到预设的延迟时间（如20秒）
2. 当缓冲区填满后，每当新的一帧进入缓冲区，最早的一帧就会被推出并发送到输出
3. 这种机制确保视频输出比输入恒定地延迟固定时间
4. 延迟时间可以根据字幕生成的需要进行调整（10-30秒范围内）

## 3. 技术实现

### 3.1 视频缓冲器实现

使用Python实现的视频帧缓冲器：

```python
# video_buffer.py
import subprocess
import numpy as np
import time
from collections import deque
import os
import argparse

def create_buffer(input_path, output_path, delay_seconds=20, frame_rate=30, resolution="1280x720"):
    """
    创建视频缓冲器，实现固定延迟
    
    参数:
        input_path: 输入视频路径
        output_path: 输出视频路径
        delay_seconds: 延迟秒数
        frame_rate: 视频帧率
        resolution: 视频分辨率
    """
    # 解析分辨率
    width, height = map(int, resolution.split('x'))
    
    # 计算帧大小和缓冲区容量
    frame_size = width * height * 3  # RGB24格式
    buffer_capacity = delay_seconds * frame_rate
    
    # 创建帧缓冲区
    frame_buffer = deque(maxlen=buffer_capacity)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # 创建命名管道（如果输出路径是管道）
    if output_path.startswith('pipe:'):
        pipe_path = output_path[5:]
        if not os.path.exists(pipe_path):
            os.mkfifo(pipe_path)
    
    # 从输入读取视频帧
    input_process = subprocess.Popen([
        'ffmpeg', '-i', input_path, '-f', 'rawvideo', 
        '-pix_fmt', 'rgb24', '-vf', f'fps={frame_rate}', '-'
    ], stdout=subprocess.PIPE)
    
    # 向输出写入延迟后的视频帧
    output_process = subprocess.Popen([
        'ffmpeg', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 
        '-s', resolution, '-r', str(frame_rate), '-i', '-', 
        '-c:v', 'libvpx-vp9', '-f', 'webm', output_path
    ], stdin=subprocess.PIPE)
    
    print(f"开始填充缓冲区，延迟设置为{delay_seconds}秒...")
    
    # 填充缓冲区
    frames_read = 0
    while frames_read < buffer_capacity:
        frame = input_process.stdout.read(frame_size)
        if not frame or len(frame) < frame_size:
            break
        frame_buffer.append(frame)
        frames_read += 1
        if frames_read % frame_rate == 0:
            print(f"缓冲区填充进度: {frames_read}/{buffer_capacity} 帧 ({frames_read/buffer_capacity*100:.1f}%)")
    
    print(f"缓冲区已填充，开始输出（延迟{delay_seconds}秒）")
    
    # 持续处理
    try:
        while True:
            # 读取新帧
            new_frame = input_process.stdout.read(frame_size)
            if not new_frame or len(new_frame) < frame_size:
                break
                
            # 将新帧添加到缓冲区（自动移除最旧的帧）
            frame_buffer.append(new_frame)
            
            # 输出最旧的帧
            output_process.stdin.write(frame_buffer[0])
            
    except KeyboardInterrupt:
        print("处理中断")
    finally:
        # 清理
        print("处理完成，清理资源...")
        input_process.terminate()
        output_process.stdin.close()
        output_process.wait()
        print("缓冲器已关闭")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="视频缓冲器，实现固定延迟")
    parser.add_argument("--input", required=True, help="输入视频路径")
    parser.add_argument("--output", required=True, help="输出视频路径")
    parser.add_argument("--delay", type=int, default=20, help="延迟秒数")
    parser.add_argument("--fps", type=int, default=30, help="视频帧率")
    parser.add_argument("--resolution", default="1280x720", help="视频分辨率")
    
    args = parser.parse_args()
    create_buffer(args.input, args.output, args.delay, args.fps, args.resolution)
```

### 3.2 字幕生成器实现

使用SubtitleGenius处理音频并生成WebVTT字幕：

```python
# subtitle_generator.py
import asyncio
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

# 导入SubtitleGenius相关模块
from subtitle_genius.audio.processor import AudioProcessor
from subtitle_genius.models.transcribe_model import TranscribeModel
from subtitle_genius.models.whisper_sagemaker_streaming import WhisperSageMakerStreamConfig
from translation_service import translation_manager

async def generate_subtitles(
    audio_input,
    subtitle_output,
    language="ar",
    target_language="zh",
    endpoint_name="your-sagemaker-endpoint",
    region="us-east-1"
):
    """
    从音频生成WebVTT字幕
    
    参数:
        audio_input: 输入音频路径或管道
        subtitle_output: 输出字幕路径或管道
        language: 源语言
        target_language: 目标翻译语言
        endpoint_name: SageMaker端点名称
        region: AWS区域
    """
    print(f"初始化字幕生成器，语言: {language} -> {target_language}")
    
    # 初始化音频处理器
    audio_processor = AudioProcessor()
    
    # 如果输入是管道，先保存到临时文件
    temp_file = None
    if audio_input.startswith('pipe:'):
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        pipe_path = audio_input[5:]
        
        # 从管道读取音频并保存
        subprocess.run([
            'ffmpeg', '-i', pipe_path, '-f', 'wav', 
            '-ar', '16000', '-ac', '1', temp_file.name
        ], check=True)
        
        audio_path = temp_file.name
    else:
        audio_path = audio_input
    
    # 加载音频文件
    audio_data = await audio_processor.process_file(audio_path)
    
    # 配置Whisper模型
    config = WhisperSageMakerStreamConfig(
        chunk_duration=30,
        overlap_duration=2,
        voice_threshold=0.01,
        sagemaker_chunk_duration=30
    )
    
    # 初始化转录模型
    model = TranscribeModel(
        backend="sagemaker_whisper",
        sagemaker_endpoint=endpoint_name,
        region_name=region,
        whisper_config=config
    )
    
    # 创建异步生成器
    async def audio_generator():
        yield audio_data
    
    # 打开字幕输出文件
    with open(subtitle_output, "w", encoding="utf-8") as f:
        # 写入WebVTT头
        f.write("WEBVTT\n\n")
        
        # 生成字幕
        subtitle_count = 0
        async for subtitle in model.transcribe_stream(audio_generator(), language=language):
            # 翻译字幕
            if target_language != language:
                translation_result = await translation_manager.translate(
                    text=subtitle.text,
                    target_lang=target_language,
                    service="bedrock"  # 或其他可用服务
                )
                subtitle.translated_text = translation_result.translated_text
            
            # 写入WebVTT格式
            f.write(f"{subtitle.format_time(subtitle.start, 'vtt')} --> {subtitle.format_time(subtitle.end, 'vtt')}\n")
            f.write(f"{subtitle.text}\n")
            if subtitle.translated_text:
                f.write(f"{subtitle.translated_text}\n")
            f.write("\n")
            
            # 确保文件立即写入（对于流式处理很重要）
            f.flush()
            os.fsync(f.fileno())
            
            subtitle_count += 1
            if subtitle_count % 5 == 0:
                print(f"已生成 {subtitle_count} 条字幕")
    
    # 清理临时文件
    if temp_file:
        os.unlink(temp_file.name)
    
    print(f"字幕生成完成，共 {subtitle_count} 条字幕")
    return subtitle_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="音频字幕生成器")
    parser.add_argument("--audio", required=True, help="输入音频路径或管道")
    parser.add_argument("--output", required=True, help="输出字幕路径")
    parser.add_argument("--language", default="ar", help="源语言")
    parser.add_argument("--target", default="zh", help="目标语言")
    parser.add_argument("--endpoint", default="your-endpoint", help="SageMaker端点")
    parser.add_argument("--region", default="us-east-1", help="AWS区域")
    
    args = parser.parse_args()
    asyncio.run(generate_subtitles(
        args.audio, args.output, args.language, 
        args.target, args.endpoint, args.region
    ))
```

### 3.3 主控制脚本

整合所有组件的主控制脚本：

```python
# dash_subtitle_pipeline.py
import subprocess
import os
import asyncio
import argparse
import tempfile
import signal
import sys
from pathlib import Path

async def process_webm_to_dash(
    input_webm,
    output_dir,
    language="ar",
    target_language="zh",
    delay_seconds=20,
    segment_duration=4,
    endpoint_name="your-sagemaker-endpoint",
    region="us-east-1"
):
    """
    处理webm文件，生成带字幕的DASH流
    
    参数:
        input_webm: 输入webm文件路径
        output_dir: 输出DASH目录
        language: 源语言
        target_language: 目标翻译语言
        delay_seconds: 视频延迟秒数
        segment_duration: DASH分段时长(秒)
        endpoint_name: SageMaker端点名称
        region: AWS区域
    """
    # 创建临时目录
    temp_dir = Path(tempfile.mkdtemp())
    audio_pipe = temp_dir / "audio.pipe"
    video_pipe = temp_dir / "video.pipe"
    subtitle_path = temp_dir / "subtitles.vtt"
    
    # 创建命名管道
    os.mkfifo(audio_pipe)
    os.mkfifo(video_pipe)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 启动进程列表
    processes = []
    
    try:
        # 1. 启动音频提取进程
        audio_process = subprocess.Popen([
            "ffmpeg", "-i", input_webm, 
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            "-f", "wav", str(audio_pipe)
        ])
        processes.append(audio_process)
        print("音频提取进程已启动")
        
        # 2. 启动字幕生成进程
        subtitle_process = subprocess.Popen([
            "python", "subtitle_generator.py",
            "--audio", f"pipe:{audio_pipe}",
            "--output", str(subtitle_path),
            "--language", language,
            "--target", target_language,
            "--endpoint", endpoint_name,
            "--region", region
        ])
        processes.append(subtitle_process)
        print("字幕生成进程已启动")
        
        # 3. 启动视频缓冲进程
        buffer_process = subprocess.Popen([
            "python", "video_buffer.py",
            "--input", input_webm,
            "--output", f"pipe:{video_pipe}",
            "--delay", str(delay_seconds)
        ])
        processes.append(buffer_process)
        print(f"视频缓冲进程已启动，延迟设置为{delay_seconds}秒")
        
        # 等待字幕文件创建
        while not os.path.exists(subtitle_path):
            await asyncio.sleep(1)
            print("等待字幕文件创建...")
        
        # 4. 启动DASH封装进程
        dash_process = subprocess.Popen([
            "ffmpeg", 
            "-i", f"pipe:{video_pipe}",  # 延迟后的视频
            "-i", str(subtitle_path),    # 字幕文件
            "-map", "0:v", "-map", "0:a", "-map", "1",  # 映射流
            "-c:v", "copy",              # 复制视频编码
            "-c:a", "aac", "-b:a", "128k",  # 音频编码
            "-c:s", "webvtt",            # 字幕编码
            "-f", "dash",                # DASH格式
            "-seg_duration", str(segment_duration),  # 分段时长
            "-streaming", "1",           # 流式输出
            "-use_template", "1",        # 使用模板
            "-use_timeline", "1",        # 使用时间线
            "-window_size", "5",         # 窗口大小
            "-adaptation_sets", "id=0,streams=v id=1,streams=a id=2,streams=s",  # 自适应集
            f"{output_dir}/manifest.mpd"  # 输出文件
        ])
        processes.append(dash_process)
        print(f"DASH封装进程已启动，输出到 {output_dir}/manifest.mpd")
        
        # 等待所有进程完成
        for p in processes:
            p.wait()
            
    except KeyboardInterrupt:
        print("处理中断")
    finally:
        # 清理进程
        for p in processes:
            try:
                p.terminate()
                p.wait(timeout=5)
            except:
                pass
        
        # 清理管道
        try:
            os.unlink(audio_pipe)
            os.unlink(video_pipe)
        except:
            pass
            
        print("处理完成，资源已清理")
    
    return f"{output_dir}/manifest.mpd"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DASH流媒体字幕处理系统")
    parser.add_argument("--input", required=True, help="输入webm文件路径")
    parser.add_argument("--output", required=True, help="输出DASH目录")
    parser.add_argument("--language", default="ar", help="源语言")
    parser.add_argument("--target", default="zh", help="目标语言")
    parser.add_argument("--delay", type=int, default=20, help="视频延迟秒数")
    parser.add_argument("--segment", type=int, default=4, help="DASH分段时长(秒)")
    parser.add_argument("--endpoint", default="your-endpoint", help="SageMaker端点")
    parser.add_argument("--region", default="us-east-1", help="AWS区域")
    
    args = parser.parse_args()
    
    # 处理Ctrl+C信号
    def signal_handler(sig, frame):
        print("接收到中断信号，正在清理...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # 运行处理流程
    asyncio.run(process_webm_to_dash(
        args.input, args.output, args.language, args.target,
        args.delay, args.segment, args.endpoint, args.region
    ))
```

## 4. 使用指南

### 4.1 环境准备

1. 安装必要的依赖：

```bash
# 安装ffmpeg
sudo apt install ffmpeg

# 安装Python依赖
pip install numpy asyncio

# 安装SubtitleGenius
# 按照项目README进行安装
```

2. 配置AWS凭证（用于SageMaker和Bedrock服务）：

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=us-east-1
```

### 4.2 运行系统

使用主控制脚本启动系统：

```bash
python dash_subtitle_pipeline.py \
  --input input.webm \
  --output ./dash_output \
  --language ar \
  --target zh \
  --delay 20 \
  --segment 4 \
  --endpoint your-sagemaker-endpoint \
  --region us-east-1
```

### 4.3 参数说明

- `--input`: 输入webm文件路径
- `--output`: 输出DASH目录
- `--language`: 源语言代码（默认：ar）
- `--target`: 目标翻译语言代码（默认：zh）
- `--delay`: 视频延迟秒数（默认：20）
- `--segment`: DASH分段时长（默认：4秒）
- `--endpoint`: SageMaker端点名称
- `--region`: AWS区域（默认：us-east-1）

### 4.4 播放DASH流

可以使用支持DASH的播放器播放生成的流：

```html
<!DOCTYPE html>
<html>
<head>
    <title>DASH播放器</title>
    <script src="https://cdn.dashjs.org/latest/dash.all.min.js"></script>
    <style>
        video {
            width: 80%;
            max-width: 1280px;
        }
    </style>
</head>
<body>
    <div>
        <video id="videoPlayer" controls></video>
    </div>
    
    <script>
        var player = dashjs.MediaPlayer().create();
        player.initialize(document.querySelector("#videoPlayer"), "http://your-server/dash_output/manifest.mpd", true);
    </script>
</body>
</html>
```

## 5. 性能优化

### 5.1 系统资源优化

1. **CPU使用优化**：
   - 使用多进程处理不同组件
   - 为ffmpeg设置线程数限制：`-threads 4`
   - 调整视频编码参数减少CPU负载

2. **内存使用优化**：
   - 使用流式处理减少内存占用
   - 定期清理缓冲区
   - 使用内存映射文件处理大型视频帧

3. **磁盘I/O优化**：
   - 使用命名管道减少磁盘写入
   - 避免不必要的临时文件
   - 使用异步I/O操作

### 5.2 延迟优化

1. **减少初始延迟**：
   - 优化缓冲区填充策略
   - 使用更高效的视频帧处理算法
   - 调整延迟参数找到最佳平衡点

2. **提高字幕生成速度**：
   - 使用更小的音频块进行处理
   - 优化Whisper模型参数
   - 使用更高效的翻译服务

## 6. 故障处理

### 6.1 常见问题及解决方案

1. **进程意外终止**：
   - 实现监控脚本检测进程状态
   - 自动重启失败的组件
   - 保存处理状态以便恢复

2. **字幕同步问题**：
   - 检查时间戳计算逻辑
   - 调整延迟参数
   - 验证WebVTT格式是否正确

3. **内存溢出**：
   - 减小缓冲区大小
   - 优化视频帧存储
   - 增加系统内存或使用交换空间

### 6.2 调试技巧

1. 启用详细日志：

```bash
python dash_subtitle_pipeline.py --input input.webm --output ./dash_output --verbose
```

2. 检查各组件状态：

```bash
# 检查进程状态
ps aux | grep python
ps aux | grep ffmpeg

# 检查管道状态
ls -la /tmp/subtitle_genius_*
```

3. 验证DASH输出：

```bash
# 检查DASH清单文件
cat ./dash_output/manifest.mpd

# 验证分段文件
ls -la ./dash_output/chunk_*.m4s
```

## 7. 扩展功能

### 7.1 多语言支持

扩展系统支持更多语言对：

```python
# 语言配置
LANGUAGE_CONFIGS = {
    "ar": {"name": "Arabic", "whisper_code": "ar", "bedrock_code": "arb"},
    "zh": {"name": "Chinese", "whisper_code": "zh", "bedrock_code": "cmn-Hans"},
    "en": {"name": "English", "whisper_code": "en", "bedrock_code": "eng"},
    "fr": {"name": "French", "whisper_code": "fr", "bedrock_code": "fra"},
    "de": {"name": "German", "whisper_code": "de", "bedrock_code": "deu"},
    "ja": {"name": "Japanese", "whisper_code": "ja", "bedrock_code": "jpn"},
    "ko": {"name": "Korean", "whisper_code": "ko", "bedrock_code": "kor"},
    "ru": {"name": "Russian", "whisper_code": "ru", "bedrock_code": "rus"},
    "es": {"name": "Spanish", "whisper_code": "es", "bedrock_code": "spa"},
}
```

### 7.2 多字幕轨道

支持多种语言的字幕轨道：

```bash
ffmpeg -i video_pipe \
       -i subtitles_zh.vtt \
       -i subtitles_en.vtt \
       -map 0:v -map 0:a -map 1 -map 2 \
       -c:v copy -c:a aac \
       -c:s webvtt -c:s webvtt \
       -metadata:s:s:0 language=chi \
       -metadata:s:s:1 language=eng \
       -f dash output/manifest.mpd
```

### 7.3 自动化部署

创建Docker容器简化部署：

```dockerfile
FROM python:3.9

# 安装ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

# 安装依赖
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

# 复制应用代码
COPY . /app/
WORKDIR /app

# 设置环境变量
ENV AWS_REGION=us-east-1

# 入口点
ENTRYPOINT ["python", "dash_subtitle_pipeline.py"]
```

## 8. 结论

本设计文档描述了一个基于ffmpeg和SubtitleGenius的实时DASH流媒体字幕处理系统。该系统通过引入固定延迟，解决了实时字幕生成与视频同步的问题。系统架构采用多进程协作的方式，确保各组件独立运行，同时保持数据流的连续性。

通过合理设置延迟参数（10-30秒范围内），系统可以在保证字幕质量的同时，提供流畅的视频播放体验。该方案充分利用了SubtitleGenius的流式处理能力和ffmpeg的DASH封装功能，适用于各种实时直播场景，如体育赛事、新闻直播等。

---

*文档版本: 1.0.0*  
*更新日期: 2025-07-07*
