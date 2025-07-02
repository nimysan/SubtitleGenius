# Amazon Transcribe 音频格式分析对话历史

## 问题背景

在 SubtitleGenius 项目中，我们发现不同的音频格式会导致 Amazon Transcribe 产生不正确的字幕结果。通过对比 `output.wav` 和 `output_16k_mono.wav` 两个文件，我们深入分析了 Amazon Transcribe 对音频格式的要求。

## 音频文件格式对比

### output.wav (原始文件)
```json
{
  "音频编码": "PCM signed 16-bit little-endian",
  "采样率": "48000 Hz",
  "声道数": "2 (立体声)",
  "位深度": "16 bit",
  "比特率": "1536000 bps",
  "文件大小": "11.5 MB",
  "时长": "60.0 秒"
}
```

### output_16k_mono.wav (优化后文件)
```json
{
  "音频编码": "PCM signed 16-bit little-endian", 
  "采样率": "16000 Hz",
  "声道数": "1 (单声道)",
  "位深度": "16 bit",
  "比特率": "256000 bps",
  "文件大小": "1.9 MB",
  "时长": "60.0 秒"
}
```

## 关键差异分析

### 1. 采样率差异
- **原始文件**: 48000 Hz (高采样率)
- **优化文件**: 16000 Hz (标准语音采样率)
- **影响**: 48kHz 对语音识别来说是过采样，会增加处理负担且可能引入噪声

### 2. 声道数差异
- **原始文件**: 2 声道 (立体声)
- **优化文件**: 1 声道 (单声道)
- **影响**: 立体声会增加数据量，对语音识别没有额外价值

### 3. 文件大小差异
- **原始文件**: 11.5 MB
- **优化文件**: 1.9 MB (减少 83%)
- **影响**: 文件大小直接影响上传速度和处理效率

## Amazon Transcribe 格式要求

### 支持的音频格式
```
✅ 推荐格式:
- WAV (PCM 16-bit)
- FLAC
- MP3
- MP4 (音频部分)
- WebM (音频部分)

✅ 推荐参数:
- 采样率: 8000Hz, 16000Hz, 22050Hz, 44100Hz
- 声道: 单声道 (推荐) 或 立体声
- 位深度: 16-bit (推荐) 或 24-bit
- 编码: PCM (无损)
```

### 流式处理最佳实践
```python
# 最佳音频参数配置
SAMPLE_RATE = 16000      # 16kHz - 语音识别标准
BYTES_PER_SAMPLE = 2     # 16-bit = 2 bytes
CHANNEL_NUMS = 1         # 单声道
CHUNK_SIZE = 1024 * 8    # 8KB 块大小
```

## 格式转换命令

### 使用 FFmpeg 转换
```bash
# 转换为 Amazon Transcribe 最佳格式
ffmpeg -i output.wav -ar 16000 -ac 1 -sample_fmt s16 output_16k_mono.wav -y

# 参数说明:
# -ar 16000    : 设置采样率为 16kHz
# -ac 1        : 设置为单声道
# -sample_fmt s16 : 设置为 16-bit PCM
# -y           : 覆盖输出文件
```

### 批量转换脚本
```bash
#!/bin/bash
# 批量转换音频文件为 Transcribe 兼容格式

for file in *.wav; do
    if [[ -f "$file" ]]; then
        output="${file%.*}_16k_mono.wav"
        echo "转换: $file -> $output"
        ffmpeg -i "$file" -ar 16000 -ac 1 -sample_fmt s16 "$output" -y
    fi
done
```

## 格式对转录质量的影响

### 1. 采样率影响
```
48kHz -> 16kHz 转换的影响:
✅ 优点:
- 减少数据量 (3倍减少)
- 提高处理速度
- 降低网络传输时间
- 符合语音识别标准

⚠️ 注意:
- 16kHz 足够捕获人声频率范围 (0-8kHz)
- 48kHz 对语音识别是过采样
```

### 2. 声道数影响
```
立体声 -> 单声道 转换的影响:
✅ 优点:
- 文件大小减半
- 处理速度提升
- 避免声道间干扰

⚠️ 注意:
- 语音识别通常不需要立体声信息
- 单声道足够保留语音特征
```

### 3. 实际测试结果对比

#### 使用 output.wav (48kHz 立体声)
```
❌ 问题:
- 转录准确率较低
- 处理时间较长
- 网络传输慢
- 可能出现音频同步问题
```

#### 使用 output_16k_mono.wav (16kHz 单声道)
```
✅ 改善:
- 转录准确率提高
- 处理速度加快
- 文件传输快速
- 音频同步准确
```

## 项目中的实现

### 音频预处理函数
```python
def preprocess_audio_for_transcribe(input_path: str, output_path: str):
    """
    为 Amazon Transcribe 预处理音频文件
    """
    import subprocess
    
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-ar', '16000',      # 16kHz 采样率
        '-ac', '1',          # 单声道
        '-sample_fmt', 's16', # 16-bit PCM
        output_path,
        '-y'                 # 覆盖输出
    ]
    
    subprocess.run(cmd, check=True)
    print(f"音频已转换: {input_path} -> {output_path}")
```

### 配置文件设置
```python
# subtitle_genius/config.py
TRANSCRIBE_AUDIO_CONFIG = {
    'sample_rate': 16000,
    'channels': 1,
    'sample_format': 's16',
    'chunk_size': 1024 * 8,
    'encoding': 'pcm'
}
```

## 最佳实践建议

### 1. 音频预处理流程
```
原始音频 -> 格式检查 -> 必要时转换 -> Transcribe 处理
```

### 2. 自动格式检测
```python
def check_audio_format(file_path):
    """检查音频格式是否适合 Transcribe"""
    # 使用 ffprobe 检查格式
    # 如果不符合要求，自动转换
    pass
```

### 3. 性能优化
- 优先使用 16kHz 单声道
- 使用 PCM 编码避免解码开销
- 合理设置 chunk_size 平衡延迟和效率

## 故障排除

### 常见问题
1. **转录结果不准确**
   - 检查采样率是否为 16kHz
   - 确认使用单声道
   - 验证音频质量

2. **处理速度慢**
   - 降低采样率到 16kHz
   - 转换为单声道
   - 减小文件大小

3. **网络传输超时**
   - 预处理音频减小文件大小
   - 使用流式处理
   - 检查网络连接

### 调试命令
```bash
# 检查音频文件信息
ffprobe -v quiet -print_format json -show_format -show_streams audio.wav

# 测试转换效果
ffmpeg -i input.wav -ar 16000 -ac 1 -sample_fmt s16 test_output.wav -y

# 比较文件大小
ls -lh *.wav
```

## 结论

通过对比 `output.wav` 和 `output_16k_mono.wav`，我们发现：

1. **格式标准化至关重要**: Amazon Transcribe 对音频格式有明确要求
2. **16kHz 单声道是最佳选择**: 平衡了质量和效率
3. **预处理是必要步骤**: 可以显著提高转录准确率
4. **文件大小优化**: 正确的格式可以减少 80%+ 的文件大小

这次分析帮助我们建立了完整的音频预处理流程，确保 SubtitleGenius 项目能够高效准确地生成字幕。

---

*此文档记录了 SubtitleGenius 项目中关于音频格式分析和 Amazon Transcribe 格式要求的详细对话历史。*
