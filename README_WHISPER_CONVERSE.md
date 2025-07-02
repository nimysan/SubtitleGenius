# Whisper Turbo SageMaker API 使用指南

基于 AWS 样例代码，使用 Amazon SageMaker Runtime 调用自定义 Whisper Turbo 端点。

**重要更新**: 由于 Converse API 不支持音频格式，现已改用 SageMaker Runtime 直接调用端点。

## 📋 文件说明

- `whisper_converse.py` - 主要的 SageMaker Runtime 客户端类
- `example_whisper_converse.py` - 简单的使用示例
- `whisper_config.py` - 配置管理
- `whisper-transcription.py` - 下载的AWS样例代码（参考用）
- `README_WHISPER_CONVERSE.md` - 本文档

## 🚀 快速开始

### 1. 配置端点信息

编辑 `whisper_config.py` 或设置环境变量：

```python
# 方法1: 编辑 whisper_config.py
DEFAULT_ENDPOINT_NAME = "your-actual-whisper-endpoint"  # 🔧 替换为实际SageMaker端点名称
DEFAULT_REGION = "us-east-1"  # 🔧 替换为你的AWS区域

# 方法2: 使用环境变量
export WHISPER_ENDPOINT_NAME="your-actual-whisper-endpoint"
export AWS_REGION="us-east-1"
```

### 2. 配置 AWS 凭证和权限

```bash
# 方法1: 环境变量
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"

# 方法2: AWS CLI
aws configure

# 方法3: IAM角色 (推荐用于EC2/Lambda)
```

**重要**: 确保IAM用户/角色有以下权限：
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:InvokeEndpoint"
            ],
            "Resource": "arn:aws:sagemaker:*:*:endpoint/your-whisper-endpoint"
        }
    ]
}
```

### 3. 运行示例

```bash
# 检查配置状态
python whisper_config.py

# 运行基本示例
python example_whisper_converse.py

# 查看帮助
python example_whisper_converse.py --help
```

## 💻 代码示例

### 基本用法

```python
from whisper_converse import WhisperSageMakerClient

# 初始化客户端
client = WhisperSageMakerClient(
    endpoint_name="your-whisper-endpoint",
    region_name="us-east-1"
)

# Arabic 语音转录（自动分块）
result = client.transcribe_audio(
    audio_path="audio.wav",
    language="ar",
    task="transcribe",
    chunk_duration=30  # 30秒分块
)

print(f"转录结果: {result['transcription']}")
print(f"处理块数: {result['chunks_processed']}")
```

### 高级用法

```python
# 翻译任务
result = client.transcribe_audio(
    audio_path="arabic_audio.wav",
    language="ar",
    task="translate",  # 翻译到英语
    chunk_duration=20  # 较短分块用于翻译
)

# 批量处理
results = client.batch_transcribe(
    audio_files=["file1.wav", "file2.wav"],
    language="ar",
    task="transcribe",
    chunk_duration=30
)

# 长音频处理
result = client.transcribe_audio(
    audio_path="long_audio.wav",
    language="ar",
    task="transcribe",
    chunk_duration=60  # 更大分块处理长音频
)
```

## 🎯 功能特性

### ✅ 支持的功能

- **音频分块处理**: 自动将长音频分割为小块，避免SageMaker负载限制
- **多格式支持**: WAV, MP3, AAC, M4A, FLAC, OGG
- **自动格式转换**: 使用FFmpeg自动转换非WAV格式
- **多语言转录**: Arabic, English, Chinese, Japanese 等
- **语音翻译**: 自动翻译到英语
- **批量处理**: 同时处理多个音频文件
- **性能监控**: 处理时间、分块统计
- **错误处理**: 完善的异常处理和重试机制

### 🎵 支持的音频格式

| 格式 | 扩展名 | 处理方式 |
|------|--------|----------|
| WAV | `.wav` | 直接处理 (推荐) |
| MP3 | `.mp3` | FFmpeg转换 |
| AAC | `.aac` | FFmpeg转换 |
| M4A | `.m4a` | FFmpeg转换 |
| FLAC | `.flac` | FFmpeg转换 |
| OGG | `.ogg` | FFmpeg转换 |

### 🌍 支持的语言

| 代码 | 语言 | Whisper代码 |
|------|------|-------------|
| `ar` | Arabic | `arabic` |
| `ar-SA` | Arabic (Saudi) | `arabic` |
| `en` | English | `english` |
| `zh` | Chinese | `chinese` |
| `ja` | Japanese | `japanese` |
| `ko` | Korean | `korean` |
| `fr` | French | `french` |
| `de` | German | `german` |
| `es` | Spanish | `spanish` |
| `ru` | Russian | `russian` |

## 🔧 配置选项

### 基本配置

```python
{
    "endpoint_name": "your-whisper-endpoint",  # 必需: SageMaker端点名称
    "region_name": "us-east-1",               # AWS区域
    "chunk_duration": 30,                     # 音频分块时长（秒）
    "task": "transcribe",                     # 任务类型
    "language": "ar"                          # 语言代码
}
```

### 分块策略

- **短音频** (< 30秒): 不分块，直接处理
- **中等音频** (30秒-5分钟): 30秒分块
- **长音频** (> 5分钟): 60秒分块
- **实时处理**: 10-15秒分块

### 任务类型

- `transcribe`: 转录 (保持原语言)
- `translate`: 翻译 (转换为英语)

## 📊 性能监控

每次调用都会返回详细的性能指标：

```python
result = client.transcribe_audio("audio.wav", language="ar")

print(f"处理时间: {result['metrics']['processing_time_seconds']}秒")
print(f"处理块数: {result['metrics']['chunks_count']}")
print(f"平均每块时间: {result['metrics']['average_chunk_time']}秒")
print(f"音频信息: {result['audio_info']}")
```

## 🛠️ 故障排除

### 常见问题

1. **SageMaker端点未找到**
   ```
   ❌ 错误: EndpointNotFound
   💡 解决: 检查端点名称和区域设置
   ```

2. **权限不足**
   ```
   ❌ 错误: AccessDenied
   💡 解决: 添加 sagemaker:InvokeEndpoint 权限
   ```

3. **音频文件过大**
   ```
   ❌ 错误: PayloadTooLarge
   💡 解决: 减小 chunk_duration 参数
   ```

4. **FFmpeg未安装**
   ```
   ❌ 错误: FFmpeg not found
   💡 解决: 安装FFmpeg或使用WAV格式
   ```

### 调试步骤

```bash
# 1. 检查配置
python whisper_config.py

# 2. 验证AWS凭证
aws sts get-caller-identity

# 3. 检查SageMaker端点状态
aws sagemaker describe-endpoint --endpoint-name your-endpoint

# 4. 测试端点连接
python -c "
from whisper_converse import WhisperSageMakerClient
client = WhisperSageMakerClient('your-endpoint', 'us-east-1')
print('✅ 客户端初始化成功')
"

# 5. 检查音频文件
ffprobe your_audio.wav
```

### FFmpeg 安装

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# CentOS/RHEL
sudo yum install ffmpeg

# 验证安装
ffmpeg -version
```

## 🔄 与现有项目集成

### 集成到 SubtitleGenius

```python
# 在 subtitle_genius/models/ 中添加
from whisper_converse import WhisperSageMakerClient

class WhisperSageMakerModel:
    def __init__(self, endpoint_name, region="us-east-1"):
        self.client = WhisperSageMakerClient(endpoint_name, region)
    
    async def transcribe(self, audio_path, language="ar"):
        result = self.client.transcribe_audio(
            audio_path=audio_path,
            language=language,
            task="transcribe",
            chunk_duration=30
        )
        return result['transcription']
    
    async def transcribe_with_timestamps(self, audio_path, language="ar"):
        result = self.client.transcribe_audio(audio_path, language)
        return {
            "text": result['transcription'],
            "chunks": result['chunk_timings'],
            "duration": sum(end - start for start, end in result['chunk_timings'])
        }
```

### 与流式处理结合

```python
# 结合现有的流式处理
from subtitle_genius.stream.processor import StreamProcessor

async def stream_with_whisper_sagemaker():
    processor = StreamProcessor()
    client = WhisperSageMakerClient("your-endpoint")
    
    # 处理音频流
    audio_stream = processor.start_microphone_stream()
    
    # 分块转录
    async for audio_chunk in audio_stream:
        # 保存临时文件
        temp_file = f"/tmp/chunk_{int(time.time())}.wav"
        with open(temp_file, 'wb') as f:
            f.write(audio_chunk)
        
        # 转录
        result = client.transcribe_audio(
            audio_path=temp_file,
            language="ar",
            chunk_duration=10  # 小分块用于实时处理
        )
        
        print(f"实时转录: {result['transcription']}")
        
        # 清理临时文件
        os.remove(temp_file)
```

## 📈 性能优化建议

1. **音频预处理**
   ```bash
   # 转换为最佳格式 (16kHz, 单声道, WAV)
   ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
   ```

2. **分块大小优化**
   ```python
   # 根据音频长度调整分块
   audio_duration = get_audio_duration(audio_path)
   if audio_duration < 60:
       chunk_duration = 30
   elif audio_duration < 300:
       chunk_duration = 45
   else:
       chunk_duration = 60
   ```

3. **并行处理**
   ```python
   import asyncio
   
   async def parallel_transcribe(files):
       tasks = []
       for file in files:
           task = asyncio.create_task(
               client.transcribe_audio(file, "ar")
           )
           tasks.append(task)
       return await asyncio.gather(*tasks)
   ```

4. **缓存结果**
   ```python
   import hashlib
   import json
   
   def cache_result(audio_path, result):
       # 使用文件哈希作为缓存键
       with open(audio_path, 'rb') as f:
           file_hash = hashlib.md5(f.read()).hexdigest()
       
       cache_file = f"cache/{file_hash}.json"
       with open(cache_file, 'w') as f:
           json.dump(result, f)
   ```

## 📝 示例输出

### 转录示例

```json
{
    "transcription": "مرحبا بكم في هذا البرنامج التعليمي حول استخدام نماذج الذكاء الاصطناعي",
    "language": "ar",
    "task": "transcribe",
    "chunks_processed": 3,
    "chunk_timings": [
        [0, 30],
        [30, 60], 
        [60, 75]
    ],
    "audio_info": {
        "format": "WAV",
        "duration": 75.2,
        "sample_rate": 16000,
        "channels": 1
    },
    "metrics": {
        "processing_time_seconds": 12.5,
        "chunks_count": 3,
        "average_chunk_time": 4.17
    }
}
```

### 翻译示例

```json
{
    "transcription": "Welcome to this educational program about using artificial intelligence models",
    "language": "ar",
    "task": "translate",
    "chunks_processed": 2,
    "metrics": {
        "processing_time_seconds": 8.3,
        "chunks_count": 2
    }
}
```

## 🆚 Converse API vs SageMaker Runtime

| 特性 | Converse API | SageMaker Runtime |
|------|--------------|-------------------|
| 音频支持 | ❌ 不支持 | ✅ 支持 |
| 分块处理 | ❌ 需要手动 | ✅ 自动分块 |
| 格式转换 | ❌ 不支持 | ✅ 自动转换 |
| 负载限制 | 较小 | 可配置 |
| 延迟 | 较低 | 中等 |
| 成本 | 按token计费 | 按调用计费 |

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License - 与主项目保持一致

## 📋 文件说明

- `whisper_converse.py` - 主要的 Converse API 客户端类
- `example_whisper_converse.py` - 简单的使用示例
- `whisper_config.py` - 配置管理
- `README_WHISPER_CONVERSE.md` - 本文档

## 🚀 快速开始

### 1. 配置端点信息

编辑 `whisper_config.py` 或设置环境变量：

```python
# 方法1: 编辑 whisper_config.py
DEFAULT_ENDPOINT_NAME = "your-actual-whisper-endpoint"  # 🔧 替换为实际端点名称
DEFAULT_REGION = "us-east-1"  # 🔧 替换为你的AWS区域

# 方法2: 使用环境变量
export WHISPER_ENDPOINT_NAME="your-actual-whisper-endpoint"
export AWS_REGION="us-east-1"
```

### 2. 配置 AWS 凭证

```bash
# 方法1: 环境变量
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"

# 方法2: AWS CLI
aws configure

# 方法3: IAM角色 (推荐用于EC2/Lambda)
```

### 3. 运行示例

```bash
# 检查配置状态
python whisper_config.py

# 运行基本示例
python example_whisper_converse.py

# 查看帮助
python example_whisper_converse.py --help
```

## 💻 代码示例

### 基本用法

```python
from whisper_converse import WhisperConverseClient

# 初始化客户端
client = WhisperConverseClient(
    endpoint_name="your-whisper-endpoint",
    region_name="us-east-1"
)

# Arabic 语音转录
result = client.transcribe_audio(
    audio_path="audio.wav",
    language="ar",
    task="transcribe"
)

print(f"转录结果: {result['transcription']}")
```

### 高级用法

```python
# 翻译任务
result = client.transcribe_audio(
    audio_path="arabic_audio.wav",
    language="ar",
    task="translate",  # 翻译到英语
    temperature=0.1
)

# 批量处理
results = client.batch_transcribe(
    audio_files=["file1.wav", "file2.wav"],
    language="ar",
    task="transcribe"
)

# 自定义参数
result = client.transcribe_audio(
    audio_path="audio.wav",
    language="ar",
    task="transcribe",
    temperature=0.0,
    max_tokens=2048,
    latency_mode="optimized"
)
```

## 🎯 功能特性

### ✅ 支持的功能

- **多语言转录**: Arabic, English, Chinese, Japanese 等
- **语音翻译**: 自动翻译到英语
- **批量处理**: 同时处理多个音频文件
- **流式处理**: 基于 Converse API 的实时处理
- **性能监控**: 延迟、Token使用统计
- **错误处理**: 完善的异常处理机制

### 🎵 支持的音频格式

- WAV (推荐)
- MP3
- M4A
- AAC
- FLAC
- OGG

### 🌍 支持的语言

| 代码 | 语言 | 说明 |
|------|------|------|
| `ar` | Arabic | 阿拉伯语 (默认) |
| `ar-SA` | Arabic (Saudi) | 沙特阿拉伯语 |
| `ar-AE` | Arabic (UAE) | 阿联酋阿拉伯语 |
| `en` | English | 英语 |
| `zh` | Chinese | 中文 |
| `ja` | Japanese | 日语 |
| `ko` | Korean | 韩语 |
| `fr` | French | 法语 |
| `de` | German | 德语 |
| `es` | Spanish | 西班牙语 |
| `ru` | Russian | 俄语 |

## 🔧 配置选项

### 基本配置

```python
{
    "endpoint_name": "your-whisper-endpoint",  # 必需
    "region_name": "us-east-1",               # AWS区域
    "temperature": 0.0,                       # 温度参数 (0.0-1.0)
    "max_tokens": 1024,                       # 最大输出tokens
    "latency_mode": "standard"                # 延迟模式
}
```

### 延迟模式

- `standard`: 标准模式，平衡质量和速度
- `optimized`: 优化模式，更快响应

### 任务类型

- `transcribe`: 转录 (保持原语言)
- `translate`: 翻译 (转换为英语)

## 📊 性能监控

每次调用都会返回性能指标：

```python
result = client.transcribe_audio("audio.wav", language="ar")

print(f"延迟: {result['metrics']['latency_ms']}ms")
print(f"输入tokens: {result['metrics']['input_tokens']}")
print(f"输出tokens: {result['metrics']['output_tokens']}")
print(f"总tokens: {result['metrics']['total_tokens']}")
```

## 🛠️ 故障排除

### 常见问题

1. **端点未找到**
   ```
   ❌ 错误: EndpointNotFound
   💡 解决: 检查端点名称和区域设置
   ```

2. **AWS凭证错误**
   ```
   ❌ 错误: InvalidCredentials
   💡 解决: 配置正确的AWS访问密钥
   ```

3. **音频格式不支持**
   ```
   ❌ 错误: UnsupportedAudioFormat
   💡 解决: 转换为支持的格式 (WAV推荐)
   ```

4. **Token限制**
   ```
   ❌ 错误: TokenLimitExceeded
   💡 解决: 增加max_tokens或分割长音频
   ```

### 调试步骤

```bash
# 1. 检查配置
python whisper_config.py

# 2. 验证AWS凭证
aws sts get-caller-identity

# 3. 测试端点连接
python -c "
from whisper_converse import WhisperConverseClient
client = WhisperConverseClient('your-endpoint', 'us-east-1')
print('✅ 客户端初始化成功')
"

# 4. 检查音频文件
ffprobe your_audio.wav
```

## 🔄 与现有项目集成

### 集成到 SubtitleGenius

```python
# 在 subtitle_genius/models/ 中添加
from whisper_converse import WhisperConverseClient

class WhisperConverseModel:
    def __init__(self, endpoint_name, region="us-east-1"):
        self.client = WhisperConverseClient(endpoint_name, region)
    
    async def transcribe(self, audio_path, language="ar"):
        result = self.client.transcribe_audio(
            audio_path=audio_path,
            language=language,
            task="transcribe"
        )
        return result['transcription']
```

### 与流式处理结合

```python
# 结合现有的流式处理
from subtitle_genius.stream.processor import StreamProcessor

async def stream_with_whisper_converse():
    processor = StreamProcessor()
    client = WhisperConverseClient("your-endpoint")
    
    # 处理音频流
    audio_stream = processor.start_microphone_stream()
    
    # 分块转录
    async for audio_chunk in audio_stream:
        result = client.transcribe_audio(
            audio_path=audio_chunk,
            language="ar"
        )
        print(f"实时转录: {result['transcription']}")
```

## 📈 性能优化建议

1. **音频预处理**
   ```bash
   # 转换为最佳格式
   ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
   ```

2. **批量处理优化**
   ```python
   # 并行处理多个文件
   import asyncio
   
   async def parallel_transcribe(files):
       tasks = [client.transcribe_audio(f, "ar") for f in files]
       return await asyncio.gather(*tasks)
   ```

3. **缓存结果**
   ```python
   import hashlib
   import json
   
   def cache_result(audio_path, result):
       cache_key = hashlib.md5(open(audio_path, 'rb').read()).hexdigest()
       with open(f"cache/{cache_key}.json", 'w') as f:
           json.dump(result, f)
   ```

## 📝 示例输出

### 转录示例

```json
{
    "transcription": "مرحبا بكم في هذا البرنامج التعليمي",
    "language": "ar",
    "task": "transcribe",
    "audio_info": {
        "format": "WAV",
        "duration": 5.2,
        "sample_rate": 16000
    },
    "metrics": {
        "latency_ms": 1250,
        "input_tokens": 45,
        "output_tokens": 12,
        "total_tokens": 57
    }
}
```

### 翻译示例

```json
{
    "transcription": "Welcome to this educational program",
    "language": "ar",
    "task": "translate",
    "metrics": {
        "latency_ms": 1450,
        "total_tokens": 62
    }
}
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License - 与主项目保持一致
