# 🎬 SubtitleGenius 流式字幕翻译功能

## 功能概述

这是一个独立的 Gradio Web 界面，提供以下功能：
- **左侧**: 上传本地 WAV 音频文件
- **中间**: 实时显示流式字幕输出
- **右侧**: 显示翻译后的中文内容

## 🚀 快速开始

### 1. 安装依赖

```bash
# 安装流式处理依赖
python install_streaming.py

# 安装 Gradio (如果还没有)
pip install gradio

# 安装翻译服务依赖
pip install aiohttp
```

### 2. 配置环境变量

```bash
# AWS Transcribe (必需)
export AWS_ACCESS_KEY_ID=your_aws_access_key
export AWS_SECRET_ACCESS_KEY=your_aws_secret_key
export AWS_REGION=us-east-1

# OpenAI 翻译 (可选，推荐)
export OPENAI_API_KEY=your_openai_api_key

# 百度翻译 (可选)
export BAIDU_TRANSLATE_APP_ID=your_baidu_app_id
export BAIDU_TRANSLATE_SECRET_KEY=your_baidu_secret_key
```

### 3. 启动界面

```bash
# 方式1: 使用启动脚本 (推荐)
python launch_streaming_translation.py

# 方式2: 直接运行
python gradio_streaming_translation.py

# 方式3: 先测试系统
python test_streaming_translation.py
```

### 4. 访问界面

打开浏览器访问: http://127.0.0.1:7861

## 📋 使用步骤

1. **上传音频文件**
   - 点击左侧的文件上传区域
   - 选择 WAV 格式的音频文件
   - 系统会自动检测并显示文件信息

2. **选择语言**
   - 从下拉菜单中选择音频的语言
   - 支持 11 种语言，默认为 Arabic (ar-SA)

3. **选择翻译服务**
   - OpenAI GPT: 质量最高，需要 API key
   - Google Translate: 免费服务，无需配置
   - 百度翻译: 需要 API 凭证
   - 简单翻译: 基础功能，用于测试

4. **开始处理**
   - 点击"🚀 开始处理"按钮
   - 系统会自动转换音频格式
   - 实时显示字幕和翻译结果

## 🔧 技术特性

### 音频处理
- **自动格式转换**: 输入音频自动转换为 16kHz 单声道 PCM 格式
- **实时流式处理**: 使用 Amazon Transcribe 流式 API
- **低延迟**: 优化的处理管道，最小化延迟

### 语言支持
- **语音识别**: 支持 11 种语言
  - Arabic (ar-SA, ar-AE)
  - English (en-US, en-GB)
  - Chinese (zh-CN)
  - Japanese (ja-JP)
  - Korean (ko-KR)
  - French (fr-FR)
  - German (de-DE)
  - Spanish (es-ES)
  - Russian (ru-RU)

### 翻译服务
- **OpenAI GPT**: 高质量翻译，支持上下文理解
- **Google Translate**: 免费服务，覆盖面广
- **百度翻译**: 对中文优化，支持多种语言对
- **简单翻译**: 基础词汇替换，用于测试

## 📁 文件结构

```
SubtitleGenius/
├── gradio_streaming_translation.py    # 主界面文件
├── translation_service.py             # 翻译服务模块
├── launch_streaming_translation.py    # 启动脚本
├── test_streaming_translation.py      # 测试脚本
└── README_STREAMING_TRANSLATION.md    # 使用说明
```

## 🔍 故障排除

### 常见问题

1. **Amazon Transcribe SDK 不可用**
   ```bash
   # 解决方案
   python install_streaming.py
   # 或手动安装
   pip install amazon-transcribe boto3
   ```

2. **AWS 凭证错误**
   ```bash
   # 检查凭证
   aws sts get-caller-identity
   
   # 设置凭证
   export AWS_ACCESS_KEY_ID=your_key
   export AWS_SECRET_ACCESS_KEY=your_secret
   export AWS_REGION=us-east-1
   ```

3. **FFmpeg 不可用**
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu
   sudo apt install ffmpeg
   
   # Windows
   # 下载并安装: https://ffmpeg.org/download.html
   ```

4. **翻译服务失败**
   - 检查 API key 是否正确设置
   - 确认网络连接正常
   - 尝试切换到其他翻译服务

### 调试命令

```bash
# 测试系统完整性
python test_streaming_translation.py

# 检查音频文件格式
ffprobe -v quiet -print_format json -show_format -show_streams your_audio.wav

# 转换音频格式
ffmpeg -i input.wav -ar 16000 -ac 1 -sample_fmt s16 output_16k_mono.wav -y

# 测试翻译服务
python -c "
import asyncio
from translation_service import translation_manager
async def test():
    result = await translation_manager.translate('Hello world', 'zh')
    print(result.translated_text)
asyncio.run(test())
"
```

## 🎯 使用场景

### 实时会议记录
- 上传会议录音
- 实时生成字幕
- 同步翻译为中文

### 教育培训
- 处理外语教学音频
- 生成双语字幕
- 辅助语言学习

### 内容创作
- 视频内容转录
- 多语言字幕生成
- 内容本地化

### 客服分析
- 电话录音转录
- 多语言客户沟通
- 服务质量分析

## 🔮 扩展功能

### 自定义翻译服务
```python
# 在 translation_service.py 中添加新的翻译器
class CustomTranslator(TranslationService):
    def __init__(self):
        super().__init__()
        self.name = "custom"
    
    async def translate(self, text, target_lang="zh", source_lang="auto"):
        # 实现自定义翻译逻辑
        pass
```

### 批量处理
```python
# 扩展支持批量文件处理
async def process_batch_files(file_list, language, translation_service):
    results = []
    for file_path in file_list:
        result = await process_audio_file(file_path, language, translation_service)
        results.append(result)
    return results
```

### 输出格式
```python
# 支持导出 SRT, VTT 等字幕格式
def export_subtitles(subtitles, format="srt"):
    if format == "srt":
        return generate_srt(subtitles)
    elif format == "vtt":
        return generate_vtt(subtitles)
```

## 📊 性能优化

### 音频预处理
- 使用 16kHz 采样率减少数据量
- 单声道处理提高效率
- 预处理缓存避免重复转换

### 流式处理
- 异步处理提高并发性能
- 实时数据流减少延迟
- 智能缓冲平衡速度和质量

### 翻译优化
- 批量翻译减少 API 调用
- 缓存常用翻译结果
- 智能服务选择和回退

## 📝 更新日志

### v1.0.0 (2025-07-01)
- ✅ 初始版本发布
- ✅ 支持 WAV 文件上传
- ✅ 实时流式字幕生成
- ✅ 多翻译服务集成
- ✅ 11 种语言支持
- ✅ 自动音频格式优化

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个功能！

## 📄 许可证

MIT License

---

*这个功能是 SubtitleGenius 项目的一部分，专注于提供高质量的实时字幕生成和翻译服务。*
