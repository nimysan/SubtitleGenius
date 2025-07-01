# Amazon Transcribe 集成指南

## 概述

Amazon Transcribe 是 AWS 提供的自动语音识别 (ASR) 服务，支持多种语言和方言。SubtitleGenius 现已集成 Amazon Transcribe，为用户提供高质量的云端语音转文本服务。

## 功能特性

- ✅ 支持多种语言和方言
- ✅ 高精度语音识别
- ✅ 自动标点符号
- ✅ 时间戳精确到毫秒
- ✅ 云端处理，无需本地GPU
- ✅ 支持长音频文件

## 配置要求

### 1. AWS 账户和权限

确保你的 AWS 账户具有以下权限：
- `transcribe:StartTranscriptionJob`
- `transcribe:GetTranscriptionJob`
- `transcribe:ListTranscriptionJobs`
- `s3:PutObject`
- `s3:GetObject`
- `s3:DeleteObject`
- `s3:CreateBucket`

### 2. 环境变量配置

在 `.env` 文件中添加以下配置：

```bash
# AWS 凭证
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_REGION=us-east-1

# S3 存储桶 (用于临时存储音频文件)
AWS_S3_BUCKET=your-subtitle-genius-bucket
```

## 使用方法

### 在 Gradio 界面中使用

1. 启动 Gradio 应用：
   ```bash
   uv run python gradio_app.py
   ```

2. 在 AI 模型下拉菜单中选择 "Amazon Transcribe"

3. 选择适当的语言（支持中文、英语、阿拉伯语等）

4. 输入视频 URL 或上传视频文件

5. 点击"生成字幕"按钮

### 在 Python 代码中使用

```python
import asyncio
from subtitle_genius import SubtitleGenerator

async def main():
    # 初始化生成器
    generator = SubtitleGenerator(
        model="amazon-transcribe",
        language="zh-CN"
    )
    
    # 处理视频文件
    subtitles = await generator.process_video("video.mp4")
    
    # 打印结果
    for subtitle in subtitles:
        print(f"[{subtitle.start:.1f}s - {subtitle.end:.1f}s] {subtitle.text}")

# 运行
asyncio.run(main())
```

## 支持的语言

| 语言 | 代码 | Transcribe 代码 |
|------|------|-----------------|
| 中文（简体） | zh-CN | zh-CN |
| 英语（美国） | en | en-US |
| 阿拉伯语 | ar | ar-SA |
| 日语 | ja | ja-JP |
| 韩语 | ko | ko-KR |
| 法语 | fr | fr-FR |
| 德语 | de | de-DE |
| 西班牙语 | es | es-ES |
| 俄语 | ru | ru-RU |

## 故障排除

### 常见错误

1. **凭证错误**: 检查 AWS_ACCESS_KEY_ID 和 AWS_SECRET_ACCESS_KEY
2. **权限不足**: 确保 IAM 用户有 Transcribe 和 S3 权限
3. **存储桶问题**: 程序会自动创建存储桶
4. **转录失败**: 检查音频文件格式和质量

### 测试连接

```bash
uv run python test_transcribe.py
```

## 成本考虑

- 标准转录：每分钟 $0.024
- 批量转录：每分钟 $0.0144（节省40%）
- 建议及时清理 S3 临时文件
