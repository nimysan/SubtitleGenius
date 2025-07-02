#!/usr/bin/env python3
"""
独立的 Gradio 流式字幕翻译页面
左边: 上传 WAV 文件
中间: 流式字幕输出
右边: 中文翻译结果
"""

import gradio as gr
import asyncio
import sys
import os
from pathlib import Path
import tempfile
import subprocess
from typing import AsyncGenerator, List, Tuple
import json
import time

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "amazon-transcribe-streaming-sdk"))

try:
    import aiofile
    from amazon_transcribe.client import TranscribeStreamingClient
    from amazon_transcribe.handlers import TranscriptResultStreamHandler
    from amazon_transcribe.model import TranscriptEvent
    from amazon_transcribe.utils import apply_realtime_delay
    TRANSCRIBE_AVAILABLE = True
except ImportError:
    TRANSCRIBE_AVAILABLE = False
    print("⚠️ Amazon Transcribe SDK 不可用，请运行: python install_streaming.py")

# 配置
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHANNEL_NUMS = 1
CHUNK_SIZE = 1024 * 8
REGION = "us-east-1"

class StreamingSubtitleHandler(TranscriptResultStreamHandler):
    """流式字幕处理器"""
    
    def __init__(self, output_stream):
        super().__init__(output_stream)
        self.subtitles = []
        self.current_text = ""
        self.subtitle_count = 0
    
    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        """处理转录事件"""
        results = transcript_event.transcript.results
        
        for result in results:
            for alt in result.alternatives:
                if alt.transcript.strip():
                    self.subtitle_count += 1
                    
                    subtitle_data = {
                        'id': self.subtitle_count,
                        'text': alt.transcript.strip(),
                        'start_time': getattr(result, 'start_time', 0),
                        'end_time': getattr(result, 'end_time', 0),
                        'is_partial': result.is_partial,
                        'timestamp': time.time()
                    }
                    
                    self.subtitles.append(subtitle_data)
                    
                    # 更新当前文本用于实时显示
                    if result.is_partial:
                        self.current_text = f"[处理中...] {alt.transcript}"
                    else:
                        self.current_text = f"[{self.subtitle_count}] {alt.transcript}"

def preprocess_audio(input_path: str) -> str:
    """预处理音频文件为 Transcribe 兼容格式"""
    output_path = input_path.replace('.wav', '_16k_mono.wav')
    
    try:
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-ar', str(SAMPLE_RATE),
            '-ac', str(CHANNEL_NUMS),
            '-sample_fmt', 's16',
            output_path,
            '-y'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ 音频预处理完成: {output_path}")
        return output_path
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 音频预处理失败: {e}")
        print(f"错误输出: {e.stderr}")
        raise Exception(f"音频预处理失败: {e.stderr}")

async def transcribe_audio_stream(audio_path: str, language: str = "ar-SA") -> AsyncGenerator[str, None]:
    """流式转录音频文件"""
    if not TRANSCRIBE_AVAILABLE:
        yield "❌ Amazon Transcribe SDK 不可用"
        return
    
    try:
        # 预处理音频
        processed_audio = preprocess_audio(audio_path)
        
        # 创建 Transcribe 客户端
        client = TranscribeStreamingClient(region=REGION)
        
        # 启动流式转录
        stream = await client.start_stream_transcription(
            language_code=language,
            media_sample_rate_hz=SAMPLE_RATE,
            media_encoding="pcm",
        )
        
        # 创建处理器
        handler = StreamingSubtitleHandler(stream.output_stream)
        
        async def write_chunks():
            """发送音频数据"""
            async with aiofile.AIOFile(processed_audio, "rb") as afp:
                reader = aiofile.Reader(afp, chunk_size=CHUNK_SIZE)
                await apply_realtime_delay(
                    stream, reader, BYTES_PER_SAMPLE, SAMPLE_RATE, CHANNEL_NUMS
                )
            await stream.input_stream.end_stream()
        
        # 启动处理任务
        transcription_task = asyncio.create_task(handler.handle_events())
        audio_task = asyncio.create_task(write_chunks())
        
        # 实时输出字幕
        start_time = time.time()
        last_count = 0
        
        while not transcription_task.done():
            await asyncio.sleep(0.5)  # 每0.5秒更新一次
            
            if handler.subtitle_count > last_count:
                # 输出新的字幕
                for subtitle in handler.subtitles[last_count:]:
                    if not subtitle['is_partial']:
                        elapsed = time.time() - start_time
                        yield f"[{elapsed:.1f}s] {subtitle['text']}"
                
                last_count = handler.subtitle_count
            
            # 输出当前处理状态
            if handler.current_text:
                yield handler.current_text
        
        # 等待任务完成
        await asyncio.gather(transcription_task, audio_task)
        
        yield f"✅ 转录完成！共生成 {handler.subtitle_count} 条字幕"
        
    except Exception as e:
        yield f"❌ 转录失败: {str(e)}"
        import traceback
        traceback.print_exc()

async def translate_to_chinese(text: str, service: str = "auto") -> str:
    """翻译文本到中文"""
    # 导入翻译服务
    try:
        from translation_service import translation_manager
    except ImportError:
        # 回退到简单翻译
        return f"[简单译] {text}"
    
    if not text or text.startswith('[') or text.startswith('❌') or text.startswith('✅'):
        return text
    
    # 清理文本 - 移除时间戳等
    clean_text = text
    if ']' in text and text.startswith('['):
        # 移除 [时间戳] 前缀
        parts = text.split(']', 1)
        if len(parts) > 1:
            clean_text = parts[1].strip()
    
    try:
        # 使用翻译服务
        result = await translation_manager.translate(
            clean_text, 
            target_lang="zh",
            service=service if service != "auto" else None
        )
        
        # 保留原始时间戳格式
        if clean_text != text:
            timestamp_part = text.replace(clean_text, '').strip()
            return f"{timestamp_part} [中文] {result.translated_text}"
        else:
            return f"[中文] {result.translated_text}"
            
    except Exception as e:
        print(f"翻译失败: {e}")
        return f"[译] {clean_text}"

def process_audio_file_wrapper(audio_file, language, translation_service, progress=gr.Progress()):
    """处理音频文件的包装函数 - 同步版本"""
    import asyncio
    
    if not audio_file:
        return "请上传音频文件", ""
    
    if not TRANSCRIBE_AVAILABLE:
        return "❌ Amazon Transcribe SDK 不可用，请运行: python install_streaming.py", ""
    
    progress(0, desc="开始处理...")
    
    try:
        # 保存上传的文件
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "input.wav")
        
        # 复制上传的文件
        import shutil
        shutil.copy2(audio_file, audio_path)
        
        progress(0.1, desc="音频文件已保存")
        
        # 流式处理
        subtitle_output = ""
        translation_output = ""
        
        progress(0.2, desc="开始转录...")
        
        # 运行异步处理
        async def async_process():
            nonlocal subtitle_output, translation_output
            async for subtitle_text in transcribe_audio_stream(audio_path, language):
                subtitle_output += subtitle_text + "\n"
                
                # 翻译最新的字幕
                if not subtitle_text.startswith('[处理中') and not subtitle_text.startswith('❌') and not subtitle_text.startswith('✅'):
                    translated = await translate_to_chinese(subtitle_text, translation_service)
                    translation_output += translated + "\n"
                
                progress(0.8, desc="转录中...")
        
        # 运行异步处理
        asyncio.run(async_process())
        
        progress(1.0, desc="完成!")
        
        # 清理临时文件
        try:
            os.remove(audio_path)
            processed_path = audio_path.replace('.wav', '_16k_mono.wav')
            if os.path.exists(processed_path):
                os.remove(processed_path)
            os.rmdir(temp_dir)
        except:
            pass
        
        return subtitle_output, translation_output
        
    except Exception as e:
        error_msg = f"❌ 处理失败: {str(e)}"
        return error_msg, error_msg

async def process_audio_file(audio_file, language, translation_service, progress=gr.Progress()):
    """处理音频文件的主函数"""
    if not audio_file:
        yield "请上传音频文件", ""
        return
    
    if not TRANSCRIBE_AVAILABLE:
        yield "❌ Amazon Transcribe SDK 不可用，请运行: python install_streaming.py", ""
        return
    
    progress(0, desc="开始处理...")
    
    try:
        # 保存上传的文件
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "input.wav")
        
        # 复制上传的文件
        import shutil
        shutil.copy2(audio_file, audio_path)
        
        progress(0.1, desc="音频文件已保存")
        
        # 流式处理
        subtitle_output = ""
        translation_output = ""
        
        progress(0.2, desc="开始转录...")
        
        async for subtitle_text in transcribe_audio_stream(audio_path, language):
            subtitle_output += subtitle_text + "\n"
            
            # 翻译最新的字幕
            if not subtitle_text.startswith('[处理中') and not subtitle_text.startswith('❌') and not subtitle_text.startswith('✅'):
                translated = await translate_to_chinese(subtitle_text, translation_service)
                translation_output += translated + "\n"
            
            # 实时更新界面
            yield subtitle_output, translation_output
            
            progress(0.8, desc="转录中...")
        
        progress(1.0, desc="完成!")
        
        # 清理临时文件
        try:
            os.remove(audio_path)
            processed_path = audio_path.replace('.wav', '_16k_mono.wav')
            if os.path.exists(processed_path):
                os.remove(processed_path)
            os.rmdir(temp_dir)
        except:
            pass
        
        # Final yield with complete results
        yield subtitle_output, translation_output
        
    except Exception as e:
        error_msg = f"❌ 处理失败: {str(e)}"
        yield error_msg, error_msg

def create_interface():
    """创建 Gradio 界面"""
    
    # 获取可用的翻译服务
    try:
        from translation_service import translation_manager
        available_services = translation_manager.get_available_services()
        translation_choices = []
        
        service_names = {
            "openai": "OpenAI GPT (推荐)",
            "google": "Google Translate (免费)",
            "baidu": "百度翻译",
            "mock": "简单翻译 (测试)"
        }
        
        for service in available_services:
            name = service_names.get(service, service)
            translation_choices.append((name, service))
        
        # 如果没有可用服务，添加默认选项
        if not translation_choices:
            translation_choices = [("简单翻译", "mock")]
            
    except ImportError:
        translation_choices = [("简单翻译", "mock")]
    
    # 自定义 CSS
    custom_css = """
    # .gradio-container {
    #     max-width: 1400px !important;
    # }
    # .subtitle-output {
    #     height: 400px !important;
    #     overflow-y: auto !important;
    #     font-family: monospace !important;
    #     font-size: 14px !important;
    #     line-height: 1.5 !important;
    # }
    # .translation-output {
    #     height: 400px !important;
    #     overflow-y: auto !important;
    #     font-family: 'Microsoft YaHei', sans-serif !important;
    #     font-size: 14px !important;
    #     line-height: 1.5 !important;
    # }
    """
    
    with gr.Blocks(css=custom_css, title="SubtitleGenius - 流式字幕翻译") as interface:
        gr.Markdown("# 🎬 SubtitleGenius - 流式字幕翻译")
        gr.Markdown("上传 WAV 音频文件，实时生成字幕并翻译为中文")
        
        with gr.Row():
            # 左侧：文件上传和控制
            with gr.Column(scale=1):
                gr.Markdown("### 📁 音频文件上传")
                
                audio_input = gr.File(
                    label="选择 WAV 音频文件",
                    file_types=[".wav"],
                    type="filepath"
                )
                
                language_select = gr.Dropdown(
                    choices=[
                        ("Arabic (Saudi Arabia)", "ar-SA"),
                        ("Arabic (UAE)", "ar-AE"),
                        ("English (US)", "en-US"),
                        ("English (UK)", "en-GB"),
                        ("Chinese (Simplified)", "zh-CN"),
                        ("Japanese", "ja-JP"),
                        ("Korean", "ko-KR"),
                        ("French", "fr-FR"),
                        ("German", "de-DE"),
                        ("Spanish", "es-ES"),
                        ("Russian", "ru-RU")
                    ],
                    value="ar-SA",
                    label="选择语言"
                )
                
                translation_service_select = gr.Dropdown(
                    choices=translation_choices,
                    value=translation_choices[0][1] if translation_choices else "mock",
                    label="翻译服务"
                )
                
                process_btn = gr.Button("🚀 开始处理", variant="primary", size="lg")
                
                gr.Markdown("### ℹ️ 使用说明")
                gr.Markdown("""
                1. 上传 WAV 格式音频文件
                2. 选择音频语言
                3. 选择翻译服务
                4. 点击"开始处理"
                5. 实时查看字幕和翻译结果
                
                **注意**: 音频会自动转换为 16kHz 单声道格式以优化识别效果
                """)
                
                # 翻译服务说明
                gr.Markdown("### 🔧 翻译服务说明")
                gr.Markdown("""
                - **OpenAI GPT**: 需要 OPENAI_API_KEY 环境变量
                - **Google Translate**: 免费服务，无需配置
                - **百度翻译**: 需要 BAIDU_TRANSLATE_APP_ID 和 BAIDU_TRANSLATE_SECRET_KEY
                - **简单翻译**: 基础词汇替换，用于测试
                """)
            
            # 中间：字幕输出
            with gr.Column(scale=2):
                gr.Markdown("### 🎤 实时字幕输出")
                subtitle_output = gr.Textbox(
                    label="字幕内容",
                    placeholder="字幕将在这里实时显示...",
                    lines=20,
                    max_lines=20,
                    elem_classes=["subtitle-output"],
                    interactive=False
                )
            
            # 右侧：中文翻译
            with gr.Column(scale=2):
                gr.Markdown("### 🇨🇳 中文翻译")
                translation_output = gr.Textbox(
                    label="中文翻译",
                    placeholder="中文翻译将在这里显示...",
                    lines=20,
                    max_lines=20,
                    elem_classes=["translation-output"],
                    interactive=False
                )
        
        # 状态显示
        with gr.Row():
            status_text = gr.Markdown("### 📊 状态: 等待上传文件...")
        
        # 事件处理
        def update_status(audio_file):
            if audio_file:
                return "### 📊 状态: 文件已上传，点击开始处理"
            return "### 📊 状态: 等待上传文件..."
        
        audio_input.change(
            fn=update_status,
            inputs=[audio_input],
            outputs=[status_text]
        )
        
        # 处理按钮事件 - 使用同步包装函数
        process_btn.click(
            fn=process_audio_file_wrapper,
            inputs=[audio_input, language_select, translation_service_select],
            outputs=[subtitle_output, translation_output],
            show_progress=True
        )
        
        # 底部信息
        gr.Markdown("""
        ---
        ### 🔧 技术特性
        - **实时处理**: 使用 Amazon Transcribe 流式 API
        - **多语言支持**: 支持 11 种语言识别
        - **自动优化**: 音频格式自动转换为最佳参数
        - **多翻译服务**: 支持 OpenAI、Google、百度等翻译服务
        
        ### 📋 支持格式
        - **输入**: WAV 音频文件
        - **输出**: 实时字幕文本 + 中文翻译
        - **优化**: 自动转换为 16kHz 单声道 PCM 格式
        
        ### 🔑 环境变量配置
        ```bash
        # OpenAI 翻译
        export OPENAI_API_KEY=your_openai_key
        
        # 百度翻译
        export BAIDU_TRANSLATE_APP_ID=your_app_id
        export BAIDU_TRANSLATE_SECRET_KEY=your_secret_key
        
        # AWS Transcribe
        export AWS_ACCESS_KEY_ID=your_aws_key
        export AWS_SECRET_ACCESS_KEY=your_aws_secret
        export AWS_REGION=us-east-1
        ```
        """)
    
    return interface

def main():
    """主函数"""
    print("🚀 启动 SubtitleGenius 流式字幕翻译界面")
    
    # 检查依赖
    if not TRANSCRIBE_AVAILABLE:
        print("⚠️ Amazon Transcribe SDK 不可用")
        print("请运行: python install_streaming.py")
        print("或手动安装: pip install amazon-transcribe boto3")
    
    # 检查 FFmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✅ FFmpeg 可用")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ FFmpeg 不可用，音频预处理可能失败")
        print("请安装 FFmpeg: https://ffmpeg.org/download.html")
    
    # 创建并启动界面
    interface = create_interface()
    
    print("🌐 启动 Web 界面...")
    interface.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()
