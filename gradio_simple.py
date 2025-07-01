#!/usr/bin/env python3
"""
SubtitleGenius 简化版 Gradio 界面
"""

import gradio as gr
import asyncio
import tempfile
import os
from pathlib import Path
import traceback

# 简化导入，避免复杂的依赖问题
try:
    from subtitle_genius import SubtitleGenerator
    from subtitle_genius.core.config import config
    SUBTITLE_GENIUS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  SubtitleGenius 导入失败: {e}")
    SUBTITLE_GENIUS_AVAILABLE = False


def process_video_simple(video_url, language, model, subtitle_format):
    """简化的视频处理函数"""
    try:
        if not SUBTITLE_GENIUS_AVAILABLE:
            return "❌ SubtitleGenius 模块未正确安装", "", ""
        
        if not video_url.strip():
            return "❌ 请输入视频URL", "", ""
        
        # 检查模型可用性
        if model == "amazon-transcribe":
            try:
                import boto3
                # 简单的凭证检查
                boto3.client('transcribe')
            except Exception as e:
                return f"❌ Amazon Transcribe 不可用: {str(e)}", "", ""
        
        # 这里应该是实际的处理逻辑
        # 为了演示，我们返回一个模拟结果
        mock_subtitles = f"""1
00:00:00,000 --> 00:00:05,000
这是使用 {model} 生成的示例字幕

2
00:00:05,000 --> 00:00:10,000
语言: {language}，格式: {subtitle_format}

3
00:00:10,000 --> 00:00:15,000
视频URL: {video_url[:50]}...
"""
        
        preview = f"✅ 模拟生成了 3 条字幕\n使用模型: {model}\n语言: {language}"
        
        return "✅ 处理完成（演示模式）", mock_subtitles, preview
        
    except Exception as e:
        error_msg = f"❌ 处理失败: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return error_msg, "", ""


def create_interface():
    """创建简化的界面"""
    
    # 简单的 CSS
    css = """
    .container { max-width: 1200px; margin: 0 auto; }
    .subtitle-preview { 
        max-height: 300px; 
        overflow-y: auto; 
        font-family: monospace; 
        background-color: #f8f9fa; 
        padding: 10px; 
        border-radius: 5px; 
    }
    """
    
    with gr.Blocks(css=css, title="SubtitleGenius") as interface:
        
        gr.Markdown("""
        # 🎬 SubtitleGenius - AI字幕生成器
        
        基于GenAI的字幕生成工具，支持多种AI模型包括 Amazon Transcribe
        """)
        
        with gr.Row():
            # 左侧输入
            with gr.Column(scale=1):
                gr.Markdown("## 📥 输入设置")
                
                video_input = gr.Textbox(
                    label="视频URL",
                    placeholder="输入视频URL（支持YouTube）",
                    lines=2
                )
                
                with gr.Row():
                    language = gr.Dropdown(
                        choices=[
                            ("中文", "zh-CN"),
                            ("英语", "en"),
                            ("阿拉伯语", "ar"),
                            ("日语", "ja"),
                            ("韩语", "ko"),
                            ("法语", "fr"),
                            ("德语", "de"),
                            ("西班牙语", "es"),
                            ("俄语", "ru")
                        ],
                        value="zh-CN",
                        label="语言"
                    )
                    
                    model = gr.Dropdown(
                        choices=[
                            ("OpenAI Whisper (本地)", "openai-whisper"),
                            ("OpenAI API", "openai-gpt"),
                            ("Claude", "claude"),
                            ("Amazon Transcribe", "amazon-transcribe")
                        ],
                        value="amazon-transcribe",
                        label="AI模型"
                    )
                
                subtitle_format = gr.Radio(
                    choices=["srt", "vtt"],
                    value="srt",
                    label="字幕格式"
                )
                
                generate_btn = gr.Button(
                    "🚀 生成字幕", 
                    variant="primary",
                    size="lg"
                )
                
                status = gr.Textbox(
                    label="状态",
                    interactive=False,
                    lines=2
                )
            
            # 右侧输出
            with gr.Column(scale=1):
                gr.Markdown("## 📝 字幕输出")
                
                preview = gr.Textbox(
                    label="处理结果",
                    lines=5,
                    interactive=False
                )
                
                subtitle_output = gr.Textbox(
                    label="字幕内容",
                    lines=15,
                    interactive=False,
                    show_copy_button=True
                )
        
        # 使用说明
        with gr.Accordion("📖 使用说明", open=False):
            gr.Markdown("""
            ### 支持的AI模型:
            - **Amazon Transcribe**: AWS云端语音识别，高精度，支持多语言
            - **OpenAI Whisper**: 本地运行，免费使用
            - **OpenAI API**: 需要API密钥，质量高
            - **Claude**: 需要API密钥，适合后处理
            
            ### 配置说明:
            在 `.env` 文件中配置相应的API密钥：
            ```
            # AWS 配置
            AWS_ACCESS_KEY_ID=your_access_key
            AWS_SECRET_ACCESS_KEY=your_secret_key
            AWS_REGION=us-east-1
            
            # OpenAI 配置
            OPENAI_API_KEY=your_openai_key
            
            # Anthropic 配置
            ANTHROPIC_API_KEY=your_anthropic_key
            ```
            
            ### 注意事项:
            - 当前为演示模式，实际功能需要正确配置API密钥
            - Amazon Transcribe 需要 AWS 凭证和相应权限
            - 支持 YouTube URL 和直接视频链接
            """)
        
        # 事件绑定
        generate_btn.click(
            fn=process_video_simple,
            inputs=[video_input, language, model, subtitle_format],
            outputs=[status, subtitle_output, preview]
        )
        
        # 示例
        gr.Examples(
            examples=[
                ["https://www.youtube.com/watch?v=dQw4w9WgXcQ", "en", "amazon-transcribe", "srt"],
                ["https://example.com/video.mp4", "zh-CN", "amazon-transcribe", "srt"],
                ["https://example.com/arabic.mp4", "ar", "amazon-transcribe", "vtt"],
            ],
            inputs=[video_input, language, model, subtitle_format],
        )
    
    return interface


def main():
    """主函数"""
    print("🚀 启动 SubtitleGenius 简化版界面...")
    
    try:
        # 检查 Gradio 版本
        print(f"✅ Gradio 版本: {gr.__version__}")
        
        # 创建界面
        interface = create_interface()
        print("✅ 界面创建成功")
        
        # 启动服务
        print("🌐 启动服务器...")
        print("📱 访问地址: http://127.0.0.1:7860")
        print("⏹️  按 Ctrl+C 停止服务")
        
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            debug=False,
            show_error=True,
            quiet=False
        )
        
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
