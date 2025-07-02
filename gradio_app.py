#!/usr/bin/env python3
"""
SubtitleGenius Gradio Web界面
"""

import gradio as gr
import asyncio
import tempfile
import os
from pathlib import Path
import requests
from urllib.parse import urlparse
import subprocess
from typing import Optional, Tuple
import traceback

from subtitle_genius import SubtitleGenerator
from subtitle_genius.core.config import config


class SubtitleWebApp:
    """字幕生成Web应用"""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        print(f"临时目录: {self.temp_dir}")
    
    def download_video(self, video_url: str) -> Optional[Path]:
        """下载视频文件"""
        try:
            # 验证URL
            parsed = urlparse(video_url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError("无效的URL格式")
            
            # 生成临时文件名
            video_filename = f"video_{hash(video_url) % 10000}.mp4"
            video_path = self.temp_dir / video_filename
            
            print(f"正在下载视频: {video_url}")
            
            # 如果是YouTube URL，使用yt-dlp
            if 'youtube.com' in video_url or 'youtu.be' in video_url:
                return self.download_youtube_video(video_url, video_path)
            else:
                # 直接下载
                response = requests.get(video_url, stream=True, timeout=30)
                response.raise_for_status()
                
                with open(video_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"视频下载完成: {video_path}")
                return video_path
                
        except Exception as e:
            print(f"下载视频失败: {e}")
            return None
    
    def download_youtube_video(self, video_url: str, output_path: Path) -> Optional[Path]:
        """使用yt-dlp下载YouTube视频"""
        try:
            # 检查yt-dlp是否安装
            subprocess.run(["yt-dlp", "--version"], check=True, capture_output=True)
            
            cmd = [
                "yt-dlp",
                "--format", "best[height<=720][ext=mp4]",
                "--output", str(output_path),
                video_url
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            if output_path.exists():
                return output_path
            else:
                print(f"yt-dlp输出: {result.stdout}")
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"yt-dlp下载失败: {e}")
            return None
        except FileNotFoundError:
            print("yt-dlp未安装，尝试直接下载...")
            return None
    
    async def generate_subtitles(
        self, 
        video_input: str, 
        language: str = "zh-CN",
        model: str = "openai-whisper",
        subtitle_format: str = "srt"
    ) -> Tuple[str, str, str]:
        """生成字幕"""
        try:
            # 检查输入
            if not video_input.strip():
                return "❌ 请输入视频URL或上传视频文件", "", ""
            
            video_path = None
            
            # 判断输入类型
            if video_input.startswith(('http://', 'https://')):
                # URL输入
                video_path = self.download_video(video_input)
                if not video_path:
                    return "❌ 视频下载失败，请检查URL是否有效", "", ""
            else:
                # 文件路径输入
                video_path = Path(video_input)
                if not video_path.exists():
                    return "❌ 视频文件不存在", "", ""
            
            # 检查API密钥
            if not config.openai_api_key and model.startswith('openai'):
                return "❌ 请在.env文件中配置OPENAI_API_KEY", "", ""
            
            if not config.anthropic_api_key and model == 'claude':
                return "❌ 请在.env文件中配置ANTHROPIC_API_KEY", "", ""
            
            # 对于 Amazon Transcribe，检查 AWS 凭证
            if model == 'amazon-transcribe':
                try:
                    import boto3
                    # 尝试创建客户端来验证凭证
                    boto3.client('transcribe', region_name=config.aws_region)
                except Exception as e:
                    return f"❌ AWS 凭证配置错误: {str(e)}\n请配置 AWS_ACCESS_KEY_ID 和 AWS_SECRET_ACCESS_KEY", "", ""
            
            # 初始化字幕生成器
            generator = SubtitleGenerator(
                model=model,
                language=language,
                output_format=subtitle_format
            )
            
            # 生成字幕
            print(f"正在处理视频: {video_path}")
            subtitles = await generator.process_video(video_path)
            
            if not subtitles:
                return "❌ 未能生成字幕，请检查视频文件", "", ""
            
            # 格式化字幕内容
            subtitle_content = generator.subtitle_formatter.format(subtitles, subtitle_format)
            
            # 生成预览
            preview_lines = []
            for i, subtitle in enumerate(subtitles[:10]):  # 只显示前10条
                preview_lines.append(
                    f"{i+1}. [{subtitle.start:.1f}s - {subtitle.end:.1f}s] {subtitle.text}"
                )
            
            preview_text = "\n".join(preview_lines)
            if len(subtitles) > 10:
                preview_text += f"\n... 还有 {len(subtitles) - 10} 条字幕"
            
            success_msg = f"✅ 成功生成 {len(subtitles)} 条字幕！"
            
            return success_msg, subtitle_content, preview_text
            
        except Exception as e:
            error_msg = f"❌ 处理失败: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return error_msg, "", ""
    
    def process_video_sync(self, video_input, language, model, subtitle_format):
        """同步包装器"""
        return asyncio.run(
            self.generate_subtitles(video_input, language, model, subtitle_format)
        )


def create_gradio_interface():
    """创建Gradio界面"""
    
    app = SubtitleWebApp()
    
    # 自定义CSS
    css = """
    .container {
        max-width: 1200px;
        margin: 0 auto;
    }
    .subtitle-preview {
        max-height: 400px;
        overflow-y: auto;
        font-family: monospace;
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 5px;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    """
    
    with gr.Blocks(css=css, title="SubtitleGenius - AI字幕生成器") as interface:
        
        gr.Markdown("""
        # 🎬 SubtitleGenius - AI字幕生成器
        
        基于GenAI的实时字幕生成工具，支持多种语言和AI模型
        """)
        
        with gr.Row():
            # 左侧输入区域
            with gr.Column(scale=1):
                gr.Markdown("## 📥 输入设置")
                
                video_input = gr.Textbox(
                    label="视频URL或文件路径",
                    placeholder="输入MP4视频URL (支持YouTube) 或本地文件路径",
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
                        value="openai-whisper",
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
                
                # 状态显示
                status = gr.Textbox(
                    label="状态",
                    interactive=False,
                    lines=3
                )
            
            # 右侧输出区域
            with gr.Column(scale=1):
                gr.Markdown("## 📝 字幕输出")
                
                # 字幕预览
                preview = gr.Textbox(
                    label="字幕预览 (前10条)",
                    lines=10,
                    interactive=False,
                    elem_classes=["subtitle-preview"]
                )
                
                # 完整字幕内容
                subtitle_output = gr.Textbox(
                    label="完整字幕内容",
                    lines=15,
                    interactive=False,
                    show_copy_button=True
                )
                
                # 下载按钮
                download_btn = gr.DownloadButton(
                    "💾 下载字幕文件",
                    visible=False
                )
        
        # 使用说明
        with gr.Accordion("📖 使用说明", open=False):
            gr.Markdown("""
            ### 支持的输入格式:
            - **YouTube URL**: https://www.youtube.com/watch?v=...
            - **直接视频URL**: https://example.com/video.mp4
            - **本地文件路径**: /path/to/video.mp4
            
            ### AI模型说明:
            - **OpenAI Whisper (本地)**: 免费，本地运行，支持多语言
            - **OpenAI API**: 需要API密钥，质量更高
            - **Claude**: 需要API密钥，可用于字幕优化
            - **Amazon Transcribe**: 需要AWS凭证，支持多语言，云端处理
            
            ### 配置API密钥:
            在项目根目录的 `.env` 文件中添加:
            ```
            OPENAI_API_KEY=your_key_here
            ANTHROPIC_API_KEY=your_key_here
            
            # AWS 配置 (用于 Amazon Transcribe)
            AWS_ACCESS_KEY_ID=your_access_key
            AWS_SECRET_ACCESS_KEY=your_secret_key
            AWS_REGION=us-east-1
            AWS_S3_BUCKET=your-bucket-name
            ```
            
            ### 支持的语言:
            中文、英语、阿拉伯语、日语、韩语、法语、德语、西班牙语、俄语等
            """)
        
        # 事件处理
        def process_and_update(video_input, language, model, subtitle_format):
            """处理视频并更新界面"""
            status_msg, subtitle_content, preview_text = app.process_video_sync(
                video_input, language, model, subtitle_format
            )
            
            # 创建下载文件
            download_file = None
            if subtitle_content:
                filename = f"subtitles.{subtitle_format}"
                filepath = app.temp_dir / filename
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(subtitle_content)
                download_file = str(filepath)
            
            return (
                status_msg,
                preview_text,
                subtitle_content,
                gr.update(visible=bool(subtitle_content), value=download_file)
            )
        
        generate_btn.click(
            fn=process_and_update,
            inputs=[video_input, language, model, subtitle_format],
            outputs=[status, preview, subtitle_output, download_btn]
        )
        
        # 示例
        gr.Examples(
            examples=[
                ["https://www.youtube.com/watch?v=dQw4w9WgXcQ", "en", "openai-whisper", "srt"],
                ["https://example.com/arabic_news.mp4", "ar", "amazon-transcribe", "srt"],
                ["/path/to/local/video.mp4", "zh-CN", "amazon-transcribe", "vtt"],
            ],
            inputs=[video_input, language, model, subtitle_format],
        )
    
    return interface


def main():
    """主函数"""
    print("🚀 启动SubtitleGenius Web界面...")
    
    # 检查依赖
    try:
        import gradio
        print(f"✅ Gradio版本: {gradio.__version__}")
    except ImportError:
        print("❌ Gradio未安装，正在安装...")
        subprocess.run(["pip", "install", "gradio>=4.40.0"], check=True)
    
    try:
        # 创建界面
        interface = create_gradio_interface()
        print("✅ Gradio 界面创建成功")
        
        # 启动服务
        print("🌐 启动 Web 服务器...")
        print("📱 访问地址: http://127.0.0.1:7860")
        
        interface.launch(
            server_name="127.0.0.1",
            server_port=7862,
            share=False,
            debug=False,
            show_error=True,
            quiet=False,
            prevent_thread_lock=False
        )
        
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("\n🔧 故障排除建议:")
        print("1. 检查端口 7860 是否被占用")
        print("2. 尝试使用不同的端口: --server-port 8080")
        print("3. 检查防火墙设置")
        print("4. 尝试启用共享模式: share=True")
        
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
