#!/usr/bin/env python3
"""
启动流式字幕翻译界面的便捷脚本
"""

import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """检查依赖项"""
    print("🔍 检查依赖项...")
    
    # 检查 Amazon Transcribe SDK
    try:
        import amazon_transcribe
        print("✅ Amazon Transcribe SDK 已安装")
    except ImportError:
        print("❌ Amazon Transcribe SDK 未安装")
        print("正在安装...")
        try:
            subprocess.run([sys.executable, "install_streaming.py"], check=True)
            print("✅ Amazon Transcribe SDK 安装完成")
        except subprocess.CalledProcessError:
            print("❌ 安装失败，请手动运行: python install_streaming.py")
            return False
    
    # 检查 Gradio
    try:
        import gradio
        print("✅ Gradio 已安装")
    except ImportError:
        print("❌ Gradio 未安装")
        print("正在安装...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "gradio"], check=True)
            print("✅ Gradio 安装完成")
        except subprocess.CalledProcessError:
            print("❌ Gradio 安装失败")
            return False
    
    # 检查 FFmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✅ FFmpeg 可用")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ FFmpeg 不可用，音频预处理可能失败")
        print("请安装 FFmpeg:")
        print("  macOS: brew install ffmpeg")
        print("  Ubuntu: sudo apt install ffmpeg")
        print("  Windows: https://ffmpeg.org/download.html")
    
    return True

def main():
    """主函数"""
    print("=" * 60)
    print("🎬 SubtitleGenius - 流式字幕翻译界面启动器")
    print("=" * 60)
    
    # 检查依赖
    if not check_dependencies():
        print("❌ 依赖检查失败，请解决上述问题后重试")
        return
    
    print("\n🚀 启动流式字幕翻译界面...")
    
    # 启动主界面
    try:
        from gradio_streaming_translation import main as run_interface
        run_interface()
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("请确保所有依赖都已正确安装")
    except KeyboardInterrupt:
        print("\n👋 用户中断，程序退出")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
