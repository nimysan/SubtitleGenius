#!/usr/bin/env python3
"""
Amazon Transcribe 流式处理 Arabic 语音识别示例
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from subtitle_genius.models.transcribe_model import TranscribeModel
from subtitle_genius.stream.processor import StreamProcessor
from subtitle_genius.core.config import config


async def test_realtime_microphone():
    """测试实时麦克风输入"""
    print("🎤 测试实时麦克风输入 (Arabic)")
    print("=" * 50)
    
    try:
        # 初始化模型和流处理器
        model = TranscribeModel(region_name=config.aws_region, use_streaming=True)
        stream_processor = StreamProcessor()
        
        if not model.use_streaming:
            print("⚠️  流式处理不可用，请安装 amazon-transcribe 包")
            return
        
        if not model.is_available():
            print("❌ Amazon Transcribe 不可用")
            return
        
        print("🚀 开始实时语音识别...")
        print("💬 请对着麦克风说话 (Arabic)，按 Ctrl+C 停止")
        
        # 创建麦克风音频流
        audio_stream = stream_processor.start_microphone_stream()
        
        # 实时转录
        subtitle_count = 0
        async for subtitle in model.transcribe_stream(audio_stream, language="ar"):
            subtitle_count += 1
            print(f"📝 [{subtitle.start:.1f}s] {subtitle.text}")
        
    except KeyboardInterrupt:
        print("\n⏹️  用户停止录音")
    except Exception as e:
        print(f"❌ 实时转录失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'stream_processor' in locals():
            stream_processor.stop_stream()


async def test_file_streaming():
    """测试文件流式处理"""
    print("📁 测试文件流式处理 (Arabic)")
    print("=" * 50)
    
    test_file = "input.mp4"
    if not Path(test_file).exists():
        print(f"⚠️  未找到测试文件: {test_file}")
        return
    
    try:
        # 初始化模型和流处理器
        model = TranscribeModel(region_name=config.aws_region, use_streaming=True)
        stream_processor = StreamProcessor()
        
        if not model.use_streaming:
            print("⚠️  流式处理不可用，请安装 amazon-transcribe 包")
            return
        
        if not model.is_available():
            print("❌ Amazon Transcribe 不可用")
            return
        
        print(f"🚀 开始流式处理文件: {test_file}")
        
        # 创建文件音频流
        audio_stream = stream_processor.process_file_stream(test_file)
        
        # 流式转录
        subtitle_count = 0
        async for subtitle in model.transcribe_stream(audio_stream, language="ar"):
            subtitle_count += 1
            print(f"📝 字幕 {subtitle_count}: [{subtitle.start:.1f}s - {subtitle.end:.1f}s] {subtitle.text}")
        
        print(f"✅ 流式处理完成！共生成 {subtitle_count} 条字幕")
        
    except Exception as e:
        print(f"❌ 文件流式处理失败: {e}")
        import traceback
        traceback.print_exc()


async def test_rtmp_streaming():
    """测试 RTMP 流处理"""
    print("📺 测试 RTMP 流处理 (Arabic)")
    print("=" * 50)
    
    # 示例 RTMP URL (需要替换为实际的流地址)
    rtmp_url = "rtmp://example.com/live/stream"
    
    print(f"⚠️  RTMP 流测试需要实际的流地址")
    print(f"   示例 URL: {rtmp_url}")
    print("   请修改 rtmp_url 变量为实际的流地址")
    
    # 如果有实际的 RTMP URL，可以取消注释以下代码
    """
    try:
        # 初始化模型和流处理器
        model = TranscribeModel(region_name=config.aws_region, use_streaming=True)
        stream_processor = StreamProcessor()
        
        if not model.use_streaming:
            print("⚠️  流式处理不可用，请安装 amazon-transcribe 包")
            return
        
        if not model.is_available():
            print("❌ Amazon Transcribe 不可用")
            return
        
        print(f"🚀 开始处理 RTMP 流: {rtmp_url}")
        
        # 创建 RTMP 音频流
        audio_stream = stream_processor.process_rtmp_stream(rtmp_url)
        
        # 流式转录
        subtitle_count = 0
        async for subtitle in model.transcribe_stream(audio_stream, language="ar"):
            subtitle_count += 1
            print(f"📝 实时字幕 {subtitle_count}: {subtitle.text}")
        
    except Exception as e:
        print(f"❌ RTMP 流处理失败: {e}")
        import traceback
        traceback.print_exc()
    """


def show_menu():
    """显示菜单"""
    print("\n🎬 SubtitleGenius - 流式处理测试菜单")
    print("=" * 50)
    print("1. 实时麦克风输入 (Arabic)")
    print("2. 文件流式处理 (Arabic)")
    print("3. RTMP 流处理 (Arabic)")
    print("4. 退出")
    print("=" * 50)


async def main():
    """主函数"""
    print("🌊 Amazon Transcribe 流式处理测试 (Arabic)")
    print("=" * 60)
    
    # 检查依赖
    try:
        from amazon_transcribe.client import TranscribeStreamingClient
        print("✅ amazon-transcribe 包已安装")
    except ImportError:
        print("❌ amazon-transcribe 包未安装")
        print("   安装命令: pip install amazon-transcribe")
        return
    
    while True:
        show_menu()
        
        try:
            choice = input("\n请选择测试选项 (1-4): ").strip()
            
            if choice == "1":
                await test_realtime_microphone()
            elif choice == "2":
                await test_file_streaming()
            elif choice == "3":
                await test_rtmp_streaming()
            elif choice == "4":
                print("👋 再见！")
                break
            else:
                print("❌ 无效选择，请输入 1-4")
                
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")
        
        input("\n按 Enter 键继续...")


if __name__ == "__main__":
    asyncio.run(main())
