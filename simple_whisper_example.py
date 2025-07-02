#!/usr/bin/env python3
"""
简单的 Whisper 流式处理示例
快速开始使用 Whisper 进行实时语音识别
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from subtitle_genius.models.transcribe_model import TranscribeModel
from subtitle_genius.models.whisper_streaming_model import WhisperStreamConfig


async def simple_whisper_example():
    """简单的 Whisper 流式处理示例"""
    
    print("🎤 Whisper 流式语音识别示例")
    print("=" * 40)
    
    # 1. 创建 Whisper 流式模型
    print("📦 初始化 Whisper 模型...")
    
    # 配置参数
    config = WhisperStreamConfig(
        chunk_duration=3.0,    # 每3秒处理一次
        overlap_duration=0.5,  # 0.5秒重叠避免截断
        voice_threshold=0.01   # 语音检测阈值
    )
    
    # 创建模型 (使用 Whisper 后端)
    model = TranscribeModel(
        backend="whisper",
        whisper_model="base",  # 可选: tiny, base, small, medium, large
        whisper_config=config
    )
    
    # 检查模型是否可用
    if not model.is_available():
        print("❌ Whisper 不可用，请安装:")
        print("   pip install openai-whisper")
        return
    
    print("✅ Whisper 模型已准备就绪")
    
    # 2. 处理音频文件 (如果存在)
    audio_file = "test_audio.wav"  # 替换为你的音频文件
    
    if Path(audio_file).exists():
        print(f"\n🎵 处理音频文件: {audio_file}")
        
        # 导入流处理器
        from subtitle_genius.stream.processor import StreamProcessor
        processor = StreamProcessor()
        
        # 创建文件音频流
        audio_stream = processor.process_file_stream(audio_file)
        
        # 流式转录
        async for subtitle in model.transcribe_stream(audio_stream, language="ar"):
            print(f"📝 [{subtitle.start:.1f}s] {subtitle.text}")
    
    else:
        print(f"\n⚠️  音频文件不存在: {audio_file}")
        print("请提供音频文件或使用麦克风模式")
    
    print("\n✅ 示例完成")


async def microphone_example():
    """麦克风实时识别示例"""
    
    print("🎤 麦克风实时语音识别")
    print("=" * 30)
    
    # 创建快速响应的配置
    config = WhisperStreamConfig(
        chunk_duration=2.0,    # 更短的处理间隔
        overlap_duration=0.3,  # 较短重叠
        voice_threshold=0.02   # 稍高的检测阈值
    )
    
    model = TranscribeModel(
        backend="whisper",
        whisper_model="base",
        whisper_config=config
    )
    
    if not model.is_available():
        print("❌ Whisper 不可用")
        return
    
    try:
        from subtitle_genius.stream.processor import StreamProcessor
        processor = StreamProcessor()
        
        print("🔴 开始录音 (按 Ctrl+C 停止)")
        print("💬 请开始说话...")
        
        # 启动麦克风
        mic_stream = processor.start_microphone_stream()
        
        # 实时转录
        async for subtitle in model.transcribe_stream(mic_stream, language="ar"):
            print(f"🗣️  {subtitle.text}")
    
    except KeyboardInterrupt:
        print("\n⏹️  录音停止")
    except Exception as e:
        print(f"❌ 错误: {e}")


async def main():
    """主函数"""
    print("🚀 Whisper 流式处理快速开始")
    print("=" * 50)
    
    choice = input("选择模式:\n  1. 文件处理\n  2. 麦克风实时\n请输入 (1/2): ").strip()
    
    if choice == "1":
        await simple_whisper_example()
    elif choice == "2":
        await microphone_example()
    else:
        print("❌ 无效选择")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 再见!")
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        print("\n💡 确保已安装依赖:")
        print("   pip install openai-whisper")
        print("   pip install pyaudio  # 用于麦克风")
