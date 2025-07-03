#!/usr/bin/env python3
"""
简单的 SageMaker Whisper 流式处理示例
快速开始使用你现有的 SageMaker Whisper 端点进行实时语音识别
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from subtitle_genius.models.transcribe_model import TranscribeModel
from subtitle_genius.models.whisper_sagemaker_streaming import WhisperSageMakerStreamConfig


# 🔧 配置你的 SageMaker 端点信息
SAGEMAKER_ENDPOINT = "endpoint-quick-start-z9afg"  # 替换为你的端点名称
AWS_REGION = "us-east-1"  # 替换为你的 AWS 区域


async def simple_sagemaker_whisper_example():
    """简单的 SageMaker Whisper 流式处理示例"""
    
    print("🎤 SageMaker Whisper 流式语音识别示例")
    print("=" * 50)
    print(f"📍 使用端点: {SAGEMAKER_ENDPOINT}")
    print(f"🌍 AWS 区域: {AWS_REGION}")
    print("=" * 50)
    
    # 1. 创建 SageMaker Whisper 流式模型
    print("📦 初始化 SageMaker Whisper 模型...")
    
    # 配置参数
    config = WhisperSageMakerStreamConfig(
        chunk_duration=10,    # 每3秒处理一次
        overlap_duration=2,  # 0.5秒重叠避免截断
        voice_threshold=0.01,  # 语音检测阈值
        sagemaker_chunk_duration=30  # SageMaker 端点处理的块大小
    )
    
    # 创建模型 (使用 SageMaker Whisper 后端)
    model = TranscribeModel(
        backend="sagemaker_whisper",
        sagemaker_endpoint=SAGEMAKER_ENDPOINT,
        region_name=AWS_REGION,
        whisper_config=config
    )
    
    # 检查模型是否可用
    if not model.is_available():
        print("❌ SageMaker Whisper 不可用，请检查:")
        print(f"   - 端点名称: {SAGEMAKER_ENDPOINT}")
        print(f"   - AWS 区域: {AWS_REGION}")
        print("   - AWS 凭证配置 (aws configure)")
        print("   - 端点是否在运行中")
        return
    
    print("✅ SageMaker Whisper 模型已准备就绪")
    
    # 2. 处理音频文件 (如果存在)
    audio_files = [
        "/Users/yexw/PycharmProjects/SubtitleGenius/ar_football_mono.wav",
        # "/Users/yexw/PycharmProjects/SubtitleGenius/test.wav"
    ]
    
    audio_file = None
    for file_path in audio_files:
        if Path(file_path).exists():
            audio_file = file_path
            break
    
    if audio_file:
        print(f"\n🎵 处理音频文件: {Path(audio_file).name}")
        
        # 导入流处理器
        from subtitle_genius.stream.processor import StreamProcessor
        processor = StreamProcessor()
        
        # 创建文件音频流
        audio_stream = processor.process_file_stream(audio_file)
        
        # 流式转录
        subtitle_count = 0
        async for subtitle in model.transcribe_stream(audio_stream, language="ar"):
            subtitle_count += 1
            print(f"📝 [{subtitle.start:.1f}s] {subtitle.text}")
        
        print(f"✅ 文件处理完成，共生成 {subtitle_count} 条字幕")
    
    else:
        print(f"\n⚠️  音频文件不存在，请提供以下文件之一:")
        for file_path in audio_files:
            print(f"   - {file_path}")
        print("或使用麦克风模式")
    
    print("\n✅ 示例完成")


async def microphone_example():
    """麦克风实时识别示例"""
    
    print("🎤 SageMaker Whisper 麦克风实时语音识别")
    print("=" * 40)
    
    # 创建快速响应的配置
    config = WhisperSageMakerStreamConfig(
        chunk_duration=2.0,    # 更短的处理间隔
        overlap_duration=0.3,  # 较短重叠
        voice_threshold=0.02,  # 稍高的检测阈值
        sagemaker_chunk_duration=30
    )
    
    model = TranscribeModel(
        backend="sagemaker_whisper",
        sagemaker_endpoint=SAGEMAKER_ENDPOINT,
        region_name=AWS_REGION,
        whisper_config=config
    )
    
    if not model.is_available():
        print("❌ SageMaker Whisper 不可用")
        return
    
    try:
        from subtitle_genius.stream.processor import StreamProcessor
        processor = StreamProcessor()
        
        print("🔴 开始录音 (按 Ctrl+C 停止)")
        print("💬 请开始说话...")
        
        # 启动麦克风
        mic_stream = processor.start_microphone_stream()
        
        # 实时转录
        subtitle_count = 0
        async for subtitle in model.transcribe_stream(mic_stream, language="ar"):
            subtitle_count += 1
            print(f"🗣️  {subtitle_count}: {subtitle.text}")
    
    except KeyboardInterrupt:
        print(f"\n⏹️  录音停止，共识别 {subtitle_count} 段语音")
    except Exception as e:
        print(f"❌ 错误: {e}")


async def batch_vs_streaming_comparison():
    """比较批处理和流式处理的效果"""
    
    print("⚖️  批处理 vs 流式处理比较")
    print("=" * 40)
    
    audio_file = "/Users/yexw/PycharmProjects/SubtitleGenius/ar_football_mono.wav"
    
    if not Path(audio_file).exists():
        print(f"❌ 测试音频文件不存在: {audio_file}")
        return
    
    # 1. 原始批处理方式
    print("🔵 原始批处理方式:")
    try:
        from whisper_converse import WhisperSageMakerClient
        
        batch_client = WhisperSageMakerClient(
            endpoint_name=SAGEMAKER_ENDPOINT,
            region_name=AWS_REGION
        )
        
        import time
        start_time = time.time()
        result = batch_client.transcribe_audio(audio_file, language="ar")
        batch_time = time.time() - start_time
        
        print(f"📝 批处理结果: {result.get('transcription', 'N/A')}")
        print(f"⏱️  批处理时间: {batch_time:.2f}秒")
        
    except Exception as e:
        print(f"❌ 批处理失败: {e}")
    
    # 2. 新的流式处理方式
    print("\n🟠 新的流式处理方式:")
    try:
        from subtitle_genius.stream.processor import StreamProcessor
        
        model = TranscribeModel(
            backend="sagemaker_whisper",
            sagemaker_endpoint=SAGEMAKER_ENDPOINT,
            region_name=AWS_REGION
        )
        
        processor = StreamProcessor()
        audio_stream = processor.process_file_stream(audio_file)
        
        start_time = time.time()
        subtitles = []
        async for subtitle in model.transcribe_stream(audio_stream, language="ar"):
            subtitles.append(subtitle.text)
            print(f"📝 流式结果: {subtitle.text}")
        
        stream_time = time.time() - start_time
        combined_text = ' '.join(subtitles)
        
        print(f"📝 流式合并结果: {combined_text}")
        print(f"⏱️  流式处理时间: {stream_time:.2f}秒")
        
    except Exception as e:
        print(f"❌ 流式处理失败: {e}")


async def main():
    """主函数"""
    print("🚀 SageMaker Whisper 流式处理快速开始")
    print("=" * 60)
    
    # 检查配置
    if SAGEMAKER_ENDPOINT == "endpoint-quick-start-z9afg":
        print("⚠️  请在脚本顶部更新你的 SageMaker 端点配置:")
        print("   SAGEMAKER_ENDPOINT = '你的端点名称'")
        print("   AWS_REGION = '你的AWS区域'")
        print()
    
    choice = input("选择模式:\n  1. 文件处理\n  2. 麦克风实时\n  3. 批处理对比\n请输入 (1/2/3): ").strip()
    
    if choice == "1":
        await simple_sagemaker_whisper_example()
    elif choice == "2":
        await microphone_example()
    elif choice == "3":
        await batch_vs_streaming_comparison()
    else:
        print("❌ 无效选择")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 再见!")
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        print("\n💡 确保已配置:")
        print("   - SageMaker 端点名称和区域")
        print("   - AWS 凭证 (aws configure)")
        print("   - 端点正在运行中")
