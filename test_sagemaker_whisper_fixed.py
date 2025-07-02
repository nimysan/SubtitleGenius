#!/usr/bin/env python3
"""
修复版本的 SageMaker Whisper 流式处理测试
直接使用正确的导入路径
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# 直接导入需要的类
from subtitle_genius.models.whisper_sagemaker_streaming import (
    WhisperSageMakerStreamingModel, 
    WhisperSageMakerStreamConfig
)


# 🔧 配置你的 SageMaker 端点信息
SAGEMAKER_ENDPOINT = "endpoint-quick-start-z9afg"  # 替换为你的端点名称
AWS_REGION = "us-east-1"  # 替换为你的 AWS 区域


async def test_direct_sagemaker_whisper():
    """直接测试 SageMaker Whisper 流式处理"""
    
    print("🎤 直接测试 SageMaker Whisper 流式处理")
    print("=" * 50)
    print(f"📍 使用端点: {SAGEMAKER_ENDPOINT}")
    print(f"🌍 AWS 区域: {AWS_REGION}")
    print("=" * 50)
    
    # 1. 创建配置
    config = WhisperSageMakerStreamConfig(
        chunk_duration=3.0,      # 每3秒处理一次
        overlap_duration=0.5,    # 0.5秒重叠
        voice_threshold=0.01,    # 语音检测阈值
        sagemaker_chunk_duration=30  # SageMaker 端点处理块大小
    )
    
    # 2. 直接创建 SageMaker Whisper 模型
    print("📦 初始化 SageMaker Whisper 模型...")
    
    try:
        model = WhisperSageMakerStreamingModel(
            endpoint_name=SAGEMAKER_ENDPOINT,
            region_name=AWS_REGION,
            config=config
        )
        print("✅ SageMaker Whisper 模型初始化成功")
        
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        print("请检查:")
        print(f"   - 端点名称: {SAGEMAKER_ENDPOINT}")
        print(f"   - AWS 区域: {AWS_REGION}")
        print("   - AWS 凭证配置")
        return
    
    # 3. 测试音频文件处理
    audio_files = [
        "/Users/yexw/PycharmProjects/SubtitleGenius/ar_football_mono.wav",
        "/Users/yexw/PycharmProjects/SubtitleGenius/test.wav"
    ]
    
    audio_file = None
    for file_path in audio_files:
        if Path(file_path).exists():
            audio_file = file_path
            break
    
    if not audio_file:
        print("❌ 没有找到测试音频文件")
        print("请提供以下文件之一:")
        for file_path in audio_files:
            print(f"   - {file_path}")
        return
    
    print(f"🎵 处理音频文件: {Path(audio_file).name}")
    
    try:
        # 导入流处理器
        from subtitle_genius.stream.processor import StreamProcessor
        processor = StreamProcessor()
        
        # 创建音频流
        audio_stream = processor.process_file_stream(audio_file)
        
        # 流式转录
        subtitle_count = 0
        async for subtitle in model.transcribe_stream(audio_stream, language="ar"):
            subtitle_count += 1
            print(f"📝 字幕 {subtitle_count}: [{subtitle.start:.1f}s-{subtitle.end:.1f}s] {subtitle.text}")
        
        print(f"✅ 处理完成，共生成 {subtitle_count} 条字幕")
        
    except Exception as e:
        print(f"❌ 音频处理失败: {e}")
        import traceback
        traceback.print_exc()


async def test_with_transcribe_model():
    """使用 TranscribeModel 统一接口测试"""
    
    print("🎤 使用 TranscribeModel 统一接口测试")
    print("=" * 50)
    
    try:
        # 直接导入并创建配置
        config = WhisperSageMakerStreamConfig(
            chunk_duration=2.0,
            overlap_duration=0.3,
            voice_threshold=0.02
        )
        
        # 使用统一接口
        from subtitle_genius.models.transcribe_model import TranscribeModel
        
        model = TranscribeModel(
            backend="sagemaker_whisper",
            sagemaker_endpoint=SAGEMAKER_ENDPOINT,
            region_name=AWS_REGION,
            whisper_config=config
        )
        
        if not model.is_available():
            print("❌ TranscribeModel 不可用")
            return
        
        print("✅ TranscribeModel 初始化成功")
        
        # 测试音频文件
        audio_file = "/Users/yexw/PycharmProjects/SubtitleGenius/ar_football_mono.wav"
        
        if Path(audio_file).exists():
            from subtitle_genius.stream.processor import StreamProcessor
            processor = StreamProcessor()
            audio_stream = processor.process_file_stream(audio_file)
            
            subtitle_count = 0
            async for subtitle in model.transcribe_stream(audio_stream, language="ar"):
                subtitle_count += 1
                print(f"📝 统一接口 {subtitle_count}: {subtitle.text}")
            
            print(f"✅ 统一接口测试完成，共 {subtitle_count} 条字幕")
        else:
            print(f"❌ 测试文件不存在: {audio_file}")
            
    except Exception as e:
        print(f"❌ 统一接口测试失败: {e}")
        import traceback
        traceback.print_exc()


async def simple_microphone_test():
    """简单的麦克风测试"""
    
    print("🎤 麦克风实时测试")
    print("=" * 30)
    
    try:
        # 创建快速响应配置
        config = WhisperSageMakerStreamConfig(
            chunk_duration=2.0,
            overlap_duration=0.3,
            voice_threshold=0.02
        )
        
        # 创建模型
        model = WhisperSageMakerStreamingModel(
            endpoint_name=SAGEMAKER_ENDPOINT,
            region_name=AWS_REGION,
            config=config
        )
        
        print("🔴 开始麦克风录音 (按 Ctrl+C 停止)")
        print("💬 请开始说话...")
        
        from subtitle_genius.stream.processor import StreamProcessor
        processor = StreamProcessor()
        mic_stream = processor.start_microphone_stream()
        
        subtitle_count = 0
        async for subtitle in model.transcribe_stream(mic_stream, language="ar"):
            subtitle_count += 1
            print(f"🗣️  实时 {subtitle_count}: {subtitle.text}")
            
    except KeyboardInterrupt:
        print(f"\n⏹️  录音停止，共识别 {subtitle_count} 段语音")
    except Exception as e:
        print(f"❌ 麦克风测试失败: {e}")


async def main():
    """主函数"""
    print("🚀 修复版本的 SageMaker Whisper 测试")
    print("=" * 60)
    
    # 检查配置
    if SAGEMAKER_ENDPOINT == "endpoint-quick-start-z9afg":
        print("⚠️  请更新脚本顶部的端点配置")
        print()
    
    tests = {
        "1": ("直接模型测试", test_direct_sagemaker_whisper),
        "2": ("统一接口测试", test_with_transcribe_model),
        "3": ("麦克风实时测试", simple_microphone_test),
    }
    
    print("选择测试:")
    for key, (name, _) in tests.items():
        print(f"  {key}. {name}")
    
    choice = input("\n请输入选择 (1/2/3): ").strip()
    
    if choice in tests:
        name, test_func = tests[choice]
        print(f"\n{'='*20} {name} {'='*20}")
        await test_func()
    else:
        print("❌ 无效选择")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 测试中断")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
