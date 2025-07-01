#!/usr/bin/env python3
"""测试新的 Amazon Transcribe 流式实现"""

import asyncio
import numpy as np
from typing import AsyncGenerator

from subtitle_genius.models.transcribe_model import TranscribeModel


async def create_test_audio_stream() -> AsyncGenerator[np.ndarray, None]:
    """创建测试音频流 - 模拟实时音频数据"""
    # 模拟 16kHz 采样率的音频数据
    sample_rate = 16000
    duration = 0.1  # 每个块 100ms
    samples_per_chunk = int(sample_rate * duration)
    
    print("🎵 开始生成测试音频流...")
    
    for i in range(50):  # 生成 5 秒的音频数据
        # 生成简单的正弦波作为测试音频
        t = np.linspace(i * duration, (i + 1) * duration, samples_per_chunk)
        frequency = 440  # A4 音符
        audio_chunk = np.sin(2 * np.pi * frequency * t) * 0.5
        
        # 转换为 int16 格式
        audio_chunk = (audio_chunk * 32767).astype(np.int16)
        
        print(f"📡 发送音频块 {i+1}/50 ({len(audio_chunk)} 样本)")
        yield audio_chunk
        
        # 模拟实时延迟
        await asyncio.sleep(duration)
    
    print("🎵 音频流生成完成")


async def test_transcribe_streaming():
    """测试流式转录功能"""
    print("🚀 开始测试 Amazon Transcribe 流式转录")
    
    # 初始化模型
    model = TranscribeModel(region_name="us-east-1")
    
    # 检查可用性
    if not model.is_available():
        print("❌ TranscribeModel 不可用，请检查:")
        print("   1. AWS 凭证配置")
        print("   2. amazon-transcribe-streaming-sdk 子模块")
        print("   3. 网络连接")
        return
    
    print("✅ TranscribeModel 可用")
    
    try:
        # 创建音频流
        audio_stream = create_test_audio_stream()
        
        # 开始流式转录 (Arabic)
        print("🎤 开始 Arabic 流式转录...")
        subtitle_count = 0
        
        async for subtitle in model.transcribe_stream(audio_stream, language="ar"):
            subtitle_count += 1
            print(f"📝 字幕 {subtitle_count}: [{subtitle.start:.1f}s-{subtitle.end:.1f}s] {subtitle.text}")
        
        print(f"✅ 转录完成，共生成 {subtitle_count} 条字幕")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


async def test_english_transcription():
    """测试英语转录"""
    print("\n🚀 测试英语转录")
    
    model = TranscribeModel(region_name="us-east-1")
    
    if not model.is_available():
        print("❌ 模型不可用")
        return
    
    try:
        audio_stream = create_test_audio_stream()
        
        print("🎤 开始 English 流式转录...")
        subtitle_count = 0
        
        async for subtitle in model.transcribe_stream(audio_stream, language="en"):
            subtitle_count += 1
            print(f"📝 字幕 {subtitle_count}: [{subtitle.start:.1f}s-{subtitle.end:.1f}s] {subtitle.text}")
        
        print(f"✅ 英语转录完成，共生成 {subtitle_count} 条字幕")
        
    except Exception as e:
        print(f"❌ 英语转录测试失败: {e}")


def test_batch_mode_removed():
    """测试批处理模式是否已移除"""
    print("\n🚀 测试批处理模式移除")
    
    model = TranscribeModel()
    
    try:
        # 这应该抛出 NotImplementedError
        result = asyncio.run(model.transcribe("test.wav"))
        print("❌ 批处理模式未正确移除")
    except NotImplementedError as e:
        print(f"✅ 批处理模式已正确移除: {e}")
    except Exception as e:
        print(f"⚠️  意外错误: {e}")


async def main():
    """主测试函数"""
    print("=" * 60)
    print("🧪 Amazon Transcribe 流式 SDK 测试")
    print("=" * 60)
    
    # 测试批处理模式移除
    test_batch_mode_removed()
    
    # 测试流式转录
    await test_transcribe_streaming()
    
    # 测试英语转录
    await test_english_transcription()
    
    print("\n" + "=" * 60)
    print("🎉 测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
