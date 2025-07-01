#!/usr/bin/env python3
"""测试麦克风实时流式转录"""

import asyncio
import numpy as np
from typing import AsyncGenerator

from subtitle_genius.models.transcribe_model import TranscribeModel

# 尝试导入音频库
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
    print("✅ sounddevice 可用")
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️  sounddevice 不可用，请安装: pip install sounddevice")

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
    print("✅ pyaudio 可用")
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("⚠️  pyaudio 不可用，请安装: pip install pyaudio")


async def create_microphone_stream_sounddevice() -> AsyncGenerator[np.ndarray, None]:
    """使用 sounddevice 创建麦克风音频流"""
    if not AUDIO_AVAILABLE:
        raise RuntimeError("sounddevice 不可用")
    
    sample_rate = 16000
    channels = 1
    blocksize = 1024
    
    print(f"🎤 开始麦克风录音 (采样率: {sample_rate}Hz, 通道: {channels})")
    print("💡 请对着麦克风说话，按 Ctrl+C 停止...")
    
    # 创建音频队列
    audio_queue = asyncio.Queue()
    
    def audio_callback(indata, frames, time, status):
        """音频回调函数"""
        if status:
            print(f"⚠️  音频状态: {status}")
        
        # 将音频数据放入队列
        audio_data = indata[:, 0] if indata.ndim > 1 else indata
        audio_int16 = (audio_data * 32767).astype(np.int16)
        asyncio.create_task(audio_queue.put(audio_int16))
    
    # 启动音频流
    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        blocksize=blocksize,
        dtype=np.float32,
        callback=audio_callback
    )
    
    try:
        with stream:
            while True:
                audio_chunk = await audio_queue.get()
                yield audio_chunk
                
    except KeyboardInterrupt:
        print("\n🛑 用户停止录音")
    except Exception as e:
        print(f"❌ 麦克风流错误: {e}")
    finally:
        print("🎤 麦克风录音结束")


async def create_test_file_stream(file_path: str) -> AsyncGenerator[np.ndarray, None]:
    """从音频文件创建流 (模拟实时)"""
    try:
        import librosa
        print(f"📁 加载音频文件: {file_path}")
        
        # 加载音频文件
        audio_data, sr = librosa.load(file_path, sr=16000, mono=True)
        
        # 转换为 int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # 分块发送
        chunk_size = 1024
        total_chunks = len(audio_int16) // chunk_size
        
        print(f"🎵 开始流式发送音频 ({total_chunks} 块)")
        
        for i in range(0, len(audio_int16), chunk_size):
            chunk = audio_int16[i:i+chunk_size]
            if len(chunk) > 0:
                yield chunk
                # 模拟实时延迟
                await asyncio.sleep(chunk_size / 16000)  # 基于采样率的延迟
        
        print("🎵 文件流发送完成")
        
    except ImportError:
        print("❌ librosa 不可用，请安装: pip install librosa")
        raise
    except Exception as e:
        print(f"❌ 文件流错误: {e}")
        raise


async def test_microphone_transcription():
    """测试麦克风实时转录"""
    print("🚀 开始麦克风实时转录测试")
    
    # 初始化模型
    model = TranscribeModel(region_name="us-east-1")
    
    if not model.is_available():
        print("❌ TranscribeModel 不可用")
        return
    
    if not AUDIO_AVAILABLE:
        print("❌ 音频库不可用，跳过麦克风测试")
        return
    
    try:
        # 创建麦克风音频流
        audio_stream = create_microphone_stream_sounddevice()
        
        # 开始实时转录
        print("🎤 开始实时 Arabic 转录...")
        subtitle_count = 0
        
        async for subtitle in model.transcribe_stream(audio_stream, language="ar"):
            subtitle_count += 1
            print(f"🗣️  实时字幕 {subtitle_count}: {subtitle.text}")
            print(f"   ⏱️  时间: {subtitle.start:.1f}s - {subtitle.end:.1f}s")
        
        print(f"✅ 实时转录完成，共生成 {subtitle_count} 条字幕")
        
    except KeyboardInterrupt:
        print("\n🛑 用户中断测试")
    except Exception as e:
        print(f"❌ 麦克风转录测试失败: {e}")
        import traceback
        traceback.print_exc()


async def test_file_streaming():
    """测试文件流式转录"""
    print("\n🚀 测试文件流式转录")
    
    # 检查是否有测试音频文件
    test_files = [
        "test.wav",
        "audio.wav", 
        "sample.wav",
        "tests/integration/assets/test.wav"  # SDK 示例文件
    ]
    
    test_file = None
    for file_path in test_files:
        from pathlib import Path
        if Path(file_path).exists():
            test_file = file_path
            break
    
    if not test_file:
        print("⚠️  未找到测试音频文件，跳过文件流测试")
        print(f"   尝试的文件: {test_files}")
        return
    
    model = TranscribeModel(region_name="us-east-1")
    
    if not model.is_available():
        print("❌ 模型不可用")
        return
    
    try:
        # 创建文件音频流
        audio_stream = create_test_file_stream(test_file)
        
        print(f"🎤 开始文件流式转录: {test_file}")
        subtitle_count = 0
        
        async for subtitle in model.transcribe_stream(audio_stream, language="en"):
            subtitle_count += 1
            print(f"📝 文件字幕 {subtitle_count}: {subtitle.text}")
            print(f"   ⏱️  时间: {subtitle.start:.1f}s - {subtitle.end:.1f}s")
        
        print(f"✅ 文件转录完成，共生成 {subtitle_count} 条字幕")
        
    except Exception as e:
        print(f"❌ 文件转录测试失败: {e}")


async def main():
    """主测试函数"""
    print("=" * 60)
    print("🎤 麦克风实时流式转录测试")
    print("=" * 60)
    
    # 检查依赖
    print("📋 检查依赖:")
    print(f"   sounddevice: {'✅' if AUDIO_AVAILABLE else '❌'}")
    print(f"   pyaudio: {'✅' if PYAUDIO_AVAILABLE else '❌'}")
    
    # 测试文件流式转录
    await test_file_streaming()
    
    # 测试麦克风实时转录
    if AUDIO_AVAILABLE:
        print("\n" + "=" * 40)
        print("准备开始麦克风测试...")
        print("按 Enter 继续，或 Ctrl+C 跳过")
        try:
            input()
            await test_microphone_transcription()
        except KeyboardInterrupt:
            print("\n⏭️  跳过麦克风测试")
    
    print("\n" + "=" * 60)
    print("🎉 测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
