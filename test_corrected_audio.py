#!/usr/bin/env python3
"""测试修正后的音频文件"""

import asyncio
import sys
from pathlib import Path

# 添加子模块路径
sys.path.insert(0, str(Path(__file__).parent / "amazon-transcribe-streaming-sdk"))

import aiofile
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
from amazon_transcribe.utils import apply_realtime_delay

# 音频参数 - 与转换后的文件匹配
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # 16-bit = 2 bytes
CHANNEL_NUMS = 1      # 单声道

# 使用转换后的音频文件
AUDIO_PATH = "output_16k_mono.wav"
CHUNK_SIZE = 1024 * 8
REGION = "us-east-1"


class MyEventHandler(TranscriptResultStreamHandler):
    def __init__(self, output_stream):
        super().__init__(output_stream)
        self.subtitle_count = 0
    
    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        """处理转录事件"""
        results = transcript_event.transcript.results
        
        for result in results:
            for alt in result.alternatives:
                if alt.transcript.strip():
                    self.subtitle_count += 1
                    if result.is_partial:
                        print(f"[部分 {self.subtitle_count}] {alt.transcript}")
                    else:
                        print(f"[完整 {self.subtitle_count}] {alt.transcript}")
                        print(f"   时间: {result.start_time:.1f}s - {result.end_time:.1f}s")


async def test_corrected_audio():
    """测试修正后的音频文件"""
    print("🚀 测试修正后的音频文件转录")
    print(f"📁 音频文件: {AUDIO_PATH}")
    print(f"🎵 参数: {SAMPLE_RATE}Hz, {CHANNEL_NUMS}声道, {BYTES_PER_SAMPLE*8}bit")
    
    # 检查文件是否存在
    if not Path(AUDIO_PATH).exists():
        print(f"❌ 音频文件不存在: {AUDIO_PATH}")
        print("请先运行以下命令转换音频格式:")
        print("ffmpeg -i output.wav -ar 16000 -ac 1 -sample_fmt s16 output_16k_mono.wav -y")
        return
    
    try:
        # 设置客户端
        client = TranscribeStreamingClient(region=REGION)
        print(f"✅ 已连接到 AWS Transcribe (区域: {REGION})")
        
        # 启动流式转录 - 使用 Arabic
        stream = await client.start_stream_transcription(
            language_code="ar-SA",  # Arabic (Saudi Arabia)
            media_sample_rate_hz=SAMPLE_RATE,
            media_encoding="pcm",
        )
        print("🎤 已启动 Arabic 流式转录")
        
        async def write_chunks():
            """写入音频数据块"""
            print("📡 开始发送音频数据...")
            try:
                async with aiofile.AIOFile(AUDIO_PATH, "rb") as afp:
                    reader = aiofile.Reader(afp, chunk_size=CHUNK_SIZE)
                    await apply_realtime_delay(
                        stream, reader, BYTES_PER_SAMPLE, SAMPLE_RATE, CHANNEL_NUMS
                    )
                await stream.input_stream.end_stream()
                print("📡 音频数据发送完成")
            except Exception as e:
                print(f"❌ 音频发送错误: {e}")
                raise
        
        # 创建事件处理器
        handler = MyEventHandler(stream.output_stream)
        
        print("🎧 开始处理转录结果...")
        print("-" * 50)
        
        # 并发执行音频发送和结果处理
        await asyncio.gather(write_chunks(), handler.handle_events())
        
        print("-" * 50)
        print(f"✅ 转录完成！共生成 {handler.subtitle_count} 条字幕")
        
    except Exception as e:
        print(f"❌ 转录失败: {e}")
        import traceback
        traceback.print_exc()


async def test_english_version():
    """测试英语版本"""
    print("\n" + "="*60)
    print("🚀 测试英语转录")
    
    if not Path(AUDIO_PATH).exists():
        print(f"❌ 音频文件不存在: {AUDIO_PATH}")
        return
    
    try:
        client = TranscribeStreamingClient(region=REGION)
        
        # 启动英语转录
        stream = await client.start_stream_transcription(
            language_code="en-US",  # English (US)
            media_sample_rate_hz=SAMPLE_RATE,
            media_encoding="pcm",
        )
        print("🎤 已启动 English 流式转录")
        
        async def write_chunks():
            async with aiofile.AIOFile(AUDIO_PATH, "rb") as afp:
                reader = aiofile.Reader(afp, chunk_size=CHUNK_SIZE)
                await apply_realtime_delay(
                    stream, reader, BYTES_PER_SAMPLE, SAMPLE_RATE, CHANNEL_NUMS
                )
            await stream.input_stream.end_stream()
        
        handler = MyEventHandler(stream.output_stream)
        
        print("🎧 开始英语转录...")
        print("-" * 50)
        
        await asyncio.gather(write_chunks(), handler.handle_events())
        
        print("-" * 50)
        print(f"✅ 英语转录完成！共生成 {handler.subtitle_count} 条字幕")
        
    except Exception as e:
        print(f"❌ 英语转录失败: {e}")


async def main():
    """主函数"""
    print("=" * 60)
    print("🧪 音频格式修正测试")
    print("=" * 60)
    
    # 测试 Arabic 转录
    await test_corrected_audio()
    
    # 测试 English 转录
    await test_english_version()
    
    print("\n" + "=" * 60)
    print("🎉 测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
