"""
测试VAD缓冲区处理器
演示如何使用VADBufferProcessor处理音频流
"""

import asyncio
import numpy as np
import logging
import argparse
from pathlib import Path
import time
import wave
import io
from typing import List, Dict, Tuple, Optional, AsyncGenerator

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入VAD处理器
from subtitle_genius.audio.vad_processor import VADBufferProcessor, VADConfig
from subtitle_genius.models.whisper_sagemaker_streaming import WhisperSageMakerStreamingModel

# 模拟SageMaker Whisper模型（如果不可用）
class MockWhisperModel:
    """模拟Whisper模型，用于测试"""
    
    def __init__(self):
        logger.info("初始化模拟Whisper模型")
        
    async def transcribe_audio(self, audio_data: np.ndarray, language: str = "zh") -> str:
        """模拟转录音频"""
        # 简单地返回音频长度信息
        duration = len(audio_data) / 16000  # 假设采样率为16kHz
        return f"音频长度: {duration:.2f}秒"


async def read_audio_file_in_chunks(file_path: str, chunk_duration: float = 3.0, sample_rate: int = 16000) -> AsyncGenerator[np.ndarray, None]:
    """
    以指定的块大小读取音频文件
    
    参数:
        file_path: 音频文件路径
        chunk_duration: 每个块的时长（秒）
        sample_rate: 采样率
        
    返回:
        音频块生成器
    """
    try:
        # 打开WAV文件
        with wave.open(file_path, 'rb') as wav_file:
            # 检查采样率
            file_sample_rate = wav_file.getframerate()
            if file_sample_rate != sample_rate:
                logger.warning(f"文件采样率 ({file_sample_rate}Hz) 与指定采样率 ({sample_rate}Hz) 不匹配")
            
            # 计算每个块的样本数
            chunk_samples = int(chunk_duration * file_sample_rate)
            
            # 读取音频数据
            while True:
                frames = wav_file.readframes(chunk_samples)
                if not frames:
                    break
                
                # 转换为numpy数组
                if wav_file.getsampwidth() == 2:  # 16-bit
                    audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                else:  # 假设为8-bit
                    audio_data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 255.0 - 0.5
                
                yield audio_data
                
                # 模拟实时处理的延迟
                await asyncio.sleep(chunk_duration * 0.5)  # 模拟处理时间
                
    except Exception as e:
        logger.error(f"读取音频文件失败: {e}")
        raise


async def process_audio_with_vad_buffer(audio_file: str, use_whisper: bool = False, language: str = "zh"):
    """
    使用VAD缓冲区处理器处理音频文件
    
    参数:
        audio_file: 音频文件路径
        use_whisper: 是否使用Whisper模型进行转录
        language: 语言代码
    """
    logger.info(f"开始处理音频文件: {audio_file}")
    
    # 创建VAD配置
    vad_config = VADConfig(
        threshold=0.5,                # VAD置信度阈值
        min_speech_duration_ms=250,   # 最小语音持续时间(毫秒)
        min_silence_duration_ms=100,  # 最小静音持续时间(毫秒)
        window_size_samples=1536,     # VAD窗口大小
        speech_pad_ms=30              # 语音片段前后填充(毫秒)
    )
    
    # 创建VAD缓冲区处理器
    vad_buffer = VADBufferProcessor(config=vad_config, buffer_size_seconds=30.0)
    
    # 初始化Whisper模型（如果需要）
    whisper_model = None
    if use_whisper:
        try:
            # 尝试初始化SageMaker Whisper模型
            from subtitle_genius.models.whisper_sagemaker_streaming import WhisperSageMakerStreamingModel
            whisper_model = WhisperSageMakerStreamingModel(
                endpoint_name="your-sagemaker-endpoint",  # 替换为实际的端点名称
                region_name="us-east-1"                   # 替换为实际的区域
            )
            logger.info("已初始化SageMaker Whisper模型")
        except Exception as e:
            logger.warning(f"初始化SageMaker Whisper模型失败: {e}")
            logger.warning("将使用模拟Whisper模型")
            whisper_model = MockWhisperModel()
    else:
        logger.info("不使用Whisper模型，仅进行VAD处理")
    
    # 读取音频文件
    chunk_duration = 3.0  # 3秒一个块
    audio_chunks = read_audio_file_in_chunks(audio_file, chunk_duration=chunk_duration)
    
    # 处理计数器
    chunk_count = 0
    segment_count = 0
    
    # 开始处理时间
    start_time = time.time()
    
    try:
        async for audio_chunk in audio_chunks:
            chunk_count += 1
            logger.info(f"处理音频块 #{chunk_count}, 大小: {len(audio_chunk)} 样本, 时长: {len(audio_chunk)/16000:.2f}秒")
            
            # 添加到VAD缓冲区
            vad_buffer.add_audio_chunk(audio_chunk)
            
            # 处理缓冲区，获取语音片段
            speech_segments = vad_buffer.process_buffer()
            
            # 处理检测到的语音片段
            for i, (segment, timestamp) in enumerate(speech_segments):
                segment_count += 1
                segment_duration = len(segment) / 16000
                
                logger.info(f"检测到语音片段 #{segment_count}: "
                           f"开始={timestamp['start']:.2f}s, "
                           f"结束={timestamp['end']:.2f}s, "
                           f"时长={segment_duration:.2f}s")
                
                # 如果启用Whisper，转录语音片段
                if use_whisper and whisper_model:
                    # 转录音频
                    transcription = await whisper_model.transcribe_audio(segment, language)
                    logger.info(f"转录结果: {transcription}")
            
            # 显示缓冲区状态
            buffer_duration = vad_buffer.get_buffer_duration()
            logger.info(f"当前缓冲区大小: {buffer_duration:.2f}秒")
            
            # 每处理5个块，显示一次统计信息
            if chunk_count % 5 == 0:
                elapsed = time.time() - start_time
                logger.info(f"已处理 {chunk_count} 个音频块, 检测到 {segment_count} 个语音片段, 耗时: {elapsed:.2f}秒")
    
    except Exception as e:
        logger.error(f"处理音频时出错: {e}")
    
    finally:
        # 处理剩余的缓冲区
        logger.info("处理剩余的缓冲区...")
        remaining_segments = vad_buffer.get_pending_segments()
        
        for i, (segment, timestamp) in enumerate(remaining_segments):
            segment_count += 1
            segment_duration = len(segment) / 16000
            
            logger.info(f"处理剩余语音片段 #{segment_count}: "
                       f"开始={timestamp['start']:.2f}s, "
                       f"结束={timestamp['end']:.2f}s, "
                       f"时长={segment_duration:.2f}s")
            
            # 如果启用Whisper，转录语音片段
            if use_whisper and whisper_model:
                # 转录音频
                transcription = await whisper_model.transcribe_audio(segment, language)
                logger.info(f"转录结果: {transcription}")
        
        # 显示最终统计信息
        elapsed = time.time() - start_time
        logger.info(f"处理完成! 共处理 {chunk_count} 个音频块, 检测到 {segment_count} 个语音片段, 总耗时: {elapsed:.2f}秒")


async def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="测试VAD缓冲区处理器")
    parser.add_argument("audio_file", help="要处理的音频文件路径")
    parser.add_argument("--use-whisper", action="store_true", help="是否使用Whisper模型进行转录")
    parser.add_argument("--language", default="zh", help="语言代码 (默认: zh)")
    args = parser.parse_args()
    
    # 检查文件是否存在
    audio_file = Path(args.audio_file)
    if not audio_file.exists():
        logger.error(f"音频文件不存在: {audio_file}")
        return
    
    # 处理音频
    await process_audio_with_vad_buffer(
        str(audio_file),
        use_whisper=args.use_whisper,
        language=args.language
    )


if __name__ == "__main__":
    asyncio.run(main())
