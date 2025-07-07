"""
增强版VAD与WebSocket集成示例
演示如何在WebSocket服务器中集成增强版VAD处理和Whisper SageMaker转录
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, AsyncGenerator
import json
import io
import wave
import time
import os
from pathlib import Path

# 导入VAD分段日志记录器
from vad_segments_log import VADSegmentLogger

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入增强版VAD处理器和Whisper模型
from enhanced_vad_processor import EnhancedVADBufferProcessor
from subtitle_genius.audio.vad_processor import VADConfig
from subtitle_genius.models.whisper_sagemaker_streaming import WhisperSageMakerStreamingModel, WhisperSageMakerStreamConfig
from subtitle_genius.subtitle.models import Subtitle

# 模拟WebSocket连接
class MockWebSocket:
    """模拟WebSocket连接，用于测试"""
    
    def __init__(self, output_vtt_path: Optional[str] = None):
        self.messages = []
        self.subtitles = []
        self.output_vtt_path = output_vtt_path
        self.first_audio_time = None  # 记录第一个音频块的时间
        self.total_duration = 0.0     # 记录总处理时间
        
    async def send_json(self, data: Dict):
        """发送JSON数据"""
        self.messages.append(data)
        logger.info(f"发送消息: {json.dumps(data, ensure_ascii=False)}")
        
        # 如果是字幕消息，保存到字幕列表
        if data.get("type") == "subtitle" and "subtitle" in data:
            subtitle_data = data["subtitle"]
            self.subtitles.append(subtitle_data)
            
            # 如果指定了输出路径，保存为VTT格式
            if self.output_vtt_path:
                self.save_subtitles_as_vtt(self.output_vtt_path)
        
    async def receive(self):
        """接收消息（模拟）"""
        # 这里可以根据需要模拟接收消息
        return {"type": "websocket.receive", "bytes": b""}
    
    def save_subtitles_as_vtt(self, output_path: str) -> None:
        """
        将字幕保存为VTT格式
        
        参数:
            output_path: 输出文件路径
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                # 写入VTT头部
                f.write("WEBVTT\n\n")
                
                # 直接使用VAD分解后的时间戳
                # 按时间顺序排序字幕
                sorted_subtitles = sorted(self.subtitles, key=lambda x: x["start"])
                
                # 写入每个字幕
                for subtitle in sorted_subtitles:
                    # 格式化时间戳
                    start = self._format_timestamp(subtitle["start"])
                    end = self._format_timestamp(subtitle["end"])
                    
                    # 写入字幕
                    f.write(f"{start} --> {end}\n")
                    f.write(f"{subtitle['text']}\n\n")
                    
            logger.info(f"字幕已保存为VTT格式: {output_path}")
            
        except Exception as e:
            logger.error(f"保存字幕为VTT格式失败: {e}")
    
    def _format_timestamp(self, seconds: float) -> str:
        """
        将秒数格式化为VTT时间戳格式 (HH:MM:SS.mmm)
        
        参数:
            seconds: 秒数
            
        返回:
            格式化后的时间戳
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


class EnhancedVADWebSocketHandler:
    """增强版VAD WebSocket处理器"""
    
    def __init__(self, 
                 client_id: str,
                 language: str = "zh",
                 vad_config: Optional[VADConfig] = None,
                 whisper_endpoint: Optional[str] = None,
                 region_name: str = "us-east-1",
                 segment_log_file: str = "logs/vad_segments.log"):
        """
        初始化增强版VAD WebSocket处理器
        
        参数:
            client_id: 客户端ID
            language: 语言代码
            vad_config: VAD配置
            whisper_endpoint: Whisper SageMaker端点名称
            region_name: AWS区域
            segment_log_file: VAD分段日志文件路径
        """
        self.client_id = client_id
        self.language = language
        
        # 初始化VAD配置
        self.vad_config = vad_config or VADConfig(
            threshold=0.5,                # VAD置信度阈值
            min_speech_duration_ms=250,   # 最小语音持续时间(毫秒)
            min_silence_duration_ms=100,  # 最小静音持续时间(毫秒)
            window_size_samples=1536,     # VAD窗口大小
            speech_pad_ms=30              # 语音片段前后填充(毫秒)
        )
        
        # 初始化增强版VAD缓冲区处理器
        self.vad_buffer = EnhancedVADBufferProcessor(config=self.vad_config, buffer_size_seconds=30.0)
        
        # 初始化Whisper模型
        self.whisper_model = None
        if whisper_endpoint:
            try:
                whisper_config = WhisperSageMakerStreamConfig(
                    sample_rate=16000,
                    sagemaker_chunk_duration=30  # SageMaker处理的块大小(秒)
                )
                self.whisper_model = WhisperSageMakerStreamingModel(
                    endpoint_name=whisper_endpoint,
                    region_name=region_name,
                    config=whisper_config
                )
                logger.info(f"已初始化SageMaker Whisper模型: {whisper_endpoint}")
            except Exception as e:
                logger.error(f"初始化SageMaker Whisper模型失败: {e}")
                self.whisper_model = None
        
        # 存储字幕
        self.subtitles: List[Subtitle] = []
        
        # 处理计数器
        self.chunk_count = 0
        self.segment_count = 0
        
        # 记录处理时间
        self.start_time = time.time()  # 记录开始处理的时间
        self.total_processed_duration = 0.0  # 记录已处理的总时长
        
        # 初始化VAD分段日志记录器
        self.segment_logger = VADSegmentLogger(segment_log_file)
        
        logger.info(f"增强版VAD WebSocket处理器初始化完成，客户端ID: {client_id}, 语言: {language}")
    
    async def process_audio_chunk(self, audio_data: np.ndarray, websocket: MockWebSocket) -> None:
        """
        处理音频块
        
        参数:
            audio_data: 音频数据，numpy数组
            websocket: WebSocket连接
        """
        self.chunk_count += 1
        chunk_duration = len(audio_data) / 16000
        logger.info(f"处理音频块 #{self.chunk_count}, 大小: {len(audio_data)} 样本, 时长: {chunk_duration:.2f}秒")
        
        # 更新总处理时长
        self.total_processed_duration += chunk_duration
        
        # 添加到VAD缓冲区
        self.vad_buffer.add_audio_chunk(audio_data)
        
        # 处理缓冲区，获取完整的语音片段
        complete_segments = self.vad_buffer.process_buffer()
        
        # 处理完整的语音片段
        for i, (segment, timestamp) in enumerate(complete_segments):
            self.segment_count += 1
            segment_duration = len(segment) / 16000
            
            # 记录VAD分段信息
            self.segment_logger.log_segment(
                segment_index=self.segment_count,
                start_time=timestamp['start'],
                end_time=timestamp['end']
            )
            
            logger.info(f"处理完整语音片段 #{self.segment_count}: "
                       f"开始={timestamp['start']:.2f}s, "
                       f"结束={timestamp['end']:.2f}s, "
                       f"时长={segment_duration:.2f}秒")
            
            # 如果有Whisper模型，转录语音片段
            if self.whisper_model:
                # 转录音频
                transcription = await self.whisper_model.transcribe_audio(segment, self.language)
                
                if transcription:
                    # 创建字幕对象
                    subtitle = Subtitle(
                        start=timestamp['start'],
                        end=timestamp['end'],
                        text=transcription
                    )
                    
                    # 添加到字幕列表
                    self.subtitles.append(subtitle)
                    
                    # 发送字幕到客户端
                    await self.send_subtitle(websocket, subtitle)
                    
                    logger.info(f"转录结果: {transcription}")
                else:
                    logger.warning(f"语音片段 #{self.segment_count} 转录失败")
            else:
                logger.info(f"未配置Whisper模型，跳过转录")
    
    async def send_subtitle(self, websocket: MockWebSocket, subtitle: Subtitle) -> None:
        """
        发送字幕到客户端
        
        参数:
            websocket: WebSocket连接
            subtitle: 字幕对象
        """
        await websocket.send_json({
            "type": "subtitle",
            "subtitle": {
                "id": f"{self.client_id}_{self.segment_count}",
                "start": subtitle.start,
                "end": subtitle.end,
                "text": subtitle.text,
                "translated_text": subtitle.translated_text,
                "correction_applied": False,
                "translation_applied": False,
                "target_language": None
            }
        })


async def process_wav_data(data: bytes) -> Optional[np.ndarray]:
    """
    处理WAV数据
    
    参数:
        data: WAV数据
        
    返回:
        音频数据，numpy数组
    """
    try:
        # 从WAV数据中读取音频
        with wave.open(io.BytesIO(data), 'rb') as wav_file:
            # 获取音频参数
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            
            # 读取音频数据
            frames = wav_file.readframes(wav_file.getnframes())
            
            # 转换为numpy数组
            if sample_width == 2:  # 16-bit
                audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            else:  # 假设为8-bit
                audio_data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 255.0 - 0.5
            
            # 如果是立体声，转换为单声道
            if channels == 2:
                audio_data = audio_data.reshape(-1, 2).mean(axis=1)
            
            return audio_data
    
    except Exception as e:
        logger.error(f"处理WAV数据失败: {e}")
        return None


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


async def simulate_websocket_connection(audio_file: str, language: str = "zh", output_vtt: Optional[str] = None, segment_log_file: str = "logs/vad_segments.log"):
    """
    模拟WebSocket连接
    
    参数:
        audio_file: 音频文件路径
        language: 语言代码
        output_vtt: 输出VTT文件路径，如果为None则不保存
        segment_log_file: VAD分段日志文件路径
    """
    logger.info(f"开始模拟WebSocket连接，处理音频文件: {audio_file}")
    
    # 创建模拟WebSocket
    websocket = MockWebSocket(output_vtt_path=output_vtt)
    
    # 创建客户端ID
    client_id = "test_client_001"
    
    # 创建增强版VAD WebSocket处理器
    handler = EnhancedVADWebSocketHandler(
        client_id=client_id,
        language=language,
        whisper_endpoint="endpoint-quick-start-z9afg",  # 替换为实际的端点名称
        segment_log_file=segment_log_file
    )
    
    # 发送连接确认
    await websocket.send_json({
        "type": "connection",
        "status": "connected",
        "client_id": client_id,
        "model": "whisper",
        "language": language
    })
    
    # 读取音频文件
    chunk_duration = 3.0  # 3秒一个块
    audio_chunks = read_audio_file_in_chunks(audio_file, chunk_duration=chunk_duration)
    
    try:
        # 处理音频块
        async for audio_chunk in audio_chunks:
            # 处理音频块
            await handler.process_audio_chunk(audio_chunk, websocket)
            
            # 显示处理统计信息
            if handler.chunk_count % 5 == 0:
                logger.info(f"已处理 {handler.chunk_count} 个音频块, 检测到 {handler.segment_count} 个语音片段")
    
    except Exception as e:
        logger.error(f"处理音频时出错: {e}")
    
    finally:
        # 发送断开连接消息
        await websocket.send_json({
            "type": "disconnect",
            "client_id": client_id
        })
        
        # 显示最终统计信息
        logger.info(f"处理完成! 共处理 {handler.chunk_count} 个音频块, 检测到 {handler.segment_count} 个语音片段")
        logger.info(f"生成 {len(handler.subtitles)} 个字幕")
        
        # 分析VAD分段重叠情况
        overlaps = handler.segment_logger.analyze_overlaps()
        if overlaps:
            logger.warning(f"检测到 {len(overlaps)} 个重叠的VAD分段")
            for i, overlap in enumerate(overlaps):
                logger.warning(f"重叠 #{i+1}: 分段 {overlap['segment1']} 和分段 {overlap['segment2']} 重叠 {overlap['overlap_duration']:.3f}秒")
        else:
            logger.info("未检测到VAD分段重叠")
        
        # 保存VAD分段信息为JSON格式
        json_file = f"{os.path.splitext(segment_log_file)[0]}.json"
        handler.segment_logger.save_json(json_file)
        logger.info(f"VAD分段信息已保存为JSON格式: {json_file}")


async def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="增强版VAD与WebSocket集成示例")
    parser.add_argument("audio_file", help="要处理的音频文件路径")
    parser.add_argument("--language", default="zh", help="语言代码 (默认: zh)")
    parser.add_argument("--output-vtt", help="输出VTT文件路径")
    parser.add_argument("--segment-log", default="logs/enhanced_vad_segments.log", help="VAD分段日志文件路径 (默认: logs/enhanced_vad_segments.log)")
    args = parser.parse_args()
    
    # 检查文件是否存在
    audio_file = Path(args.audio_file)
    if not audio_file.exists():
        logger.error(f"音频文件不存在: {audio_file}")
        return
    
    # 如果未指定输出VTT文件路径，则使用默认路径
    output_vtt = args.output_vtt
    if not output_vtt:
        # 使用音频文件名作为VTT文件名
        output_vtt = str(audio_file.with_suffix(".enhanced.vtt"))
        logger.info(f"未指定输出VTT文件路径，使用默认路径: {output_vtt}")
    
    # 确保日志目录存在
    log_dir = os.path.dirname(args.segment_log)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 模拟WebSocket连接
    await simulate_websocket_connection(
        str(audio_file),
        language=args.language,
        output_vtt=output_vtt,
        segment_log_file=args.segment_log
    )


if __name__ == "__main__":
    asyncio.run(main())
