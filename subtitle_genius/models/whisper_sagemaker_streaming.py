"""
基于 SageMaker Whisper 的流式处理封装
将现有的 SageMaker Whisper 实现封装为流式处理模式
"""

import asyncio
import numpy as np
import io
import wave
import time
from typing import AsyncGenerator, Optional, List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import logging
from dataclasses import dataclass
from collections import deque

# 导入现有的 SageMaker Whisper 客户端
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from sagemaker_whisper import WhisperSageMakerClient, chunk_audio

from ..subtitle.models import Subtitle
from ..audio.vad_processor import SileroVADProcessor, VADConfig

logger = logging.getLogger(__name__)


@dataclass
class WhisperSageMakerStreamConfig:
    """SageMaker Whisper 流式处理配置"""
    chunk_duration: float = 30      # 每次处理的音频长度(秒)
    overlap_duration: float = 3    # 重叠时间(秒)
    sample_rate: int = 16000         # 采样率
    min_silence_duration: float = 0.3 # 最小静音时长
    voice_threshold: float = 0.01    # 语音活动检测阈值
    max_buffer_size: int = 10        # 最大缓冲区大小
    sagemaker_chunk_duration: int = 30  # SageMaker 处理的块大小(秒)
    
    # VAD相关配置
    use_vad: bool = True            # 是否使用VAD
    vad_threshold: float = 0.5      # VAD置信度阈值
    vad_min_speech_duration_ms: int = 250  # 最小语音持续时间(毫秒)
    vad_min_silence_duration_ms: int = 100  # 最小静音持续时间(毫秒)
    vad_window_size_samples: int = 1536  # VAD窗口大小
    vad_speech_pad_ms: int = 30     # 语音片段前后填充(毫秒)
    vad_max_merge_gap_ms: int = 500  # 最大合并间隔(毫秒)


class WhisperSageMakerStreamBuffer:
    """SageMaker Whisper 流式缓冲区管理"""
    
    def __init__(self, config: WhisperSageMakerStreamConfig):
        self.config = config
        self.chunk_size = int(config.chunk_duration * config.sample_rate)
        self.overlap_size = int(config.overlap_duration * config.sample_rate)
        
        self.buffer = deque(maxlen=config.max_buffer_size * self.chunk_size)
        self.processed_samples = 0
        self.last_result: Optional[Dict] = None
        self.start_time = time.time()
        
    def add_chunk(self, audio_chunk: np.ndarray):
        """添加音频块到缓冲区"""
        # 确保音频数据是 float32 格式
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        
        self.buffer.extend(audio_chunk.flatten())
        
    def ready_for_processing(self) -> bool:
        """检查是否准备好处理"""
        return len(self.buffer) >= self.chunk_size
        
    def get_processing_chunk(self) -> Optional[bytes]:
        """获取待处理的音频块，转换为 WAV 格式"""
        if len(self.buffer) < self.chunk_size:
            return None
            
        # 提取音频数据
        audio_data = np.array(list(self.buffer)[:self.chunk_size], dtype=np.float32)
        
        # 转换为 WAV 格式
        wav_data = self._convert_to_wav(audio_data)
        return wav_data
        
    def _convert_to_wav(self, audio_data: np.ndarray) -> bytes:
        """将音频数据转换为 WAV 格式"""
        # 转换为 16-bit PCM
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # 创建 WAV 文件
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.config.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        return wav_buffer.getvalue()
        
    def advance(self):
        """推进缓冲区，移除已处理的部分"""
        advance_size = self.chunk_size - self.overlap_size
        
        # 移除已处理的音频（保留重叠部分）
        for _ in range(min(advance_size, len(self.buffer))):
            self.buffer.popleft()
            
        self.processed_samples += advance_size
        
    def get_current_time(self) -> float:
        """获取当前处理时间点"""
        return self.processed_samples / self.config.sample_rate
        
    def is_voice_active(self, audio_data: np.ndarray) -> bool:
        """检测语音活动"""
        if audio_data is None or len(audio_data) == 0:
            return False
            
        # 简单的能量检测
        energy = np.mean(np.abs(audio_data))
        return energy > self.config.voice_threshold


class WhisperSageMakerStreamingModel:
    """基于 SageMaker 的 Whisper 流式处理模型"""
    
    def __init__(self, 
                 endpoint_name: str,
                 region_name: str = "us-east-1",
                 config: Optional[WhisperSageMakerStreamConfig] = None):
        self.endpoint_name = endpoint_name
        self.region_name = region_name
        self.config = config or WhisperSageMakerStreamConfig()
        
        # 初始化 SageMaker 客户端
        logger.info(f"Initializing SageMaker Whisper client: {endpoint_name}")
        self.sagemaker_client = WhisperSageMakerClient(
            endpoint_name=endpoint_name,
            region_name=region_name
        )
        
        # 线程池用于异步处理
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    async def transcribe_audio(self, audio_data: np.ndarray, language: str = "zh") -> str:
        """
        转录单个音频片段
        
        参数:
            audio_data: 音频数据，numpy数组
            language: 语言代码
            
        返回:
            转录文本
        """
        try:
            # 确保音频数据是float32格式
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # 转换为WAV格式
            # 转换为16-bit PCM
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # 创建WAV文件
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # 单声道
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.config.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            wav_data = wav_buffer.getvalue()
            
            # 异步转录
            result = await self._async_transcribe(wav_data, language)
            
            if result and result.get('transcription'):
                return result.get('transcription', '').strip()
            else:
                return ""
                
        except Exception as e:
            logger.error(f"转录音频失败: {e}")
            return ""
    
    async def transcribe_stream(self, 
                              audio_stream: AsyncGenerator[np.ndarray, None],
                              language: str = "ar") -> AsyncGenerator[Subtitle, None]:
        """流式转录音频"""
        logger.info(f"Starting SageMaker Whisper streaming transcription for language: {language}")
        
        try:
            # 直接处理传入的音频流
            # 这里假设音频流已经经过VAD处理，每个块都是一个语音片段
            async for audio_chunk in audio_stream:
                # 确保音频数据是float32格式
                if audio_chunk.dtype != np.float32:
                    audio_chunk = audio_chunk.astype(np.float32)
                
                # 转换为WAV格式
                # 转换为16-bit PCM
                audio_int16 = (audio_chunk * 32767).astype(np.int16)
                
                # 创建WAV文件
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # 单声道
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(self.config.sample_rate)
                    wav_file.writeframes(audio_int16.tobytes())
                
                wav_data = wav_buffer.getvalue()
                
                # 异步转录
                result = await self._async_transcribe(wav_data, language)
                
                if result and result.get('transcription'):
                    text = result.get('transcription', '').strip()
                    
                    if text:
                        # 创建字幕对象（时间戳将在外部设置）
                        subtitle = Subtitle(
                            start=0,  # 将在外部设置
                            end=0,    # 将在外部设置
                            text=text
                        )
                        
                        logger.debug(f"生成字幕: {subtitle}")
                        yield subtitle
                    
        except Exception as e:
            logger.error(f"Error in SageMaker Whisper streaming: {e}")
            raise
            
    async def _async_transcribe(self, wav_data: bytes, language: str) -> Dict[str, Any]:
        """异步转录音频数据"""
        loop = asyncio.get_event_loop()
        
        try:
            result = await loop.run_in_executor(
                self.executor,
                self._transcribe_wav_data,
                wav_data,
                language
            )
            return result
        except Exception as e:
            logger.error(f"SageMaker transcription error: {e}")
            return {}
            
    def _transcribe_wav_data(self, wav_data: bytes, language: str) -> Dict[str, Any]:
        """使用 SageMaker 转录 WAV 数据"""
        try:
            # 将 WAV 数据分块（如果需要）
            chunks = chunk_audio(wav_data, self.config.sagemaker_chunk_duration)
            
            if not chunks:
                return {}
            
            # 处理第一个块（对于流式处理，通常只有一个小块）
            chunk_data = chunks[0]
            
            # 调用 SageMaker 端点
            from sagemaker_whisper import transcribe_chunk
            result = transcribe_chunk(
                self.sagemaker_client.sagemaker_runtime,
                chunk_data,
                self.endpoint_name,
                language=self.sagemaker_client._convert_language_code(language),
                task="transcribe"
            )
            
            # 添加调试信息
            logger.debug(f"SageMaker result type: {type(result)}")
            logger.debug(f"SageMaker result: {result}")
            
            # 提取文本
            if isinstance(result, dict):
                if 'text' in result:
                    text = result['text']
                    # 处理文本可能是列表的情况
                    if isinstance(text, list):
                        text = ' '.join(str(item) for item in text)
                    elif not isinstance(text, str):
                        text = str(text)
                elif 'transcription' in result:
                    text = result['transcription']
                    if isinstance(text, list):
                        text = ' '.join(str(item) for item in text)
                    elif not isinstance(text, str):
                        text = str(text)
                else:
                    # 尝试从其他可能的字段提取文本
                    text = str(result)
                
                return {
                    'transcription': text,
                    'raw_result': result
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"SageMaker chunk transcription error: {e}")
            return {}
            
    def _filter_overlap(self, current_result: Dict, previous_result: Optional[Dict]) -> str:
        """过滤重叠内容"""
        current_text = current_result.get('transcription', '').strip()
        
        if not previous_result:
            return current_text
            
        previous_text = previous_result.get('transcription', '').strip()
        
        if not previous_text or not current_text:
            return current_text
            
        # 简单的重叠检测和过滤
        current_words = current_text.split()
        previous_words = previous_text.split()
        
        # 查找重叠点
        overlap_point = self._find_text_overlap(current_words, previous_words)
        
        # 返回新增部分
        new_words = current_words[overlap_point:]
        return ' '.join(new_words)
        
    def _find_text_overlap(self, current_words: List[str], previous_words: List[str]) -> int:
        """查找文本重叠点"""
        if not previous_words:
            return 0
            
        # 从后往前查找重叠
        max_overlap = min(len(current_words), len(previous_words), 5)  # 最多检查5个词
        
        for i in range(max_overlap, 0, -1):
            if current_words[:i] == previous_words[-i:]:
                return i
                
        return 0
        
    async def _process_remaining_buffer(self, buffer: WhisperSageMakerStreamBuffer, language: str) -> Optional[Subtitle]:
        """处理剩余缓冲区内容，返回生成的字幕（如果有）"""
        if len(buffer.buffer) <= 0:
            return None
            
        remaining_audio = np.array(list(buffer.buffer), dtype=np.float32)
        
        # 使用VAD处理剩余音频
        if self.config.use_vad:
            vad_config = VADConfig(
                sample_rate=self.config.sample_rate,
                threshold=self.config.vad_threshold,
                min_speech_duration_ms=self.config.vad_min_speech_duration_ms,
                min_silence_duration_ms=self.config.vad_min_silence_duration_ms,
                window_size_samples=self.config.vad_window_size_samples,
                speech_pad_ms=self.config.vad_speech_pad_ms
            )
            vad_processor = SileroVADProcessor(vad_config)
            
            if vad_processor.is_available():
                speech_segments, speech_timestamps = vad_processor.extract_speech_segments(remaining_audio)
                
                # 如果检测到语音片段
                if speech_segments:
                    # 处理最后一个语音片段
                    segment = speech_segments[-1]
                    timestamp = speech_timestamps[-1]
                    
                    # 计算片段时间
                    segment_start = buffer.get_current_time() + timestamp['start'] / self.config.sample_rate
                    segment_end = buffer.get_current_time() + timestamp['end'] / self.config.sample_rate
                    
                    # 转换为WAV格式
                    wav_data = buffer._convert_to_wav(segment)
                    
                    # 异步转录
                    result = await self._async_transcribe(wav_data, language)
                    
                    if result and result.get('transcription'):
                        new_text = self._filter_overlap(result, buffer.last_result)
                        
                        if new_text.strip():
                            subtitle = Subtitle(
                                start=segment_start,
                                end=segment_end,
                                text=new_text.strip()
                            )
                            logger.info(f"最终字幕: {subtitle}")
                            return subtitle
                return None
        
        # 如果VAD不可用或未启用，使用原有的处理逻辑
        if buffer.is_voice_active(remaining_audio):
            wav_data = buffer._convert_to_wav(remaining_audio)
            result = await self._async_transcribe(wav_data, language)
            
            if result and result.get('transcription'):
                new_text = self._filter_overlap(result, buffer.last_result)
                
                if new_text.strip():
                    # 计算实际的结束时间
                    end_time = buffer.get_current_time() + len(remaining_audio) / self.config.sample_rate
                    
                    subtitle = Subtitle(
                        start=buffer.get_current_time(),
                        end=end_time,
                        text=new_text.strip()
                    )
                    logger.info(f"最终字幕: {subtitle}")
                    return subtitle
        
        return None  # 如果没有生成字幕，返回None
                        
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
