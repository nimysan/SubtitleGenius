"""
Whisper 流式处理模型
实现基于缓冲区的伪流式处理
"""

import asyncio
import numpy as np
import whisper
from typing import AsyncGenerator, Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import logging
from dataclasses import dataclass
from collections import deque
import time

from ..subtitle.models import Subtitle

logger = logging.getLogger(__name__)


@dataclass
class WhisperStreamConfig:
    """Whisper 流式处理配置"""
    chunk_duration: float = 3.0      # 每次处理的音频长度(秒)
    overlap_duration: float = 0.5    # 重叠时间(秒)
    sample_rate: int = 16000         # 采样率
    min_silence_duration: float = 0.3 # 最小静音时长
    voice_threshold: float = 0.01    # 语音活动检测阈值
    max_buffer_size: int = 10        # 最大缓冲区大小


class WhisperStreamBuffer:
    """Whisper 流式缓冲区管理"""
    
    def __init__(self, config: WhisperStreamConfig):
        self.config = config
        self.chunk_size = int(config.chunk_duration * config.sample_rate)
        self.overlap_size = int(config.overlap_duration * config.sample_rate)
        
        self.buffer = deque(maxlen=config.max_buffer_size * self.chunk_size)
        self.processed_samples = 0
        self.last_result: Optional[Dict] = None
        self.start_time = time.time()
        
    def add_chunk(self, audio_chunk: np.ndarray):
        """添加音频块到缓冲区"""
        self.buffer.extend(audio_chunk.flatten())
        
    def ready_for_processing(self) -> bool:
        """检查是否准备好处理"""
        return len(self.buffer) >= self.chunk_size
        
    def get_processing_chunk(self) -> np.ndarray:
        """获取待处理的音频块"""
        if len(self.buffer) < self.chunk_size:
            return None
            
        # 提取音频数据
        audio_data = np.array(list(self.buffer)[:self.chunk_size], dtype=np.float32)
        return audio_data
        
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


class WhisperStreamingModel:
    """Whisper 流式处理模型"""
    
    def __init__(self, 
                 model_name: str = "base",
                 config: Optional[WhisperStreamConfig] = None,
                 device: str = "cpu"):
        self.model_name = model_name
        self.config = config or WhisperStreamConfig()
        self.device = device
        
        # 加载 Whisper 模型
        logger.info(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name, device=device)
        
        # 线程池用于异步处理
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    async def transcribe_stream(self, 
                              audio_stream: AsyncGenerator[np.ndarray, None],
                              language: str = "ar") -> AsyncGenerator[Subtitle, None]:
        """流式转录音频"""
        buffer = WhisperStreamBuffer(self.config)
        
        logger.info(f"Starting Whisper streaming transcription for language: {language}")
        
        try:
            async for audio_chunk in audio_stream:
                # 添加音频到缓冲区
                buffer.add_chunk(audio_chunk)
                
                # 当缓冲区准备好时处理
                while buffer.ready_for_processing():
                    audio_data = buffer.get_processing_chunk()
                    
                    if audio_data is None:
                        break
                        
                    # 检测语音活动
                    if not buffer.is_voice_active(audio_data):
                        buffer.advance()
                        continue
                    
                    # 异步转录
                    result = await self._async_transcribe(audio_data, language)
                    
                    if result and result.get('text', '').strip():
                        # 过滤重叠内容
                        new_text = self._filter_overlap(result, buffer.last_result)
                        
                        if new_text.strip():
                            subtitle = Subtitle(
                                start=buffer.get_current_time(),
                                end=buffer.get_current_time() + self.config.chunk_duration,
                                text=new_text.strip()
                            )
                            
                            logger.debug(f"Generated subtitle: {subtitle}")
                            yield subtitle
                        
                        buffer.last_result = result
                    
                    buffer.advance()
                    
        except Exception as e:
            logger.error(f"Error in Whisper streaming: {e}")
            raise
        finally:
            # 处理剩余缓冲区内容
            await self._process_remaining_buffer(buffer, language)
            
    async def _async_transcribe(self, audio_data: np.ndarray, language: str) -> Dict[str, Any]:
        """异步转录音频数据"""
        loop = asyncio.get_event_loop()
        
        try:
            result = await loop.run_in_executor(
                self.executor,
                self._transcribe_chunk,
                audio_data,
                language
            )
            return result
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {}
            
    def _transcribe_chunk(self, audio_data: np.ndarray, language: str) -> Dict[str, Any]:
        """转录音频块"""
        try:
            # 预处理音频
            audio_data = self._preprocess_audio(audio_data)
            
            # Whisper 转录
            result = self.model.transcribe(
                audio_data,
                language=language,
                task="transcribe",
                fp16=False,
                verbose=False
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Chunk transcription error: {e}")
            return {}
            
    def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """预处理音频数据"""
        # 归一化
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
            
        # 简单降噪（可以根据需要增强）
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        return audio_data
        
    def _filter_overlap(self, current_result: Dict, previous_result: Optional[Dict]) -> str:
        """过滤重叠内容"""
        current_text = current_result.get('text', '').strip()
        
        if not previous_result:
            return current_text
            
        previous_text = previous_result.get('text', '').strip()
        
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
        
    async def _process_remaining_buffer(self, buffer: WhisperStreamBuffer, language: str):
        """处理剩余缓冲区内容"""
        if len(buffer.buffer) > 0:
            remaining_audio = np.array(list(buffer.buffer), dtype=np.float32)
            
            if buffer.is_voice_active(remaining_audio):
                result = await self._async_transcribe(remaining_audio, language)
                
                if result and result.get('text', '').strip():
                    new_text = self._filter_overlap(result, buffer.last_result)
                    
                    if new_text.strip():
                        subtitle = Subtitle(
                            start=buffer.get_current_time(),
                            end=buffer.get_current_time() + len(remaining_audio) / buffer.config.sample_rate,
                            text=new_text.strip()
                        )
                        logger.debug(f"Final subtitle: {subtitle}")
                        
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
