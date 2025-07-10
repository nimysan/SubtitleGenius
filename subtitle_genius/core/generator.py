"""字幕生成器核心类"""

import asyncio
from typing import AsyncGenerator, List, Optional, Union
from pathlib import Path
from loguru import logger

from ..models.whisper_sagemaker_streaming import WhisperSageMakerStreamingModel
from ..audio.processor import AudioProcessor
from ..subtitle.formatter import SubtitleFormatter
from ..subtitle.models import Subtitle
from .config import config


class SubtitleGenerator:
    """字幕生成器主类"""
    
    def __init__(
        self,
        model: str = "whisper-sagemaker-streaming",
        language: str = "zh-CN",
        output_format: str = "srt"
    ):
        self.model_name = model
        self.language = language
        self.output_format = output_format
        
        # 初始化组件
        self.audio_processor = AudioProcessor()
        self.subtitle_formatter = SubtitleFormatter()
        
        # 初始化AI模型
        self._init_model()
        
        logger.info(f"SubtitleGenerator initialized with model: {model}")
    
    def _init_model(self) -> None:
        """初始化AI模型"""
        if self.model_name == "whisper-sagemaker-streaming":
             self.model = WhisperSageMakerStreamingModel(
                endpoint_name=config.sagemaker_endpoint_name,
                region_name=config.aws_region
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
    async def generate_realtime(
        self, 
        audio_stream
    ) -> AsyncGenerator[Subtitle, None]:
        """实时生成字幕"""
        logger.info("Starting real-time subtitle generation")
        
        buffer = []
        
        async for audio_chunk in audio_stream:
            buffer.append(audio_chunk)
            
            # 当缓冲区达到指定大小时处理
            if len(buffer) >= config.real_time_buffer_size:
                try:
                    # 合并音频块
                    combined_audio = self.audio_processor.combine_chunks(buffer)
                    
                    # 生成字幕
                    if hasattr(self.model, 'transcribe'):
                        subtitles = await self.model.transcribe(
                            combined_audio, 
                            self.language
                        )
                    elif hasattr(self.model, 'transcribe_audio'):
                        # 使用transcribe_audio方法
                        text = await self.model.transcribe_audio(combined_audio, self.language)
                        # 创建单个字幕
                        duration = len(combined_audio) / self.audio_processor.sample_rate
                        subtitles = [Subtitle(start=0, end=duration, text=text)] if text else []
                    else:
                        logger.error(f"Model {self.model_name} does not implement transcribe or transcribe_audio method")
                        subtitles = []
                    
                    # 返回生成的字幕
                    for subtitle in subtitles:
                        yield subtitle
                    
                    # 清空缓冲区
                    buffer = []
                    
                except Exception as e:
                    logger.error(f"Error in real-time processing: {e}")
                    continue