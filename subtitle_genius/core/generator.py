"""字幕生成器核心类"""

import asyncio
from typing import AsyncGenerator, List, Optional, Union
from pathlib import Path
from loguru import logger

from ..models.whisper_model import WhisperModel
from ..models.openai_model import OpenAIModel
from ..models.claude_model import ClaudeModel
from ..models.transcribe_model import TranscribeModel
from ..audio.processor import AudioProcessor
from ..subtitle.formatter import SubtitleFormatter
from ..subtitle.models import Subtitle
from .config import config


class SubtitleGenerator:
    """字幕生成器主类"""
    
    def __init__(
        self,
        model: str = "openai-whisper",
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
        if self.model_name == "openai-whisper":
            self.model = WhisperModel()
        elif self.model_name == "openai-gpt":
            self.model = OpenAIModel()
        elif self.model_name == "claude":
            self.model = ClaudeModel()
        elif self.model_name == "amazon-transcribe":
            self.model = TranscribeModel(region_name=config.aws_region)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    async def generate_from_file(self, file_path: Union[str, Path]) -> List[Subtitle]:
        """从音频文件生成字幕"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        logger.info(f"Processing audio file: {file_path}")
        
        # 处理音频文件
        audio_data = await self.audio_processor.process_file(file_path)
        
        # 生成字幕
        subtitles = await self.model.transcribe(audio_data, self.language)
        
        logger.info(f"Generated {len(subtitles)} subtitles")
        return subtitles
    
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
                    subtitles = await self.model.transcribe(
                        combined_audio, 
                        self.language
                    )
                    
                    # 返回生成的字幕
                    for subtitle in subtitles:
                        yield subtitle
                    
                    # 清空缓冲区
                    buffer = []
                    
                except Exception as e:
                    logger.error(f"Error in real-time processing: {e}")
                    continue
    
    def save_subtitles(
        self, 
        subtitles: List[Subtitle], 
        output_path: Union[str, Path]
    ) -> None:
        """保存字幕到文件"""
        output_path = Path(output_path)
        
        formatted_content = self.subtitle_formatter.format(
            subtitles, 
            self.output_format
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_content)
        
        logger.info(f"Subtitles saved to: {output_path}")
    
    async def process_video(
        self, 
        video_path: Union[str, Path], 
        output_path: Optional[Union[str, Path]] = None
    ) -> List[Subtitle]:
        """处理视频文件并生成字幕"""
        video_path = Path(video_path)
        
        if output_path is None:
            output_path = video_path.with_suffix(f'.{self.output_format}')
        
        # 从视频中提取音频
        audio_data = await self.audio_processor.extract_from_video(video_path)
        
        # 生成字幕
        subtitles = await self.model.transcribe(audio_data, self.language)
        
        # 保存字幕
        self.save_subtitles(subtitles, output_path)
        
        return subtitles
