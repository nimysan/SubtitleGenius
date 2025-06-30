"""OpenAI GPT模型集成"""

import openai
from typing import List, Any
from loguru import logger

from .base import BaseModel
from ..subtitle.models import Subtitle
from ..core.config import config


class OpenAIModel(BaseModel):
    """OpenAI GPT模型"""
    
    def __init__(self):
        if not config.openai_api_key:
            raise ValueError("OpenAI API key not configured")
        
        openai.api_key = config.openai_api_key
        self.model = config.openai_model
        logger.info(f"OpenAI model '{self.model}' initialized")
    
    async def transcribe(self, audio_data: Any, language: str) -> List[Subtitle]:
        """使用OpenAI API进行音频转录"""
        try:
            # 使用OpenAI的Whisper API
            response = await openai.Audio.atranscribe(
                model="whisper-1",
                file=audio_data,
                language=language.split('-')[0],
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
            
            subtitles = []
            
            # 处理API响应
            for segment in response.get('segments', []):
                subtitle = Subtitle(
                    start=segment['start'],
                    end=segment['end'],
                    text=segment['text'].strip()
                )
                subtitles.append(subtitle)
            
            logger.info(f"OpenAI transcribed {len(subtitles)} segments")
            return subtitles
            
        except Exception as e:
            logger.error(f"OpenAI transcription failed: {e}")
            raise
    
    def is_available(self) -> bool:
        """检查OpenAI API是否可用"""
        return bool(config.openai_api_key)
