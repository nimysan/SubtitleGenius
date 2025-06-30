"""OpenAI Whisper模型集成"""

import whisper
from typing import List, Any
from loguru import logger

from .base import BaseModel
from ..subtitle.models import Subtitle
from ..core.config import config


class WhisperModel(BaseModel):
    """OpenAI Whisper模型"""
    
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """加载Whisper模型"""
        try:
            self.model = whisper.load_model(self.model_size)
            logger.info(f"Whisper model '{self.model_size}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    async def transcribe(self, audio_data: Any, language: str) -> List[Subtitle]:
        """使用Whisper转录音频"""
        if not self.model:
            raise RuntimeError("Whisper model not loaded")
        
        try:
            # 使用Whisper进行转录
            result = self.model.transcribe(
                audio_data,
                language=language.split('-')[0],  # 'zh-CN' -> 'zh'
                word_timestamps=True
            )
            
            subtitles = []
            
            # 处理转录结果
            for segment in result.get('segments', []):
                subtitle = Subtitle(
                    start=segment['start'],
                    end=segment['end'],
                    text=segment['text'].strip()
                )
                subtitles.append(subtitle)
            
            logger.info(f"Whisper transcribed {len(subtitles)} segments")
            return subtitles
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            raise
    
    def is_available(self) -> bool:
        """检查Whisper模型是否可用"""
        return self.model is not None
