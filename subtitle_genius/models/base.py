"""AI模型基类"""

from abc import ABC, abstractmethod
from typing import List, Any
from ..subtitle.models import Subtitle


class BaseModel(ABC):
    """AI模型基类"""
    
    @abstractmethod
    async def transcribe(self, audio_data: Any, language: str) -> List[Subtitle]:
        """转录音频为字幕"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查模型是否可用"""
        pass
