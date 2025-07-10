"""AI模型集成模块"""

from .base import BaseModel
from .whisper_sagemaker_streaming import WhisperSageMakerStreamingModel

__all__ = [
    'WhisperSageMakerStreamingModel'
]
