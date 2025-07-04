"""
字幕纠错模块

在transcribe之后、translate之前进行字幕纠错
"""

from .base import SubtitleCorrectionService, CorrectionInput, CorrectionOutput
from .basic_corrector import BasicCorrectionService
from .bedrock_corrector import BedrockCorrectionService
from .utils import correct_subtitle

__all__ = [
    'SubtitleCorrectionService',
    'CorrectionInput', 
    'CorrectionOutput',
    'BasicCorrectionService',
    'BedrockCorrectionService',
    'correct_subtitle'
]

__version__ = "1.0.0"
