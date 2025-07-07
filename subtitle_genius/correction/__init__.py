"""
字幕纠错模块

提供多语言字幕纠错服务，支持：
- Amazon Bedrock Claude纠错
- OpenAI LLM纠错  
- 基础规则纠错

支持语言：阿拉伯语(ar)、中文(zh)、英语(en)、西班牙语(es)等
在transcribe之后、translate之前进行字幕纠错
"""

from .base import SubtitleCorrectionService, CorrectionInput, CorrectionOutput
from .basic_corrector import BasicCorrectionService
from .bedrock_corrector import BedrockCorrectionService
from .llm_corrector import LLMCorrectionService
from .factory import CorrectionServiceFactory, create_corrector
from .utils import correct_subtitle

__all__ = [
    'SubtitleCorrectionService',
    'CorrectionInput', 
    'CorrectionOutput',
    'BasicCorrectionService',
    'BedrockCorrectionService',
    'LLMCorrectionService',
    'CorrectionServiceFactory',
    'create_corrector',
    'correct_subtitle'
]

__version__ = "1.1.0"
