"""
字幕翻译模块

提供多种翻译服务的统一接口
"""

from .base import TranslationService, TranslationInput, TranslationOutput
from .google_translator import GoogleTranslator
from .bedrock_translator import BedrockTranslator
from .utils import translate_text, batch_translate

__all__ = [
    'TranslationService',
    'TranslationInput',
    'TranslationOutput',
    'GoogleTranslator', 
    'BedrockTranslator',
    'translate_text',
    'batch_translate'
]

__version__ = "1.0.0"
