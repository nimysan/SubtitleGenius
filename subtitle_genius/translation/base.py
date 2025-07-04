"""
字幕翻译基础接口和数据类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class TranslationInput:
    """翻译输入"""
    text: str  # 待翻译文本
    source_language: str = "ar"  # 源语言
    target_language: str = "zh"  # 目标语言
    context: Optional[str] = None  # 上下文信息


@dataclass 
class TranslationOutput:
    """翻译输出"""
    original_text: str  # 原始文本
    translated_text: str  # 翻译后文本
    source_language: str  # 源语言
    target_language: str  # 目标语言
    confidence: float  # 翻译置信度 0-1
    service_name: str  # 使用的翻译服务名称
    translation_details: Optional[str] = None  # 翻译详情说明


class TranslationService(ABC):
    """翻译服务接口"""
    
    @abstractmethod
    async def translate(self, input_data: TranslationInput) -> TranslationOutput:
        """
        翻译文本
        
        Args:
            input_data: 翻译输入数据
            
        Returns:
            TranslationOutput: 翻译结果
        """
        pass
    
    def get_service_name(self) -> str:
        """获取服务名称"""
        return self.__class__.__name__
