"""
翻译工具函数
"""

from typing import Optional
from .base import TranslationInput, TranslationOutput
from .bedrock_translator import BedrockTranslator


async def translate_text(
    text: str,
    source_language: str = "ar",
    target_language: str = "zh",
    context: Optional[str] = None,
    service: str = "bedrock"
) -> str:
    """
    便捷的翻译函数
    
    Args:
        text: 待翻译文本
        source_language: 源语言
        target_language: 目标语言
        context: 上下文信息
        service: 翻译服务类型
        
    Returns:
        str: 翻译后的文本
    """
    
    # 根据服务类型选择翻译器
    if service == "bedrock":
        translator = BedrockTranslator()
    elif service == "google":
        from .google_translator import GoogleTranslator
        translator = GoogleTranslator()
    elif service == "openai":
        from .openai_translator import OpenAITranslator
        translator = OpenAITranslator()
    else:
        # 默认使用Bedrock
        translator = BedrockTranslator()
    
    input_data = TranslationInput(
        text=text,
        source_language=source_language,
        target_language=target_language,
        context=context
    )
    
    result = await translator.translate(input_data)
    return result.translated_text


async def batch_translate(
    texts: list[str],
    source_language: str = "ar",
    target_language: str = "zh",
    context: Optional[str] = None,
    service: str = "bedrock"
) -> list[str]:
    """
    批量翻译函数
    
    Args:
        texts: 待翻译文本列表
        source_language: 源语言
        target_language: 目标语言
        context: 上下文信息
        service: 翻译服务类型
        
    Returns:
        list[str]: 翻译后的文本列表
    """
    results = []
    
    for text in texts:
        translated = await translate_text(
            text=text,
            source_language=source_language,
            target_language=target_language,
            context=context,
            service=service
        )
        results.append(translated)
    
    return results
