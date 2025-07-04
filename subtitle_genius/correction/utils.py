"""
字幕纠错工具函数
"""

from typing import List, Optional
from .base import CorrectionInput, CorrectionOutput
from .basic_corrector import BasicCorrectionService


async def correct_subtitle(
    current_subtitle: str,
    history_subtitles: List[str] = None,
    scene_description: str = "通用",
    language: str = "ar"
) -> str:
    """
    便捷的字幕纠错函数
    
    Args:
        current_subtitle: 当前字幕
        history_subtitles: 历史字幕列表
        scene_description: 场景描述
        language: 语言代码
        
    Returns:
        str: 纠正后的字幕
    """
    service = BasicCorrectionService()
    
    input_data = CorrectionInput(
        current_subtitle=current_subtitle,
        history_subtitles=history_subtitles or [],
        scene_description=scene_description,
        language=language
    )
    
    result = await service.correct(input_data)
    return result.corrected_subtitle
