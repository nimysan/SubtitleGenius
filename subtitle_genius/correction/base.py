"""
字幕纠错基础接口和数据类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CorrectionInput:
    """纠错输入"""
    current_subtitle: str  # 当前字幕词条
    history_subtitles: List[str]  # 同一视频的历史字幕词条
    scene_description: str  # 场景说明，如"足球比赛"、"新闻播报"
    language: str = "ar"  # 语言代码


@dataclass 
class CorrectionOutput:
    """纠错输出"""
    corrected_subtitle: str  # 纠正后的字幕词条
    has_correction: bool  # 是否进行了纠正
    confidence: float  # 纠正置信度 0-1
    correction_details: Optional[str] = None  # 纠正详情说明


class SubtitleCorrectionService(ABC):
    """字幕纠错服务接口"""
    
    @abstractmethod
    async def correct(self, input_data: CorrectionInput) -> CorrectionOutput:
        """
        纠正字幕
        
        Args:
            input_data: 纠错输入数据
            
        Returns:
            CorrectionOutput: 纠错结果
        """
        pass
    
    def get_service_name(self) -> str:
        """获取服务名称"""
        return self.__class__.__name__
