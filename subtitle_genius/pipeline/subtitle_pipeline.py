"""字幕处理流水线模块"""

from typing import List

from ..subtitle.models import Subtitle
from ..correction.base import CorrectionInput, CorrectionOutput
from ..correction.factory import create_corrector
from ..translation.utils import translate_text


class SubtitlePipeline:
    """
    字幕处理流水线
    
    维护字幕条目列表，提供校正和翻译功能，输出处理后的字幕
    """
    
    def __init__(
        self,
        source_language: str = "ar",
        target_language: str = "zh",
        correction_enabled: bool = True,
        correction_service: str = "bedrock",
        translation_enabled: bool = False,
        translation_service: str = "bedrock",
        scene_description: str = "",
        output_format: str = "srt"
    ):
        """
        初始化字幕处理流水线
        
        Args:
            source_language: 源语言
            target_language: 目标语言
            correction_enabled: 是否启用校正
            correction_service: 校正服务类型
            translation_enabled: 是否启用翻译
            translation_service: 翻译服务类型
            scene_description: 场景描述
            output_format: 输出格式
        """
        self.subtitles: List[Subtitle] = []
        self.source_language = source_language
        self.target_language = target_language
        self.correction_enabled = correction_enabled
        self.translation_enabled = translation_enabled
        self.scene_description = scene_description
        self.output_format = output_format
        
        # 初始化校正服务(如果启用)
        self.correction_service = None
        if self.correction_enabled:
            self.correction_service = create_corrector(
                service_type=correction_service,
                language=source_language
            )
        
        # 翻译服务类型
        self.translation_service = translation_service
        
        # 历史字幕，用于校正
        self.history_subtitles: List[str] = []
    
    def add_subtitle(self, subtitle: Subtitle) -> None:
        """
        添加字幕条目
        
        Args:
            subtitle: 字幕条目
        """
        self.subtitles.append(subtitle)
    
    
    def clear_subtitles(self) -> None:
        """清空字幕条目列表"""
        self.subtitles = []
        self.history_subtitles = []
    
    async def correct_subtitle(self, subtitle: Subtitle) -> Subtitle:
        """
        校正单个字幕
        
        Args:
            subtitle: 原始字幕
            
        Returns:
            Subtitle: 校正后的字幕
        """
        # 如果未启用校正或字幕为空，直接返回
        if not self.correction_enabled or not subtitle.text.strip() or not self.correction_service:
            return subtitle
        
        # 准备校正输入
        input_data = CorrectionInput(
            current_subtitle=subtitle.text,
            history_subtitles=self.history_subtitles[-5:],  # 最近5条历史字幕
            scene_description=self.scene_description,
            language=self.source_language
        )
        
        try:
            # 执行校正
            correction_result = await self.correction_service.correct(input_data)
            
            # 更新字幕文本
            subtitle.text = correction_result.corrected_subtitle
            
            # 添加到历史
            self.history_subtitles.append(subtitle.text)
            
        except Exception as e:
            print(f"Error in subtitle correction: {e}")
        
        return subtitle
    
    async def translate_subtitle(self, subtitle: Subtitle) -> Subtitle:
        """
        翻译单个字幕
        
        Args:
            subtitle: 字幕
            
        Returns:
            Subtitle: 翻译后的字幕
        """
        # 如果未启用翻译或字幕为空，直接返回
        if not self.translation_enabled or not subtitle.text.strip():
            return subtitle
        
        try:
            # 翻译
            translated_text = await translate_text(
                text=subtitle.text,
                source_language=self.source_language,
                target_language=self.target_language,
                context=self.scene_description,
                service=self.translation_service
            )
            
            # 更新字幕的翻译文本
            subtitle.translated_text = translated_text
            
        except Exception as e:
            print(f"Error in subtitle translation: {e}")
        
        return subtitle
    
    async def process_subtitle(self, subtitle: Subtitle) -> Subtitle:
        """
        处理单个字幕(校正和翻译)
        
        Args:
            subtitle: 原始字幕
            
        Returns:
            Subtitle: 处理后的字幕
        """
        # 校正
        if self.correction_enabled and self.correction_service:
            subtitle = await self.correct_subtitle(subtitle)
        
        # 翻译
        if self.translation_enabled:
            subtitle = await self.translate_subtitle(subtitle)
        
        return subtitle
