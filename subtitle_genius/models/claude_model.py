"""Anthropic Claude模型集成"""

import anthropic
from typing import List, Any
from loguru import logger

from .base import BaseModel
from ..subtitle.models import Subtitle
from ..core.config import config


class ClaudeModel(BaseModel):
    """Anthropic Claude模型"""
    
    def __init__(self):
        if not config.anthropic_api_key:
            raise ValueError("Anthropic API key not configured")
        
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        self.model = config.anthropic_model
        logger.info(f"Claude model '{self.model}' initialized")
    
    async def transcribe(self, audio_data: Any, language: str) -> List[Subtitle]:
        """
        注意: Claude目前不直接支持音频转录
        这里作为示例，实际使用时需要先用其他工具转录，然后用Claude优化文本
        """
        try:
            # 这里应该先使用其他工具(如Whisper)进行基础转录
            # 然后使用Claude进行文本优化和分段
            
            # 示例: 假设已有基础转录文本
            base_transcription = "这里是基础转录文本"  # 实际应从其他模型获取
            
            # 使用Claude优化字幕
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": f"""
                        请将以下转录文本优化为适合字幕显示的格式:
                        1. 修正语法错误
                        2. 添加标点符号
                        3. 按语义分段，每段不超过{config.max_subtitle_length}字符
                        4. 保持原意不变
                        
                        转录文本: {base_transcription}
                        
                        请以JSON格式返回，包含start, end, text字段。
                        """
                    }
                ]
            )
            
            # 解析Claude的响应并转换为Subtitle对象
            # 这里需要根据实际响应格式进行解析
            subtitles = []  # 实际实现时需要解析Claude的响应
            
            logger.info(f"Claude processed {len(subtitles)} segments")
            return subtitles
            
        except Exception as e:
            logger.error(f"Claude processing failed: {e}")
            raise
    
    def is_available(self) -> bool:
        """检查Claude API是否可用"""
        return bool(config.anthropic_api_key)
