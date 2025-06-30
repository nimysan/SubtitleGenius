"""字幕格式化器"""

from typing import List
from loguru import logger

from .models import Subtitle


class SubtitleFormatter:
    """字幕格式化器"""
    
    def format(self, subtitles: List[Subtitle], format_type: str) -> str:
        """格式化字幕列表"""
        if format_type.lower() == "srt":
            return self.to_srt(subtitles)
        elif format_type.lower() in ["vtt", "webvtt"]:
            return self.to_vtt(subtitles)
        else:
            raise ValueError(f"Unsupported subtitle format: {format_type}")
    
    def to_srt(self, subtitles: List[Subtitle]) -> str:
        """转换为SRT格式"""
        srt_content = []
        
        for i, subtitle in enumerate(subtitles, 1):
            srt_content.append(subtitle.to_srt(i))
        
        result = "\n".join(srt_content)
        logger.info(f"Formatted {len(subtitles)} subtitles to SRT format")
        return result
    
    def to_vtt(self, subtitles: List[Subtitle]) -> str:
        """转换为WebVTT格式"""
        vtt_content = ["WEBVTT\n"]
        
        for subtitle in subtitles:
            vtt_content.append(subtitle.to_vtt())
        
        result = "\n".join(vtt_content)
        logger.info(f"Formatted {len(subtitles)} subtitles to WebVTT format")
        return result
    
    def optimize_subtitles(self, subtitles: List[Subtitle]) -> List[Subtitle]:
        """优化字幕"""
        optimized = []
        
        for subtitle in subtitles:
            # 文本清理
            cleaned_text = self._clean_text(subtitle.text)
            
            # 长度检查和分割
            if len(cleaned_text) > 80:  # 假设最大长度为80字符
                split_subtitles = self._split_long_subtitle(subtitle, cleaned_text)
                optimized.extend(split_subtitles)
            else:
                optimized.append(Subtitle(
                    start=subtitle.start,
                    end=subtitle.end,
                    text=cleaned_text
                ))
        
        logger.info(f"Optimized {len(subtitles)} subtitles to {len(optimized)} subtitles")
        return optimized
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除多余空格
        text = " ".join(text.split())
        
        # 移除特殊字符
        text = text.replace("  ", " ")
        
        return text.strip()
    
    def _split_long_subtitle(self, subtitle: Subtitle, text: str) -> List[Subtitle]:
        """分割过长的字幕"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= 80:  # +1 for space
                current_chunk.append(word)
                current_length += len(word) + 1
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        # 为每个块创建字幕
        split_subtitles = []
        duration_per_chunk = subtitle.duration / len(chunks)
        
        for i, chunk in enumerate(chunks):
            start_time = subtitle.start + i * duration_per_chunk
            end_time = start_time + duration_per_chunk
            
            split_subtitles.append(Subtitle(
                start=start_time,
                end=end_time,
                text=chunk
            ))
        
        return split_subtitles
