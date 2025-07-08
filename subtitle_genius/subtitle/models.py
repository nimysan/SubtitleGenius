"""字幕数据模型"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class Subtitle:
    """字幕条目"""
    start: float  # 开始时间(秒)
    end: float    # 结束时间(秒)
    text: str     # 字幕文本
    translated_text: Optional[str] = None  # 翻译后的文本
    
    def __post_init__(self):
        """验证数据"""
        if self.start < 0:
            raise ValueError("Start time cannot be negative")
        # if self.end <= self.start:
        #     raise ValueError("End time must be greater than start time")
        if not self.text.strip():
            raise ValueError("Subtitle text cannot be empty")
    
    @property
    def duration(self) -> float:
        """字幕持续时间"""
        return self.end - self.start
    
    def format_time(self, time_seconds: float, format_type: str = "srt") -> str:
        """格式化时间"""
        hours = int(time_seconds // 3600)
        minutes = int((time_seconds % 3600) // 60)
        seconds = int(time_seconds % 60)
        milliseconds = int((time_seconds % 1) * 1000)
        
        if format_type == "srt":
            return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
        elif format_type == "vtt":
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
        else:
            raise ValueError(f"Unsupported time format: {format_type}")
    
    def to_srt(self, index: int) -> str:
        """转换为SRT格式"""
        start_time = self.format_time(self.start, "srt")
        end_time = self.format_time(self.end, "srt")
        return f"{index}\n{start_time} --> {end_time}\n{self.text}\n"
    
    def to_vtt(self) -> str:
        """转换为WebVTT格式"""
        start_time = self.format_time(self.start, "vtt")
        end_time = self.format_time(self.end, "vtt")
        return f"{start_time} --> {end_time}\n{self.text}\n"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "translated_text": self.translated_text
        }
