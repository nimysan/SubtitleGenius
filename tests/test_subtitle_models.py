"""测试字幕模型"""

import pytest
from subtitle_genius.subtitle.models import Subtitle


class TestSubtitle:
    """测试字幕模型"""
    
    def test_valid_subtitle(self):
        """测试有效字幕"""
        subtitle = Subtitle(start=0.0, end=2.0, text="Hello world")
        assert subtitle.start == 0.0
        assert subtitle.end == 2.0
        assert subtitle.text == "Hello world"
        assert subtitle.duration == 2.0
    
    def test_invalid_start_time(self):
        """测试无效开始时间"""
        with pytest.raises(ValueError, match="Start time cannot be negative"):
            Subtitle(start=-1.0, end=2.0, text="Hello world")
    
    def test_invalid_end_time(self):
        """测试无效结束时间"""
        with pytest.raises(ValueError, match="End time must be greater than start time"):
            Subtitle(start=2.0, end=1.0, text="Hello world")
    
    def test_empty_text(self):
        """测试空文本"""
        with pytest.raises(ValueError, match="Subtitle text cannot be empty"):
            Subtitle(start=0.0, end=2.0, text="")
    
    def test_format_time_srt(self):
        """测试SRT时间格式"""
        subtitle = Subtitle(start=0.0, end=2.0, text="Hello world")
        formatted = subtitle.format_time(3661.5, "srt")
        assert formatted == "01:01:01,500"
    
    def test_format_time_vtt(self):
        """测试VTT时间格式"""
        subtitle = Subtitle(start=0.0, end=2.0, text="Hello world")
        formatted = subtitle.format_time(3661.5, "vtt")
        assert formatted == "01:01:01.500"
    
    def test_to_srt(self):
        """测试转换为SRT格式"""
        subtitle = Subtitle(start=0.0, end=2.0, text="Hello world")
        srt = subtitle.to_srt(1)
        expected = "1\n00:00:00,000 --> 00:00:02,000\nHello world\n"
        assert srt == expected
    
    def test_to_vtt(self):
        """测试转换为VTT格式"""
        subtitle = Subtitle(start=0.0, end=2.0, text="Hello world")
        vtt = subtitle.to_vtt()
        expected = "00:00:00.000 --> 00:00:02.000\nHello world\n"
        assert vtt == expected
