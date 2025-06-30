"""测试字幕生成器"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path

from subtitle_genius.core.generator import SubtitleGenerator
from subtitle_genius.subtitle.models import Subtitle


class TestSubtitleGenerator:
    """测试字幕生成器"""
    
    def test_init(self):
        """测试初始化"""
        generator = SubtitleGenerator()
        assert generator.model_name == "openai-whisper"
        assert generator.language == "zh-CN"
        assert generator.output_format == "srt"
    
    def test_init_with_params(self):
        """测试带参数初始化"""
        generator = SubtitleGenerator(
            model="openai-gpt",
            language="en-US",
            output_format="vtt"
        )
        assert generator.model_name == "openai-gpt"
        assert generator.language == "en-US"
        assert generator.output_format == "vtt"
    
    @pytest.mark.asyncio
    async def test_generate_from_file_not_found(self):
        """测试文件不存在的情况"""
        generator = SubtitleGenerator()
        
        with pytest.raises(FileNotFoundError):
            await generator.generate_from_file("nonexistent.wav")
    
    def test_save_subtitles(self, tmp_path):
        """测试保存字幕"""
        generator = SubtitleGenerator()
        
        subtitles = [
            Subtitle(start=0.0, end=2.0, text="Hello world"),
            Subtitle(start=2.0, end=4.0, text="This is a test")
        ]
        
        output_file = tmp_path / "test.srt"
        generator.save_subtitles(subtitles, output_file)
        
        assert output_file.exists()
        content = output_file.read_text(encoding='utf-8')
        assert "Hello world" in content
        assert "This is a test" in content
