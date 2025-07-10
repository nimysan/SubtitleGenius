"""
SubtitleGenius - 基于GenAI的实时MP4音频流字幕生成工具
"""

__version__ = "0.1.0"
__author__ = "234aini@gmail.com"

from .core.generator import SubtitleGenerator
from .core.config import Config
from .core.segment_processor import AudioSegmentProcessor
from .core.segment_processor_config import AudioSegmentProcessorConfig, segment_processor_config

__all__ = ["SubtitleGenerator", "Config", "AudioSegmentProcessor", "AudioSegmentProcessorConfig", "segment_processor_config"]
