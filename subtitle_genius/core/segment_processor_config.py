"""音频段处理器配置模块"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


def get_env_var(name, default):
    """从环境变量获取值，如果不存在则返回默认值"""
    return os.environ.get(name, default)


def get_env_bool(name, default):
    """从环境变量获取布尔值，如果不存在则返回默认值"""
    value = os.environ.get(name, str(default).lower())
    return value.lower() in ('true', '1', 'yes', 'y', 't')


def get_env_int(name, default):
    """从环境变量获取整数值，如果不存在则返回默认值"""
    try:
        return int(os.environ.get(name, default))
    except (ValueError, TypeError):
        return default


class AudioSegmentProcessorConfig:
    """音频段处理器配置类"""
    
    def __init__(self):
        # 转录配置
        self.transcribe_model = get_env_var("TRANSCRIBE_MODEL", "openai-whisper")
        self.source_language = get_env_var("SOURCE_LANGUAGE", "ar")
        
        # 校正配置
        self.correction_enabled = get_env_bool("CORRECTION_ENABLED", True)
        self.correction_service = get_env_var("CORRECTION_SERVICE", "bedrock")
        self.correction_model_id = get_env_var("CORRECTION_MODEL_ID", None)
        self.scene_description = get_env_var("SCENE_DESCRIPTION", "")
        
        # 翻译配置
        self.translation_enabled = get_env_bool("TRANSLATION_ENABLED", False)
        self.translation_service = get_env_var("TRANSLATION_SERVICE", "bedrock")
        self.target_language = get_env_var("TARGET_LANGUAGE", "zh")
        self.translation_model_id = get_env_var("TRANSLATION_MODEL_ID", None)
        
        # 输出配置
        self.output_format = get_env_var("OUTPUT_FORMAT", "srt")
        self.include_original = get_env_bool("INCLUDE_ORIGINAL", True)
        self.include_translation = get_env_bool("INCLUDE_TRANSLATION", True)
        
        # 音频段处理配置
        self.segment_duration_ms = get_env_int("SEGMENT_DURATION_MS", 1000)  # 音频段长度(毫秒)
        self.segment_overlap_ms = get_env_int("SEGMENT_OVERLAP_MS", 200)     # 音频段重叠(毫秒)
        self.min_segment_duration_ms = get_env_int("MIN_SEGMENT_DURATION_MS", 500)  # 最小音频段长度
    
    def get_correction_kwargs(self) -> dict:
        """获取校正服务的参数"""
        kwargs = {}
        if self.correction_model_id:
            kwargs["model_id"] = self.correction_model_id
        return kwargs
    
    def get_translation_kwargs(self) -> dict:
        """获取翻译服务的参数"""
        kwargs = {}
        if self.translation_model_id:
            kwargs["model_id"] = self.translation_model_id
        return kwargs
    
    def __str__(self):
        """返回配置的字符串表示"""
        return (
            f"AudioSegmentProcessorConfig("
            f"transcribe_model={self.transcribe_model}, "
            f"source_language={self.source_language}, "
            f"correction_enabled={self.correction_enabled}, "
            f"translation_enabled={self.translation_enabled})"
        )


# 全局音频段处理器配置实例
segment_processor_config = AudioSegmentProcessorConfig()
