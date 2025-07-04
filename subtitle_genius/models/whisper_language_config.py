"""
Whisper模型的语言特定配置
根据不同语言优化Whisper模型参数
"""

from dataclasses import dataclass
from typing import Dict, Any
from .whisper_sagemaker_streaming import WhisperSageMakerStreamConfig


@dataclass
class LanguageConfig:
    """语言特定配置"""
    voice_threshold: float
    chunk_duration: float
    overlap_duration: float
    min_silence_duration: float
    sagemaker_chunk_duration: int
    # SageMaker Whisper特定参数
    temperature: float = 0.0
    best_of: int = 1
    beam_size: int = 1
    patience: float = 1.0
    length_penalty: float = 1.0
    suppress_tokens: str = "-1"
    initial_prompt: str = ""
    condition_on_previous_text: bool = True
    fp16: bool = True
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6


# 语言特定配置映射
LANGUAGE_CONFIGS: Dict[str, LanguageConfig] = {
    "ar": LanguageConfig(  # 阿拉伯语
        voice_threshold=0.005,
        chunk_duration=25.0,
        overlap_duration=2.0,
        min_silence_duration=0.3,
        sagemaker_chunk_duration=25,
        temperature=0.0,
        initial_prompt="هذا نص باللغة العربية",  # 阿拉伯语提示
        no_speech_threshold=0.5,  # 阿拉伯语语音检测阈值
        compression_ratio_threshold=2.2,
        logprob_threshold=-0.8
    ),
    
    "en": LanguageConfig(  # 英语
        voice_threshold=0.01,
        chunk_duration=30.0,
        overlap_duration=3.0,
        min_silence_duration=0.4,
        sagemaker_chunk_duration=30,
        temperature=0.0,
        initial_prompt="This is English text",
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0
    ),
    
    "zh": LanguageConfig(  # 中文
        voice_threshold=0.008,
        chunk_duration=20.0,
        overlap_duration=2.5,
        min_silence_duration=0.2,
        sagemaker_chunk_duration=20,
        temperature=0.0,
        initial_prompt="这是中文文本",
        no_speech_threshold=0.5,
        compression_ratio_threshold=2.0,
        logprob_threshold=-0.9
    ),
    
    "fr": LanguageConfig(  # 法语
        voice_threshold=0.009,
        chunk_duration=28.0,
        overlap_duration=2.8,
        min_silence_duration=0.35,
        sagemaker_chunk_duration=28,
        temperature=0.0,
        initial_prompt="Ceci est un texte français",
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.3,
        logprob_threshold=-0.95
    ),
    
    "es": LanguageConfig(  # 西班牙语
        voice_threshold=0.009,
        chunk_duration=28.0,
        overlap_duration=2.8,
        min_silence_duration=0.35,
        sagemaker_chunk_duration=28,
        temperature=0.0,
        initial_prompt="Este es texto en español",
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.3,
        logprob_threshold=-0.95
    )
}

# 默认配置（用于未知语言）
DEFAULT_CONFIG = LanguageConfig(
    voice_threshold=0.01,
    chunk_duration=30.0,
    overlap_duration=3.0,
    min_silence_duration=0.4,
    sagemaker_chunk_duration=30,
    temperature=0.0,
    initial_prompt="",
    no_speech_threshold=0.6,
    compression_ratio_threshold=2.4,
    logprob_threshold=-1.0
)


def get_language_config(language: str) -> LanguageConfig:
    """获取指定语言的配置"""
    return LANGUAGE_CONFIGS.get(language.lower(), DEFAULT_CONFIG)


def create_whisper_config(language: str) -> WhisperSageMakerStreamConfig:
    """为指定语言创建Whisper流式配置"""
    lang_config = get_language_config(language)
    
    return WhisperSageMakerStreamConfig(
        chunk_duration=lang_config.chunk_duration,
        overlap_duration=lang_config.overlap_duration,
        voice_threshold=lang_config.voice_threshold,
        min_silence_duration=lang_config.min_silence_duration,
        sagemaker_chunk_duration=lang_config.sagemaker_chunk_duration
    )


def get_sagemaker_whisper_params(language: str) -> Dict[str, Any]:
    """获取SageMaker Whisper的语言特定参数"""
    lang_config = get_language_config(language)
    
    return {
        "temperature": lang_config.temperature,
        "best_of": lang_config.best_of,
        "beam_size": lang_config.beam_size,
        "patience": lang_config.patience,
        "length_penalty": lang_config.length_penalty,
        "suppress_tokens": lang_config.suppress_tokens,
        "initial_prompt": lang_config.initial_prompt,
        "condition_on_previous_text": lang_config.condition_on_previous_text,
        "fp16": lang_config.fp16,
        "compression_ratio_threshold": lang_config.compression_ratio_threshold,
        "logprob_threshold": lang_config.logprob_threshold,
        "no_speech_threshold": lang_config.no_speech_threshold
    }


def get_correction_scene_description(language: str) -> str:
    """根据语言获取纠错场景描述"""
    scene_descriptions = {
        "ar": "足球比赛解说",
        "en": "football match commentary",
        "zh": "足球比赛解说",
        "fr": "commentaire de match de football",
        "es": "comentario de partido de fútbol"
    }
    return scene_descriptions.get(language.lower(), "sports commentary")
