"""
Whisper Turbo SageMaker API 配置文件
"""

import os
from typing import Dict, Any

class WhisperConfig:
    """Whisper SageMaker配置管理类"""
    
    # 🔧 在这里配置你的SageMaker端点信息
    DEFAULT_ENDPOINT_NAME = "endpoint-quick-start-z9afg"  # 替换为你的实际SageMaker端点名称
    DEFAULT_REGION = "us-east-1"  # 替换为你的AWS区域
    
    # 支持的语言配置
    SUPPORTED_LANGUAGES = {
        "ar": "Arabic (阿拉伯语)",
        "ar-SA": "Arabic (Saudi Arabia)",
        "ar-AE": "Arabic (UAE)",
        "en": "English (英语)",
        "en-US": "English (US)",
        "en-GB": "English (UK)",
        "zh": "Chinese (中文)",
        "zh-CN": "Chinese (Simplified)",
        "ja": "Japanese (日语)",
        "ko": "Korean (韩语)",
        "fr": "French (法语)",
        "de": "German (德语)",
        "es": "Spanish (西班牙语)",
        "ru": "Russian (俄语)"
    }
    
    # Whisper语言代码映射
    WHISPER_LANGUAGE_MAP = {
        "ar": "arabic",
        "ar-SA": "arabic", 
        "ar-AE": "arabic",
        "en": "english",
        "en-US": "english",
        "en-GB": "english",
        "zh": "chinese",
        "zh-CN": "chinese",
        "ja": "japanese",
        "ko": "korean",
        "fr": "french",
        "de": "german",
        "es": "spanish",
        "ru": "russian"
    }
    
    # 默认参数
    DEFAULT_PARAMS = {
        "chunk_duration": 30,  # 音频分块时长（秒）
        "task": "transcribe",  # 或 "translate"
        "top_p": 0.9
    }
    
    # 音频格式支持
    SUPPORTED_AUDIO_FORMATS = {
        ".wav": "wav",
        ".mp3": "mp3", 
        ".m4a": "m4a",
        ".aac": "aac",
        ".flac": "flac",
        ".ogg": "ogg"
    }
    
    # SageMaker配置
    SAGEMAKER_CONFIG = {
        "max_payload_size": 2 * 1024 * 1024,  # 2MB安全限制
        "content_type": "application/json",
        "accept": "application/json"
    }
    
    @classmethod
    def get_endpoint_name(cls) -> str:
        """获取SageMaker端点名称，优先使用环境变量"""
        return os.getenv("WHISPER_ENDPOINT_NAME", cls.DEFAULT_ENDPOINT_NAME)
    
    @classmethod
    def get_region(cls) -> str:
        """获取AWS区域，优先使用环境变量"""
        return os.getenv("AWS_REGION", cls.DEFAULT_REGION)
    
    @classmethod
    def get_whisper_language(cls, language_code: str) -> str:
        """转换为Whisper支持的语言代码"""
        return cls.WHISPER_LANGUAGE_MAP.get(language_code, "english")
    
    @classmethod
    def get_language_name(cls, language_code: str) -> str:
        """获取语言显示名称"""
        return cls.SUPPORTED_LANGUAGES.get(language_code, language_code)
    
    @classmethod
    def is_language_supported(cls, language_code: str) -> bool:
        """检查语言是否支持"""
        return language_code in cls.SUPPORTED_LANGUAGES
    
    @classmethod
    def get_audio_format(cls, file_extension: str) -> str:
        """根据文件扩展名获取音频格式"""
        return cls.SUPPORTED_AUDIO_FORMATS.get(file_extension.lower(), "wav")
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """验证配置并返回状态"""
        config_status = {
            "valid": True,
            "issues": [],
            "config": {}
        }
        
        # 检查端点名称
        endpoint_name = cls.get_endpoint_name()
        if endpoint_name == "your-whisper-turbo-endpoint":
            config_status["valid"] = False
            config_status["issues"].append("请设置实际的SageMaker端点名称")
        
        config_status["config"]["endpoint_name"] = endpoint_name
        config_status["config"]["region"] = cls.get_region()
        config_status["config"]["service"] = "SageMaker Runtime"
        
        # 检查AWS凭证
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        
        if not aws_access_key or not aws_secret_key:
            config_status["issues"].append("AWS凭证未配置")
        
        return config_status

# 预设配置模板
WHISPER_PRESETS = {
    "arabic_transcription": {
        "language": "ar",
        "task": "transcribe",
        "chunk_duration": 30
    },
    
    "arabic_translation": {
        "language": "ar", 
        "task": "translate",
        "chunk_duration": 20  # 较短分块用于翻译
    },
    
    "english_transcription": {
        "language": "en",
        "task": "transcribe", 
        "chunk_duration": 30
    },
    
    "multilingual_detection": {
        "language": "auto",  # 自动检测
        "task": "transcribe",
        "chunk_duration": 30
    },
    
    "long_audio_processing": {
        "language": "ar",
        "task": "transcribe",
        "chunk_duration": 60  # 长音频用更大的分块
    },
    
    "real_time_processing": {
        "language": "ar",
        "task": "transcribe", 
        "chunk_duration": 10  # 实时处理用小分块
    }
}

def print_config_status():
    """打印配置状态"""
    print("🔧 Whisper Turbo SageMaker 配置状态")
    print("=" * 40)
    
    status = WhisperConfig.validate_config()
    
    print(f"📍 端点名称: {status['config']['endpoint_name']}")
    print(f"🌍 AWS区域: {status['config']['region']}")
    print(f"🛠️  服务类型: {status['config']['service']}")
    
    if status["valid"]:
        print("✅ 配置有效")
    else:
        print("❌ 配置问题:")
        for issue in status["issues"]:
            print(f"   - {issue}")
    
    print(f"\n📋 支持的语言 ({len(WhisperConfig.SUPPORTED_LANGUAGES)}):")
    for code, name in list(WhisperConfig.SUPPORTED_LANGUAGES.items())[:5]:
        whisper_lang = WhisperConfig.get_whisper_language(code)
        print(f"   {code}: {name} -> {whisper_lang}")
    print("   ...")
    
    print(f"\n🎵 支持的音频格式:")
    for ext, fmt in WhisperConfig.SUPPORTED_AUDIO_FORMATS.items():
        print(f"   {ext} -> {fmt}")
    
    print(f"\n⚙️ SageMaker配置:")
    print(f"   最大负载: {WhisperConfig.SAGEMAKER_CONFIG['max_payload_size'] / 1024 / 1024:.1f}MB")
    print(f"   内容类型: {WhisperConfig.SAGEMAKER_CONFIG['content_type']}")
    
    print(f"\n🎯 预设配置:")
    for preset_name, config in list(WHISPER_PRESETS.items())[:3]:
        print(f"   {preset_name}: {config}")
    print("   ...")

if __name__ == "__main__":
    print_config_status()
