"""配置管理模块"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Config(BaseSettings):
    """应用配置类"""
    
    # OpenAI配置
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4-turbo-preview", env="OPENAI_MODEL")
    
    # Anthropic配置
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-sonnet-20240229", env="ANTHROPIC_MODEL")
    
    # 音频处理配置
    audio_sample_rate: int = Field(default=16000, env="AUDIO_SAMPLE_RATE")
    audio_chunk_size: int = Field(default=1024, env="AUDIO_CHUNK_SIZE")
    audio_format: str = Field(default="wav", env="AUDIO_FORMAT")
    
    # 字幕配置
    subtitle_language: str = Field(default="ar", env="SUBTITLE_LANGUAGE")  # 默认使用 Arabic
    subtitle_format: str = Field(default="srt", env="SUBTITLE_FORMAT")
    max_subtitle_length: int = Field(default=80, env="MAX_SUBTITLE_LENGTH")
    
    # 日志配置
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/subtitle_genius.log", env="LOG_FILE")
    
    # 实时处理配置
    real_time_buffer_size: int = Field(default=5, env="REAL_TIME_BUFFER_SIZE")
    processing_interval: float = Field(default=1.0, env="PROCESSING_INTERVAL")
    
    # AWS配置
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    aws_s3_bucket: str = Field(default="subtitle-genius-temp", env="AWS_S3_BUCKET")
    
    # SageMaker配置
    sagemaker_endpoint_name: str = Field(default="endpoint-quick-start-z9afg", env="SAGEMAKER_ENDPOINT_NAME")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 全局配置实例
config = Config()
