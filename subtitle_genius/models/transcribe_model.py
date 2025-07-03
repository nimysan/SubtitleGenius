"""Amazon Transcribe 流式模型实现 - 使用 amazon-transcribe-streaming-sdk"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Any, Optional, AsyncGenerator
import numpy as np

# 设置日志记录器
logger = logging.getLogger(__name__)

# 添加子模块路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "amazon-transcribe-streaming-sdk"))

from .base import BaseModel
from ..subtitle.models import Subtitle
from ..core.config import config

# 导入 SageMaker Whisper 流式处理
try:
    from .whisper_sagemaker_streaming import WhisperSageMakerStreamingModel, WhisperSageMakerStreamConfig
    SAGEMAKER_WHISPER_AVAILABLE = True
    logger.info("✅ SageMaker Whisper 流式处理模块已加载")
except ImportError as e:
    SAGEMAKER_WHISPER_AVAILABLE = False
    logger.warning(f"⚠️  SageMaker Whisper 流式处理模块不可用: {e}")
    logger.warning("   请确保 sagemaker_whisper.py 文件存在且可访问")

# 导入 amazon-transcribe-streaming-sdk
try:
    from amazon_transcribe.client import TranscribeStreamingClient
    from amazon_transcribe.handlers import TranscriptResultStreamHandler
    from amazon_transcribe.model import TranscriptEvent
    STREAMING_AVAILABLE = True
    logger.info("✅ Amazon Transcribe Streaming SDK 已加载")
except ImportError as e:
    STREAMING_AVAILABLE = False
    logger.warning(f"⚠️  Amazon Transcribe Streaming SDK 不可用: {e}")
    logger.warning("   请确保子模块已正确初始化: git submodule update --init --recursive")


class TranscribeModel(BaseModel):
    """统一的流式转录模型 - 支持 Amazon Transcribe 和 SageMaker Whisper"""
    
    def __init__(self, 
                 region_name: str = "us-east-1",
                 backend: str = "transcribe",  # "transcribe" 或 "sagemaker_whisper"
                 sagemaker_endpoint: Optional[str] = None,
                 whisper_config: Optional['WhisperSageMakerStreamConfig'] = None):
        """初始化转录模型
        
        Args:
            region_name: AWS 区域名称 (默认: us-east-1)
            backend: 后端选择 ("transcribe" 或 "sagemaker_whisper")
            sagemaker_endpoint: SageMaker Whisper 端点名称
            whisper_config: SageMaker Whisper 流式配置
        """
        self.region_name = region_name
        self.backend = backend
        self.streaming_client = None
        self.sagemaker_whisper_model = None
        
        if backend == "transcribe":
            self._initialize_transcribe_client()
        elif backend == "sagemaker_whisper":
            if not sagemaker_endpoint:
                raise ValueError("使用 sagemaker_whisper 后端时必须提供 sagemaker_endpoint 参数")
            self._initialize_sagemaker_whisper_model(sagemaker_endpoint, whisper_config)
        else:
            raise ValueError(f"不支持的后端: {backend}. 请选择 'transcribe' 或 'sagemaker_whisper'")
    
    def _initialize_transcribe_client(self):
        """初始化 AWS Transcribe 流式客户端"""
        if not STREAMING_AVAILABLE:
            logger.error("❌ Amazon Transcribe Streaming SDK 不可用")
            return
            
        try:
            # 初始化流式客户端
            self.streaming_client = TranscribeStreamingClient(region=self.region_name)
            logger.info(f"✅ Transcribe 流式客户端已初始化 (区域: {self.region_name})")
            
        except Exception as e:
            logger.error(f"❌ 初始化 Transcribe 流式客户端失败: {e}")
            self.streaming_client = None
    
    def _initialize_sagemaker_whisper_model(self, endpoint_name: str, config: Optional['WhisperSageMakerStreamConfig']):
        """初始化 SageMaker Whisper 流式模型"""
        if not SAGEMAKER_WHISPER_AVAILABLE:
            logger.error("❌ SageMaker Whisper 流式处理不可用")
            return
            
        try:
            self.sagemaker_whisper_model = WhisperSageMakerStreamingModel(
                endpoint_name=endpoint_name,
                region_name=self.region_name,
                config=config or WhisperSageMakerStreamConfig()
            )
            logger.info(f"✅ SageMaker Whisper 流式模型已初始化 (端点: {endpoint_name})")
            
        except Exception as e:
            logger.error(f"❌ 初始化 SageMaker Whisper 流式模型失败: {e}")
            self.sagemaker_whisper_model = None
    
    def is_available(self) -> bool:
        """检查模型是否可用"""
        if self.backend == "transcribe":
            return STREAMING_AVAILABLE and self.streaming_client is not None
        elif self.backend == "sagemaker_whisper":
            return SAGEMAKER_WHISPER_AVAILABLE and self.sagemaker_whisper_model is not None
        return False
    
    async def transcribe(self, audio_data: Any, language: str = "ar") -> List[Subtitle]:
        """批处理模式已移除 - 请使用 transcribe_stream 进行流式处理
        
        Args:
            audio_data: 音频文件路径或音频数据
            language: 语言代码 (默认: ar - Arabic)
            
        Returns:
            字幕列表
        """
        raise NotImplementedError(
            "批处理模式已移除。请使用 transcribe_stream() 进行流式处理。\n"
            "示例: async for subtitle in model.transcribe_stream(audio_stream, language='ar')"
        )
    
    async def transcribe_stream(
        self, 
        audio_stream: AsyncGenerator[np.ndarray, None], 
        language: str = "ar"
    ) -> AsyncGenerator[Subtitle, None]:
        """统一的流式转录接口
        
        Args:
            audio_stream: 音频数据流 (numpy数组)
            language: 语言代码 (默认: ar - Arabic)
            
        Yields:
            实时字幕
        """
        if not self.is_available():
            raise RuntimeError(f"{self.backend} 后端不可用，请检查配置")
        
        logger.info(f"🎤 使用 {self.backend.upper()} 后端进行流式转录 (语言: {language})")
        
        if self.backend == "transcribe":
            async for subtitle in self._transcribe_stream_aws(audio_stream, language):
                yield subtitle
        elif self.backend == "sagemaker_whisper":
            async for subtitle in self._transcribe_stream_sagemaker_whisper(audio_stream, language):
                logger.debug(f"----> title is {subtitle}")
                yield subtitle
    
    async def _transcribe_stream_sagemaker_whisper(
        self, 
        audio_stream: AsyncGenerator[np.ndarray, None], 
        language: str = "ar"
    ) -> AsyncGenerator[Subtitle, None]:
        """使用 SageMaker Whisper 进行流式转录"""
        async for subtitle in self.sagemaker_whisper_model.transcribe_stream(audio_stream, language):
            yield subtitle
    
    async def _transcribe_stream_aws(
        self, 
        audio_stream: AsyncGenerator[np.ndarray, None], 
        language: str = "ar"
    ) -> AsyncGenerator[Subtitle, None]:
        """使用 Amazon Transcribe 流式转录音频
        
        Args:
            audio_stream: 音频数据流 (numpy数组)
            language: 语言代码 (默认: ar - Arabic)
            
        Yields:
            实时字幕
        """
        if not self.is_available():
            raise RuntimeError("Amazon Transcribe Streaming SDK 不可用，请检查配置")
        
        try:
            # 转换语言代码
            language_code = self._convert_language_code(language)
            logger.info(f"🎤 开始流式转录 (语言: {language_code})")
            
            # 启动流式转录
            stream = await self.streaming_client.start_stream_transcription(
                language_code=language_code,
                media_sample_rate_hz=getattr(config, 'audio_sample_rate', 16000),
                media_encoding="pcm"
            )
            
            # 创建字幕处理器
            subtitle_handler = SubtitleStreamHandler()
            
            # 音频写入任务
            async def write_audio_chunks():
                try:
                    async for audio_chunk in audio_stream:
                        # 将 numpy 数组转换为 PCM 字节数据
                        if isinstance(audio_chunk, np.ndarray):
                            # 确保数据类型为 int16
                            if audio_chunk.dtype != np.int16:
                                # 假设输入是 float32 范围 [-1, 1]
                                audio_chunk = (audio_chunk * 32767).astype(np.int16)
                            audio_bytes = audio_chunk.tobytes()
                        else:
                            audio_bytes = audio_chunk
                        
                        # 发送音频数据
                        await stream.input_stream.send_audio_event(audio_chunk=audio_bytes)
                    
                    # 结束音频流
                    await stream.input_stream.end_stream()
                    logger.info("🎤 音频流发送完成")
                    
                except Exception as e:
                    logger.error(f"❌ 音频流写入错误: {e}")
                    raise
            
            # 处理转录结果
            async def handle_transcription_results():
                try:
                    async for event in stream.output_stream:
                        if isinstance(event, TranscriptEvent):
                            # 处理转录事件
                            for result in event.transcript.results:
                                if not result.is_partial:  # 只处理完整结果
                                    for alternative in result.alternatives:
                                        if alternative.transcript.strip():
                                            subtitle = Subtitle(
                                                start=result.start_time,
                                                end=result.end_time,
                                                text=alternative.transcript.strip()
                                            )
                                            subtitle_handler.add_subtitle(subtitle)
                                            
                except Exception as e:
                    logger.error(f"❌ 转录结果处理错误: {e}")
                    raise
            
            # 并发执行音频写入和结果处理
            write_task = asyncio.create_task(write_audio_chunks())
            handle_task = asyncio.create_task(handle_transcription_results())
            
            # 等待音频写入完成
            await write_task
            
            # 等待处理完成或超时
            try:
                await asyncio.wait_for(handle_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("⏰ 转录结果处理超时，返回已收集的字幕")
            
            # 返回收集到的字幕
            for subtitle in subtitle_handler.get_subtitles():
                yield subtitle
                
        except Exception as e:
            logger.error(f"❌ 流式转录失败: {e}")
            raise
    
    def _convert_language_code(self, language: str) -> str:
        """转换语言代码为 Transcribe 支持的格式"""
        language_mapping = {
            'ar': 'ar-SA',          # Arabic (Saudi Arabia) - 默认
            'ar-SA': 'ar-SA',       # Arabic (Saudi Arabia)
            'ar-AE': 'ar-AE',       # Arabic (UAE)
            'zh-CN': 'zh-CN',       # Chinese (Simplified)
            'zh': 'zh-CN',
            'en': 'en-US',          # English (US)
            'en-US': 'en-US',
            'en-GB': 'en-GB',       # English (UK)
            'ja': 'ja-JP',          # Japanese
            'ko': 'ko-KR',          # Korean
            'fr': 'fr-FR',          # French
            'de': 'de-DE',          # German
            'es': 'es-ES',          # Spanish
            'ru': 'ru-RU',          # Russian
        }
        
        return language_mapping.get(language, 'ar-SA')  # 默认使用 Arabic


class SubtitleStreamHandler:
    """流式字幕处理器 - 收集和管理实时字幕"""
    
    def __init__(self):
        self.subtitles: List[Subtitle] = []
        self.current_subtitle: Optional[Subtitle] = None
    
    def add_subtitle(self, subtitle: Subtitle):
        """添加新字幕"""
        logger.info(f"📝 [{subtitle.start:.1f}s-{subtitle.end:.1f}s] {subtitle.text}")
        self.subtitles.append(subtitle)
    
    def get_subtitles(self) -> List[Subtitle]:
        """获取所有收集到的字幕"""
        return self.subtitles.copy()
    
    def clear(self):
        """清空字幕缓存"""
        self.subtitles.clear()
        self.current_subtitle = None
