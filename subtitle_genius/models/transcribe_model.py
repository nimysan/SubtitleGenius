"""Amazon Transcribe 流式模型实现 - 使用 amazon-transcribe-streaming-sdk"""

import asyncio
import sys
from pathlib import Path
from typing import List, Any, Optional, AsyncGenerator
import numpy as np

# 添加子模块路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "amazon-transcribe-streaming-sdk"))

from .base import BaseModel
from ..subtitle.models import Subtitle
from ..core.config import config

# 导入 amazon-transcribe-streaming-sdk
try:
    from amazon_transcribe.client import TranscribeStreamingClient
    from amazon_transcribe.handlers import TranscriptResultStreamHandler
    from amazon_transcribe.model import TranscriptEvent
    STREAMING_AVAILABLE = True
    print("✅ Amazon Transcribe Streaming SDK 已加载")
except ImportError as e:
    STREAMING_AVAILABLE = False
    print(f"⚠️  Amazon Transcribe Streaming SDK 不可用: {e}")
    print("   请确保子模块已正确初始化: git submodule update --init --recursive")


class TranscribeModel(BaseModel):
    """Amazon Transcribe 流式模型 - 专注于实时语音识别"""
    
    def __init__(self, region_name: str = "us-east-1"):
        """初始化 Transcribe 流式客户端
        
        Args:
            region_name: AWS 区域名称 (默认: us-east-1)
        """
        self.region_name = region_name
        self.streaming_client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """初始化 AWS 流式客户端"""
        if not STREAMING_AVAILABLE:
            print("❌ Amazon Transcribe Streaming SDK 不可用")
            return
            
        try:
            # 初始化流式客户端
            self.streaming_client = TranscribeStreamingClient(region=self.region_name)
            print(f"✅ Transcribe 流式客户端已初始化 (区域: {self.region_name})")
            
        except Exception as e:
            print(f"❌ 初始化 Transcribe 流式客户端失败: {e}")
            self.streaming_client = None
    
    def is_available(self) -> bool:
        """检查模型是否可用"""
        return STREAMING_AVAILABLE and self.streaming_client is not None
    
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
            print(f"🎤 开始流式转录 (语言: {language_code})")
            
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
                    print("🎤 音频流发送完成")
                    
                except Exception as e:
                    print(f"❌ 音频流写入错误: {e}")
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
                    print(f"❌ 转录结果处理错误: {e}")
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
                print("⏰ 转录结果处理超时，返回已收集的字幕")
            
            # 返回收集到的字幕
            for subtitle in subtitle_handler.get_subtitles():
                yield subtitle
                
        except Exception as e:
            print(f"❌ 流式转录失败: {e}")
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
        print(f"📝 [{subtitle.start:.1f}s-{subtitle.end:.1f}s] {subtitle.text}")
        self.subtitles.append(subtitle)
    
    def get_subtitles(self) -> List[Subtitle]:
        """获取所有收集到的字幕"""
        return self.subtitles.copy()
    
    def clear(self):
        """清空字幕缓存"""
        self.subtitles.clear()
        self.current_subtitle = None
