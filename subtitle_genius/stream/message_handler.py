"""WebSocket消息处理模块，用于处理WebSocket消息"""

import logging
import json
import asyncio
import time
import numpy as np
import io
import soundfile as sf

from typing import Dict, List, Optional, Any, AsyncGenerator

from fastapi import WebSocket, WebSocketDisconnect

from .vac_processor import VACProcessor
from .subtitle_processor import SubtitleProcessor
from ..subtitle.models import Subtitle
from ..models.transcribe_model import TranscribeModel
from ..models.whisper_language_config import get_sagemaker_whisper_params, create_whisper_config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageHandler:
    """WebSocket消息处理器，用于处理WebSocket消息"""
    
    def __init__(
        self, 
        subtitle_processor: SubtitleProcessor,
        vac_processor: VACProcessor,
        whisper_model: Optional[TranscribeModel] = None,
        transcribe_model: Optional[TranscribeModel] = None
    ):
        """初始化消息处理器
        
        Args:
            subtitle_processor: 字幕处理器实例
            vac_processor: VAC处理器实例
            whisper_model: Whisper模型实例
            transcribe_model: Transcribe模型实例
        """
        self.subtitle_processor = subtitle_processor
        self.vac_processor = vac_processor
        self.whisper_model = whisper_model
        self.transcribe_model = transcribe_model
        
        # 活跃连接管理
        self.active_connections: Dict[str, WebSocket] = {}
    
    def register_connection(self, client_id: str, websocket: WebSocket):
        """注册WebSocket连接
        
        Args:
            client_id: 客户端ID
            websocket: WebSocket连接
        """
        self.active_connections[client_id] = websocket
        self.subtitle_processor.register_client(client_id)
    
    def unregister_connection(self, client_id: str):
        """注销WebSocket连接
        
        Args:
            client_id: 客户端ID
        """
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        self.subtitle_processor.unregister_client(client_id)
    
    async def update_whisper_model_language(self, language: str):
        """动态更新Whisper模型的语言配置
        
        Args:
            language: 语言代码
        """
        try:
            if self.whisper_model and hasattr(self.whisper_model, 'set_language'):
                # 如果模型支持动态语言设置
                await self.whisper_model.set_language(language)
                logger.info(f"Whisper模型语言已更新为: {language}")
            else:
                # 使用语言特定配置
                logger.info(f"为语言 {language} 重新配置Whisper模型")
                
                # 获取语言特定配置
                config = create_whisper_config(language)
                sagemaker_params = get_sagemaker_whisper_params(language)
                
                logger.info(f"语言 {language} 的Whisper配置: {config}")
                logger.info(f"语言 {language} 的SageMaker参数: {sagemaker_params}")
                
                # 更新模型配置
                if hasattr(self.whisper_model, 'whisper_config'):
                    self.whisper_model.whisper_config = config
                    logger.info(f"Whisper配置已更新")
                
                # 如果模型支持设置SageMaker参数
                if hasattr(self.whisper_model, 'set_sagemaker_params'):
                    self.whisper_model.set_sagemaker_params(sagemaker_params)
                    logger.info(f"SageMaker参数已更新")
                
        except Exception as e:
            logger.error(f"更新Whisper模型语言配置失败: {e}")
    
    async def handle_text_message(self, client_id: str, message_text: str):
        """处理文本消息
        
        Args:
            client_id: 客户端ID
            message_text: 消息文本
            
        Returns:
            Dict: 处理结果
            None: 如果消息不需要响应
        """
        try:
            message_data = json.loads(message_text)
            logger.info(f"解析JSON成功，消息类型: {message_data.get('type')}")
            
            if message_data.get('type') == 'audio_with_timestamp':
                # 处理时间戳信息
                await self.subtitle_processor.process_timestamp_message(client_id, message_data)
                return {
                    "type": "timestamp_received",
                    "chunk_index": message_data.get('timestamp', {}).get('chunk_index', 'unknown')
                }
            else:
                logger.warning(f"未知的文本消息类型: {message_data.get('type')}")
                return None
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}, 消息内容: {message_text[:100]}...")
            return {
                "type": "error",
                "message": f"JSON解析失败: {str(e)}"
            }
            
    def bytes_to_audio_data(self, binary_data):
        """
        将WebSocket接收到的二进制WAV数据转换为与sf.read()返回格式相同的格式
        
        参数:
            binary_data: WebSocket接收到的二进制数据
            
        返回:
            tuple: (audio_data, sample_rate) - 与sf.read()返回格式相同
        """
        # 使用io.BytesIO创建一个内存文件对象
        wav_io = io.BytesIO(binary_data)
        
        # 使用soundfile直接从内存中读取WAV数据
        # 这会自动处理WAV头部并提取采样率
        audio_data, sample_rate = sf.read(wav_io)
        
        return audio_data, sample_rate

    async def handle_binary_message(
        self, 
        client_id: str, 
        binary_data: bytes,
        language: str = "ar",
        enable_correction: bool = True,
        enable_translation: bool = True,
        target_language: str = "en",
        pending_timestamp: Optional[Dict] = None,
        current_chunk_index: int = 0
    ):
        """处理二进制消息（音频数据）
        
        Args:
            client_id: 客户端ID
            binary_data: 二进制数据
            language: 语言代码
            enable_correction: 是否启用纠错
            enable_translation: 是否启用翻译
            target_language: 目标翻译语言
            pending_timestamp: 待处理的时间戳信息
            current_chunk_index: 当前chunk索引
            
        Returns:
            Dict: 处理结果，包含音频处理状态和可能的字幕
            None: 如果音频数据无效或不需要处理
        """
        try:
            audio_data, sample_rate = self.bytes_to_audio_data(binary_data)
            
            print(f"音频数据: 形状={audio_data.shape}, 类型={audio_data.dtype}")
            print(f"采样率: {sample_rate} Hz")
            print(f"音频时长: {len(audio_data)}长度")
            
            # 调整VAC处理器参数，使其更容易检测到语音
            # 降低阈值，减少最小静音持续时间
            original_threshold = self.vac_processor.threshold
            original_min_silence = self.vac_processor.min_silence_duration_ms
            
            # 临时调整参数以提高灵敏度
            self.vac_processor.threshold = 0.2  # 降低阈值，使其更容易检测到语音
            self.vac_processor.min_silence_duration_ms = 200  # 减少最小静音持续时间
            
            print(f"调整VAC处理器参数: 阈值={self.vac_processor.threshold}, 最小静音持续时间={self.vac_processor.min_silence_duration_ms}ms")
            
            # 定义语音段检测回调函数
            detected_segments = []
            
            def on_speech_segment(segment):
                """语音段检测回调函数"""
                print(f"检测到完整语音段: 开始={segment['start']:.3f}s, 结束={segment['end']:.3f}s, 时长={segment['duration']:.3f}s")
                # 保存检测到的语音段
                detected_segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'duration': segment['duration']
                })
            
            # 设置回调函数
            self.vac_processor.on_speech_segment = on_speech_segment
            
            # 将单个音频数据转换为迭代器，以便传递给process_streaming_audio
            def audio_stream_generator():
                # 打印音频数据的统计信息，帮助调试
                print(f"音频数据统计: 最小值={np.min(audio_data):.4f}, 最大值={np.max(audio_data):.4f}, 均值={np.mean(audio_data):.4f}, 标准差={np.std(audio_data):.4f}")
                
                # 检查音频数据是否包含语音（简单检查：是否有足够的变化）
                if np.std(audio_data) < 0.01:
                    print("警告: 音频数据变化很小，可能不包含语音")
                
                yield audio_data
            
            # 创建音频流迭代器
            audio_stream = audio_stream_generator()
            
            # 调用vac_processor.process_streaming_audio处理音频数据
            # 设置end_stream_flag为False，避免立即结束流
            print("开始处理音频流...")
            self.vac_processor.process_streaming_audio(
                audio_stream=audio_stream,
                end_stream_flag=False,  # 不立即结束流
                return_segments=False  # 不需要返回段，因为我们使用回调函数
            )
            print("音频流处理完成")
            
            # 恢复原始参数
            self.vac_processor.threshold = original_threshold
            self.vac_processor.min_silence_duration_ms = original_min_silence
            
            # 返回处理结果
            if detected_segments:
                print(f"总共检测到 {len(detected_segments)} 个语音段")
                return {
                    "type": "audio_processed",
                    "segments": detected_segments,
                    "count": len(detected_segments)
                }
            else:
                print("未检测到语音段")
                return {
                    "type": "audio_processed",
                    "segments": [],
                    "count": 0
                }
                
        except Exception as e:
            logger.error(f"处理音频数据失败: {e}")
            import traceback
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            return {
                "type": "error",
                "message": f"处理音频数据失败: {str(e)}"
            }
    
    async def send_message(self, client_id: str, message: Dict):
        """发送消息到客户端
        
        Args:
            client_id: 客户端ID
            message: 消息内容
            
        Returns:
            bool: 发送是否成功
        """
        if client_id not in self.active_connections:
            logger.warning(f"客户端 {client_id} 不存在或已断开连接")
            return False
        
        try:
            await self.active_connections[client_id].send_json(message)
            return True
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return False
    
    async def handle_save_subtitles(self, client_id: str, filename: str, language: str = "auto"):
        """处理保存字幕请求
        
        Args:
            client_id: 客户端ID
            filename: 文件名
            language: 语言代码
            
        Returns:
            Dict: 处理结果
        """
        try:
            # 保存字幕
            success = self.subtitle_processor.save_subtitles(client_id, filename, language)
            
            if success:
                return {
                    "type": "success",
                    "message": "字幕已保存",
                    "files": [
                        f"{filename}_{language}.srt",
                        f"{filename}_{language}.vtt",
                        f"{filename}_{language}.json"
                    ]
                }
            else:
                return {
                    "type": "error",
                    "message": "保存字幕失败"
                }
        except Exception as e:
            logger.error(f"保存字幕失败: {e}")
            return {
                "type": "error",
                "message": f"保存字幕失败: {str(e)}"
            }
