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
            # 将音频数据添加到VAC处理器的连续缓冲区
            if not self.vac_processor.add_audio_chunk(binary_data):
                logger.warning("添加音频数据到VAC处理器失败")
                return None
            
            # 获取VAC处理器的缓冲区统计信息
            buffer_stats = self.vac_processor.get_buffer_stats()
            logger.debug(f"VAC缓冲区统计: {buffer_stats}")
            
                        # 将音频数据转换为numpy数组
                 # 转换为与sf.read()相同的格式
            
            
            # 检查是否有待处理的语音段
            results = []
            
            # 处理所有待处理的语音段
            while self.vac_processor.has_pending_segments():
                # 获取下一个语音段
                segment = self.vac_processor.get_next_voice_segment()
                if not segment:
                    break
                
                audio_data, segment_start_time, segment_end_time = segment
                
                # 创建时间戳信息
                segment_timestamp = {
                    'chunk_index': current_chunk_index,
                    'start_time': segment_start_time - self.vac_processor.continuous_buffer_start_time if self.vac_processor.continuous_buffer_start_time else segment_start_time,
                    'end_time': segment_end_time - self.vac_processor.continuous_buffer_start_time if self.vac_processor.continuous_buffer_start_time else segment_end_time,
                    'duration': segment_end_time - segment_start_time,
                    'total_samples_processed': len(audio_data),
                    'audio_start_time': segment_start_time - self.vac_processor.continuous_buffer_start_time if self.vac_processor.continuous_buffer_start_time else segment_start_time,
                    'processing_start_time': time.time(),
                    'current_time': time.time(),
                    'is_relative_time': True  # 标记为相对时间
                }
                
                # 记录音频数据信息
                audio_duration = len(audio_data) / self.vac_processor.SAMPLING_RATE
                
                logger.info("=====> 传入SageMaker Transcribe的语音段:")
                logger.info(f"=====> 音频长度: {len(audio_data)} 样本, {audio_duration:.2f} 秒")
                logger.info(f"=====> 时间范围: start={segment_start_time:.2f}s, end={segment_end_time:.2f}s")
                logger.info(f"=====> Chunk索引: {current_chunk_index}")
                
                # 将关键参数单独打印到segments.log文件
                with open('segments.log', 'a') as f:
                    f.write(f"SEGMENT: chunk_index={current_chunk_index}, audio_length={len(audio_data)}, duration={audio_duration:.2f}s, start={segment_start_time:.2f}s, end={segment_end_time:.2f}s, relative_start={(segment_start_time - self.vac_processor.continuous_buffer_start_time if self.vac_processor.continuous_buffer_start_time else segment_start_time):.2f}s, relative_end={(segment_end_time - self.vac_processor.continuous_buffer_start_time if self.vac_processor.continuous_buffer_start_time else segment_end_time):.2f}s\n")
                
                # 创建异步生成器
                async def audio_generator():
                    logger.info(f"=====> 生成音频数据: {len(audio_data)} 样本, {audio_duration:.2f} 秒")
                    yield audio_data
                
                # 使用模型处理音频
                if self.whisper_model and self.whisper_model.is_available():
                    logger.info("开始使用Whisper模型处理语音段...")
                    # 为当前语言更新模型配置
                    await self.update_whisper_model_language(language)
                    
                    # 记录处理开始时间
                    process_start_time = time.time()
                    logger.info(f"=====> 开始处理音频: {process_start_time:.2f}s")
                    
                    async for subtitle in self.whisper_model.transcribe_stream(
                        audio_generator(), language=language
                    ):
                        # 记录处理结束时间
                        process_end_time = time.time()
                        process_duration = process_end_time - process_start_time
                        logger.info(f"=====> 音频处理完成: 耗时 {process_duration:.2f}s")
                        logger.info(f"=====> 识别结果: '{subtitle.text}'")
                        
                        # 应用时间戳到字幕
                        subtitle = await self.subtitle_processor.apply_timestamp_to_subtitle(subtitle, segment_timestamp)
                        logger.info(f"应用时间戳到字幕: start={subtitle.start:.2f}s, end={subtitle.end:.2f}s")
                        
                        # 处理字幕（纠错和翻译）
                        result = await self.subtitle_processor.process_subtitle(
                            subtitle, client_id, 
                            language=language,
                            enable_correction=enable_correction,
                            enable_translation=enable_translation,
                            target_language=target_language
                        )
                        
                        # 添加到结果列表
                        if result:
                            if result.get("type") == "split_subtitles":
                                # 如果是拆分字幕，添加所有拆分结果
                                results.extend(result.get("subtitles", []))
                            else:
                                # 单个字幕结果
                                results.append(result)
                else:
                    logger.warning("Whisper模型不可用")
                    return {
                        "type": "error",
                        "message": "Whisper模型不可用"
                    }
            
            # 如果有处理结果，返回
            if results:
                return {
                    "type": "audio_processed",
                    "status": "success",
                    "chunk_index": current_chunk_index,
                    "results": results
                }
            else:
                # 如果没有处理结果，但有音频数据添加成功，返回None表示正在累积数据
                return None
                
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
