"""
连续音频处理器模块

这个模块提供了一个独立的连续音频处理器，可以被多个组件使用，
包括WebSocket服务器、CLI工具等。

主要功能：
- 处理连续音频流
- 集成VAC（Voice Activity Detection）处理器
- 支持多种AI模型（Whisper、Claude等）
- 提供字幕生成和优化功能
- 支持多语言处理和翻译
"""

import asyncio
import json
import logging
import os
import io
import uuid
import numpy as np
import soundfile as sf
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Iterator, Callable
from websockets.server import WebSocketServerProtocol

# 导入VAC处理器
from subtitle_genius.stream.vac_processor import VACProcessor
# 导入SageMaker Whisper流式处理模型
from subtitle_genius.models.whisper_sagemaker_streaming import WhisperSageMakerStreamingModel
# 导入SubtitlePipeline和Subtitle
from subtitle_genius.pipeline.subtitle_pipeline import SubtitlePipeline
from subtitle_genius.subtitle.models import Subtitle

# 配置日志
logger = logging.getLogger(__name__)

# 定义处理块大小 (512*8 = 4096字节)
PROCESSING_CHUNK_SIZE = 512 * 8


class ContinuousAudioProcessor:
    """处理连续音频流的处理器"""
    
    def __init__(self, config: Dict[str, Any] = None, result_callback: Optional[Callable] = None):
        """
        初始化连续音频处理器
        
        Args:
            config: 配置参数字典
            result_callback: 结果回调函数，用于处理生成的字幕
        """
        # 存储主事件循环，用于线程安全操作
        try:
            self.main_loop = asyncio.get_running_loop()
            logger.info("已获取主事件循环引用")
        except RuntimeError:
            logger.warning("初始化时未找到运行中的事件循环，将在首次使用时获取")
            self.main_loop = None
        
        # 处理配置参数
        self.config = config or {}
        self.source_language = self.config.get('language', 'zh')
        self.target_language = self.config.get('target_language', 'en')
        self.correction_enabled = self.config.get('correction', True)
        self.translation_enabled = self.config.get('translation', False)
        self.model_type = self.config.get('model', 'whisper')
        self.scene_description = self.config.get('scene_description', '足球比赛')
        self.client_id = self.config.get('client_id', str(uuid.uuid4()))
        
        # 结果回调函数
        self.result_callback = result_callback
        
        logger.info(f"音频处理器配置: {self.config}")
        
        # 创建SubtitlePipeline实例，用于处理字幕
        self.subtitle_pipeline = SubtitlePipeline(
            source_language=self.source_language,
            target_language=self.target_language,
            correction_enabled=self.correction_enabled,
            correction_service="bedrock",
            translation_enabled=self.translation_enabled,
            translation_service="bedrock",
            scene_description=self.scene_description,
            output_format="srt"
        )
        logger.info(f"SubtitlePipeline初始化完成，配置: 源语言={self.source_language}, 目标语言={self.target_language}, 纠错={self.correction_enabled}, 翻译={self.translation_enabled}")
            
        # 初始化VAC处理器，设置回调函数
        self.vac_processor = VACProcessor(
            threshold=0.3,
            min_silence_duration_ms=300,
            speech_pad_ms=100,
            sample_rate=16000,
            processing_chunk_size=512,  # VAC处理器的处理块大小
            no_audio_input_threshold=5.0,
            on_speech_segment=self._on_speech_segment_sync
        )
        
        # 存储活跃的连接（如果使用WebSocket）
        self.active_connections = {}
        # 音频缓冲区 - 每个流一个缓冲区
        self.audio_buffers = {}
        # 处理任务
        self.processing_tasks = {}
        # 结束标志
        self.end_flags = {}
        # 音频队列 - 用于VAC处理器
        self.vac_queues = {}
        
        # WebVTT文件路径
        self.webvtt_file = "test.webvtt"
        # 初始化WebVTT文件
        self._init_webvtt_file()
        
        # 初始化SageMaker Whisper模型
        try:
            # 从环境变量或配置中获取端点名称和区域
            endpoint_name = os.environ.get("SAGEMAKER_WHISPER_ENDPOINT", "endpoint-quick-start-z9afg")
            region_name = os.environ.get("AWS_REGION", "us-east-1")
            
            self.whisper_model = WhisperSageMakerStreamingModel(
                endpoint_name=endpoint_name,
                region_name=region_name
            )
            logger.info(f"SageMaker Whisper模型初始化成功: {endpoint_name}")
        except Exception as e:
            logger.error(f"初始化SageMaker Whisper模型失败: {e}")
            self.whisper_model = None
        
        logger.info("ContinuousAudioProcessor初始化完成")
    
    def _init_webvtt_file(self):
        """初始化WebVTT文件"""
        with open(self.webvtt_file, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
        logger.info(f"已初始化WebVTT文件: {self.webvtt_file}")
    
    def _format_timestamp(self, seconds: float) -> str:
        """将秒数格式化为WebVTT时间戳格式 (HH:MM:SS.mmm)"""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int(td.microseconds / 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    
    def _write_subtitle_to_webvtt(self, start: float, end: float, text: str):
        """将字幕写入WebVTT文件"""
        start_timestamp = self._format_timestamp(start)
        end_timestamp = self._format_timestamp(end)
        
        with open(self.webvtt_file, "a", encoding="utf-8") as f:
            f.write(f"{start_timestamp} --> {end_timestamp}\n")
            f.write(f"{text}\n\n")
        
        logger.info(f"已将字幕写入WebVTT文件: {start_timestamp} --> {end_timestamp}: {text}")
    
    async def start_stream(self, stream_id: str, connection_ref: Any = None):
        """
        开始一个新的音频流处理
        
        Args:
            stream_id: 流的唯一标识符
            connection_ref: 连接引用（可以是WebSocket连接或其他类型的连接）
        """
        logger.info(f"开始新的音频流: {stream_id}")
        
        # 确保我们有主事件循环的引用
        if self.main_loop is None:
            self.main_loop = asyncio.get_running_loop()
            logger.info("已在start_stream中获取主事件循环引用")
        
        # 存储连接信息
        if connection_ref:
            self.active_connections[stream_id] = connection_ref
        
        # 初始化音频缓冲区
        self.audio_buffers[stream_id] = bytearray()
        # 创建VAC处理队列
        self.vac_queues[stream_id] = asyncio.Queue()
        # 设置结束标志
        self.end_flags[stream_id] = False
        
        # 创建处理任务 - 确保立即启动VAC处理器
        self.processing_tasks[stream_id] = asyncio.create_task(
            self._process_audio_stream(stream_id)
        )
        
        # 确保任务已经启动
        logger.info(f"VAC处理任务已创建: {self.processing_tasks[stream_id]}")
        
        # 添加任务完成的回调，以便在任务完成时记录日志
        self.processing_tasks[stream_id].add_done_callback(
            lambda t: logger.info(f"VAC处理任务完成状态: {'成功' if not t.exception() else f'失败: {t.exception()}'}")
        )
        
        return stream_id
    
    async def stop_stream(self, stream_id: str):
        """停止音频流处理"""
        if stream_id in self.processing_tasks:
            logger.info(f"停止音频流: {stream_id}")
            
            # 设置结束标志
            self.end_flags[stream_id] = True
            
            # 向队列添加None表示结束
            if stream_id in self.vac_queues:
                await self.vac_queues[stream_id].put(None)
            
            # 等待处理任务完成
            try:
                await asyncio.wait_for(self.processing_tasks[stream_id], timeout=2.0)
            except asyncio.TimeoutError:
                # 如果超时，取消任务
                self.processing_tasks[stream_id].cancel()
                try:
                    await self.processing_tasks[stream_id]
                except asyncio.CancelledError:
                    pass
            
            # 清理资源
            del self.processing_tasks[stream_id]
            if stream_id in self.vac_queues:
                del self.vac_queues[stream_id]
            if stream_id in self.audio_buffers:
                del self.audio_buffers[stream_id]
            if stream_id in self.end_flags:
                del self.end_flags[stream_id]
            if stream_id in self.active_connections:
                del self.active_connections[stream_id]
            
            # 记录WebVTT文件已完成
            logger.info(f"WebVTT文件已完成: {self.webvtt_file}")
    
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
         # 确保音频是float32格式
        logger.debug(f"data is {audio_data.dtype}")
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        return audio_data, sample_rate
    
    async def process_audio(self, stream_id: str, binary_data: bytes):
        """
        处理音频数据 - 这是音频数据从外部进入处理流程的入口
        
        流程：
        1. 接收音频数据 → 添加到缓冲区
        2. 当缓冲区达到处理块大小时 → 提取块 → 添加到VAC队列
        3. VAC队列 → ThreadSafeIterator → VAC处理器
        """
        if stream_id not in self.audio_buffers:
            raise ValueError(f"找不到流 {stream_id}")
        
        try:
            audio_data, sample_rate = self.bytes_to_audio_data(binary_data)
            
            logger.debug(f"音频数据: 形状={audio_data.shape}, 类型={audio_data.dtype}")
            logger.debug(f"采样率: {sample_rate} Hz")
            logger.debug(f"音频时长: {len(audio_data)}长度")
            
            # 步骤1: 将音频数据添加到缓冲区
            self.audio_buffers[stream_id].extend(audio_data)
            buffer_size = len(self.audio_buffers[stream_id])
            
            logger.debug(f"-------》 接收到音频数据: {len(audio_data)}字节, 当前缓冲区大小: {len(self.audio_buffers[stream_id])}字节 处理入口是 {PROCESSING_CHUNK_SIZE}")
            
            # 步骤2: 当缓冲区大小达到处理块大小时，处理数据
            while len(self.audio_buffers[stream_id]) >= PROCESSING_CHUNK_SIZE:
                # 从缓冲区提取一个处理块
                chunk_data = self.audio_buffers[stream_id][:PROCESSING_CHUNK_SIZE]
                # 更新缓冲区，移除已处理的数据
                self.audio_buffers[stream_id] = self.audio_buffers[stream_id][PROCESSING_CHUNK_SIZE:]
                
                # 转换为numpy数组 - VAC处理器需要numpy数组格式
                audio_np = np.frombuffer(chunk_data, dtype=np.float32)
                
                logger.debug(f"处理音频块: 大小={len(audio_np)}, 准备添加到VAC队列")
                
                # 步骤3: 将处理块添加到VAC队列 - 这是关键步骤，将数据传递给VAC处理器
                # VAC队列是连接音频缓冲区和VAC处理器的桥梁
                await self.vac_queues[stream_id].put(audio_np)
                logger.debug(f"音频块已添加到VAC队列，当前队列大小: {self.vac_queues[stream_id].qsize()}")
            
            return {
                "status": "processing",
                "timestamp": datetime.now().isoformat(),
                "chunk_size": len(audio_data),
                "buffer_size": len(self.audio_buffers[stream_id]),
                "queue_size": self.vac_queues[stream_id].qsize()
            }
        
        except Exception as e:
            logger.error(f"处理音频数据时出错: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def process_audio_from_numpy(self, stream_id: str, audio_data: np.ndarray):
        """
        直接处理numpy音频数据
        
        Args:
            stream_id: 流ID
            audio_data: numpy音频数据数组
        """
        if stream_id not in self.audio_buffers:
            raise ValueError(f"找不到流 {stream_id}")
        
        try:
            # 确保音频是float32格式
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            logger.debug(f"处理numpy音频数据: 形状={audio_data.shape}, 类型={audio_data.dtype}")
            
            # 将音频数据添加到缓冲区
            self.audio_buffers[stream_id].extend(audio_data.tobytes())
            
            # 当缓冲区大小达到处理块大小时，处理数据
            while len(self.audio_buffers[stream_id]) >= PROCESSING_CHUNK_SIZE:
                # 从缓冲区提取一个处理块
                chunk_data = self.audio_buffers[stream_id][:PROCESSING_CHUNK_SIZE]
                # 更新缓冲区，移除已处理的数据
                self.audio_buffers[stream_id] = self.audio_buffers[stream_id][PROCESSING_CHUNK_SIZE:]
                
                # 转换为numpy数组
                audio_np = np.frombuffer(chunk_data, dtype=np.float32)
                
                # 将处理块添加到VAC队列
                await self.vac_queues[stream_id].put(audio_np)
                logger.debug(f"numpy音频块已添加到VAC队列，当前队列大小: {self.vac_queues[stream_id].qsize()}")
            
            return {
                "status": "processing",
                "timestamp": datetime.now().isoformat(),
                "chunk_size": len(audio_data),
                "buffer_size": len(self.audio_buffers[stream_id]),
                "queue_size": self.vac_queues[stream_id].qsize()
            }
        
        except Exception as e:
            logger.error(f"处理numpy音频数据时出错: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    async def _process_audio_stream(self, stream_id: str):
        """处理音频流 - 这是VAC处理器被触发的关键方法"""
        logger.info(f"开始音频流处理: {stream_id}")
        
        try:
            # 创建一个可以在线程间传递的迭代器
            # 这个迭代器是关键 - 它从VAC队列中获取数据并提供给VAC处理器
            class ThreadSafeIterator:
                def __init__(self, queue, loop, end_flag):
                    self.queue = queue  # VAC队列，存储待处理的音频块
                    self.loop = loop    # 事件循环，用于在线程间安全地获取队列数据
                    self.end_flag = end_flag  # 结束标志
                    self.chunks_processed = 0  # 处理的块数量
                    logger.info("线程安全迭代器初始化 - 将作为VAC处理器的数据源")
                
                def __iter__(self):
                    # 使迭代器可迭代
                    logger.info("VAC处理器调用了迭代器的__iter__方法")
                    return self
                
                def __next__(self):
                    # 这个方法是VAC处理器获取数据的入口点
                    # 每当VAC处理器需要下一个音频块时，它会调用这个方法
                    try:
                        logger.debug(f"VAC处理器请求下一个音频块 (已处理: {self.chunks_processed})")
                        
                        # 使用run_coroutine_threadsafe从队列获取数据
                        # 这是线程安全的方式，因为VAC处理器在单独的线程中运行
                        future = asyncio.run_coroutine_threadsafe(
                            self.queue.get(), self.loop
                        )
                        # 等待数据，最多5秒
                        logger.debug("等待队列中的音频数据...")
                        chunk = future.result(timeout=5)  # 5秒超时
                        
                        # None表示流结束
                        if chunk is None:
                            logger.info("检测到流结束标记(None)，停止迭代")
                            raise StopIteration
                        
                        self.chunks_processed += 1
                        logger.debug(f"VAC处理器获取到音频块 #{self.chunks_processed}, 大小={len(chunk)}")
                        
                        # 返回音频块给VAC处理器
                        # 这里是VAC处理器实际获取数据的地方
                        return chunk
                    
                    except asyncio.TimeoutError:
                        # 超时，检查结束标志
                        if self.end_flag[0]:
                            logger.info("由于结束标志流已结束")
                            raise StopIteration
                        # 返回静音块以保持流程
                        logger.info("等待音频数据超时，返回静音块以保持VAC处理器运行")
                        silent_chunk = np.zeros(PROCESSING_CHUNK_SIZE // 4, dtype=np.float32)  # float32是4字节
                        self.chunks_processed += 1
                        return silent_chunk
                    
                    except Exception as e:
                        logger.error(f"迭代器中出错: {e}")
                        raise StopIteration
            
            # 获取当前事件循环
            loop = asyncio.get_running_loop()
            
            # 创建一个可变的结束标志引用
            end_flag_ref = [self.end_flags[stream_id]]
            
            # 创建迭代器 - 这是VAC处理器的数据源
            audio_iterator = ThreadSafeIterator(self.vac_queues[stream_id], loop, end_flag_ref)
            
            # 在单独的线程中运行VAC处理器
            # 这里是VAC处理器被实际触发的地方
            logger.info(f"启动VAC处理器处理流 {stream_id} - 这里是VAC处理器被触发的地方")
            
            # 确保VAC处理器已正确初始化
            if self.vac_processor is None:
                logger.error("VAC处理器未初始化!")
                return
            
            # 记录VAC处理器的状态
            logger.info(f"VAC处理器状态: threshold={self.vac_processor.threshold}, "
                       f"min_silence_duration_ms={self.vac_processor.min_silence_duration_ms}")
            
            # 使用try-except包装VAC处理器调用，以捕获任何错误
            try:
                logger.info("开始调用VAC处理器的process_streaming_audio方法")
                result = await loop.run_in_executor(
                    None,  # 使用默认执行器
                    lambda: self.vac_processor.process_streaming_audio(
                        audio_stream=audio_iterator,  # 传入迭代器作为数据源
                        end_stream_flag=False,  # 不立即结束流
                        return_segments=True     # 返回语音段
                    )
                )
                logger.info(f"VAC处理器返回结果: {result}")
            except Exception as e:
                logger.error(f"VAC处理器调用失败: {e}")
                # 重新抛出异常，以便上层捕获
                raise
            
            logger.info(f"音频流处理完成: {stream_id}")
        
        except asyncio.CancelledError:
            logger.info(f"音频流处理被取消: {stream_id}")
        except Exception as e:
            logger.error(f"音频流处理出错: {e}")
            # 打印详细的堆栈跟踪
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
    
    async def _on_speech_segment(self, segment: Dict[str, Any]):
        """语音段检测回调"""
        logger.info(f"_on_speech_segment 检测到语音段: {segment['start']:.2f}秒 - {segment['end']:.2f}秒, 持续时间: {segment['duration']:.2f}秒")
        
        # 验证音频数据完整性
        has_audio = 'audio_bytes' in segment and len(segment['audio_bytes']) > 0
        audio_size = len(segment['audio_bytes']) if has_audio else 0
        
        # 获取音频元数据
        audio_metadata = segment.get('audio_metadata', {})
        completeness = audio_metadata.get('completeness', 0) if audio_metadata else 0
        
        # 记录音频数据信息
        if has_audio:
            logger.info(f"音频数据: {audio_size} 字节, 完整性: {completeness:.1f}%")
            if completeness < 90:
                logger.warning(f"音频数据不完整，可能影响后续处理")
        else:
            logger.warning(f"语音段没有音频数据")
            return
        
        # 存储音频数据并获取段ID
        segment_id = self._store_audio_segment(segment)
        logger.info(f"已存储音频段，ID: {segment_id}")
        
        # 使用AudioSegmentProcessor处理音频段
        transcript = ""
        translated_text = None
        
        # 使用whisper_sagemaker转录并使用SubtitlePipeline优化
        try:
            # 将音频数据转换为numpy数组
            audio_data = np.frombuffer(segment['audio_bytes'], dtype=np.float32)
            
            # 使用whisper_sagemaker模型转录
            if self.whisper_model:
                # 转录音频
                logger.info(f"使用WhisperSageMakerStreamingModel转录音频段: {segment['start']:.2f}s - {segment['end']:.2f}s")
                transcript = await self.whisper_model.transcribe(
                    audio_data, 
                    language=self.source_language  # 使用配置的源语言
                )
                logger.info(f"转录结果: {transcript}")
                
                # 创建Subtitle对象
                subtitle = Subtitle(
                    start=segment['start'],
                    end=segment['end'],
                    text=transcript
                )
                
                processed_subtitle = await self.subtitle_pipeline.process_subtitle(subtitle)
                
                # 从处理后的字幕中获取结果
                transcript = processed_subtitle.text
                translated_text = processed_subtitle.translated_text
                
                logger.info(f"处理后的字幕: {transcript}")
                if translated_text:
                    logger.info(f"翻译后的字幕: {translated_text}")
                
                # 将字幕写入WebVTT文件
                self._write_subtitle_to_webvtt(segment['start'], segment['end'], transcript)
                logger.info(f"已将字幕写入WebVTT文件: {segment['start']:.2f}s - {segment['end']:.2f}s: {transcript}")
            else:
                logger.warning("WhisperSageMakerStreamingModel未初始化，无法转录")
                transcript = f"[未转录的语音段: {segment['start']:.2f}s - {segment['end']:.2f}s]"
                translated_text = None
        except Exception as e:
            logger.error(f"处理音频段时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            transcript = f"[转录错误: {str(e)}]"
            translated_text = None
        
        # 准备字幕数据
        subtitle_data = {
            "id": segment_id or str(uuid.uuid4()),
            "start": segment['start'],
            "end": segment['end'],
            "text": transcript if transcript else f"检测到语音，从 {segment['start']:.2f}秒 到 {segment['end']:.2f}秒",
            "translated_text": translated_text,  # 可能为None
            "language": self.source_language,  # 添加语言信息
            "confidence": 0.9,  # 默认置信度
            "is_final": segment.get('is_final', True)
        }
        
        # 如果没有翻译文本，从字幕数据中移除该字段（避免显示null）
        if not translated_text:
            subtitle_data.pop("translated_text", None)
        
        # 准备结果数据
        result_data = {
            "type": "subtitle",
            "subtitle": subtitle_data,
            "timestamp": datetime.now().isoformat(),
            "client_id": self.client_id
        }
        
        # 如果有音频段ID，添加到结果中
        if segment_id:
            result_data["segment_id"] = segment_id
        
        # 调用结果回调函数（如果提供）
        if self.result_callback:
            try:
                if asyncio.iscoroutinefunction(self.result_callback):
                    await self.result_callback(result_data)
                else:
                    self.result_callback(result_data)
            except Exception as e:
                logger.error(f"调用结果回调函数时出错: {e}")
        
        # 为所有活跃连接发送结果（如果有WebSocket连接）
        for stream_id, connection in self.active_connections.items():
            if hasattr(connection, 'send'):  # 检查是否是WebSocket连接
                # 获取事件循环 - 优先使用存储的主循环
                loop = self.main_loop
                
                # 如果主循环未设置，尝试获取当前运行的循环
                if loop is None:
                    try:
                        loop = asyncio.get_running_loop()
                        # 更新主循环引用以便后续使用
                        self.main_loop = loop
                        logger.info("已更新主事件循环引用")
                    except RuntimeError:
                        logger.error("无法获取事件循环，无法发送语音段结果")
                        continue
                
                # 使用线程安全的方式调度协程
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._send_result(connection, stream_id, result_data),
                        loop
                    )
                    logger.debug(f"已安排发送字幕结果到客户端 {stream_id}")
                except Exception as e:
                    logger.error(f"安排发送字幕结果时出错: {e}")
    
    def _on_speech_segment_sync(self, segment: Dict[str, Any]):
        """
        语音段检测同步回调 - 这是VAC处理器实际调用的方法
        它会创建一个异步任务来处理语音段
        """
        # 获取事件循环 - 优先使用存储的主循环
        loop = self.main_loop
        
        # 如果主循环未设置，尝试获取当前运行的循环
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
                # 更新主循环引用以便后续使用
                self.main_loop = loop
                logger.info("已更新主事件循环引用")
            except RuntimeError:
                logger.error("无法获取事件循环，无法处理语音段")
                return
        
        # 使用线程安全的方式调度异步处理方法
        try:
            asyncio.run_coroutine_threadsafe(
                self._on_speech_segment(segment),
                loop
            )
            logger.debug(f"已安排异步处理语音段")
        except Exception as e:
            logger.error(f"安排异步处理语音段时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    async def _send_result(self, websocket: WebSocketServerProtocol, stream_id: str, result: Dict[str, Any]):
        """发送结果到客户端 - 按照前端期望的格式"""
        try:
            # 直接发送结果数据，因为在_on_speech_segment中已经格式化为前端期望的格式
            await websocket.send(json.dumps(result))
            
            logger.info(f"✅ 已向客户端发送字幕: {result.get('type')} - {result.get('subtitle', {}).get('text', 'N/A')}")
            logger.debug(f"完整消息内容: {result}")
            
        except Exception as e:
            logger.error(f"向客户端发送结果时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # 存储最近的语音段音频数据，用于按需获取
    # 格式: {segment_id: {'audio_bytes': bytes, 'metadata': dict}}
    _recent_audio_segments = {}
    _max_stored_segments = 50  # 最多存储的段数
    
    def _store_audio_segment(self, segment: Dict[str, Any]) -> str:
        """存储语音段音频数据，返回段ID"""
        if 'audio_bytes' not in segment or not segment['audio_bytes']:
            return None
            
        # 生成唯一ID
        segment_id = str(uuid.uuid4())
        
        # 存储音频数据和元数据
        self._recent_audio_segments[segment_id] = {
            'audio_bytes': segment['audio_bytes'],
            'metadata': {
                'start': segment['start'],
                'end': segment['end'],
                'duration': segment['duration'],
                'sample_rate': segment.get('sample_rate', 16000),
                'audio_format': segment.get('audio_format', 'float32'),
                'num_channels': segment.get('num_channels', 1),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # 如果存储的段数超过限制，删除最旧的
        if len(self._recent_audio_segments) > self._max_stored_segments:
            oldest_key = next(iter(self._recent_audio_segments))
            del self._recent_audio_segments[oldest_key]
            
        return segment_id
        
    async def handle_audio_request(self, websocket: WebSocketServerProtocol, segment_id: str):
        """处理音频数据请求"""
        if segment_id not in self._recent_audio_segments:
            await websocket.send(json.dumps({
                "type": "audio_data_response",
                "status": "error",
                "error": "找不到请求的音频段",
                "segment_id": segment_id
            }))
            return
            
        # 获取存储的音频数据
        audio_data = self._recent_audio_segments[segment_id]
        
        # 将音频数据转换为Base64
        import base64
        audio_base64 = base64.b64encode(audio_data['audio_bytes']).decode('utf-8')
        
        # 发送响应
        await websocket.send(json.dumps({
            "type": "audio_data_response",
            "status": "success",
            "segment_id": segment_id,
            "audio_base64": audio_base64,
            "metadata": audio_data['metadata']
        }))
        
        logger.info(f"已发送音频段 {segment_id} 的数据，大小: {len(audio_data['audio_bytes'])} 字节")
    
    def set_result_callback(self, callback: Callable):
        """设置结果回调函数"""
        self.result_callback = callback
    
    def get_active_streams(self) -> List[str]:
        """获取活跃的流ID列表"""
        return list(self.active_connections.keys())
    
    def get_stream_status(self, stream_id: str) -> Dict[str, Any]:
        """获取流的状态信息"""
        if stream_id not in self.audio_buffers:
            return {"status": "not_found"}
        
        return {
            "status": "active",
            "buffer_size": len(self.audio_buffers[stream_id]),
            "queue_size": self.vac_queues[stream_id].qsize() if stream_id in self.vac_queues else 0,
            "has_processing_task": stream_id in self.processing_tasks,
            "is_ended": self.end_flags.get(stream_id, False)
        }
