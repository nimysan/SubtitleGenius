import asyncio
import json
import logging
import os
import io
import uuid
import numpy as np
import soundfile as sf
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Iterator

import websockets
from websockets.server import WebSocketServerProtocol

# 导入VAC处理器
from subtitle_genius.stream.vac_processor import VACProcessor
# 导入SageMaker Whisper流式处理模型
from subtitle_genius.models.whisper_sagemaker_streaming import WhisperSageMakerStreamingModel
# 导入SubtitlePipeline和Subtitle
from subtitle_genius.pipeline.subtitle_pipeline import SubtitlePipeline
from subtitle_genius.subtitle.models import Subtitle

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("websocket_server")

# 存储接收到的音频文件的目录
AUDIO_DIR = "received_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# 定义处理块大小 (512*8 = 4096字节)
PROCESSING_CHUNK_SIZE = 512 * 8



class ContinuousAudioProcessor:
    """处理连续音频流的处理器"""
    
    def __init__(self):
        # 存储主事件循环，用于线程安全操作
        try:
            self.main_loop = asyncio.get_running_loop()
            logger.info("已获取主事件循环引用")
        except RuntimeError:
            logger.warning("初始化时未找到运行中的事件循环，将在首次使用时获取")
            self.main_loop = None
        
        # 创建SubtitlePipeline实例，用于处理字幕
        self.subtitle_pipeline = SubtitlePipeline(
            source_language="zh",  # 源语言：中文
            target_language="zh",  # 目标语言：中文
            correction_enabled=True,  # 启用纠正
            correction_service="bedrock",
            translation_enabled=False,  # 不启用翻译
            translation_service="bedrock",
            scene_description="足球比赛",  # 场景描述：足球比赛
            output_format="srt"
        )
        logger.info("SubtitlePipeline初始化完成")
            
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
        # 存储活跃的连接
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
    
    async def start_stream(self, stream_id: str, websocket: WebSocketServerProtocol):
        """开始一个新的音频流处理"""
        logger.info(f"开始新的音频流: {stream_id}")
        
        # 确保我们有主事件循环的引用
        if self.main_loop is None:
            self.main_loop = asyncio.get_running_loop()
            logger.info("已在start_stream中获取主事件循环引用")
        
        # 存储连接信息
        self.active_connections[stream_id] = websocket
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
        处理音频数据 - 这是音频数据从WebSocket进入处理流程的入口
        
        流程：
        1. 接收音频数据 → 添加到缓冲区
        2. 当缓冲区达到处理块大小时 → 提取块 → 添加到VAC队列
        3. VAC队列 → ThreadSafeIterator → VAC处理器
        """
        if stream_id not in self.active_connections:
            raise ValueError(f"找不到流 {stream_id}")
        
        try:
            
            audio_data, sample_rate = self.bytes_to_audio_data(binary_data)
            
            print(f"音频数据: 形状={audio_data.shape}, 类型={audio_data.dtype}")
            print(f"采样率: {sample_rate} Hz")
            print(f"音频时长: {len(audio_data)}长度")
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
                    language="zh"  # 使用中文
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
        
        # 为所有活跃连接发送结果
        for stream_id, websocket in self.active_connections.items():
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
                    return
            
            # 准备结果数据
            result_data = {
                "type": "speech_segment",
                "start": segment['start'],
                "end": segment['end'],
                "duration": segment['duration'],
                "transcript": transcript if transcript else f"检测到语音，从 {segment['start']:.2f}秒 到 {segment['end']:.2f}秒",
                "translated_text": translated_text,  # 添加翻译结果
                "timestamp": datetime.now().isoformat(),
                "has_audio": has_audio,
                "audio_size": audio_size,
                "audio_format": segment.get('audio_format', 'float32'),
                "sample_rate": segment.get('sample_rate', 16000),
                "num_channels": segment.get('num_channels', 1),
                "completeness": completeness,
                "is_final": segment.get('is_final', False),
                "has_translation": translated_text is not None  # 添加是否有翻译的标志
            }
            
            # 如果有音频段ID，添加到结果中
            if segment_id:
                result_data["segment_id"] = segment_id
            
            # 使用线程安全的方式调度协程
            try:
                asyncio.run_coroutine_threadsafe(
                    self._send_result(websocket, stream_id, result_data),
                    loop
                )
                logger.debug(f"已安排发送语音段结果到客户端 {stream_id}")
            except Exception as e:
                logger.error(f"安排发送语音段结果时出错: {e}")
    
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
        """发送结果到客户端"""
        try:
            # 如果需要，可以将音频数据转换为Base64以便通过JSON发送
            # 注意：这里我们不直接发送音频数据，因为它可能很大，而是提供一个标志让客户端知道有音频可用
            # 客户端可以通过单独的请求获取音频数据
            
            # 创建响应数据
            response_data = {
                "type": "transcription_result",
                "stream_id": stream_id,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
            # 发送JSON响应
            await websocket.send(json.dumps(response_data))
            
            logger.debug(f"已向客户端 {stream_id} 发送结果: {result['type']}")
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


class WebSocketServer:
    """WebSocket服务器，处理前端连接"""
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.audio_processor = ContinuousAudioProcessor()
        self.active_connections = {}
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """处理新的WebSocket连接"""
        connection_id = str(uuid.uuid4())
        self.active_connections[connection_id] = websocket
        
        try:
            logger.info(f"新连接建立: {connection_id}")
            await websocket.send(json.dumps({
                "type": "connection_established",
                "connection_id": connection_id
            }))
            
            stream_id = None
            
            async for message in websocket:
                if isinstance(message, str):
                    # 处理文本消息
                    new_stream_id = await self._handle_text_message(connection_id, message, websocket)
                    if new_stream_id:
                        stream_id = new_stream_id
                else:
                    # 处理二进制消息（音频数据）
                    new_stream_id = await self._handle_binary_message(connection_id, message, websocket, stream_id)
                    if new_stream_id:
                        stream_id = new_stream_id
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"连接关闭: {connection_id}")
        except Exception as e:
            logger.error(f"处理连接时出错: {e}")
        finally:
            # 清理资源
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
            if stream_id:
                await self.audio_processor.stop_stream(stream_id)
    
    async def _handle_text_message(self, connection_id: str, message: str, websocket: WebSocketServerProtocol) -> Optional[str]:
        """处理文本消息"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            logger.debug(f"收到来自 {connection_id} 的文本消息: {message_type}")
            
            if message_type == "start_stream":
                # 开始新的音频流
                stream_id = str(uuid.uuid4())
                await self.audio_processor.start_stream(stream_id, websocket)
                await websocket.send(json.dumps({
                    "type": "stream_started",
                    "stream_id": stream_id
                }))
                return stream_id
            
            elif message_type == "stop_stream":
                # 停止现有的音频流
                stream_id = data.get("stream_id")
                if stream_id:
                    await self.audio_processor.stop_stream(stream_id)
                    await websocket.send(json.dumps({
                        "type": "stream_stopped",
                        "stream_id": stream_id
                    }))
            
            elif message_type == "ping":
                # 简单的ping-pong测试
                await websocket.send(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }))
            
            elif message_type == "get_audio_data":
                # 处理音频数据请求
                segment_id = data.get("segment_id")
                if segment_id:
                    # 直接调用音频处理器的handle_audio_request方法
                    if hasattr(self.audio_processor, 'handle_audio_request'):
                        await self.audio_processor.handle_audio_request(websocket, segment_id)
                    else:
                        logger.error("音频处理器没有handle_audio_request方法")
                        await websocket.send(json.dumps({
                            "type": "error",
                            "error": "服务器不支持音频数据请求功能"
                        }))
                else:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "error": "请求音频数据时未提供segment_id"
                    }))
            
            else:
                logger.warning(f"未知消息类型: {message_type}")
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": f"未知消息类型: {message_type}"
                }))
            
            return None
              
        
        except json.JSONDecodeError:
            logger.error(f"收到无效的JSON: {message}")
            await websocket.send(json.dumps({
                "type": "error",
                "error": "无效的JSON格式"
            }))
        except Exception as e:
            logger.error(f"处理文本消息时出错: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "error": str(e)
            }))
        
        return None
    
    async def _handle_binary_message(self, connection_id: str, message: bytes, websocket: WebSocketServerProtocol, stream_id: Optional[str]) -> str:
        """处理二进制消息（音频数据）"""
        try:
            # 保存音频块到文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{AUDIO_DIR}/{connection_id}_{timestamp}.wav"
            
            with open(filename, "wb") as f:
                f.write(message)
            
            logger.info(f"收到来自 {connection_id} 的音频块: {len(message)}字节, 保存到 {filename}")
            
            # 如果没有流ID，自动创建一个
            if not stream_id:
                stream_id = str(uuid.uuid4())
                logger.info(f"为音频数据自动创建流: {stream_id}")
                await self.audio_processor.start_stream(stream_id, websocket)
                
                # 通知客户端新流已创建
                await websocket.send(json.dumps({
                    "type": "stream_started",
                    "stream_id": stream_id,
                    "auto_created": True
                }))
            
            # 处理音频数据
            result = await self.audio_processor.process_audio(stream_id, message)
            
            # 发送处理结果给客户端
            await websocket.send(json.dumps({
                "type": "audio_processing",
                "stream_id": stream_id,
                "result": result
            }))
            
            return stream_id
        
        except Exception as e:
            logger.error(f"处理二进制消息时出错: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "error": str(e)
            }))
            return stream_id if stream_id else ""
    
    async def start_server(self):
        """启动WebSocket服务器"""
        server = await websockets.serve(
            self.handle_connection,
            self.host,
            self.port
        )
        
        logger.info(f"WebSocket服务器已启动，地址: ws://{self.host}:{self.port}")
        
        return server


async def main():
    """主入口点"""
    server = WebSocketServer()
    ws_server = await server.start_server()
    
    try:
        # 保持服务器运行直到被中断
        await asyncio.Future()
    except asyncio.CancelledError:
        logger.info("服务器被取消")
    finally:
        # 关闭所有活跃的流
        for stream_id in list(server.audio_processor.active_connections.keys()):
            await server.audio_processor.stop_stream(stream_id)
        
        # 关闭WebSocket服务器
        ws_server.close()
        await ws_server.wait_closed()
        logger.info("WebSocket服务器已关闭")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("服务器被用户停止")
