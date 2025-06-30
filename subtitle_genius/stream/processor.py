"""实时流处理器"""

import asyncio
import numpy as np
from typing import AsyncGenerator, Optional
from loguru import logger
import pyaudio
import threading
import queue

from ..core.config import config


class StreamProcessor:
    """实时音频流处理器"""
    
    def __init__(self):
        self.sample_rate = config.audio_sample_rate
        self.chunk_size = config.audio_chunk_size
        self.format = pyaudio.paFloat32
        self.channels = 1
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
        logger.info("StreamProcessor initialized")
    
    async def start_microphone_stream(self) -> AsyncGenerator[np.ndarray, None]:
        """开始麦克风音频流"""
        try:
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.is_recording = True
            self.stream.start_stream()
            
            logger.info("Microphone stream started")
            
            while self.is_recording:
                try:
                    # 从队列中获取音频数据
                    audio_data = self.audio_queue.get(timeout=1.0)
                    yield audio_data
                except queue.Empty:
                    continue
                    
        except Exception as e:
            logger.error(f"Microphone stream error: {e}")
            raise
        finally:
            self.stop_stream()
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音频回调函数"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # 将音频数据转换为numpy数组
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_queue.put(audio_data)
        
        return (None, pyaudio.paContinue)
    
    async def process_rtmp_stream(self, rtmp_url: str) -> AsyncGenerator[np.ndarray, None]:
        """处理RTMP流"""
        try:
            import ffmpeg
            
            # 使用ffmpeg处理RTMP流
            process = (
                ffmpeg
                .input(rtmp_url)
                .audio
                .output('pipe:', format='f32le', acodec='pcm_f32le', ar=self.sample_rate)
                .run_async(pipe_stdout=True, pipe_stderr=True)
            )
            
            logger.info(f"Started processing RTMP stream: {rtmp_url}")
            
            while True:
                # 读取音频数据
                audio_bytes = process.stdout.read(self.chunk_size * 4)  # 4 bytes per float32
                
                if not audio_bytes:
                    break
                
                # 转换为numpy数组
                audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
                yield audio_data
                
                await asyncio.sleep(0.01)  # 小延迟避免CPU占用过高
                
        except Exception as e:
            logger.error(f"RTMP stream processing error: {e}")
            raise
    
    async def process_file_stream(self, file_path: str) -> AsyncGenerator[np.ndarray, None]:
        """处理文件流（模拟实时）"""
        try:
            import librosa
            
            # 加载音频文件
            audio_data, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            
            logger.info(f"Processing file stream: {file_path}")
            
            # 按块发送音频数据
            for i in range(0, len(audio_data), self.chunk_size):
                chunk = audio_data[i:i + self.chunk_size]
                
                # 如果块不够大，用零填充
                if len(chunk) < self.chunk_size:
                    chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)))
                
                yield chunk
                
                # 模拟实时延迟
                await asyncio.sleep(self.chunk_size / self.sample_rate)
                
        except Exception as e:
            logger.error(f"File stream processing error: {e}")
            raise
    
    def stop_stream(self) -> None:
        """停止音频流"""
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        logger.info("Audio stream stopped")
    
    def __del__(self):
        """析构函数"""
        self.stop_stream()
        if hasattr(self, 'audio'):
            self.audio.terminate()
