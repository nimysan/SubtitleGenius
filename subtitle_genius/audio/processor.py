"""音频处理器"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Union, Any
from loguru import logger
import ffmpeg

from ..core.config import config


class AudioProcessor:
    """音频处理器"""
    
    def __init__(self):
        self.sample_rate = config.audio_sample_rate
        self.chunk_size = config.audio_chunk_size
        logger.info("AudioProcessor initialized")
    
    async def process_file(self, file_path: Path) -> np.ndarray:
        """处理音频文件"""
        try:
            # 使用librosa加载音频文件
            audio_data, sr = librosa.load(
                str(file_path), 
                sr=self.sample_rate,
                mono=True
            )
            
            logger.info(f"Loaded audio file: {file_path}, duration: {len(audio_data)/sr:.2f}s")
            return audio_data
            
        except Exception as e:
            logger.error(f"Failed to process audio file {file_path}: {e}")
            raise
    
    async def extract_from_video(self, video_path: Path) -> np.ndarray:
        """从视频文件中提取音频"""
        try:
            # 使用ffmpeg从视频中提取音频
            audio_data, _ = (
                ffmpeg
                .input(str(video_path))
                .audio
                .output('pipe:', format='wav', acodec='pcm_s16le', ar=self.sample_rate)
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            # 将字节数据转换为numpy数组
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0  # 归一化
            
            logger.info(f"Extracted audio from video: {video_path}")
            return audio_array
            
        except Exception as e:
            logger.error(f"Failed to extract audio from video {video_path}: {e}")
            raise
    
    def combine_chunks(self, audio_chunks: List[np.ndarray]) -> np.ndarray:
        """合并音频块"""
        try:
            combined = np.concatenate(audio_chunks)
            logger.debug(f"Combined {len(audio_chunks)} audio chunks")
            return combined
            
        except Exception as e:
            logger.error(f"Failed to combine audio chunks: {e}")
            raise
    
    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """预处理音频数据"""
        try:
            # 音频预处理步骤
            # 1. 去除静音部分
            audio_data = self._remove_silence(audio_data)
            
            # 2. 音量归一化
            audio_data = self._normalize_volume(audio_data)
            
            # 3. 降噪 (可选)
            # audio_data = self._denoise(audio_data)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            raise
    
    def _remove_silence(self, audio_data: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """移除静音部分"""
        # 简单的静音检测和移除
        non_silent = np.abs(audio_data) > threshold
        return audio_data[non_silent]
    
    def _normalize_volume(self, audio_data: np.ndarray) -> np.ndarray:
        """音量归一化"""
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data / max_val
        return audio_data
    
    def save_audio(self, audio_data: np.ndarray, output_path: Path) -> None:
        """保存音频数据到文件"""
        try:
            sf.write(
                str(output_path),
                audio_data,
                self.sample_rate,
                format='WAV'
            )
            logger.info(f"Audio saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save audio to {output_path}: {e}")
            raise
