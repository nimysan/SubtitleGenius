"""
语音活动检测(VAD)处理器
使用Silero-VAD模型进行语音检测
"""

import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
from collections import Counter, deque
from typing import List, Dict, Tuple, Optional, Union, Generator
from dataclasses import dataclass
from pathlib import Path
import io
import wave
import time

logger = logging.getLogger(__name__)

class VADIterator:
    def __init__(self,
                 model,
                 threshold: float = 0.5,
                 sampling_rate: int = 16000,
                 min_silence_duration_ms: int = 500,
                 speech_pad_ms: int = 100
                 ):
        """
        流式处理模拟类

        参数
        ----------
        model: 预加载的 .jit/.onnx silero VAD 模型

        threshold: float (默认 - 0.5)
            语音阈值。Silero VAD 为每个音频块输出语音概率，高于此值的概率被视为语音。
            最好为每个数据集单独调整此参数，但"懒惰"的 0.5 对大多数数据集都相当不错。

        sampling_rate: int (默认 - 16000)
            目前 silero VAD 模型支持 8000 和 16000 采样率

        min_silence_duration_ms: int (默认 - 500 毫秒)
            在每个语音块结束时，等待 min_silence_duration_ms 才将其分离

        speech_pad_ms: int (默认 - 100 毫秒)
            最终语音块的每一侧都由 speech_pad_ms 填充
        """

        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate

        if sampling_rate not in [8000, 16000]:
            raise ValueError('VADIterator 不支持 [8000, 16000] 以外的采样率')

        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.reset_states()

    def reset_states(self):
        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    @torch.no_grad()
    def __call__(self, x, return_seconds=False, time_resolution: int = 1):
        """
        x: torch.Tensor
            音频块（参见仓库中的示例）

        return_seconds: bool (默认 - False)
            是否以秒为单位返回时间戳（默认 - 样本）

        time_resolution: int (默认 - 1)
            请求为秒时语音坐标的时间分辨率
        """

        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x)
            except:
                raise TypeError("音频无法转换为张量。请手动转换")

        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)
        self.current_sample += window_size_samples

        speech_prob = self.model(x, self.sampling_rate).item()

        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0

        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            speech_start = max(0, self.current_sample - self.speech_pad_samples - window_size_samples)
            return {'start': int(speech_start) if not return_seconds else round(speech_start / self.sampling_rate, time_resolution)}

        if (speech_prob < self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            else:
                speech_end = self.temp_end + self.speech_pad_samples - window_size_samples
                self.temp_end = 0
                self.triggered = False
                return {'end': int(speech_end) if not return_seconds else round(speech_end / self.sampling_rate, time_resolution)}

        return None


class FixedVADIterator(VADIterator):
    '''
    修复VADIterator，允许处理任意长度的音频，而不仅仅是恰好512帧。
    如果一次处理的音频很长并且检测到多个有声段，
    则__call__返回第一个段的开始，以及最后一个段的结束（或中间，表示没有结束）。
    '''

    def reset_states(self):
        super().reset_states()
        self.buffer = np.array([], dtype=np.float32)

    def __call__(self, x, return_seconds=False):
        self.buffer = np.append(self.buffer, x) 
        ret = None
        while len(self.buffer) >= 512:
            r = super().__call__(self.buffer[:512], return_seconds=return_seconds)
            self.buffer = self.buffer[512:]
            if ret is None:
                ret = r
            elif r is not None:
                if 'end' in r:
                    ret['end'] = r['end']  # 后面的结束点
                if 'start' in r and 'end' in ret:  # 有一个更早的开始点
                    # 删除结束点，将此段与前一段合并
                    del ret['end']
        return ret if ret != {} else None

@dataclass
class VADConfig:
    """VAD配置"""
    sample_rate: int = 16000        # 采样率
    threshold: float = 0.5          # VAD置信度阈值
    min_speech_duration_ms: int = 250  # 最小语音持续时间(毫秒)
    min_silence_duration_ms: int = 100  # 最小静音持续时间(毫秒)
    window_size_samples: int = 1536  # VAD窗口大小
    speech_pad_ms: int = 30         # 语音片段前后填充(毫秒)


class StreamingVADProcessor:
    """使用VADIterator处理流式音频"""
    
    def __init__(self, config: Optional[VADConfig] = None):
        """初始化流式VAD处理器"""
        self.config = config or VADConfig()
        self.sample_rate = self.config.sample_rate
        
        # 检测设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载Silero VAD模型
        self._model_loaded = False
        try:
            # 使用简化的模型加载逻辑
            logger.info("加载Silero VAD模型")
            self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
            
            # 将模型移动到指定设备
            self.model = self.model.to(self.device)
            
            # 获取VAD函数
            (self.get_speech_timestamps, self.save_audio, self.read_audio, 
             self.VADIterator, self.collect_chunks) = self.utils
            
            # 创建FixedVADIterator实例
            self.vad_iterator = FixedVADIterator(
                model=self.model,
                threshold=self.config.threshold,
                sampling_rate=self.sample_rate,
                min_silence_duration_ms=self.config.min_silence_duration_ms,
                speech_pad_ms=self.config.speech_pad_ms
            )
            
            self._model_loaded = True
            logger.info(f"Silero VAD模型加载成功，使用设备: {self.device}")
            
        except Exception as e:
            logger.error(f"Silero VAD初始化失败: {e}")
            logger.error("请尝试手动下载模型:")
            logger.error("1. 打开Python交互式终端")
            logger.error("2. 运行: import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=True)")
            logger.error("3. 或者下载模型文件到: ~/.cache/torch/hub/snakers4_silero-vad_master")
            self._model_loaded = False
        
        # 存储检测到的语音片段
        self.speech_segments = []
        
        # 记录已处理的音频时长
        self.total_processed_duration = 0.0
        
        # 存储已处理的语音片段ID，用于去重
        self.processed_segment_ids = set()
        
        logger.info(f"流式VAD处理器初始化完成，使用设备: {self.device} - {self.vad_iterator}")
    
    def is_available(self) -> bool:
        """检查VAD模型是否可用"""
        return self._model_loaded
    
    def process_chunk(self, audio_chunk: np.ndarray) -> List[Dict]:
        """
        处理音频块，返回检测到的语音片段
        
        参数:
            audio_chunk: 音频数据，numpy数组
            
        返回:
            检测到的语音片段列表
        """
        if not self._model_loaded:
            logger.warning("VAD模型未加载，无法处理音频")
            return []
        
        # 确保音频数据是float32格式
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        
        # 使用FixedVADIterator处理音频块
        # FixedVADIterator可以处理任意长度的音频，不需要分割成固定大小的块
        speech_dict = []
        try:
            # 转换为PyTorch张量
            audio_tensor = torch.from_numpy(audio_chunk).to(self.device)
            
            # 使用FixedVADIterator处理音频
            result = self.vad_iterator(audio_tensor)
            if result:
                speech_dict.append(result)
        except Exception as e:
            logger.error(f"处理音频块失败: {e}")
        
        # 更新已处理的音频时长
        chunk_duration = len(audio_chunk) / self.sample_rate
        self.total_processed_duration += chunk_duration
        
        # 如果检测到语音片段
        if speech_dict:
            # 提取语音片段
            speech_segments = []
            
            for segment in speech_dict:
                # 计算绝对时间戳
                absolute_start = self.total_processed_duration - chunk_duration + segment['start'] / self.sample_rate
                absolute_end = self.total_processed_duration - chunk_duration + segment['end'] / self.sample_rate
                
                # 创建片段ID
                segment_id = f"{absolute_start:.3f}_{absolute_end:.3f}"
                
                # 如果片段未处理过，添加到结果列表
                if segment_id not in self.processed_segment_ids:
                    # 提取音频片段
                    start_sample = segment['start']
                    end_sample = segment['end']
                    
                    if end_sample <= len(audio_chunk):
                        speech_segment = audio_chunk[start_sample:end_sample]
                        
                        # 创建结果字典
                        result_segment = {
                            'audio': speech_segment,
                            'start': absolute_start,
                            'end': absolute_end,
                            'duration': absolute_end - absolute_start
                        }
                        
                        speech_segments.append(result_segment)
                        self.processed_segment_ids.add(segment_id)
                        
                        logger.info(f"检测到语音片段: 开始={absolute_start:.2f}秒, 结束={absolute_end:.2f}秒, "
                                   f"时长={(absolute_end - absolute_start):.2f}秒")
            
            return speech_segments
        
        return []
    
    def reset(self):
        """重置VADIterator"""
        self.vad_iterator.reset_states()
        self.speech_segments = []
        self.total_processed_duration = 0.0
        self.processed_segment_ids = set()
        logger.info("VADIterator已重置")
    
    def extract_speech_segments(self, audio_data: np.ndarray) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        从音频数据中提取语音片段
        
        参数:
            audio_data: 音频数据，numpy数组
            
        返回:
            语音片段列表和时间戳列表
        """
        if not self._model_loaded:
            logger.warning("VAD模型未加载，无法提取语音片段")
            return [], []
        
        # 确保音频数据是float32格式
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # 使用get_speech_timestamps获取语音时间戳
        # 直接处理整个音频数据，不需要分割成固定大小的块
        try:
            # 转换为PyTorch张量
            audio_tensor = torch.from_numpy(audio_data).to(self.device)
            
            # 使用get_speech_timestamps获取语音时间戳
            all_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.model,
                threshold=self.config.threshold,
                sampling_rate=self.sample_rate,
                min_silence_duration_ms=self.config.min_silence_duration_ms,
                speech_pad_ms=self.config.speech_pad_ms
            )
        except Exception as e:
            logger.error(f"提取语音片段失败: {e}")
            all_timestamps = []
        
        # 合并相邻的时间戳
        merged_timestamps = []
        if all_timestamps:
            current_ts = all_timestamps[0].copy()
            for ts in all_timestamps[1:]:
                # 如果当前时间戳的结束时间与下一个时间戳的开始时间相差不大，合并它们
                if ts['start'] - current_ts['end'] < 1000:  # 1000个样本，约62.5毫秒
                    current_ts['end'] = ts['end']
                else:
                    merged_timestamps.append(current_ts)
                    current_ts = ts.copy()
            merged_timestamps.append(current_ts)
        
        speech_timestamps = merged_timestamps
        
        # 提取语音片段
        speech_segments = []
        for timestamp in speech_timestamps:
            start_sample = timestamp['start']
            end_sample = timestamp['end']
            
            if end_sample <= len(audio_data):
                speech_segment = audio_data[start_sample:end_sample]
                speech_segments.append(speech_segment)
        
        return speech_segments, speech_timestamps
    
    def merge_segments(self, speech_segments: List[np.ndarray], max_gap_ms: int = 500) -> List[np.ndarray]:
        """
        合并相近的语音片段
        
        参数:
            speech_segments: 语音片段列表
            max_gap_ms: 最大间隔(毫秒)
            
        返回:
            合并后的语音片段列表
        """
        if len(speech_segments) <= 1:
            return speech_segments
        
        # 这里简化处理，实际上需要更复杂的逻辑
        # 在实际实现中，需要考虑时间戳、采样率等因素
        # 这里简单地返回原始片段
        return speech_segments
    
    def convert_to_wav(self, audio_data: np.ndarray) -> bytes:
        """
        将音频数据转换为WAV格式
        
        参数:
            audio_data: 音频数据，numpy数组
            
        返回:
            WAV格式的二进制数据
        """
        # 确保音频数据是float32格式
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # 转换为16-bit PCM
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # 创建WAV文件
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        return wav_buffer.getvalue()


class VADBufferProcessor:
    """VAD缓冲区处理器，用于管理音频缓冲区和VAD处理"""
    
    def __init__(self, config: Optional[VADConfig] = None, buffer_size_seconds: float = 30.0):
        """
        初始化VAD缓冲区处理器
        
        参数:
            config: VAD配置，如果为None则使用默认配置
            buffer_size_seconds: 缓冲区大小（秒）
        """
        self.config = config or VADConfig()
        self.sample_rate = self.config.sample_rate
        self.buffer_size = int(buffer_size_seconds * self.sample_rate)
        
        # 初始化VAD处理器
        self.vad_processor = StreamingVADProcessor(self.config)
        
        # 初始化音频缓冲区
        self.buffer = np.array([], dtype=np.float32)
        
        # 跟踪已处理的样本数
        self.processed_samples = 0
        
        # 存储检测到的语音片段
        self.pending_segments = []
        
        # 记录开始时间
        self.start_time = time.time()
        
        logger.info(f"VAD缓冲区处理器初始化完成，缓冲区大小: {buffer_size_seconds}秒")
        
    def add_audio_chunk(self, audio_chunk: np.ndarray) -> None:
        """
        添加音频块到缓冲区
        
        参数:
            audio_chunk: 音频数据，numpy数组
        """
        # 确保音频数据是float32格式
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        
        # 添加到缓冲区
        self.buffer = np.concatenate([self.buffer, audio_chunk])
        
        # 如果缓冲区超过最大大小，移除最早的数据
        if len(self.buffer) > self.buffer_size:
            excess = len(self.buffer) - self.buffer_size
            self.buffer = self.buffer[excess:]
            self.processed_samples += excess
            
        logger.debug(f"添加音频块到缓冲区，当前缓冲区大小: {len(self.buffer)/self.sample_rate:.2f}秒")
    
    def process_buffer(self) -> List[Tuple[np.ndarray, Dict]]:
        """
        处理缓冲区中的音频，使用VAD检测语音片段
        
        返回:
            检测到的语音片段列表，每个元素为(音频数据, 时间戳信息)
        """
        if not self.vad_processor.is_available():
            logger.warning("VAD处理器不可用，无法处理缓冲区")
            return []
        
        if len(self.buffer) == 0:
            logger.debug("缓冲区为空，无法处理")
            return []
        
        # 使用VAD处理缓冲区中的音频
        speech_segments, speech_timestamps = self.vad_processor.extract_speech_segments(self.buffer)
        
        # 如果没有检测到语音片段，返回空列表
        if not speech_segments:
            logger.debug("未检测到语音片段")
            return []
        
        # 合并相近的语音片段
        if len(speech_segments) > 1:
            merged_segments = self.vad_processor.merge_segments(
                speech_segments, 
                max_gap_ms=500  # 可配置的参数
            )
            logger.debug(f"合并后的语音片段数: {len(merged_segments)}")
            
            # 重新获取时间戳（合并后的片段需要重新计算时间戳）
            if len(merged_segments) != len(speech_segments):
                # 这里简化处理，实际应该重新计算合并后的时间戳
                # 但由于我们已经有了原始时间戳，可以基于原始时间戳进行处理
                speech_segments = merged_segments
        
        # 创建结果列表
        result = []
        for i, (segment, timestamp) in enumerate(zip(speech_segments, speech_timestamps)):
            # 计算绝对时间戳（相对于处理开始的时间）
            abs_timestamp = {
                'start': self.processed_samples / self.sample_rate + timestamp['start'] / self.sample_rate,
                'end': self.processed_samples / self.sample_rate + timestamp['end'] / self.sample_rate,
                'duration': (timestamp['end'] - timestamp['start']) / self.sample_rate,
                'segment_index': i
            }
            
            logger.debug(f"检测到语音片段 {i+1}/{len(speech_segments)}: "
                        f"{abs_timestamp['start']:.2f}s - {abs_timestamp['end']:.2f}s, "
                        f"时长: {abs_timestamp['duration']:.2f}s")
            
            result.append((segment, abs_timestamp))
        
        # 将检测到的语音片段添加到待处理列表
        self.pending_segments.extend(result)
        
        return result
    
    def get_pending_segments(self) -> List[Tuple[np.ndarray, Dict]]:
        """
        获取待处理的语音片段
        
        返回:
            待处理的语音片段列表
        """
        segments = self.pending_segments
        self.pending_segments = []
        return segments
    
    def clear_buffer(self) -> None:
        """清空缓冲区"""
        self.buffer = np.array([], dtype=np.float32)
        self.pending_segments = []
        logger.debug("缓冲区已清空")
        
    def get_buffer_duration(self) -> float:
        """获取当前缓冲区的时长（秒）"""
        return len(self.buffer) / self.sample_rate
    
    def get_elapsed_time(self) -> float:
        """获取从初始化到现在的时间（秒）"""
        return time.time() - self.start_time
    
    def convert_to_wav(self, audio_data: np.ndarray) -> bytes:
        """
        将音频数据转换为WAV格式
        
        参数:
            audio_data: 音频数据，numpy数组
            
        返回:
            WAV格式的二进制数据
        """
        # 确保音频数据是float32格式
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # 转换为16-bit PCM
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # 创建WAV文件
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        return wav_buffer.getvalue()
    
    def process_audio_stream(self, audio_chunks: Generator[np.ndarray, None, None]) -> Generator[Tuple[np.ndarray, Dict], None, None]:
        """
        处理音频流，返回检测到的语音片段
        
        参数:
            audio_chunks: 音频块生成器
            
        返回:
            语音片段生成器，每个元素为(音频数据, 时间戳信息)
        """
        for chunk in audio_chunks:
            # 添加音频块到缓冲区
            self.add_audio_chunk(chunk)
            
            # 处理缓冲区，获取语音片段
            segments = self.process_buffer()
            
            # 返回检测到的语音片段
            for segment in segments:
                yield segment
