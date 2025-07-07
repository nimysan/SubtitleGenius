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

@dataclass
class VADConfig:
    """VAD配置"""
    sample_rate: int = 16000        # 采样率
    threshold: float = 0.5          # VAD置信度阈值
    min_speech_duration_ms: int = 250  # 最小语音持续时间(毫秒)
    min_silence_duration_ms: int = 100  # 最小静音持续时间(毫秒)
    window_size_samples: int = 1536  # VAD窗口大小
    speech_pad_ms: int = 30         # 语音片段前后填充(毫秒)


class SileroVADProcessor:
    """Silero VAD处理器"""
    
    def __init__(self, config: Optional[VADConfig] = None):
        """
        初始化Silero VAD处理器
        
        参数:
            config: VAD配置，如果为None则使用默认配置
        """
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
            
            self._model_loaded = True
            logger.info(f"Silero VAD模型加载成功，使用设备: {self.device}")
            
            logger.info(f"Silero VAD初始化成功，使用设备: {self.device}")
            
        except Exception as e:
            logger.error(f"Silero VAD初始化失败: {e}")
            logger.error("请尝试手动下载模型:")
            logger.error("1. 打开Python交互式终端")
            logger.error("2. 运行: import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=True)")
            logger.error("3. 或者下载模型文件到: ~/.cache/torch/hub/snakers4_silero-vad_master")
            self._model_loaded = False
    
    def is_available(self) -> bool:
        """检查VAD模型是否可用"""
        return self._model_loaded
    
    def get_speech_segments(self, audio_data: np.ndarray) -> List[Dict[str, int]]:
        """
        获取语音片段的时间戳
        
        参数:
            audio_data: 音频数据，numpy数组
            
        返回:
            语音片段列表，格式为[{'start': start_sample, 'end': end_sample}, ...]
        """
        if not self._model_loaded:
            logger.warning("VAD模型未加载，无法获取语音片段")
            return []
        
        # 确保音频数据是float32格式
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # 转换为PyTorch张量
        audio_tensor = torch.from_numpy(audio_data).to(self.device)
        
        # 获取语音时间戳
        try:
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.model,
                threshold=self.config.threshold,
                sampling_rate=self.sample_rate,
                min_speech_duration_ms=self.config.min_speech_duration_ms,
                min_silence_duration_ms=self.config.min_silence_duration_ms,
                window_size_samples=self.config.window_size_samples,
                speech_pad_ms=self.config.speech_pad_ms
            )
            
            logger.debug(f"检测到 {len(speech_timestamps)} 个语音片段")
            return speech_timestamps
            
        except Exception as e:
            logger.error(f"获取语音时间戳失败: {e}")
            return []
    
    def extract_speech_segments(self, audio_data: np.ndarray) -> Tuple[List[np.ndarray], List[Dict[str, int]]]:
        """
        提取语音片段
        
        参数:
            audio_data: 音频数据，numpy数组
            
        返回:
            (语音片段列表, 时间戳列表)
        """
        speech_timestamps = self.get_speech_segments(audio_data)
        speech_segments = []
        
        for segment in speech_timestamps:
            start_sample = segment['start']
            end_sample = segment['end']
            
            # 提取语音片段
            if end_sample <= len(audio_data):
                speech_segment = audio_data[start_sample:end_sample]
                speech_segments.append(speech_segment)
        
        return speech_segments, speech_timestamps
    
    def merge_segments(self, segments: List[np.ndarray], max_gap_ms: int = 500) -> List[np.ndarray]:
        """
        合并相近的语音片段
        
        参数:
            segments: 语音片段列表
            max_gap_ms: 最大间隔(毫秒)，小于此间隔的片段将被合并
            
        返回:
            合并后的语音片段列表
        """
        if not segments or len(segments) < 2:
            return segments
            
        max_gap_samples = int(max_gap_ms * self.sample_rate / 1000)
        merged_segments = []
        current_segment = segments[0]
        
        for i in range(1, len(segments)):
            # 如果当前片段与下一个片段间隔小于阈值，则合并
            if len(segments[i-1]) + max_gap_samples >= len(segments[i]):
                # 创建合并后的片段
                gap_size = max_gap_samples - len(segments[i-1])
                if gap_size > 0:
                    # 添加静音填充
                    silence = np.zeros(gap_size, dtype=np.float32)
                    current_segment = np.concatenate([current_segment, silence, segments[i]])
                else:
                    current_segment = np.concatenate([current_segment, segments[i]])
            else:
                # 间隔过大，保存当前片段并开始新片段
                merged_segments.append(current_segment)
                current_segment = segments[i]
        
        # 添加最后一个片段
        merged_segments.append(current_segment)
        
        return merged_segments
    
    def simple_energy_vad(self, audio_data: np.ndarray, threshold: float = 0.01) -> bool:
        """
        简单的能量检测VAD（备用方法）
        
        参数:
            audio_data: 音频数据
            threshold: 能量阈值
            
        返回:
            是否检测到语音
        """
        if audio_data is None or len(audio_data) == 0:
            return False
            
        # 简单的能量检测
        energy = np.mean(np.abs(audio_data))
        return energy > threshold
    
    def analyze_speech_segments(self, speech_timestamps: List[Dict], 
                               audio_length_seconds: float,
                               output_dir: Optional[Union[str, Path]] = None,
                               generate_plot: bool = True) -> Dict:
        """
        分析语音段的详细信息
        
        参数:
            speech_timestamps: 语音段时间戳列表
            audio_length_seconds: 音频总长度（秒）
            output_dir: 输出目录，用于保存分析结果和图表
            generate_plot: 是否生成可视化图表
            
        返回:
            包含分析结果的字典
        """
        # 计算语音段数量
        segment_count = len(speech_timestamps)
        logger.info(f"总语音段数量: {segment_count}")
        
        if segment_count == 0:
            logger.warning("未检测到语音段")
            return {"segment_count": 0}
        
        # 计算每个语音段的长度
        segment_lengths = []
        total_speech_duration = 0
        segment_details = []
        
        for i, segment in enumerate(speech_timestamps):
            start = segment.get('start', 0)
            end = segment.get('end', 0)
            
            # 如果时间戳是样本索引而不是秒，则转换为秒
            if isinstance(start, int) and start > 1000:  # 假设大于1000的是样本索引
                start = start / self.sample_rate
            if isinstance(end, int) and end > 1000:
                end = end / self.sample_rate
                
            duration = end - start
            segment_lengths.append(duration)
            total_speech_duration += duration
            
            segment_details.append({
                "index": i+1,
                "start": start,
                "end": end,
                "duration": duration
            })
            
            logger.debug(f"语音段 {i+1}: 开始={start:.2f}秒, 结束={end:.2f}秒, 持续={duration:.2f}秒")
        
        # 计算统计信息
        avg_length = np.mean(segment_lengths)
        min_length = np.min(segment_lengths)
        max_length = np.max(segment_lengths)
        median_length = np.median(segment_lengths)
        std_length = np.std(segment_lengths)
        
        # 语音占比
        speech_ratio = (total_speech_duration / audio_length_seconds) * 100
        
        # 统计信息
        stats = {
            "segment_count": segment_count,
            "avg_length": avg_length,
            "min_length": min_length,
            "max_length": max_length,
            "median_length": median_length,
            "std_length": std_length,
            "total_speech_duration": total_speech_duration,
            "audio_length_seconds": audio_length_seconds,
            "speech_ratio": speech_ratio
        }
        
        logger.info(f"平均语音段长度: {avg_length:.2f}秒")
        logger.info(f"最短语音段长度: {min_length:.2f}秒")
        logger.info(f"最长语音段长度: {max_length:.2f}秒")
        logger.info(f"语音占比: {speech_ratio:.2f}%")
        
        # 语音段长度分布
        # 创建长度区间
        bins = [0, 1, 2, 3, 5, 10, float('inf')]
        bin_labels = ['0-1秒', '1-2秒', '2-3秒', '3-5秒', '5-10秒', '10秒以上']
        
        # 统计每个区间的语音段数量
        distribution = [0] * len(bin_labels)
        for length in segment_lengths:
            for i in range(len(bins) - 1):
                if bins[i] <= length < bins[i + 1]:
                    distribution[i] += 1
                    break
        
        # 分布统计
        distribution_stats = {}
        for i, count in enumerate(distribution):
            percentage = (count / segment_count) * 100
            distribution_stats[bin_labels[i]] = {
                "count": count,
                "percentage": percentage
            }
            logger.info(f"{bin_labels[i]}: {count}段 ({percentage:.1f}%)")
        
        # 计算语音段间隔
        gap_stats = {}
        if segment_count > 1:
            gaps = []
            for i in range(1, len(speech_timestamps)):
                gap = speech_timestamps[i].get('start', 0) - speech_timestamps[i-1].get('end', 0)
                # 如果是样本索引，转换为秒
                if gap > 1000:
                    gap = gap / self.sample_rate
                gaps.append(gap)
            
            avg_gap = np.mean(gaps)
            max_gap = np.max(gaps)
            
            gap_stats = {
                "avg_gap": avg_gap,
                "max_gap": max_gap
            }
            
            logger.info(f"平均间隔: {avg_gap:.2f}秒")
            logger.info(f"最大间隔: {max_gap:.2f}秒")
        
        # 可视化语音段分布
        plot_path = None
        if generate_plot:
            try:
                plt.figure(figsize=(10, 6))
                plt.hist(segment_lengths, bins=10, alpha=0.7, color='blue')
                plt.title('语音段长度分布直方图')
                plt.xlabel('语音段长度 (秒)')
                plt.ylabel('频次')
                plt.grid(True, alpha=0.3)
                
                # 保存图表
                if output_dir:
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                    plot_path = str(output_path / 'vad_segment_distribution.png')
                else:
                    plot_path = 'vad_segment_distribution.png'
                    
                plt.savefig(plot_path)
                logger.info(f"已保存语音段长度分布直方图到 '{plot_path}'")
                plt.close()
            except Exception as e:
                logger.error(f"无法生成可视化: {e}")
        
        # 返回完整的分析结果
        result = {
            "stats": stats,
            "segment_details": segment_details,
            "distribution": distribution_stats,
            "gaps": gap_stats,
            "plot_path": plot_path
        }
        
        return result
    
    def process_audio_file(self, audio_path: Union[str, Path], 
                          analyze: bool = True,
                          output_dir: Optional[Union[str, Path]] = None) -> Dict:
        """
        处理音频文件，检测语音段并进行分析
        
        参数:
            audio_path: 音频文件路径
            analyze: 是否分析语音段
            output_dir: 输出目录
            
        返回:
            处理结果字典
        """
        if not self._model_loaded:
            logger.error("VAD模型未加载，无法处理音频")
            return {"error": "VAD模型未加载"}
        
        try:
            # 读取音频文件
            wav = self.read_audio(str(audio_path))
            
            # 获取音频长度（秒）
            audio_length_seconds = len(wav) / self.sample_rate
            
            # 获取语音时间戳
            speech_timestamps = self.get_speech_timestamps(
                wav,
                self.model,
                return_seconds=True,  # 返回秒为单位的时间戳
                threshold=self.config.threshold,
                sampling_rate=self.sample_rate,
                min_speech_duration_ms=self.config.min_speech_duration_ms,
                min_silence_duration_ms=self.config.min_silence_duration_ms,
                window_size_samples=self.config.window_size_samples,
                speech_pad_ms=self.config.speech_pad_ms
            )
            
            result = {
                "audio_path": str(audio_path),
                "audio_length_seconds": audio_length_seconds,
                "speech_timestamps": speech_timestamps,
                "segment_count": len(speech_timestamps)
            }
            
            # 分析语音段
            if analyze and speech_timestamps:
                analysis = self.analyze_speech_segments(
                    speech_timestamps, 
                    audio_length_seconds,
                    output_dir=output_dir
                )
                result["analysis"] = analysis
            
            return result
            
        except Exception as e:
            logger.error(f"处理音频文件失败: {e}")
            return {"error": str(e)}


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
        self.vad_processor = SileroVADProcessor(self.config)
        
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
