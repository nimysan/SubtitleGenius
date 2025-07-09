"""
Voice Activity Detection (VAD) 处理器
基于Silero VAD模型的流式语音活动检测
"""

import os
import sys
import time
import numpy as np
import torch
from typing import Iterator, List, Dict, Any, Optional, Callable
from loguru import logger
from collections import deque

# 添加whisper_streaming目录到路径
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'whisper_streaming'))
from silero_vad_iterator import FixedVADIterator


class VACProcessor:
    """
    Voice Activity Detection 处理器
    
    基于Silero VAD模型的流式语音活动检测，支持实时音频流处理
    """
    
    def __init__(
        self,
        threshold: float = 0.3,
        min_silence_duration_ms: int = 300,
        speech_pad_ms: int = 100,
        sample_rate: int = 16000,
        processing_chunk_size: int = 512,
        no_audio_input_threshold: float = 0.5,
        on_speech_segment: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        初始化VAC处理器
        
        Args:
            threshold: 语音检测阈值 (0.0-1.0)
            min_silence_duration_ms: 最小静音持续时间(ms)
            speech_pad_ms: 语音段填充时间(ms)
            sample_rate: 音频采样率
            processing_chunk_size: 处理块大小，必须是512的整数倍
            no_audio_input_threshold: 无音频输入阈值(秒)
            on_speech_segment: 语音段检测完成时的回调函数
        """
        self.threshold = threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.sample_rate = sample_rate
        self.processing_chunk_size = processing_chunk_size
        self.no_audio_input_threshold = no_audio_input_threshold
        self.on_speech_segment = on_speech_segment
        
        # 验证处理块大小
        if self.processing_chunk_size % 512 != 0:
            raise ValueError(
                f"processing_chunk_size必须是512的整数倍！当前值为{self.processing_chunk_size}。"
                f"推荐值：512, 1024, 1536, 2048等。"
            )
        
        # 初始化模型和VAD迭代器
        self._model = None
        self._vad_iterator = None
        
        # 音频缓存和状态跟踪
        self._audio_buffer = deque()  # 存储音频数据
        self._current_start_time = None  # 当前语音段的开始时间
        self._current_start_sample = None  # 当前语音段的开始样本位置
        
        logger.info(f"VACProcessor initialized with parameters:")
        logger.info(f"  threshold: {self.threshold}")
        logger.info(f"  min_silence_duration_ms: {self.min_silence_duration_ms}")
        logger.info(f"  speech_pad_ms: {self.speech_pad_ms}")
        logger.info(f"  sample_rate: {self.sample_rate}")
        logger.info(f"  processing_chunk_size: {self.processing_chunk_size}")
        logger.info(f"  no_audio_input_threshold: {self.no_audio_input_threshold}")
    
    def _load_model(self):
        """加载Silero VAD模型"""
        if self._model is None:
            logger.info("Loading Silero VAD model...")
            self._model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad'
            )
            logger.info("Silero VAD model loaded successfully")
        return self._model
    
    def _create_vad_iterator(self):
        """创建VAD迭代器"""
        if self._vad_iterator is None:
            model = self._load_model()
            self._vad_iterator = FixedVADIterator(
                model=model,
                threshold=self.threshold,
                sampling_rate=self.sample_rate,
                min_silence_duration_ms=self.min_silence_duration_ms,
                speech_pad_ms=self.speech_pad_ms
            )
            logger.info("VAD iterator created successfully")
        return self._vad_iterator
    
    def reset_vad_state(self):
        """重置VAD状态"""
        if self._vad_iterator is not None:
            self._vad_iterator.reset_states()
            logger.debug("VAD state reset")
    
    def process_streaming_audio(
        self, 
        audio_stream: Iterator[np.ndarray],
        return_segments: bool = True
    ) -> List[Dict[str, Any]]:
        """
        处理流式音频数据，进行语音活动检测
        
        Args:
            audio_stream: 音频流迭代器，每个元素是numpy数组
            return_segments: 是否返回语音段格式（包含start、end、duration）
            
        Returns:
            语音检测结果列表
            - 如果return_segments=True: [{'start': float, 'end': float, 'duration': float}, ...]
            - 如果return_segments=False: [{'start': float}, {'end': float}, ...]
        """
        logger.info("开始流式VAD处理")
        logger.info(f"为避免VAD时间戳膨胀问题，请确保音频流中的块大小是512的整数倍")
        
        # 创建VAD迭代器
        vad = self._create_vad_iterator()
        
        # 处理变量
        results = []
        last_audio_time = time.time()
        total_samples_processed = 0
        stream_ended = False
        
        # 重置音频缓存和状态
        self._audio_buffer.clear()
        self._current_start_time = None
        self._current_start_sample = None
        
        logger.info(f"开始处理音频流，参数:")
        logger.info(f"  threshold={self.threshold}")
        logger.info(f"  min_silence_duration_ms={self.min_silence_duration_ms}")
        logger.info(f"  speech_pad_ms={self.speech_pad_ms}")
        logger.info(f"  no_audio_input_threshold={self.no_audio_input_threshold}秒")
        
        try:
            # 处理流式音频
            for audio_chunk in audio_stream:
                # 更新最后接收音频的时间
                last_audio_time = time.time()
                
                # 将音频块添加到缓存
                self._audio_buffer.append((audio_chunk.copy(), total_samples_processed))
                
                # 处理音频块
                for i in range(0, len(audio_chunk), self.processing_chunk_size):
                    chunk = audio_chunk[i:i+self.processing_chunk_size]
                    
                    # 如果需要，用零填充
                    if len(chunk) < self.processing_chunk_size:
                        chunk = np.pad(chunk, (0, self.processing_chunk_size - len(chunk)), 'constant')
                    
                    # 使用VAD迭代器处理块
                    result = vad(chunk, return_seconds=True)
                    
                    total_samples_processed += len(chunk)
                    
                    if result:
                        print(f"------->>>vad result is {result}")
                        results.append(result)
                        
                        # 🎯 关键逻辑：处理start和end事件
                        if 'start' in result:
                            # 只有在没有活跃语音段时才记录新的开始
                            if self._current_start_time is None:
                                self._current_start_time = result['start']
                                self._current_start_sample = int(result['start'] * self.sample_rate)
                                logger.info(f"检测到语音开始: {self._current_start_time:.2f}s")
                            else:
                                # 如果已经有活跃的语音段，忽略新的start事件
                                logger.debug(f"忽略重复的start事件: {result['start']:.2f}s (当前活跃段: {self._current_start_time:.2f}s)")
                            
                        elif 'end' in result and self._current_start_time is not None:
                            # 检测到语音结束，发射事件
                            end_time = result['end']
                            end_sample = int(end_time * self.sample_rate)
                            
                            # 提取对应的音频数据
                            audio_bytes = self._extract_audio_segment(
                                self._current_start_sample, 
                                end_sample
                            )
                            
                            # 创建语音段事件数据
                            speech_segment = {
                                'start': self._current_start_time,
                                'end': end_time,
                                'duration': end_time - self._current_start_time,
                                'audio_bytes': audio_bytes,
                                'sample_rate': self.sample_rate
                            }
                            
                            logger.info(f"检测到语音结束: {end_time:.2f}s, 时长: {speech_segment['duration']:.2f}s")
                            
                            # 🚀 发射事件 - 确保只触发一次
                            if self.on_speech_segment:
                                try:
                                    self.on_speech_segment(speech_segment)
                                    logger.debug(f"✅ 语音段事件已触发: {self._current_start_time:.2f}s-{end_time:.2f}s")
                                except Exception as e:
                                    logger.error(f"事件回调执行失败: {e}")
                            
                            # 🔧 重要：立即重置状态，防止重复触发
                            self._current_start_time = None
                            self._current_start_sample = None
                            
                        elif 'end' in result and self._current_start_time is None:
                            # 收到end事件但没有对应的start事件，记录警告
                            logger.warning(f"收到孤立的end事件: {result['end']:.2f}s (没有对应的start事件)")
                
                # 检查是否超过无音频输入阈值
                if time.time() - last_audio_time > self.no_audio_input_threshold:
                    stream_ended = True
                    break
            
            # 标记流已结束
            stream_ended = True
            
        except Exception as e:
            print(f"流处理中断: {e}")
            stream_ended = True
        
        # 🔧 修复：音频流结束时的处理
        if stream_ended:
            print(f"音频流已结束，正在进行最终处理...")
            
            # 如果VAD仍处于触发状态，强制结束当前语音段
            if vad.triggered and self._current_start_time is not None:
                # 使用当前处理的总样本数计算结束时间
                end_time = total_samples_processed / self.sample_rate
                end_sample = total_samples_processed
                
                # 提取音频数据
                audio_bytes = self._extract_audio_segment(
                    self._current_start_sample, 
                    end_sample
                )
                
                # 创建语音段事件数据
                speech_segment = {
                    'start': self._current_start_time,
                    'end': end_time,
                    'duration': end_time - self._current_start_time,
                    'audio_bytes': audio_bytes,
                    'sample_rate': self.sample_rate
                }
                
                results.append({'end': end_time})
                print(f"检测到未结束的语音段，强制结束于 {end_time:.2f}秒")
                
                # 🚀 发射最终事件
                if self.on_speech_segment:
                    try:
                        self.on_speech_segment(speech_segment)
                    except Exception as e:
                        logger.error(f"最终事件回调执行失败: {e}")
            
            # 强制刷新VAD状态，确保所有缓冲的结果都被输出
            try:
                # 发送一个静音块来触发任何待处理的结束事件
                silent_chunk = np.zeros(self.processing_chunk_size, dtype=np.float32)
                final_result = vad(silent_chunk, return_seconds=True)
                if final_result:
                    print(f"最终刷新结果: {final_result}")
                    results.append(final_result)
            except Exception as e:
                print(f"最终刷新时出错: {e}")
        
        # 根据需要返回不同格式
        if return_segments:
            return self._convert_to_segments(results)
        else:
            return results
    
    def _extract_audio_segment(self, start_sample: int, end_sample: int) -> bytes:
        """
        从音频缓存中提取指定范围的音频数据
        
        Args:
            start_sample: 开始样本位置
            end_sample: 结束样本位置
            
        Returns:
            音频数据的字节表示
        """
        try:
            # 收集指定范围内的音频数据
            audio_segments = []
            current_sample = 0
            
            for audio_chunk, chunk_start_sample in self._audio_buffer:
                chunk_end_sample = chunk_start_sample + len(audio_chunk)
                
                # 检查这个块是否与目标范围重叠
                if chunk_end_sample > start_sample and chunk_start_sample < end_sample:
                    # 计算在这个块内的相对位置
                    relative_start = max(0, start_sample - chunk_start_sample)
                    relative_end = min(len(audio_chunk), end_sample - chunk_start_sample)
                    
                    # 提取相关部分
                    segment = audio_chunk[relative_start:relative_end]
                    audio_segments.append(segment)
            
            if audio_segments:
                # 合并所有音频段
                combined_audio = np.concatenate(audio_segments)
                # 转换为字节
                return combined_audio.astype(np.float32).tobytes()
            else:
                logger.warning(f"无法找到样本范围 {start_sample}-{end_sample} 的音频数据")
                return b''
                
        except Exception as e:
            logger.error(f"提取音频段时出错: {e}")
            return b''
    
    def _convert_to_segments(self, vad_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将VAD结果转换为语音段格式
        
        Args:
            vad_results: VAD原始结果 [{'start': float}, {'end': float}, ...]
            
        Returns:
            语音段列表 [{'start': float, 'end': float, 'duration': float}, ...]
        """
        segments = []
        start_time = None
        
        for result in vad_results:
            if 'start' in result:
                start_time = result['start']
            elif 'end' in result and start_time is not None:
                segments.append({
                    'start': start_time,
                    'end': result['end'],
                    'duration': result['end'] - start_time
                })
                start_time = None
        
        return segments


# 便利函数，保持向后兼容性
def create_vac_processor(
    threshold: float = 0.3,
    min_silence_duration_ms: int = 300,
    speech_pad_ms: int = 100,
    sample_rate: int = 16000,
    on_speech_segment: Optional[Callable[[Dict[str, Any]], None]] = None
) -> VACProcessor:
    """
    创建VAC处理器的便利函数
    
    Args:
        threshold: 语音检测阈值
        min_silence_duration_ms: 最小静音持续时间
        speech_pad_ms: 语音段填充时间
        sample_rate: 采样率
        on_speech_segment: 语音段检测完成时的回调函数
        
    Returns:
        VACProcessor实例
    """
    return VACProcessor(
        threshold=threshold,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        sample_rate=sample_rate,
        on_speech_segment=on_speech_segment
    )
