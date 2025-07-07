"""
增强版VAD处理器
用于处理实时流式音频数据，解决重叠问题
"""

import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import io
import wave

# 导入原始VAD处理器
from subtitle_genius.audio.vad_processor import VADBufferProcessor as OriginalVADBufferProcessor
from subtitle_genius.audio.vad_processor import VADConfig, StreamingVADProcessor

logger = logging.getLogger(__name__)

class EnhancedVADBufferProcessor:
    """增强版VAD缓冲区处理器，用于管理音频缓冲区和VAD处理"""
    
    def __init__(self, config: Optional[VADConfig] = None, buffer_size_seconds: float = 30.0):
        """
        初始化增强版VAD缓冲区处理器
        
        参数:
            config: VAD配置，如果为None则使用默认配置
            buffer_size_seconds: 缓冲区大小（秒）
        """
        # 初始化原始VAD处理器
        self.original_processor = OriginalVADBufferProcessor(config, buffer_size_seconds)
        
        # 存储待处理的语音片段
        # 格式: [(音频数据, 时间戳信息, 检测块编号), ...]
        self.pending_segments = []
        
        # 存储已处理的语音片段ID，用于去重
        self.processed_segment_ids = set()
        
        # 当前块编号
        self.current_chunk = 0
        
        # 安全边界（秒），用于判断语音片段是否完整
        self.safety_margin = 0.5
        
        logger.info(f"增强版VAD缓冲区处理器初始化完成，缓冲区大小: {buffer_size_seconds}秒")
    
    def add_audio_chunk(self, audio_data: np.ndarray) -> None:
        """
        添加音频块到缓冲区
        
        参数:
            audio_data: 音频数据，numpy数组
        """
        # 更新当前块编号
        self.current_chunk += 1
        
        # 添加到原始处理器的缓冲区
        self.original_processor.add_audio_chunk(audio_data)
    
    def process_buffer(self) -> List[Tuple[np.ndarray, Dict]]:
        """
        处理缓冲区中的音频，使用VAD检测语音片段
        
        返回:
            完整的语音片段列表，每个元素为(音频数据, 时间戳信息)
        """
        # 使用原始处理器处理缓冲区
        new_segments = self.original_processor.process_buffer()
        
        # 将新检测到的语音片段添加到待处理列表
        for segment, timestamp in new_segments:
            segment_id = f"{timestamp['start']:.3f}_{timestamp['end']:.3f}"
            if segment_id not in self.processed_segment_ids:
                self.pending_segments.append((segment, timestamp, self.current_chunk))
        
        # 合并重叠的语音片段
        self._merge_overlapping_segments()
        
        # 获取完整的语音片段
        complete_segments = self._get_complete_segments()
        
        return complete_segments
    
    def _merge_overlapping_segments(self) -> None:
        """合并重叠的语音片段"""
        if len(self.pending_segments) <= 1:
            return
        
        # 按开始时间排序
        self.pending_segments.sort(key=lambda x: x[1]['start'])
        
        # 合并重叠的片段
        i = 0
        while i < len(self.pending_segments) - 1:
            segment1, timestamp1, chunk1 = self.pending_segments[i]
            segment2, timestamp2, chunk2 = self.pending_segments[i + 1]
            
            # 检查是否重叠
            if self._segments_overlap(timestamp1, timestamp2):
                # 合并片段
                merged_segment, merged_timestamp = self._merge_segments(segment1, timestamp1, segment2, timestamp2)
                
                # 替换原片段
                self.pending_segments[i] = (merged_segment, merged_timestamp, min(chunk1, chunk2))
                self.pending_segments.pop(i + 1)
            else:
                i += 1
    
    def _segments_overlap(self, timestamp1: Dict, timestamp2: Dict) -> bool:
        """
        检查两个语音片段是否重叠
        
        参数:
            timestamp1: 第一个语音片段的时间戳信息
            timestamp2: 第二个语音片段的时间戳信息
            
        返回:
            是否重叠
        """
        # 如果一个片段的开始时间小于另一个片段的结束时间，且结束时间大于另一个片段的开始时间，则重叠
        return (timestamp1['start'] < timestamp2['end'] and timestamp1['end'] > timestamp2['start'])
    
    def _merge_segments(self, segment1: np.ndarray, timestamp1: Dict, 
                       segment2: np.ndarray, timestamp2: Dict) -> Tuple[np.ndarray, Dict]:
        """
        合并两个语音片段
        
        参数:
            segment1: 第一个语音片段的音频数据
            timestamp1: 第一个语音片段的时间戳信息
            segment2: 第二个语音片段的音频数据
            timestamp2: 第二个语音片段的时间戳信息
            
        返回:
            合并后的语音片段和时间戳信息
        """
        # 计算合并后的开始和结束时间
        start = min(timestamp1['start'], timestamp2['start'])
        end = max(timestamp1['end'], timestamp2['end'])
        
        # 创建合并后的时间戳信息
        merged_timestamp = {
            'start': start,
            'end': end,
            'duration': end - start
        }
        
        # 合并音频数据（这里简化处理，实际上需要更复杂的逻辑）
        # 在实际实现中，需要考虑采样率、对齐等问题
        # 这里简单地使用时间较长的片段
        if (timestamp1['end'] - timestamp1['start']) >= (timestamp2['end'] - timestamp2['start']):
            merged_segment = segment1
        else:
            merged_segment = segment2
        
        logger.info(f"合并重叠语音片段: {timestamp1['start']:.2f}-{timestamp1['end']:.2f} + "
                   f"{timestamp2['start']:.2f}-{timestamp2['end']:.2f} -> "
                   f"{merged_timestamp['start']:.2f}-{merged_timestamp['end']:.2f}")
        
        return merged_segment, merged_timestamp
    
    def _get_complete_segments(self) -> List[Tuple[np.ndarray, Dict]]:
        """
        获取完整的语音片段
        
        返回:
            完整的语音片段列表，每个元素为(音频数据, 时间戳信息)
        """
        complete_segments = []
        remaining_segments = []
        
        # 获取当前缓冲区的总时长
        buffer_duration = self.original_processor.get_buffer_duration()
        
        for segment, timestamp, chunk in self.pending_segments:
            # 判断语音片段是否完整
            if self._is_segment_complete(timestamp, chunk):
                # 创建片段ID
                segment_id = f"{timestamp['start']:.3f}_{timestamp['end']:.3f}"
                
                # 如果片段未处理过，添加到完整片段列表
                if segment_id not in self.processed_segment_ids:
                    complete_segments.append((segment, timestamp))
                    self.processed_segment_ids.add(segment_id)
                    
                    logger.info(f"检测到完整语音片段: {timestamp['start']:.2f}-{timestamp['end']:.2f}, "
                               f"时长: {timestamp['end'] - timestamp['start']:.2f}秒")
            else:
                # 如果不完整，保留在待处理列表
                remaining_segments.append((segment, timestamp, chunk))
        
        # 更新待处理列表
        self.pending_segments = remaining_segments
        
        return complete_segments
    
    def _is_segment_complete(self, timestamp: Dict, chunk: int) -> bool:
        """
        判断语音片段是否完整
        
        参数:
            timestamp: 语音片段的时间戳信息
            chunk: 检测到该片段的块编号
            
        返回:
            是否完整
        """
        # 获取当前缓冲区的总时长
        buffer_duration = self.original_processor.get_buffer_duration()
        
        # 语音片段结束时间
        end_time = timestamp['end']
        
        # 条件1: 语音片段结束时间距离缓冲区末尾有足够距离
        condition1 = (buffer_duration - end_time) >= self.safety_margin
        
        # 条件2: 语音片段在至少一个块之前被检测到
        condition2 = (self.current_chunk - chunk) >= 1
        
        # 条件3: 语音片段持续时间合理（不是太短也不是太长）
        duration = timestamp['end'] - timestamp['start']
        condition3 = (duration >= 0.2 and duration <= 10.0)
        
        return condition1 and condition2 and condition3
    
    def get_buffer_duration(self) -> float:
        """获取当前缓冲区的时长（秒）"""
        return self.original_processor.get_buffer_duration()
    
    def get_elapsed_time(self) -> float:
        """获取从初始化到现在的时间（秒）"""
        return self.original_processor.get_elapsed_time()
    
    def convert_to_wav(self, audio_data: np.ndarray) -> bytes:
        """
        将音频数据转换为WAV格式
        
        参数:
            audio_data: 音频数据，numpy数组
            
        返回:
            WAV格式的二进制数据
        """
        return self.original_processor.convert_to_wav(audio_data)
    
    def clear_buffer(self) -> None:
        """清空缓冲区"""
        self.original_processor.clear_buffer()
        self.pending_segments = []
