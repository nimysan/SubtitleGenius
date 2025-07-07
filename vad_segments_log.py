"""
VAD分段日志记录器
用于记录VAD分段的时间戳信息
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import time
import os

# 配置日志
logger = logging.getLogger(__name__)

class VADSegmentLogger:
    """VAD分段日志记录器"""
    
    def __init__(self, log_file: str = "vad_segments.log"):
        """
        初始化VAD分段日志记录器
        
        参数:
            log_file: 日志文件路径
        """
        self.log_file = log_file
        self.segments = []
        
        # 确保日志文件目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 清空日志文件
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("# VAD分段日志\n")
            f.write("# 格式: [序号] 开始时间 --> 结束时间 (持续时间)\n\n")
    
    def log_segment(self, segment_index: int, start_time: float, end_time: float) -> None:
        """
        记录VAD分段信息
        
        参数:
            segment_index: 分段序号
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
        """
        duration = end_time - start_time
        
        # 格式化时间戳
        start_str = self._format_timestamp(start_time)
        end_str = self._format_timestamp(end_time)
        
        # 记录分段信息
        segment_info = {
            "index": segment_index,
            "start": start_time,
            "end": end_time,
            "duration": duration,
            "start_str": start_str,
            "end_str": end_str,
            "timestamp": time.time()
        }
        
        self.segments.append(segment_info)
        
        # 写入日志文件
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{segment_index}] {start_str} --> {end_str} ({duration:.3f}秒)\n")
    
    def analyze_overlaps(self) -> List[Dict]:
        """
        分析分段重叠情况
        
        返回:
            重叠的分段列表
        """
        overlaps = []
        
        # 按开始时间排序
        sorted_segments = sorted(self.segments, key=lambda x: x["start"])
        
        # 检查重叠
        for i in range(len(sorted_segments) - 1):
            current = sorted_segments[i]
            next_segment = sorted_segments[i + 1]
            
            if current["end"] > next_segment["start"]:
                overlap = {
                    "segment1": current["index"],
                    "segment2": next_segment["index"],
                    "overlap_start": next_segment["start"],
                    "overlap_end": min(current["end"], next_segment["end"]),
                    "overlap_duration": min(current["end"], next_segment["end"]) - next_segment["start"]
                }
                overlaps.append(overlap)
        
        # 写入重叠分析结果
        if overlaps:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write("\n# 分段重叠分析\n")
                for i, overlap in enumerate(overlaps):
                    f.write(f"重叠 #{i+1}: 分段 {overlap['segment1']} 和分段 {overlap['segment2']} 重叠 {overlap['overlap_duration']:.3f}秒\n")
                    f.write(f"  - 重叠区间: {self._format_timestamp(overlap['overlap_start'])} --> {self._format_timestamp(overlap['overlap_end'])}\n")
        else:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write("\n# 分段重叠分析\n")
                f.write("未检测到分段重叠\n")
        
        return overlaps
    
    def save_json(self, json_file: Optional[str] = None) -> None:
        """
        将分段信息保存为JSON格式
        
        参数:
            json_file: JSON文件路径，如果为None则使用日志文件路径加上.json后缀
        """
        if json_file is None:
            json_file = f"{os.path.splitext(self.log_file)[0]}.json"
        
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump({
                "segments": self.segments,
                "overlaps": self.analyze_overlaps()
            }, f, ensure_ascii=False, indent=2)
    
    def _format_timestamp(self, seconds: float) -> str:
        """
        将秒数格式化为时间戳格式 (HH:MM:SS.mmm)
        
        参数:
            seconds: 秒数
            
        返回:
            格式化后的时间戳
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
