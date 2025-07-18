"""
VAC处理器的指标收集模块
监控语音活动检测和分段处理的关键指标
"""

import logging
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, Summary

from .metrics_manager import get_metrics_manager

logger = logging.getLogger("subtitle_genius.metrics.vac")

# VAC指标前缀
METRIC_PREFIX = "subtitle_genius_vac"

# VAC指标定义
VAC_METRICS = {
    # 计数器类指标
    "speech_segments_total": {
        "type": "counter",
        "description": "Total number of speech segments detected",
        "labels": ["status"]  # status: success, error, incomplete
    },
    "vad_events_total": {
        "type": "counter",
        "description": "Total number of VAD events (start/end)",
        "labels": ["event_type"]  # event_type: start, end, orphaned_end
    },
    
    # 直方图类指标
    "speech_segment_duration_seconds": {
        "type": "histogram",
        "description": "Duration of detected speech segments in seconds",
        "buckets": [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
    },
    "speech_segment_processing_time_seconds": {
        "type": "histogram",
        "description": "Time taken to process a speech segment in seconds",
        "buckets": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    },
    "audio_buffer_size_bytes": {
        "type": "histogram",
        "description": "Size of audio buffer in bytes",
        "buckets": [1024, 4096, 16384, 65536, 262144, 1048576]
    },
    
    # 仪表盘类指标
    "audio_buffer_chunks": {
        "type": "gauge",
        "description": "Current number of chunks in the audio buffer"
    },
    "audio_buffer_duration_seconds": {
        "type": "gauge",
        "description": "Current duration of audio in buffer in seconds"
    },
    "audio_completeness_percent": {
        "type": "gauge",
        "description": "Percentage of audio samples found vs required for the last segment",
        "labels": ["segment_id"]
    },
    
    # 摘要类指标
    "speech_segment_samples": {
        "type": "summary",
        "description": "Number of samples in speech segments"
    }
}


class VACMetrics:
    """
    VAC处理器的指标收集类
    
    负责收集和更新VAC处理器的性能和操作指标
    """
    
    def __init__(self, pushgateway_url: Optional[str] = None):
        """
        初始化VAC指标收集器
        
        Args:
            pushgateway_url: Prometheus Pushgateway的URL
        """
        self.metrics_manager = get_metrics_manager(pushgateway_url)
        self.metrics = {}
        
        # 注册所有VAC指标
        self._register_metrics()
        
        logger.info("VAC metrics collector initialized")
    
    def _register_metrics(self):
        """注册所有VAC指标"""
        for name, config in VAC_METRICS.items():
            metric_name = f"{METRIC_PREFIX}_{name}"
            metric_type = config["type"]
            description = config["description"]
            labels = config.get("labels", [])
            
            if metric_type == "counter":
                self.metrics[name] = self.metrics_manager.register_counter(
                    metric_name, description, labels
                )
            elif metric_type == "gauge":
                self.metrics[name] = self.metrics_manager.register_gauge(
                    metric_name, description, labels
                )
            elif metric_type == "histogram":
                buckets = config.get("buckets")
                self.metrics[name] = self.metrics_manager.register_histogram(
                    metric_name, description, labels, buckets
                )
            elif metric_type == "summary":
                self.metrics[name] = self.metrics_manager.register_summary(
                    metric_name, description, labels
                )
        
        logger.debug(f"Registered {len(self.metrics)} VAC metrics")
    
    def observe_speech_segment(self, segment: Dict[str, Any], processing_time: float = None):
        """
        记录语音段指标
        
        Args:
            segment: 语音段数据
            processing_time: 处理时间（秒）
        """
        try:
            # 增加语音段计数
            status = "success"
            if segment.get("audio_metadata", {}).get("completeness", 100) < 80:
                status = "incomplete"
            
            self.metrics["speech_segments_total"].labels(status=status).inc()
            
            # 记录语音段时长
            duration = segment.get("duration", 0)
            self.metrics["speech_segment_duration_seconds"].observe(duration)
            
            # 记录样本数
            samples = segment.get("audio_metadata", {}).get("samples_found", 0)
            self.metrics["speech_segment_samples"].observe(samples)
            
            # 记录音频完整性
            completeness = segment.get("audio_metadata", {}).get("completeness", 100)
            segment_id = f"{segment.get('start', 0):.2f}-{segment.get('end', 0):.2f}"
            self.metrics["audio_completeness_percent"].labels(segment_id=segment_id).set(completeness)
            
            # 记录处理时间（如果提供）
            if processing_time is not None:
                self.metrics["speech_segment_processing_time_seconds"].observe(processing_time)
            
            logger.debug(f"Recorded metrics for speech segment: {segment_id}")
        
        except Exception as e:
            logger.error(f"Failed to record speech segment metrics: {e}")
    
    def observe_vad_event(self, event_type: str):
        """
        记录VAD事件
        
        Args:
            event_type: 事件类型 (start, end, orphaned_end)
        """
        try:
            self.metrics["vad_events_total"].labels(event_type=event_type).inc()
        except Exception as e:
            logger.error(f"Failed to record VAD event metric: {e}")
    
    def update_buffer_metrics(self, buffer_chunks: int, buffer_duration: float, buffer_size_bytes: int):
        """
        更新缓冲区指标
        
        Args:
            buffer_chunks: 缓冲区中的块数量
            buffer_duration: 缓冲区中的音频时长（秒）
            buffer_size_bytes: 缓冲区大小（字节）
        """
        try:
            self.metrics["audio_buffer_chunks"].set(buffer_chunks)
            self.metrics["audio_buffer_duration_seconds"].set(buffer_duration)
            self.metrics["audio_buffer_size_bytes"].observe(buffer_size_bytes)
        except Exception as e:
            logger.error(f"Failed to update buffer metrics: {e}")


# 全局VAC指标实例
_vac_metrics_instance = None

def get_vac_metrics(pushgateway_url: Optional[str] = None) -> VACMetrics:
    """
    获取全局VAC指标实例
    
    Args:
        pushgateway_url: Prometheus Pushgateway的URL
        
    Returns:
        VACMetrics实例
    """
    global _vac_metrics_instance
    if _vac_metrics_instance is None:
        _vac_metrics_instance = VACMetrics(pushgateway_url)
    return _vac_metrics_instance


def register_vac_metrics(vac_processor):
    """
    为VAC处理器注册指标收集
    
    这个函数修改VAC处理器的回调函数，添加指标收集功能
    
    Args:
        vac_processor: VACProcessor实例
    """
    metrics = get_vac_metrics()
    original_on_speech_segment = vac_processor.on_speech_segment
    
    # 包装原始回调函数
    def metrics_wrapper(segment):
        start_time = None
        try:
            # 记录处理开始时间
            start_time = time.time()
            
            # 记录语音段指标
            metrics.observe_speech_segment(segment)
            
            # 调用原始回调函数
            if original_on_speech_segment:
                original_on_speech_segment(segment)
                
            # 记录处理时间
            if start_time:
                processing_time = time.time() - start_time
                metrics.metrics["speech_segment_processing_time_seconds"].observe(processing_time)
                
        except Exception as e:
            logger.error(f"Error in metrics wrapper: {e}")
            # 确保原始回调仍然被调用
            if original_on_speech_segment and not start_time:
                original_on_speech_segment(segment)
    
    # 替换回调函数
    vac_processor.on_speech_segment = metrics_wrapper
    
    # 添加缓冲区指标更新
    original_process_streaming_audio = vac_processor.process_streaming_audio
    
    def process_streaming_audio_with_metrics(*args, **kwargs):
        result = original_process_streaming_audio(*args, **kwargs)
        
        # 更新缓冲区指标
        try:
            buffer_chunks = len(vac_processor._audio_buffer)
            buffer_duration = buffer_chunks * vac_processor.processing_chunk_size / vac_processor.sample_rate
            buffer_size_bytes = sum(len(chunk[0]) * 4 for chunk in vac_processor._audio_buffer)  # float32 = 4 bytes
            
            metrics.update_buffer_metrics(buffer_chunks, buffer_duration, buffer_size_bytes)
        except Exception as e:
            logger.error(f"Failed to update buffer metrics: {e}")
        
        return result
    
    # 替换处理函数
    vac_processor.process_streaming_audio = process_streaming_audio_with_metrics
    
    logger.info(f"Metrics collection registered for VAC processor")
    
    return metrics
