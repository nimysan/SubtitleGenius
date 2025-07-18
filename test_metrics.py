#!/usr/bin/env python
"""
指标模块测试脚本
测试VAC处理器的指标收集和推送功能
"""

import os
import sys
import time
import numpy as np
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_metrics")

# 导入VAC处理器和指标模块
from subtitle_genius.stream.vac_processor import VACProcessor
from subtitle_genius.metrics.vac_metrics import register_vac_metrics, get_vac_metrics
from subtitle_genius.metrics.metrics_manager import get_metrics_manager

def on_speech_segment(segment):
    """语音段回调函数"""
    start = segment['start']
    end = segment['end']
    duration = segment['duration']
    completeness = segment['audio_metadata']['completeness']
    
    logger.info(f"语音段: {start:.2f}s - {end:.2f}s (时长: {duration:.2f}s, 完整性: {completeness:.1f}%)")


def generate_test_audio(duration_seconds=30, sample_rate=16000):
    """生成测试音频数据"""
    # 创建一个包含多个语音段的测试音频
    total_samples = int(duration_seconds * sample_rate)
    audio = np.zeros(total_samples, dtype=np.float32)
    
    # 添加几个语音段 (用噪声模拟语音)
    speech_segments = [
        (2, 4),    # 2秒语音
        (7, 10),   # 3秒语音
        (15, 20),  # 5秒语音
        (23, 28)   # 5秒语音
    ]
    
    for start, end in speech_segments:
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        # 使用随机噪声模拟语音
        audio[start_sample:end_sample] = np.random.normal(0, 0.1, end_sample - start_sample)
    
    return audio


def test_metrics():
    """测试指标收集和推送功能"""
    logger.info("开始测试指标收集和推送功能")
    
    # 确保使用正确的Pushgateway URL
    pushgateway_url = os.environ.get('PROMETHEUS_PUSHGATEWAY_URL', 'localhost:9529')
    logger.info(f"使用Pushgateway: {pushgateway_url}")
    
    # 获取指标管理器
    metrics_manager = get_metrics_manager(pushgateway_url)
    
    # 创建VAC处理器
    vac = VACProcessor(
        threshold=0.3,
        min_silence_duration_ms=300,
        speech_pad_ms=100,
        sample_rate=16000,
        on_speech_segment=on_speech_segment
    )
    
    # 注册指标收集
    vac_metrics = register_vac_metrics(vac)
    logger.info("已注册VAC指标收集")
    
    # 生成测试音频
    logger.info("生成测试音频数据...")
    audio = generate_test_audio(duration_seconds=30)
    
    # 将音频分成块进行处理
    chunk_size = 1600  # 0.1秒
    chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]
    
    # 创建音频流迭代器
    def audio_stream():
        for chunk in chunks:
            yield chunk
            time.sleep(0.05)  # 模拟实时流
    
    # 处理音频流
    logger.info("开始处理音频流...")
    start_time = time.time()
    vac.process_streaming_audio(audio_stream())
    processing_time = time.time() - start_time
    
    logger.info(f"处理完成，耗时: {processing_time:.2f}秒")
    
    # 手动推送指标
    logger.info("手动推送指标到Pushgateway...")
    metrics_manager.push_metrics(job="subtitle_genius_test")
    
    logger.info("测试完成")


if __name__ == "__main__":
    test_metrics()
