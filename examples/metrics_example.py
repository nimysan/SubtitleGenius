#!/usr/bin/env python
"""
指标收集示例脚本
展示如何使用prometheus-client监控VAC处理器的指标
"""

import os
import sys
import time
import numpy as np
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from subtitle_genius.stream.vac_processor import VACProcessor
from subtitle_genius.metrics.vac_metrics import register_vac_metrics

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("metrics_example")

def on_speech_segment(segment):
    """语音段回调函数"""
    start = segment['start']
    end = segment['end']
    duration = segment['duration']
    completeness = segment['audio_metadata']['completeness']
    
    logger.info(f"语音段: {start:.2f}s - {end:.2f}s (时长: {duration:.2f}s, 完整性: {completeness:.1f}%)")


def generate_test_audio(duration_seconds=60, sample_rate=16000):
    """生成测试音频数据"""
    # 创建一个包含多个语音段的测试音频
    total_samples = int(duration_seconds * sample_rate)
    audio = np.zeros(total_samples, dtype=np.float32)
    
    # 添加几个语音段 (用噪声模拟语音)
    speech_segments = [
        (5, 8),    # 3秒语音
        (12, 17),  # 5秒语音
        (25, 28),  # 3秒语音
        (35, 45),  # 10秒语音
        (50, 55)   # 5秒语音
    ]
    
    for start, end in speech_segments:
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        # 使用随机噪声模拟语音
        audio[start_sample:end_sample] = np.random.normal(0, 0.1, end_sample - start_sample)
    
    return audio


def main():
    """主函数"""
    logger.info("启动VAC处理器指标收集示例")
    
    # 设置Pushgateway URL (可以从环境变量获取)
    pushgateway_url = os.environ.get('PROMETHEUS_PUSHGATEWAY_URL', 'localhost:9529')
    logger.info(f"使用Pushgateway: {pushgateway_url}")
    
    # 创建VAC处理器
    vac = VACProcessor(
        threshold=0.3,
        min_silence_duration_ms=300,
        speech_pad_ms=100,
        sample_rate=16000,
        on_speech_segment=on_speech_segment
    )
    
    # 注册指标收集
    metrics = register_vac_metrics(vac)
    logger.info("已注册VAC指标收集")
    
    # 生成测试音频
    logger.info("生成测试音频数据...")
    audio = generate_test_audio(duration_seconds=60)
    
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
    vac.process_streaming_audio(audio_stream())
    
    logger.info("处理完成，等待最终指标推送...")
    time.sleep(2)  # 等待最终指标推送
    
    logger.info("示例结束")


if __name__ == "__main__":
    main()
