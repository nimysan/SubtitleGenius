#!/usr/bin/env python
"""
VAD实现对比测试脚本
比较test_vac_processor.py和websocket_server.py中的VAD实现差异
"""

import os
import sys
import time
import numpy as np
import soundfile as sf
import torch
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from subtitle_genius.stream.vac_processor import VACProcessor
from test_vac_processor import test_streaming_vad

# 统一的VAD参数
THRESHOLD = 0.3
MIN_SILENCE_DURATION_MS = 300
SPEECH_PAD_MS = 100
SAMPLE_RATE = 16000
PROCESSING_CHUNK_SIZE = 2048  # 4 * 512，确保是512的整数倍


def load_audio_file(file_path, sample_rate=SAMPLE_RATE):
    """
    加载音频文件并转换为正确格式
    
    Args:
        file_path: 音频文件路径
        sample_rate: 目标采样率
        
    Returns:
        tuple: (audio_data, sample_rate)
    """
    logger.info(f"加载音频文件: {file_path}")
    
    # 加载音频文件
    audio_data, file_sample_rate = sf.read(file_path)
    
    # 打印音频属性
    print(f"\n===== 音频文件属性 =====")
    print(f"文件: {file_path}")
    print(f"采样率: {file_sample_rate} Hz")
    print(f"通道数: {1 if len(audio_data.shape) == 1 else audio_data.shape[1]}")
    print(f"时长: {len(audio_data)/file_sample_rate:.2f} 秒")
    print(f"数据类型: {audio_data.dtype}")
    
    # 确保音频是单声道
    if len(audio_data.shape) > 1:
        print(f"转换立体声为单声道")
        audio_data = audio_data[:, 0]
    
    # 确保音频是float32格式
    if audio_data.dtype != np.float32:
        print(f"转换 {audio_data.dtype} 为 float32")
        audio_data = audio_data.astype(np.float32)
    
    # 如果需要，重采样到目标采样率
    if file_sample_rate != sample_rate:
        import librosa
        print(f"重采样从 {file_sample_rate}Hz 到 {sample_rate}Hz")
        audio_data = librosa.resample(audio_data, orig_sr=file_sample_rate, target_sr=sample_rate)
    
    logger.info(f"音频长度: {len(audio_data)/sample_rate:.2f} 秒")
    return audio_data, sample_rate


def process_with_websocket_style(audio_data, sample_rate=SAMPLE_RATE):
    """
    使用类似websocket_server.py的方式处理音频
    
    Args:
        audio_data: 音频数据
        sample_rate: 采样率
        
    Returns:
        list: 语音段列表
    """
    print(f"\n===== 使用WebSocket风格处理 =====")
    print(f"参数: threshold={THRESHOLD}, min_silence_duration_ms={MIN_SILENCE_DURATION_MS}, speech_pad_ms={SPEECH_PAD_MS}")
    print(f"处理块大小: {PROCESSING_CHUNK_SIZE} 样本 ({PROCESSING_CHUNK_SIZE/sample_rate:.3f}秒)")
    
    # 创建VAC处理器
    vac_processor = VACProcessor(
        threshold=THRESHOLD,
        min_silence_duration_ms=MIN_SILENCE_DURATION_MS,
        speech_pad_ms=SPEECH_PAD_MS,
        sample_rate=sample_rate,
        processing_chunk_size=512,  # VAC内部处理块大小
        no_audio_input_threshold=0.5
    )
    
    # 创建音频流生成器
    def audio_stream_generator():
        for i in range(0, len(audio_data), PROCESSING_CHUNK_SIZE):
            chunk = audio_data[i:min(i+PROCESSING_CHUNK_SIZE, len(audio_data))]
            # 如果需要，用零填充到处理块大小
            if len(chunk) < PROCESSING_CHUNK_SIZE:
                chunk = np.pad(chunk, (0, PROCESSING_CHUNK_SIZE - len(chunk)), 'constant')
            yield chunk
    
    # 处理音频流
    websocket_results = vac_processor.process_streaming_audio(
        audio_stream=audio_stream_generator(),
        end_stream_flag=True,
        return_segments=True
    )
    
    print(f"WebSocket风格处理检测到 {len(websocket_results)} 个语音段")
    return websocket_results


def compare_vad_implementations(audio_file):
    """
    对比两种VAD实现的时间戳差异
    
    Args:
        audio_file: 音频文件路径
    """
    print(f"\n===== VAD实现对比 =====")
    print(f"音频文件: {audio_file}")
    print(f"统一参数: threshold={THRESHOLD}, min_silence_duration_ms={MIN_SILENCE_DURATION_MS}, speech_pad_ms={SPEECH_PAD_MS}")
    
    # 加载音频文件
    audio_data, sample_rate = load_audio_file(audio_file)
    
    # 方法1: test_streaming_vad
    print(f"\n方法1: test_streaming_vad")
    # 确保chunk_duration对应的样本数是512的整数倍
    chunk_samples = PROCESSING_CHUNK_SIZE
    chunk_duration = chunk_samples / sample_rate
    print(f"块大小: {chunk_samples} 样本 ({chunk_duration:.3f}秒)")
    
    streaming_results = test_streaming_vad(
        audio_file,
        chunk_duration=chunk_duration,
        sample_rate=sample_rate
    )
    
    # 方法2: 模拟websocket_server处理
    print(f"\n方法2: 模拟websocket_server处理")
    websocket_results = process_with_websocket_style(audio_data, sample_rate)
    
    # 对比结果
    print(f"\n===== 结果对比 =====")
    print(f"test_streaming_vad 检测到的语音段数量: {len(streaming_results)}")
    print(f"websocket_server 检测到的语音段数量: {len(websocket_results)}")
    
    # 如果段数不同，只比较共有的部分
    min_segments = min(len(streaming_results), len(websocket_results))
    if min_segments == 0:
        print("没有可比较的语音段")
        return
    
    # 计算时间戳差异
    timestamp_diffs = []
    print(f"\n{'段#':<4} {'test_streaming_vad':<25} {'websocket_server':<25} {'差异(start/end)':<20}")
    print("-" * 75)
    
    for i in range(min_segments):
        stream_seg = streaming_results[i]
        ws_seg = websocket_results[i]
        
        start_diff = abs(stream_seg['start'] - ws_seg['start'])
        end_diff = abs(stream_seg['end'] - ws_seg['end'])
        timestamp_diffs.extend([start_diff, end_diff])
        
        print(f"{i+1:<4} {stream_seg['start']:.2f}s-{stream_seg['end']:.2f}s ({stream_seg['duration']:.2f}s) "
              f"{ws_seg['start']:.2f}s-{ws_seg['end']:.2f}s ({ws_seg['duration']:.2f}s) "
              f"{start_diff:.3f}s/{end_diff:.3f}s")
    
    # 统计差异
    if timestamp_diffs:
        avg_timestamp_diff = sum(timestamp_diffs) / len(timestamp_diffs)
        max_timestamp_diff = max(timestamp_diffs)
        min_timestamp_diff = min(timestamp_diffs)
        print(f"\n时间戳差异统计:")
        print(f"  平均差异: {avg_timestamp_diff:.3f}s")
        print(f"  最大差异: {max_timestamp_diff:.3f}s")
        print(f"  最小差异: {min_timestamp_diff:.3f}s")
        
        # 如果差异较大，给出警告和建议
        if avg_timestamp_diff > 0.1:  # 超过100毫秒的平均差异
            print(f"\n⚠️  警告: 检测到显著的时间戳差异!")
            print(f"可能的原因:")
            print(f"  1. 处理块大小不一致 - 确保两种实现使用相同的处理块大小")
            print(f"  2. VAD参数不一致 - 确保两种实现使用相同的VAD参数")
            print(f"  3. 时间戳计算方式不同 - 检查VAD迭代器的时间戳计算逻辑")
            print(f"  4. 音频流处理方式不同 - 统一音频流处理逻辑")
            
            print(f"\n建议修复:")
            print(f"  1. 统一处理块大小为512的整数倍，推荐: {PROCESSING_CHUNK_SIZE}")
            print(f"  2. 统一VAD参数: threshold={THRESHOLD}, min_silence_duration_ms={MIN_SILENCE_DURATION_MS}, speech_pad_ms={SPEECH_PAD_MS}")
            print(f"  3. 确保两种实现使用相同的VAD迭代器类")
            print(f"  4. 创建共享的音频流处理函数，确保一致的处理逻辑")
        else:
            print(f"\n✅ 时间戳差异在可接受范围内")
    
    # 可视化对比
    plot_vad_comparison(streaming_results, websocket_results, audio_data, sample_rate)


def plot_vad_comparison(streaming_results, websocket_results, audio_data, sample_rate):
    """
    可视化对比两种VAD实现的结果
    
    Args:
        streaming_results: test_streaming_vad的结果
        websocket_results: websocket_server的结果
        audio_data: 音频数据
        sample_rate: 采样率
    """
    plt.figure(figsize=(15, 8))
    
    # 绘制音频波形作为背景
    audio_duration = len(audio_data) / sample_rate
    time_axis = np.arange(len(audio_data)) / sample_rate
    
    # 归一化音频数据并添加偏移
    audio_normalized = audio_data / np.max(np.abs(audio_data)) * 0.3
    plt.plot(time_axis, audio_normalized + 2, color='gray', alpha=0.3, linewidth=0.5)
    
    # 绘制test_streaming_vad的结果
    for i, segment in enumerate(streaming_results):
        plt.plot([segment['start'], segment['end']], [3, 3], 'b-', linewidth=3)
        plt.text((segment['start'] + segment['end']) / 2, 3.1, 
                f"{segment['duration']:.1f}s", 
                ha='center', fontsize=8)
    
    # 绘制websocket_server的结果
    for i, segment in enumerate(websocket_results):
        plt.plot([segment['start'], segment['end']], [2, 2], 'g-', linewidth=3)
        plt.text((segment['start'] + segment['end']) / 2, 2.1, 
                f"{segment['duration']:.1f}s", 
                ha='center', fontsize=8)
    
    # 添加网格线
    grid_interval = 10  # 秒
    for i in range(0, int(audio_duration) + grid_interval, grid_interval):
        plt.axvline(x=i, color='gray', linestyle='--', alpha=0.3)
        plt.text(i, 1.5, f"{i}s", ha='center', fontsize=8)
    
    # 设置y轴刻度
    plt.yticks([2, 3], ['WebSocket风格', 'test_streaming_vad'])
    
    # 添加图例
    plt.plot([], [], 'b-', linewidth=3, label='test_streaming_vad')
    plt.plot([], [], 'g-', linewidth=3, label='WebSocket风格')
    plt.legend(loc='upper right')
    
    plt.xlabel('时间 (秒)')
    plt.title('VAD实现对比')
    plt.grid(True, axis='x', alpha=0.3)
    plt.xlim(0, audio_duration * 1.05)
    plt.ylim(1.5, 3.5)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig('vad_implementation_comparison.png', dpi=150)
    print(f"VAD实现对比图已保存到vad_implementation_comparison.png")


def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        # 默认使用chinese_180s.wav
        audio_file = "chinese_180s.wav"
        if not os.path.exists(audio_file):
            print(f"错误: 找不到默认音频文件 {audio_file}")
            print(f"用法: python vad_comparison.py [音频文件路径]")
            return
    
    # 对比VAD实现
    compare_vad_implementations(audio_file)


if __name__ == "__main__":
    main()
