import torch
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt
from collections import Counter
torch.set_num_threads(1)

def analyze_speech_segments(speech_timestamps: List[Dict], audio_length_seconds: float):
    """
    分析语音段的详细信息
    
    Args:
        speech_timestamps: 语音段时间戳列表
        audio_length_seconds: 音频总长度（秒）
    """
    # 计算语音段数量
    segment_count = len(speech_timestamps)
    print(f"\n===== 语音段分析 =====")
    print(f"总语音段数量: {segment_count}")
    
    if segment_count == 0:
        print("未检测到语音段")
        return
    
    # 计算每个语音段的长度
    segment_lengths = []
    total_speech_duration = 0
    
    print("\n各语音段详情:")
    print(f"{'序号':<6}{'开始时间(秒)':<15}{'结束时间(秒)':<15}{'持续时间(秒)':<15}")
    print("-" * 50)
    
    for i, segment in enumerate(speech_timestamps):
        start = segment['start']
        end = segment['end']
        duration = end - start
        segment_lengths.append(duration)
        total_speech_duration += duration
        
        print(f"{i+1:<6}{start:<15.2f}{end:<15.2f}{duration:<15.2f}")
    
    # 计算统计信息
    avg_length = np.mean(segment_lengths)
    min_length = np.min(segment_lengths)
    max_length = np.max(segment_lengths)
    median_length = np.median(segment_lengths)
    std_length = np.std(segment_lengths)
    
    # 语音占比
    speech_ratio = (total_speech_duration / audio_length_seconds) * 100
    
    print("\n===== 统计信息 =====")
    print(f"平均语音段长度: {avg_length:.2f} 秒")
    print(f"最短语音段长度: {min_length:.2f} 秒")
    print(f"最长语音段长度: {max_length:.2f} 秒")
    print(f"中位语音段长度: {median_length:.2f} 秒")
    print(f"语音段长度标准差: {std_length:.2f} 秒")
    print(f"总语音时长: {total_speech_duration:.2f} 秒")
    print(f"音频总长度: {audio_length_seconds:.2f} 秒")
    print(f"语音占比: {speech_ratio:.2f}%")
    
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
    
    print("\n===== 语音段长度分布 =====")
    for i, count in enumerate(distribution):
        percentage = (count / segment_count) * 100
        print(f"{bin_labels[i]}: {count} 段 ({percentage:.1f}%)")
    
    # 计算语音段间隔
    if segment_count > 1:
        gaps = []
        for i in range(1, len(speech_timestamps)):
            gap = speech_timestamps[i]['start'] - speech_timestamps[i-1]['end']
            gaps.append(gap)
        
        avg_gap = np.mean(gaps)
        max_gap = np.max(gaps)
        
        print("\n===== 语音段间隔 =====")
        print(f"平均间隔: {avg_gap:.2f} 秒")
        print(f"最大间隔: {max_gap:.2f} 秒")

    # 可视化语音段分布
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(segment_lengths, bins=10, alpha=0.7, color='blue')
        plt.title('语音段长度分布直方图')
        plt.xlabel('语音段长度 (秒)')
        plt.ylabel('频次')
        plt.grid(True, alpha=0.3)
        plt.savefig('vad_segment_distribution.png')
        print("\n已保存语音段长度分布直方图到 'vad_segment_distribution.png'")
    except Exception as e:
        print(f"\n无法生成可视化: {e}")

# 加载模型和工具
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
(get_speech_timestamps, _, read_audio, VADIterator, collect_chunks) = utils

# 读取音频文件
audio_path = '/Users/yexw/PycharmProjects/SubtitleGenius/chinese_a_b.wav'
wav = read_audio(audio_path)

# 获取音频长度（秒）
audio_length_seconds = len(wav) / 16000  # 假设采样率为16kHz

# 获取语音时间戳
speech_timestamps = get_speech_timestamps(
  wav,
  model,
  return_seconds=True,  # Return speech timestamps in seconds (default is samples)
)

# 打印原始时间戳
print("原始语音时间戳:")
print(speech_timestamps)

# 分析语音段
analyze_speech_segments(speech_timestamps, audio_length_seconds)
