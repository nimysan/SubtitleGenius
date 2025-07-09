#!/usr/bin/env python
"""
脚本用于绘制Batch VAD和Continuous VAD的对比图
"""

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import os

# 定义Batch VAD语音段
batch_segments = [
    {'start': 0.86, 'end': 1.60},
    {'start': 52.92, 'end': 60.74},
    {'start': 60.96, 'end': 68.55},
    {'start': 70.52, 'end': 71.68},
    {'start': 73.92, 'end': 96.55},
    {'start': 96.70, 'end': 97.38},
    {'start': 97.79, 'end': 102.34},
    {'start': 102.65, 'end': 104.93},
    {'start': 105.12, 'end': 108.80},
    {'start': 109.56, 'end': 111.08},
    {'start': 111.23, 'end': 121.70},
    {'start': 122.56, 'end': 143.68},
    {'start': 144.00, 'end': 144.68},
    {'start': 145.40, 'end': 147.46},
    {'start': 147.90, 'end': 150.08},
    {'start': 150.49, 'end': 153.12},
    {'start': 153.28, 'end': 159.33},
    {'start': 159.45, 'end': 159.75},
    {'start': 160.06, 'end': 163.01},
    {'start': 163.39, 'end': 165.12},
    {'start': 167.42, 'end': 168.13},
    {'start': 170.04, 'end': 180.03}
]

# 定义Continuous VAD语音段
continuous_segments = [
    {'start': 0.90, 'end': 1.60, 'duration': 0.70},
    {'start': 52.90, 'end': 60.70, 'duration': 7.80},
    {'start': 61.00, 'end': 68.50, 'duration': 7.50},
    {'start': 70.50, 'end': 71.70, 'duration': 1.20},
    {'start': 73.90, 'end': 96.50, 'duration': 22.60},
    {'start': 96.70, 'end': 97.30, 'duration': 0.60},
    {'start': 97.80, 'end': 102.30, 'duration': 4.50},
    {'start': 102.70, 'end': 104.90, 'duration': 2.20},
    {'start': 105.10, 'end': 108.80, 'duration': 3.70},
    {'start': 109.60, 'end': 111.00, 'duration': 1.40},
    {'start': 111.20, 'end': 121.70, 'duration': 10.50},
    {'start': 122.60, 'end': 143.70, 'duration': 21.10},
    {'start': 144.00, 'end': 144.60, 'duration': 0.60},
    {'start': 145.40, 'end': 147.40, 'duration': 2.00},
    {'start': 147.90, 'end': 150.10, 'duration': 2.20},
    {'start': 150.50, 'end': 153.10, 'duration': 2.60},
    {'start': 153.30, 'end': 159.30, 'duration': 6.00},
    {'start': 159.50, 'end': 159.70, 'duration': 0.20},
    {'start': 160.10, 'end': 163.00, 'duration': 2.90},
    {'start': 163.40, 'end': 165.10, 'duration': 1.70},
    {'start': 167.40, 'end': 168.10, 'duration': 0.70}
]

def plot_vad_comparison(batch_segments, continuous_segments, audio_file=None):
    """
    绘制Batch VAD和Continuous VAD的对比图
    
    Args:
        batch_segments: Batch VAD语音段列表
        continuous_segments: Continuous VAD语音段列表
        audio_file: 音频文件路径（可选）
    """
    plt.figure(figsize=(20, 10))
    
    # 计算最大时间范围
    max_time = max(
        max([seg['end'] for seg in batch_segments]),
        max([seg['end'] for seg in continuous_segments])
    )
    
    # 如果有音频文件，绘制波形
    audio_data = None
    if audio_file and os.path.exists(audio_file):
        try:
            audio_data, sample_rate = sf.read(audio_file)
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]  # 转换为单声道
            
            # 归一化并缩放波形
            audio_data = audio_data / np.max(np.abs(audio_data)) * 0.3
            
            # 绘制波形（半透明）
            time_axis = np.arange(len(audio_data)) / sample_rate
            plt.plot(time_axis, audio_data + 1.5, color='gray', alpha=0.3, linewidth=0.5)
        except Exception as e:
            print(f"无法加载音频文件: {e}")
    
    # # 绘制Batch VAD语音段
    # for i, segment in enumerate(batch_segments):
    #     plt.plot([segment['start'], segment['end']], [2, 2], 'b-', linewidth=6)
    #     # 添加持续时间标签
    #     duration = segment['end'] - segment['start']
    #     if duration > 2:  # 只为较长的段添加标签
    #         plt.text((segment['start'] + segment['end']) / 2, 2.1, 
    #                 f"{duration:.1f}s", 
    #                 ha='center', fontsize=8)
    
    # 绘制Continuous VAD语音段
    for i, segment in enumerate(continuous_segments):
        plt.plot([segment['start'], segment['end']], [1, 1], 'g-', linewidth=6)
        # 添加持续时间标签
        if segment['duration'] > 2:  # 只为较长的段添加标签
            plt.text((segment['start'] + segment['end']) / 2, 1.1, 
                    f"{segment['duration']:.1f}s", 
                    ha='center', fontsize=8)
    
    # 添加垂直网格线（每10秒一条）
    for i in range(0, int(max_time) + 10, 10):
        plt.axvline(x=i, color='gray', linestyle='--', alpha=0.3)
        plt.text(i, 0.5, f"{i}s", ha='center', fontsize=8)
    
    # 设置Y轴刻度和标签
    plt.yticks([1, 2], ['Continuous', 'Batch'])
    plt.xlabel('Time (seconds)')
    plt.title('Batch VAD vs Continuous VAD Comparison')
    
    # 添加图例
    batch_line = plt.Line2D([0], [0], color='blue', linewidth=4, label='Batch VAD')
    continuous_line = plt.Line2D([0], [0], color='green', linewidth=4, label='Continuous VAD')
    plt.legend(handles=[batch_line, continuous_line], loc='upper right')
    
    # 设置坐标轴范围
    plt.xlim(0, max_time + 5)
    plt.ylim(0.5, 2.5)
    
    # 添加网格
    plt.grid(True, axis='x', alpha=0.3)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig('batch_vs_continuous_vad.png', dpi=150)
    print("对比图已保存为 batch_vs_continuous_vad.png")
    
    # 显示图像
    plt.show()

def calculate_statistics(batch_segments, continuous_segments):
    """计算并打印统计信息"""
    # 计算总语音时长
    batch_duration = sum(seg['end'] - seg['start'] for seg in batch_segments)
    continuous_duration = sum(seg['duration'] for seg in continuous_segments)
    
    # 计算平均语音段长度
    batch_avg_duration = batch_duration / len(batch_segments) if batch_segments else 0
    continuous_avg_duration = continuous_duration / len(continuous_segments) if continuous_segments else 0
    
    # 计算重叠
    overlap_count = 0
    for batch_seg in batch_segments:
        batch_start = batch_seg['start']
        batch_end = batch_seg['end']
        
        for cont_seg in continuous_segments:
            cont_start = cont_seg['start']
            cont_end = cont_seg['end']
            
            # 检查是否有重叠
            if (cont_start <= batch_end and cont_end >= batch_start):
                overlap_count += 1
                break
    
    overlap_ratio = overlap_count / len(batch_segments) if batch_segments else 0
    
    # 打印统计信息
    print("\n===== VAD Method Comparison =====")
    print(f"Batch VAD detected segments: {len(batch_segments)}")
    print(f"Continuous VAD detected segments: {len(continuous_segments)}")
    print(f"\nBatch VAD total speech duration: {batch_duration:.2f} seconds")
    print(f"Continuous VAD total speech duration: {continuous_duration:.2f} seconds")
    print(f"Duration difference: {abs(batch_duration - continuous_duration):.2f} seconds ({abs(batch_duration - continuous_duration) / batch_duration * 100 if batch_duration > 0 else 0:.2f}%)")
    print(f"\nAverage segment length:")
    print(f"- Batch: {batch_avg_duration:.2f} seconds")
    print(f"- Continuous: {continuous_avg_duration:.2f} seconds")
    print(f"\nOverlap ratio: {overlap_ratio:.2%}")

if __name__ == "__main__":
    # 尝试加载音频文件（如果存在）
    audio_file = "chinese_180s.wav"
    
    # 计算统计信息
    calculate_statistics(batch_segments, continuous_segments)
    
    # 绘制对比图
    plot_vad_comparison(batch_segments, continuous_segments, audio_file)
