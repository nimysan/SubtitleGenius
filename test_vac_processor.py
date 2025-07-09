#!/usr/bin/env python
"""
测试VAC处理器
使用chinese_90s.wav作为输入，每2秒送入一次，输出经过VAC处理后识别的分片
"""

import os
import sys
import logging
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import time
from pathlib import Path
import json
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.WARNING,  # 将日志级别改为WARNING，减少输出
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('vac_test.log')
    ]
)
logger = logging.getLogger(__name__)

# 导入VAC处理器
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from subtitle_genius.stream.vac_processor import VACProcessor

def load_audio_file(file_path, chunk_duration=2.0, sample_rate=16000):
    """
    加载音频文件并按时间分块（每块2秒）
    
    Args:
        file_path: 音频文件路径
        chunk_duration: 每个块的持续时间（秒）
        sample_rate: 采样率
        
    Returns:
        chunks: 音频块列表
        sample_rate: 采样率
    """
    logger.info(f"加载音频文件: {file_path}")
    
    # 加载音频文件
    audio_data, file_sample_rate = sf.read(file_path)
    
    # 确保音频是单声道
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]
    
    # 确保音频是float32格式
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    # 重采样到16kHz（如果需要）
    if file_sample_rate != sample_rate:
        import librosa
        logger.info(f"重采样音频从 {file_sample_rate}Hz 到 {sample_rate}Hz")
        audio_data = librosa.resample(audio_data, orig_sr=file_sample_rate, target_sr=sample_rate)
    
    # 计算每个块的样本数
    chunk_size = int(chunk_duration * sample_rate)
    
    # 分块
    chunks = []
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i+chunk_size]
        chunks.append(chunk)
    
    logger.info(f"音频长度: {len(audio_data)/sample_rate:.2f}秒, 分为 {len(chunks)} 个块，每块 {chunk_duration} 秒")
    return chunks, sample_rate

def convert_to_raw_bytes(audio_chunk):
    """
    将音频块转换为原始字节
    
    Args:
        audio_chunk: 音频块（numpy数组）
        
    Returns:
        raw_bytes: 原始字节
    """
    # 确保音频是float32格式
    if audio_chunk.dtype != np.float32:
        audio_chunk = audio_chunk.astype(np.float32)
    
    # 转换为16-bit PCM
    audio_int16 = (audio_chunk * 32767).astype(np.int16)
    
    # 转换为字节
    raw_bytes = audio_int16.tobytes()
    
    return raw_bytes

def visualize_input_output_transformation(chunks, segments, chunk_duration=2.0):
    """
    可视化输入块到输出语音段的转换
    
    Args:
        chunks: 输入音频块列表
        segments: 输出语音段列表
        chunk_duration: 每个块的持续时间（秒）
    """
    # 计算输入块的时间范围
    input_chunks = []
    for i in range(len(chunks)):
        start_time = i * chunk_duration
        end_time = (i + 1) * chunk_duration
        input_chunks.append({
            "id": f"chunk_{i:03d}",
            "start_time": start_time,
            "end_time": end_time
        })
    
    # 打印输入到输出的转换表格
    print("\n===== 输入块到输出段的转换 =====")
    print(f"{'输入块数':<10} {'输出段数':<10} {'输入总时长(秒)':<20} {'输出总时长(秒)':<20} {'压缩比':<10}")
    
    input_total_duration = len(chunks) * chunk_duration
    output_total_duration = sum(segment["duration"] for segment in segments)
    compression_ratio = input_total_duration / output_total_duration if output_total_duration > 0 else float('inf')
    
    print(f"{len(chunks):<10} {len(segments):<10} {input_total_duration:<20.2f} {output_total_duration:<20.2f} {compression_ratio:<10.2f}")
    
    # 打印输入块和输出段的时间分布
    print("\n输入块和输出段的时间分布:")
    print(f"{'时间范围(秒)':<20} {'输入块':<40} {'输出段':<40}")
    print("-" * 100)
    
    # 创建时间轴（以2秒为间隔）
    max_time = max(input_chunks[-1]["end_time"], segments[-1]["end"] if segments else 0)
    time_ranges = [(i*2, (i+1)*2) for i in range(int(max_time/2) + 1)]
    
    for start, end in time_ranges:
        # 找出在此时间范围内的输入块
        input_in_range = [chunk["id"] for chunk in input_chunks 
                         if chunk["start_time"] < end and chunk["end_time"] > start]
        
        # 找出在此时间范围内的输出段
        output_in_range = [str(segment["id"]) for segment in segments 
                          if segment["start"] < end and segment["end"] > start]
        
        if input_in_range or output_in_range:
            input_str = ", ".join(input_in_range) if input_in_range else "-"
            output_str = ", ".join(output_in_range) if output_in_range else "-"
            print(f"{start:02.0f}-{end:02.0f}s{'':<14} {input_str:<40} {output_str:<40}")
    
    print("=" * 100)


def test_batch_vad():
    """使用batch方式处理整个音频文件，获取语音段时间戳"""
    from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
    model = load_silero_vad()
    wav = read_audio('/Users/yexw/PycharmProjects/SubtitleGenius/chinese_90s.wav')
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        return_seconds=True,  # Return speech timestamps in seconds (default is samples)
        )
    print("\n===== Batch VAD 处理结果 =====")
    print(speech_timestamps)
    return speech_timestamps
    
def main():
    # 先运行batch处理，获取参考结果
    batch_results = test_batch_vad()
    
    # 将batch结果转换为更易读的格式
    print("\n===== Batch VAD 语音段详情 =====")
    print(f"{'序号':<6}{'开始时间(秒)':<15}{'结束时间(秒)':<15}{'持续时间(秒)':<15}")
    print("-" * 50)
    for i, segment in enumerate(batch_results):
        start = segment['start']
        end = segment['end']
        duration = end - start
        print(f"{i+1:<6}{start:<15.2f}{end:<15.2f}{duration:<15.2f}")
    
    print("\n===== 开始流式 VAC 处理 =====")
    """主函数 - 测试VAC处理器的输入到输出转换"""
    # 音频文件路径
    audio_file = "chinese_90s.wav"
    chunk_duration = 3  # 每个块2秒
    
    # 创建输出目录
    output_dir = Path("vac_test_output")
    output_dir.mkdir(exist_ok=True)
    
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 加载音频文件，每2秒一个块
    chunks, sample_rate = load_audio_file(audio_file, chunk_duration=chunk_duration)
    
    # 创建VAC处理器
    vac_processor = VACProcessor()
    
    # 存储识别的语音段
    segments = []
    
    # 处理音频块
    print(f"音频文件: {audio_file}")
    print(f"总块数: {len(chunks)} (每块 {chunk_duration} 秒)")
    print(f"采样率: {sample_rate}Hz")
    print("=" * 50)
    
    # 表头
    print(f"{'序号':<6}{'开始时间(秒)':<15}{'结束时间(秒)':<15}{'持续时间(秒)':<15}{'样本数':<10}")
    print("-" * 65)
    
    # 处理音频块
    for i, chunk in enumerate(chunks):
        # 转换为原始字节
        raw_bytes = convert_to_raw_bytes(chunk)
        
        # 添加到VAC处理器
        vac_processor.add_audio_chunk(raw_bytes)
        
        # 获取所有待处理的语音段
        while vac_processor.has_pending_segments():
            segment = vac_processor.get_next_voice_segment()
            if segment:
                audio_data, start_time, end_time = segment
                
                # 创建段落ID
                segment_id = len(segments) + 1
                duration = end_time - start_time
                
                # 添加到段落列表
                segments.append({
                    "id": segment_id,
                    "start": start_time,  # 使用与batch处理相同的键名
                    "end": end_time,      # 使用与batch处理相同的键名
                    "duration": duration,
                    "samples": len(audio_data)
                })
                
                # 打印关键信息
                print(f"{segment_id:<6}{start_time:<15.2f}{end_time:<15.2f}{duration:<15.2f}{len(audio_data):<10}")
    
    # 处理剩余的语音段
    print("\n----- 处理剩余的语音段 -----")
    # 注意：VACProcessor没有flush方法，直接处理剩余段落
    
    while vac_processor.has_pending_segments():
        segment = vac_processor.get_next_voice_segment()
        if segment:
            audio_data, start_time, end_time = segment
            
            # 创建段落ID
            segment_id = len(segments) + 1
            duration = end_time - start_time
            
            # 添加到段落列表
            segments.append({
                "id": segment_id,
                "start": start_time,  # 使用与batch处理相同的键名
                "end": end_time,      # 使用与batch处理相同的键名
                "duration": duration,
                "samples": len(audio_data)
            })
            
            # 打印关键信息
            print(f"{segment_id:<6}{start_time:<15.2f}{end_time:<15.2f}{duration:<15.2f}{len(audio_data):<10}")
    
    # 将流式处理结果转换为与batch处理相同的格式
    vac_results = []
    for segment in segments:
        vac_results.append({
            'start': segment['start'],
            'end': segment['end']
        })
    
    # 打印统计信息
    durations = [segment["duration"] for segment in segments]
    total_duration = sum(durations)
    avg_duration = np.mean(durations) if durations else 0
    min_duration = min(durations) if durations else 0
    max_duration = max(durations) if durations else 0
    
    print("\n===== VAC流式处理统计信息 =====")
    print(f"总段落数: {len(segments)}")
    print(f"总持续时间: {total_duration:.2f}秒")
    print(f"平均持续时间: {avg_duration:.2f}秒")
    print(f"最短持续时间: {min_duration:.2f}秒")
    print(f"最长持续时间: {max_duration:.2f}秒")
    
    # 比较batch处理和流式处理的结果
    print("\n===== Batch处理 vs VAC流式处理 =====")
    print(f"Batch处理检测到的语音段数量: {len(batch_results)}")
    print(f"VAC流式处理检测到的语音段数量: {len(vac_results)}")
    
    # 计算两种方法检测结果的重叠度
    overlap_count = 0
    for batch_seg in batch_results:
        batch_start = batch_seg['start']
        batch_end = batch_seg['end']
        
        for vac_seg in vac_results:
            vac_start = vac_seg['start']
            vac_end = vac_seg['end']
            
            # 检查是否有重叠
            if (vac_start <= batch_end and vac_end >= batch_start):
                overlap_count += 1
                break
    
    overlap_ratio = overlap_count / len(batch_results) if batch_results else 0
    print(f"重叠语音段数量: {overlap_count}")
    print(f"重叠比例: {overlap_ratio:.2%}")
    
    # 可视化输入到输出的转换
    visualize_input_output_transformation(chunks, segments, chunk_duration)
    
    print("=" * 50)
    
    # 直接比较两种方法的输出格式
    print("\n===== 输出格式比较 =====")
    print("Batch VAD 输出格式:")
    print(batch_results[:3] if len(batch_results) > 3 else batch_results)
    print("\nVAC流式处理 输出格式:")
    print(vac_results[:3] if len(vac_results) > 3 else vac_results)
    
    # 保存段落信息到JSON文件
    result_dir = output_dir / timestamp
    result_dir.mkdir(exist_ok=True)
    
    # 保存VAC流式处理结果
    segments_file = result_dir / "vac_segments.json"
    with open(segments_file, 'w', encoding='utf-8') as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)
    
    # 保存batch处理结果
    batch_file = result_dir / "batch_segments.json"
    with open(batch_file, 'w', encoding='utf-8') as f:
        json.dump(batch_results, f, indent=2, ensure_ascii=False)
    
    # 保存比较结果
    comparison_file = result_dir / "comparison.json"
    comparison_data = {
        "batch_count": len(batch_results),
        "vac_count": len(vac_results),
        "overlap_count": overlap_count,
        "overlap_ratio": overlap_ratio
    }
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {result_dir}")
    print("测试完成")

if __name__ == "__main__":
    main()
