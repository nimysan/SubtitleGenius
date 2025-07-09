#!/usr/bin/env python
"""
测试VAD (Voice Activity Detection) 在音频上的表现
比较批处理和流式处理的结果
支持通过命令行参数指定音频文件路径
"""

import os
import sys
import numpy as np
import soundfile as sf
import torch
import logging
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加whisper_streaming目录到路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'whisper_streaming'))
from silero_vad_iterator import FixedVADIterator

def test_batch_vad(audio_path):
    """使用batch方式处理整个音频文件，获取语音段时间戳"""
    from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
    model = load_silero_vad()
    wav = read_audio(audio_path)
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        return_seconds=True,  # Return speech timestamps in seconds (default is samples)
        )
    print("\n===== Batch VAD 处理结果 =====")
    print(speech_timestamps)
    return speech_timestamps

def load_audio_file(file_path, sample_rate=16000):
    """加载音频文件并转换为正确的格式"""
    logger.info(f"Loading audio file: {file_path}")
    
    # 加载音频文件
    audio_data, file_sample_rate = sf.read(file_path)
    
    # 打印音频属性
    print(f"\n===== Audio File Properties =====")
    print(f"File: {file_path}")
    print(f"Sample rate: {file_sample_rate} Hz")
    print(f"Channels: {1 if len(audio_data.shape) == 1 else audio_data.shape[1]}")
    print(f"Duration: {len(audio_data)/file_sample_rate:.2f} seconds")
    print(f"Data type: {audio_data.dtype}")
    print(f"Min value: {np.min(audio_data)}")
    print(f"Max value: {np.max(audio_data)}")
    print(f"Mean value: {np.mean(audio_data)}")
    print(f"RMS value: {np.sqrt(np.mean(np.square(audio_data)))}")
    
    # 确保音频是单声道
    if len(audio_data.shape) > 1:
        print(f"Converting stereo to mono")
        audio_data = audio_data[:, 0]
    
    # 确保音频是float32格式
    if audio_data.dtype != np.float32:
        print(f"Converting {audio_data.dtype} to float32")
        audio_data = audio_data.astype(np.float32)
    
    # 如果需要，重采样到16kHz
    if file_sample_rate != sample_rate:
        import librosa
        print(f"Resampling from {file_sample_rate}Hz to {sample_rate}Hz")
        audio_data = librosa.resample(audio_data, orig_sr=file_sample_rate, target_sr=sample_rate)
    
    # 绘制波形
    output_dir = Path("vad_test_output")
    output_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(audio_data)) / sample_rate, audio_data)
    plt.title('Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    
    # 使用音频文件名作为波形图文件名
    audio_filename = os.path.basename(file_path)
    waveform_filename = f"{os.path.splitext(audio_filename)[0]}_waveform.png"
    waveform_path = output_dir / waveform_filename
    plt.savefig(waveform_path)
    print(f"Waveform saved to {waveform_path}")
    
    logger.info(f"Audio length: {len(audio_data)/sample_rate:.2f} seconds")
    return audio_data, sample_rate

def analyze_with_fixed_vad(audio_data, sample_rate=16000, threshold=0.5, min_silence_duration_ms=500, speech_pad_ms=100):
    """
    使用FixedVADIterator分析音频数据
    
    Args:
        audio_data: 音频数据（numpy数组）
        sample_rate: 采样率
        threshold: 语音阈值（0.0-1.0）
        min_silence_duration_ms: 最小静音持续时间（毫秒）
        speech_pad_ms: 语音段填充时间（毫秒）
    """
    # 加载Silero VAD模型
    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad'
    )
    
    # 创建VAD迭代器
    vad = FixedVADIterator(
        model=model,
        threshold=threshold,
        sampling_rate=sample_rate,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms
    )
    
    # 处理音频块
    chunk_size = 512  # Silero VAD要求的确切大小
    results = []
    current_time = 0.0
    
    print(f"\n===== Processing audio with FixedVADIterator =====")
    print(f"Parameters: threshold={threshold}, min_silence_duration_ms={min_silence_duration_ms}, speech_pad_ms={speech_pad_ms}")
    print(f"Audio length: {len(audio_data)/sample_rate:.2f} seconds")
    print(f"Processing in chunks of {chunk_size} samples ({chunk_size/sample_rate:.2f} seconds)")
    
    # 按块处理
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i+chunk_size]
        
        # 如果需要，用零填充
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
        
        # 获取此块的原始语音概率
        chunk_prob = model(torch.tensor([chunk]), sample_rate).item()
        
        # 使用VAD迭代器处理块
        result = vad(chunk, return_seconds=True)
        
        # 只打印有结果的块，避免输出过多
        if result:
            print(f"Chunk at {current_time:.2f}s: prob={chunk_prob:.4f}, result={result}")
        
        if result:
            # 根据当前位置调整时间戳
            if 'start' in result:
                result['start'] += current_time
            if 'end' in result:
                result['end'] += current_time
            
            results.append(result)
        
        current_time += chunk_size / sample_rate
    
    # 如果有开始但没有结束，则强制添加结束
    has_start = any('start' in r for r in results)
    has_end = any('end' in r for r in results)
    
    if has_start and not has_end:
        print("\nDetected speech start but no end. Adding end at the end of audio.")
        results.append({'end': len(audio_data) / sample_rate})
    
    return results

def compare_batch_and_streaming(batch_results, streaming_results):
    """比较批处理和流式处理VAD结果"""
    print("\n===== Batch vs Streaming VAD 比较 =====")
    
    # 将流式结果转换为段
    streaming_segments = []
    start_time = None
    
    for result in streaming_results:
        if 'start' in result:
            start_time = result['start']
        elif 'end' in result and start_time is not None:
            streaming_segments.append({
                'start': start_time,
                'end': result['end'],
                'duration': result['end'] - start_time
            })
            start_time = None
    
    # 打印比较
    print(f"Batch VAD 检测到的语音段数量: {len(batch_results)}")
    print(f"Streaming VAD 检测到的语音段数量: {len(streaming_segments)}")
    
    # 打印批处理段
    print("\nBatch VAD 语音段:")
    print(f"{'#':<5} {'Start (s)':<15} {'End (s)':<15} {'Duration (s)':<15}")
    print("-" * 50)
    
    for i, segment in enumerate(batch_results):
        duration = segment['end'] - segment['start']
        print(f"{i+1:<5} {segment['start']:<15.2f} {segment['end']:<15.2f} {duration:<15.2f}")
    
    # 打印流式段
    print("\nStreaming VAD 语音段:")
    print(f"{'#':<5} {'Start (s)':<15} {'End (s)':<15} {'Duration (s)':<15}")
    print("-" * 50)
    
    for i, segment in enumerate(streaming_segments):
        print(f"{i+1:<5} {segment['start']:<15.2f} {segment['end']:<15.2f} {segment['duration']:<15.2f}")
    
    # 计算重叠
    overlap_count = 0
    for batch_seg in batch_results:
        batch_start = batch_seg['start']
        batch_end = batch_seg['end']
        
        for stream_seg in streaming_segments:
            stream_start = stream_seg['start']
            stream_end = stream_seg['end']
            
            # 检查是否有重叠
            if (stream_start <= batch_end and stream_end >= batch_start):
                overlap_count += 1
                break
    
    overlap_ratio = overlap_count / len(batch_results) if batch_results else 0
    print(f"\n重叠语音段数量: {overlap_count}")
    print(f"重叠比例: {overlap_ratio:.2%}")
    
    # 比较总时长
    batch_duration = sum(seg['end'] - seg['start'] for seg in batch_results)
    streaming_duration = sum(seg['duration'] for seg in streaming_segments)
    
    print(f"\nBatch VAD 总语音时长: {batch_duration:.2f} 秒")
    print(f"Streaming VAD 总语音时长: {streaming_duration:.2f} 秒")
    print(f"时长差异: {abs(batch_duration - streaming_duration):.2f} 秒 ({abs(batch_duration - streaming_duration) / batch_duration * 100 if batch_duration > 0 else 0:.2f}%)")

def save_results_to_markdown(audio_path, batch_results, best_params, best_results):
    """将测试结果保存为Markdown文件"""
    output_dir = Path("vad_test_output")
    output_dir.mkdir(exist_ok=True)
    
    # 使用音频文件名作为输出文件名
    audio_filename = os.path.basename(audio_path)
    output_filename = f"{os.path.splitext(audio_filename)[0]}_vad_analysis.md"
    output_path = output_dir / output_filename
    
    # 创建Markdown内容
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# VAD 测试分析: {audio_filename}\n\n")
        
        f.write("## 音频文件信息\n\n")
        f.write(f"- **文件名**: {audio_filename}\n")
        
        # 计算批处理总时长
        batch_duration = sum(seg['end'] - seg['start'] for seg in batch_results)
        f.write(f"- **总语音时长**: {batch_duration:.2f} 秒\n")
        f.write(f"- **检测到的语音段数量**: {len(batch_results)}\n\n")
        
        f.write("## 批处理 VAD 结果\n\n")
        f.write("| # | 开始时间 (秒) | 结束时间 (秒) | 持续时间 (秒) |\n")
        f.write("|---|------------|------------|------------|\n")
        
        for i, segment in enumerate(batch_results):
            duration = segment['end'] - segment['start']
            f.write(f"| {i+1} | {segment['start']:.2f} | {segment['end']:.2f} | {duration:.2f} |\n")
        
        # 如果找到了最佳参数
        if best_params and best_results:
            f.write("\n## 最佳流式处理参数\n\n")
            f.write(f"- **threshold**: {best_params['threshold']}\n")
            f.write(f"- **min_silence_duration_ms**: {best_params['min_silence_duration_ms']}\n")
            f.write(f"- **speech_pad_ms**: {best_params['speech_pad_ms']}\n\n")
            
            # 将流式结果转换为段
            streaming_segments = []
            start_time = None
            
            for result in best_results:
                if 'start' in result:
                    start_time = result['start']
                elif 'end' in result and start_time is not None:
                    streaming_segments.append({
                        'start': start_time,
                        'end': result['end'],
                        'duration': result['end'] - start_time
                    })
                    start_time = None
            
            f.write("## 流式处理 VAD 结果\n\n")
            f.write("| # | 开始时间 (秒) | 结束时间 (秒) | 持续时间 (秒) |\n")
            f.write("|---|------------|------------|------------|\n")
            
            for i, segment in enumerate(streaming_segments):
                f.write(f"| {i+1} | {segment['start']:.2f} | {segment['end']:.2f} | {segment['duration']:.2f} |\n")
            
            # 计算重叠
            overlap_count = 0
            for batch_seg in batch_results:
                batch_start = batch_seg['start']
                batch_end = batch_seg['end']
                
                for stream_seg in streaming_segments:
                    stream_start = stream_seg['start']
                    stream_end = stream_seg['end']
                    
                    # 检查是否有重叠
                    if (stream_start <= batch_end and stream_end >= batch_start):
                        overlap_count += 1
                        break
            
            overlap_ratio = overlap_count / len(batch_results) if batch_results else 0
            
            # 计算总时长
            streaming_duration = sum(seg['duration'] for seg in streaming_segments)
            
            f.write("\n## 比较结果\n\n")
            f.write(f"- **批处理检测到的语音段数量**: {len(batch_results)}\n")
            f.write(f"- **流式处理检测到的语音段数量**: {len(streaming_segments)}\n")
            f.write(f"- **重叠语音段数量**: {overlap_count}\n")
            f.write(f"- **重叠比例**: {overlap_ratio:.2%}\n")
            f.write(f"- **批处理总语音时长**: {batch_duration:.2f} 秒\n")
            f.write(f"- **流式处理总语音时长**: {streaming_duration:.2f} 秒\n")
            f.write(f"- **时长差异**: {abs(batch_duration - streaming_duration):.2f} 秒 ({abs(batch_duration - streaming_duration) / batch_duration * 100 if batch_duration > 0 else 0:.2f}%)\n\n")
        else:
            f.write("\n## 流式处理结果\n\n")
            f.write("未找到有效的参数组合，所有测试的参数组合都无法与批处理结果产生良好的重叠。\n\n")
            f.write("可能的原因：\n\n")
            f.write("1. 流式处理和批处理的VAD算法有根本差异\n")
            f.write("2. 需要测试更多的参数组合\n")
            f.write("3. 音频特性可能不适合当前的VAD算法\n")
            f.write("4. 可能需要自定义VAD算法以适应特定的音频类型\n\n")
        
        f.write("## 关键参数分析\n\n")
        f.write("### 1. threshold (语音阈值)\n\n")
        f.write("- 较低的阈值 (0.2-0.3) 可以检测到更多的语音段，但可能会包含噪音\n")
        f.write("- 较高的阈值 (0.5-0.7) 只检测明显的语音，但可能会漏掉轻声部分\n")
        f.write("- 最佳阈值通常在 0.3 左右，这是检测语音和避免噪音之间的平衡点\n\n")
        
        f.write("### 2. min_silence_duration_ms (最小静音持续时间)\n\n")
        f.write("- 较短的时间 (200-300ms) 会将短暂的停顿视为语音段之间的分隔\n")
        f.write("- 较长的时间 (500ms以上) 会将短暂停顿的语音视为一个连续段\n")
        f.write("- 对于中文解说，300-500ms 通常是合适的，因为中文语速较快，停顿较短\n\n")
        
        f.write("### 3. speech_pad_ms (语音段填充时间)\n\n")
        f.write("- 较短的填充 (50ms) 会使语音段更精确，但可能会截断词语的开头和结尾\n")
        f.write("- 较长的填充 (100ms以上) 会保留更完整的语音，但可能会包含一些静音\n")
        f.write("- 100ms 通常是一个好的平衡点，确保词语的开头和结尾不被截断\n\n")
        
        f.write("## 结论\n\n")
        if best_params and best_results:
            if overlap_ratio > 0.8:
                f.write("流式处理VAD与批处理VAD结果非常接近，参数配置良好。\n")
            elif overlap_ratio > 0.5:
                f.write("流式处理VAD与批处理VAD结果有一定差异，但总体表现可接受。\n")
            else:
                f.write("流式处理VAD与批处理VAD结果差异较大，可能需要进一步调整参数。\n")
            
            f.write("\n最佳参数组合为: threshold={}, min_silence_duration_ms={}, speech_pad_ms={}。\n".format(
                best_params['threshold'], best_params['min_silence_duration_ms'], best_params['speech_pad_ms']
            ))
        else:
            f.write("所有测试的参数组合都无法与批处理结果产生良好的重叠。建议尝试更多的参数组合或考虑自定义VAD算法。\n")
            f.write("\n可能需要考虑的改进方向：\n")
            f.write("1. 实现强制分段机制，即使没有明显的静音也可以在适当位置分割长语音段\n")
            f.write("2. 添加基于语音概率变化的分段逻辑，而不仅仅依赖于阈值\n")
            f.write("3. 考虑添加最大段长限制，防止单个段落过长\n")
            f.write("4. 尝试不同的VAD算法或预处理步骤\n")
    
    print(f"\n分析结果已保存到: {output_path}")
    return output_path

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试VAD在音频上的表现')
    parser.add_argument('--audio_path', type=str, default='/Users/yexw/PycharmProjects/SubtitleGenius/chinese_90s.wav',
                        help='音频文件路径')
    args = parser.parse_args()
    
    audio_path = args.audio_path
    
    print(f"测试音频文件: {audio_path}")
    
    # 加载音频文件
    audio_data, sample_rate = load_audio_file(audio_path)
    
    # 运行批处理VAD
    batch_results = test_batch_vad(audio_path)
    
    # 测试不同参数组合的流式VAD
    print("\n===== 测试不同参数组合 =====")
    
    # 定义要测试的参数组合
    param_combinations = [
        {"threshold": 0.3, "min_silence_duration_ms": 300, "speech_pad_ms": 100},
        {"threshold": 0.3, "min_silence_duration_ms": 500, "speech_pad_ms": 100},
        {"threshold": 0.5, "min_silence_duration_ms": 300, "speech_pad_ms": 100},
        {"threshold": 0.5, "min_silence_duration_ms": 500, "speech_pad_ms": 100},
        {"threshold": 0.2, "min_silence_duration_ms": 200, "speech_pad_ms": 50},
        # 添加更多参数组合
        {"threshold": 0.1, "min_silence_duration_ms": 200, "speech_pad_ms": 50},
        {"threshold": 0.4, "min_silence_duration_ms": 400, "speech_pad_ms": 80},
        {"threshold": 0.6, "min_silence_duration_ms": 600, "speech_pad_ms": 120},
    ]
    
    best_params = None
    best_overlap = 0
    best_results = None
    
    for params in param_combinations:
        print(f"\n参数组合: {params}")
        streaming_results = analyze_with_fixed_vad(
            audio_data, 
            sample_rate=sample_rate,
            threshold=params["threshold"],
            min_silence_duration_ms=params["min_silence_duration_ms"],
            speech_pad_ms=params["speech_pad_ms"]
        )
        
        # 将流式结果转换为段以进行比较
        streaming_segments = []
        start_time = None
        
        for result in streaming_results:
            if 'start' in result:
                start_time = result['start']
            elif 'end' in result and start_time is not None:
                streaming_segments.append({
                    'start': start_time,
                    'end': result['end'],
                    'duration': result['end'] - start_time
                })
                start_time = None
        
        # 计算重叠
        overlap_count = 0
        for batch_seg in batch_results:
            batch_start = batch_seg['start']
            batch_end = batch_seg['end']
            
            for stream_seg in streaming_segments:
                stream_start = stream_seg['start']
                stream_end = stream_seg['end']
                
                # 检查是否有重叠
                if (stream_start <= batch_end and stream_end >= batch_start):
                    overlap_count += 1
                    break
        
        overlap_ratio = overlap_count / len(batch_results) if batch_results else 0
        print(f"重叠比例: {overlap_ratio:.2%}")
        
        # 如果此组合有更好的重叠，则更新最佳参数
        if overlap_ratio > best_overlap:
            best_overlap = overlap_ratio
            best_params = params
            best_results = streaming_results
    
    # 打印最佳参数
    print(f"\n===== 最佳参数组合 =====")
    if best_params:
        print(f"参数: {best_params}")
        print(f"重叠比例: {best_overlap:.2%}")
        
        # 使用最佳参数比较批处理和流式结果
        if best_results:
            compare_batch_and_streaming(batch_results, best_results)
    else:
        print("未找到有效的参数组合，所有测试的参数组合都无法与批处理结果产生良好的重叠。")
    
    # 保存结果到Markdown文件
    output_path = save_results_to_markdown(audio_path, batch_results, best_params, best_results)
    
    # 打印关键发现
    print("\n===== 关键参数分析 =====")
    print("1. threshold (语音阈值):")
    print("   - 较低的阈值 (0.2-0.3) 可以检测到更多的语音段，但可能会包含噪音")
    print("   - 较高的阈值 (0.5-0.7) 只检测明显的语音，但可能会漏掉轻声部分")
    print("   - 最佳阈值通常在 0.3 左右，这是检测语音和避免噪音之间的平衡点")
    
    print("\n2. min_silence_duration_ms (最小静音持续时间):")
    print("   - 较短的时间 (200-300ms) 会将短暂的停顿视为语音段之间的分隔")
    print("   - 较长的时间 (500ms以上) 会将短暂停顿的语音视为一个连续段")
    print("   - 对于中文解说，300-500ms 通常是合适的，因为中文语速较快，停顿较短")
    
    print("\n3. speech_pad_ms (语音段填充时间):")
    print("   - 较短的填充 (50ms) 会使语音段更精确，但可能会截断词语的开头和结尾")
    print("   - 较长的填充 (100ms以上) 会保留更完整的语音，但可能会包含一些静音")
    print("   - 100ms 通常是一个好的平衡点，确保词语的开头和结尾不被截断")
    
    print("\n4. 批处理 vs 流式处理:")
    print("   - 批处理可以看到整个音频，因此可以做出更全局的决策")
    print("   - 流式处理是实时的，只能基于当前和过去的数据做决策")
    print("   - 流式处理需要更精细的参数调整才能达到与批处理相似的效果")
    
    if best_params:
        print("\n5. 关键发现:")
        print(f"   - 最佳参数组合: threshold={best_params['threshold']}, "
              f"min_silence_duration_ms={best_params['min_silence_duration_ms']}, "
              f"speech_pad_ms={best_params['speech_pad_ms']}")
        print(f"   - 重叠比例: {best_overlap:.2%}")
        if best_overlap > 0.8:
            print("   - 流式处理VAD与批处理VAD结果非常接近，参数配置良好")
        elif best_overlap > 0.5:
            print("   - 流式处理VAD与批处理VAD结果有一定差异，但总体表现可接受")
        else:
            print("   - 流式处理VAD与批处理VAD结果差异较大，可能需要进一步调整参数")
    else:
        print("\n5. 关键发现:")
        print("   - 所有测试的参数组合都无法与批处理结果产生良好的重叠")
        print("   - 可能需要尝试更多的参数组合或考虑自定义VAD算法")
        print("   - 建议实现强制分段机制，即使没有明显的静音也可以在适当位置分割长语音段")
        print("   - 考虑添加基于语音概率变化的分段逻辑，而不仅仅依赖于阈值")

if __name__ == "__main__":
    main()
