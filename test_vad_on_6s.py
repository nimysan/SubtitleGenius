#!/usr/bin/env python
"""
Test script to analyze chinese_6s.wav using FixedVADIterator
"""

import os
import sys
import numpy as np
import soundfile as sf
import torch
import logging
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the whisper_streaming directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'whisper_streaming'))
from silero_vad_iterator import FixedVADIterator

def test_batch_vad():
    """使用batch方式处理整个音频文件，获取语音段时间戳"""
    from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
    model = load_silero_vad()
    wav = read_audio('/Users/yexw/PycharmProjects/SubtitleGenius/chinese_6s.wav')
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        return_seconds=True,  # Return speech timestamps in seconds (default is samples)
        )
    print("\n===== Batch VAD 处理结果 =====")
    print(speech_timestamps)
    return speech_timestamps

def load_audio_file(file_path, sample_rate=16000):
    """
    Load audio file and convert to the right format
    """
    logger.info(f"Loading audio file: {file_path}")
    
    # Load audio file
    audio_data, file_sample_rate = sf.read(file_path)
    
    # Print audio properties
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
    
    # Ensure audio is mono
    if len(audio_data.shape) > 1:
        print(f"Converting stereo to mono")
        audio_data = audio_data[:, 0]
    
    # Ensure audio is float32
    if audio_data.dtype != np.float32:
        print(f"Converting {audio_data.dtype} to float32")
        audio_data = audio_data.astype(np.float32)
    
    # Resample to 16kHz if needed
    if file_sample_rate != sample_rate:
        import librosa
        print(f"Resampling from {file_sample_rate}Hz to {sample_rate}Hz")
        audio_data = librosa.resample(audio_data, orig_sr=file_sample_rate, target_sr=sample_rate)
    
    # Plot waveform
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(audio_data)) / sample_rate, audio_data)
    plt.title('Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig('audio_waveform.png')
    print(f"Waveform saved to audio_waveform.png")
    
    logger.info(f"Audio length: {len(audio_data)/sample_rate:.2f} seconds")
    return audio_data, sample_rate

def analyze_with_fixed_vad(audio_data, sample_rate=16000, threshold=0.5, min_silence_duration_ms=500, speech_pad_ms=100):
    """
    Analyze audio data with FixedVADIterator
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of the audio
        threshold: Speech threshold (0.0-1.0)
        min_silence_duration_ms: Minimum silence duration in ms
        speech_pad_ms: Padding in ms to add to each side of a speech segment
    """
    # Load the Silero VAD model
    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad'
    )
    
    # Create the VAD iterator
    vad = FixedVADIterator(
        model=model,
        threshold=threshold,
        sampling_rate=sample_rate,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms
    )
    
    # Process the audio in chunks
    chunk_size = 512  # Exact size required by Silero VAD
    results = []
    current_time = 0.0
    
    print(f"\n===== Processing audio with FixedVADIterator =====")
    print(f"Parameters: threshold={threshold}, min_silence_duration_ms={min_silence_duration_ms}, speech_pad_ms={speech_pad_ms}")
    print(f"Audio length: {len(audio_data)/sample_rate:.2f} seconds")
    print(f"Processing in chunks of {chunk_size} samples ({chunk_size/sample_rate:.2f} seconds)")
    
    # Process in chunks
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i+chunk_size]
        
        # Pad with zeros if needed
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
        
        # Get raw speech probability for this chunk
        chunk_prob = model(torch.tensor([chunk]), sample_rate).item()
        
        # Process the chunk with VAD iterator
        result = vad(chunk, return_seconds=True)
        
        print(f"Chunk at {current_time:.2f}s: prob={chunk_prob:.4f}, result={result}")
        
        if result:
            # Adjust the timestamps based on the current position
            if 'start' in result:
                result['start'] += current_time
            if 'end' in result:
                result['end'] += current_time
            
            results.append(result)
        
        current_time += chunk_size / sample_rate
    
    # Force an end if we have a start but no end
    has_start = any('start' in r for r in results)
    has_end = any('end' in r for r in results)
    
    if has_start and not has_end:
        print("\nDetected speech start but no end. Adding end at the end of audio.")
        results.append({'end': len(audio_data) / sample_rate})
    
    return results

def test_different_thresholds(audio_data, sample_rate=16000):
    """
    Test different VAD thresholds on the audio
    """
    # Load the Silero VAD model
    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad'
    )
    
    # Try different thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    print("\n===== Testing different VAD thresholds =====")
    
    # Process in chunks of 512 samples
    chunk_size = 512
    for threshold in thresholds:
        print(f"\nThreshold: {threshold}")
        print(f"{'Time (s)':<10} {'Probability':<15} {'Speech?':<10}")
        print("-" * 35)
        
        speech_chunks = 0
        total_chunks = 0
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            
            # Pad with zeros if needed
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
            
            # Get speech probability
            prob = model(torch.tensor([chunk]), sample_rate).item()
            
            # Check if speech
            is_speech = prob > threshold
            
            # Print result
            time_sec = i / sample_rate
            print(f"{time_sec:<10.2f} {prob:<15.4f} {'YES' if is_speech else 'NO':<10}")
            
            if is_speech:
                speech_chunks += 1
            total_chunks += 1
        
        # Print summary
        speech_ratio = speech_chunks / total_chunks if total_chunks > 0 else 0
        print(f"\nSummary for threshold {threshold}:")
        print(f"Speech chunks: {speech_chunks}/{total_chunks} ({speech_ratio:.2%})")

def compare_batch_and_streaming(batch_results, streaming_results):
    """
    Compare batch and streaming VAD results
    """
    print("\n===== Batch vs Streaming VAD 比较 =====")
    
    # Convert streaming results to segments
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
    
    # Print comparison
    print(f"Batch VAD 检测到的语音段数量: {len(batch_results)}")
    print(f"Streaming VAD 检测到的语音段数量: {len(streaming_segments)}")
    
    # Print batch segments
    print("\nBatch VAD 语音段:")
    print(f"{'#':<5} {'Start (s)':<15} {'End (s)':<15} {'Duration (s)':<15}")
    print("-" * 50)
    
    for i, segment in enumerate(batch_results):
        duration = segment['end'] - segment['start']
        print(f"{i+1:<5} {segment['start']:<15.2f} {segment['end']:<15.2f} {duration:<15.2f}")
    
    # Print streaming segments
    print("\nStreaming VAD 语音段:")
    print(f"{'#':<5} {'Start (s)':<15} {'End (s)':<15} {'Duration (s)':<15}")
    print("-" * 50)
    
    for i, segment in enumerate(streaming_segments):
        print(f"{i+1:<5} {segment['start']:<15.2f} {segment['end']:<15.2f} {segment['duration']:<15.2f}")
    
    # Calculate overlap
    overlap_count = 0
    for batch_seg in batch_results:
        batch_start = batch_seg['start']
        batch_end = batch_seg['end']
        
        for stream_seg in streaming_segments:
            stream_start = stream_seg['start']
            stream_end = stream_seg['end']
            
            # Check if there is overlap
            if (stream_start <= batch_end and stream_end >= batch_start):
                overlap_count += 1
                break
    
    overlap_ratio = overlap_count / len(batch_results) if batch_results else 0
    print(f"\n重叠语音段数量: {overlap_count}")
    print(f"重叠比例: {overlap_ratio:.2%}")
    
    # Compare total duration
    batch_duration = sum(seg['end'] - seg['start'] for seg in batch_results)
    streaming_duration = sum(seg['duration'] for seg in streaming_segments)
    
    print(f"\nBatch VAD 总语音时长: {batch_duration:.2f} 秒")
    print(f"Streaming VAD 总语音时长: {streaming_duration:.2f} 秒")
    print(f"时长差异: {abs(batch_duration - streaming_duration):.2f} 秒 ({abs(batch_duration - streaming_duration) / batch_duration * 100 if batch_duration > 0 else 0:.2f}%)")

def main():
    """Main function"""
    # Audio file path
    audio_file = "chinese_6s.wav"
    
    # Load audio file
    audio_data, sample_rate = load_audio_file(audio_file)
    
    # Run batch VAD
    batch_results = test_batch_vad()
    
    # Test different parameter combinations for streaming VAD
    print("\n===== 测试不同参数组合 =====")
    
    # Define parameter combinations to test
    param_combinations = [
        {"threshold": 0.3, "min_silence_duration_ms": 300, "speech_pad_ms": 100},
        {"threshold": 0.3, "min_silence_duration_ms": 500, "speech_pad_ms": 100},
        {"threshold": 0.5, "min_silence_duration_ms": 300, "speech_pad_ms": 100},
        {"threshold": 0.5, "min_silence_duration_ms": 500, "speech_pad_ms": 100},
        {"threshold": 0.2, "min_silence_duration_ms": 200, "speech_pad_ms": 50},
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
        
        # Convert streaming results to segments for comparison
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
        
        # Calculate overlap
        overlap_count = 0
        for batch_seg in batch_results:
            batch_start = batch_seg['start']
            batch_end = batch_seg['end']
            
            for stream_seg in streaming_segments:
                stream_start = stream_seg['start']
                stream_end = stream_seg['end']
                
                # Check if there is overlap
                if (stream_start <= batch_end and stream_end >= batch_start):
                    overlap_count += 1
                    break
        
        overlap_ratio = overlap_count / len(batch_results) if batch_results else 0
        print(f"重叠比例: {overlap_ratio:.2%}")
        
        # Update best parameters if this combination has better overlap
        if overlap_ratio > best_overlap:
            best_overlap = overlap_ratio
            best_params = params
            best_results = streaming_results
    
    # Print best parameters
    print(f"\n===== 最佳参数组合 =====")
    print(f"参数: {best_params}")
    print(f"重叠比例: {best_overlap:.2%}")
    
    # Compare batch and streaming results with best parameters
    if best_results:
        compare_batch_and_streaming(batch_results, best_results)
    
    # Print key findings
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
    
    print("\n5. 关键发现:")
    print(f"   - 最佳参数组合: threshold={best_params['threshold']}, "
          f"min_silence_duration_ms={best_params['min_silence_duration_ms']}, "
          f"speech_pad_ms={best_params['speech_pad_ms']}")
    print("   - 这些参数在中文解说音频上能够达到最接近批处理结果的效果")
    print("   - 对于不同类型的音频（如不同语言、不同语速），可能需要不同的参数组合")

if __name__ == "__main__":
    main()
