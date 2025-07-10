#!/usr/bin/env python
"""
Test script to analyze chinese_90s.wav using FixedVADIterator with chunk processing
This script processes a long WAV file in chunks and applies VAD to each chunk,
ensuring consistent VAD effects across the entire audio.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the whisper_streaming directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'whisper_streaming'))
from silero_vad_iterator import FixedVADIterator

def load_silero_vad():
    """
    Load the Silero VAD model
    
    Returns:
        model: Silero VAD model
    """
    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad'
    )
    return model

def read_audio(file_path, sampling_rate=16000):
    """
    Read audio file and convert to the right format
    
    Args:
        file_path: Path to the audio file
        sampling_rate: Target sampling rate
        
    Returns:
        audio_data: Audio data as float32 numpy array
    """
    # Load audio file
    audio_data, file_sample_rate = sf.read(file_path)
    
    # Ensure audio is mono
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]
    
    # Ensure audio is float32
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    # Resample to target sampling rate if needed
    if file_sample_rate != sampling_rate:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=file_sample_rate, target_sr=sampling_rate)
    
    return audio_data

def get_speech_timestamps(audio_data, model, threshold=0.5, sampling_rate=16000, 
                         min_silence_duration_ms=500, speech_pad_ms=100, return_seconds=False):
    """
    Get speech timestamps from audio data using Silero VAD
    
    Args:
        audio_data: Audio data as float32 numpy array
        model: Silero VAD model
        threshold: Speech threshold (0.0-1.0)
        sampling_rate: Audio sampling rate
        min_silence_duration_ms: Minimum silence duration in ms
        speech_pad_ms: Padding in ms to add to each side of a speech segment
        return_seconds: Whether to return timestamps in seconds (default is samples)
        
    Returns:
        speech_timestamps: List of dicts with 'start' and 'end' keys
    """
    # Convert to tensor
    audio_tensor = torch.tensor(audio_data)
    
    # Get speech timestamps
    speech_timestamps = []
    
    # Process in chunks of 512 samples
    chunk_size = 512
    window_size_samples = chunk_size
    
    # Initialize variables
    triggered = False
    speeches = []
    current_speech = {}
    neg_threshold = threshold - 0.15
    temp_end = 0
    
    # Convert to samples
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i+chunk_size]
        
        # Pad with zeros if needed
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
        
        # Get speech probability
        speech_prob = model(torch.tensor([chunk]), sampling_rate).item()
        
        # Update triggered state
        if (speech_prob >= threshold) and temp_end:
            temp_end = 0
            
        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech['start'] = max(0, i - speech_pad_samples)
            continue
            
        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = i + chunk_size
            
            if i + chunk_size - temp_end < min_silence_samples:
                continue
            
            current_speech['end'] = temp_end + speech_pad_samples
            speeches.append(current_speech)
            current_speech = {}
            temp_end = 0
            triggered = False
    
    # Add the last speech segment if triggered
    if triggered and 'start' in current_speech:
        current_speech['end'] = len(audio_data)
        speeches.append(current_speech)
    
    # Convert to seconds if requested
    if return_seconds:
        for speech in speeches:
            speech['start'] = speech['start'] / sampling_rate
            speech['end'] = speech['end'] / sampling_rate
    
    return speeches

def test_batch_vad(audio_file):
    """使用batch方式处理整个音频文件，获取语音段时间戳"""
    model = load_silero_vad()
    wav = read_audio(audio_file)
    
    # 使用指定参数
    threshold = 0.3
    min_silence_duration_ms = 300
    speech_pad_ms = 100
    
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        threshold=threshold,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=True,  # Return speech timestamps in seconds (default is samples)
    )
    print("\n===== Batch VAD 处理结果 =====")
    print(f"参数: threshold={threshold}, min_silence_duration_ms={min_silence_duration_ms}, speech_pad_ms={speech_pad_ms}")
    print(f"检测到 {len(speech_timestamps)} 个语音段")
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
    plt.figure(figsize=(15, 4))
    plt.plot(np.arange(len(audio_data)) / sample_rate, audio_data)
    plt.title('Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig('audio_waveform_90s.png')
    print(f"Waveform saved to audio_waveform_90s.png")
    
    logger.info(f"Audio length: {len(audio_data)/sample_rate:.2f} seconds")
    return audio_data, sample_rate

def analyze_with_fixed_vad_chunks(audio_data, sample_rate=16000, chunk_duration=10.0, 
                                 threshold=0.5, min_silence_duration_ms=500, speech_pad_ms=100):
    """
    Analyze audio data with FixedVADIterator using chunk processing
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of the audio
        chunk_duration: Duration of each chunk in seconds
        threshold: Speech threshold (0.0-1.0)
        min_silence_duration_ms: Minimum silence duration in ms
        speech_pad_ms: Padding in ms to add to each side of a speech segment
    """
    # Load the Silero VAD model
    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad'
    )
    
    # Calculate chunk size in samples
    chunk_size_samples = int(chunk_duration * sample_rate)
    
    # Create the VAD iterator
    vad = FixedVADIterator(
        model=model,
        threshold=threshold,
        sampling_rate=sample_rate,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms
    )
    
    # Process the audio in chunks
    processing_chunk_size = 512  # Exact size required by Silero VAD
    results = []
    
    print(f"\n===== Processing audio with FixedVADIterator in chunks =====")
    print(f"Parameters: threshold={threshold}, min_silence_duration_ms={min_silence_duration_ms}, speech_pad_ms={speech_pad_ms}")
    print(f"Audio length: {len(audio_data)/sample_rate:.2f} seconds")
    print(f"Processing in chunks of {chunk_duration} seconds ({chunk_size_samples} samples)")
    print(f"VAD processing chunk size: {processing_chunk_size} samples ({processing_chunk_size/sample_rate:.2f} seconds)")
    
    # Process in larger chunks (e.g., 10 seconds each)
    num_chunks = int(np.ceil(len(audio_data) / chunk_size_samples))
    
    for chunk_idx in tqdm(range(num_chunks), desc="Processing chunks"):
        # Extract the current chunk
        start_idx = chunk_idx * chunk_size_samples
        end_idx = min(start_idx + chunk_size_samples, len(audio_data))
        current_chunk = audio_data[start_idx:end_idx]
        
        # Reset VAD state for each new chunk
        vad.reset_states()
        
        # Process this chunk with VAD in smaller processing chunks
        chunk_results = []
        current_time_offset = start_idx / sample_rate
        
        # Process the current chunk in smaller processing chunks
        for i in range(0, len(current_chunk), processing_chunk_size):
            proc_chunk = current_chunk[i:i+processing_chunk_size]
            
            # Pad with zeros if needed
            if len(proc_chunk) < processing_chunk_size:
                proc_chunk = np.pad(proc_chunk, (0, processing_chunk_size - len(proc_chunk)), 'constant')
            
            # Process the chunk with VAD iterator
            result = vad(proc_chunk, return_seconds=True)
            
            if result:
                # Adjust the timestamps based on the current position
                if 'start' in result:
                    result['start'] += current_time_offset + (i / sample_rate)
                if 'end' in result:
                    result['end'] += current_time_offset + (i / sample_rate)
                
                chunk_results.append(result)
        
        # Force an end if we have a start but no end
        has_start = any('start' in r for r in chunk_results)
        has_end = any('end' in r for r in chunk_results)
        
        if has_start and not has_end:
            chunk_results.append({
                'end': current_time_offset + (len(current_chunk) / sample_rate)
            })
        
        # Add chunk results to overall results
        results.extend(chunk_results)
    
    # Post-process results to merge segments that span across chunks
    merged_results = []
    current_segment = None
    
    for result in results:
        if 'start' in result:
            if current_segment is None:
                current_segment = {'start': result['start']}
            else:
                # If we already have a start but no end, keep the earlier start
                pass
        elif 'end' in result:
            if current_segment is not None:
                # Ensure end time is after start time
                if result['end'] > current_segment['start']:
                    current_segment['end'] = result['end']
                    merged_results.append(current_segment)
                else:
                    # Skip invalid segments where end is before start
                    print(f"Warning: Skipping invalid segment with end ({result['end']}) before start ({current_segment['start']})")
                current_segment = None
    
    # Handle any remaining segment without an end
    if current_segment is not None:
        current_segment['end'] = len(audio_data) / sample_rate
        merged_results.append(current_segment)
    
    return merged_results

def analyze_with_fixed_vad_continuous(audio_data, sample_rate=16000, 
                                     threshold=0.5, min_silence_duration_ms=500, speech_pad_ms=100):
    """
    Analyze audio data with FixedVADIterator in a continuous manner
    
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
    processing_chunk_size = 512  # Exact size required by Silero VAD
    results = []
    last_chunk_index = (len(audio_data) - 1) // processing_chunk_size
    
    print(f"\n===== Processing audio with FixedVADIterator continuously =====")
    print(f"Parameters: threshold={threshold}, min_silence_duration_ms={min_silence_duration_ms}, speech_pad_ms={speech_pad_ms}")
    print(f"Audio length: {len(audio_data)/sample_rate:.2f} seconds")
    print(f"VAD processing chunk size: {processing_chunk_size} samples ({processing_chunk_size/sample_rate:.2f} seconds)")
    
    # Process the entire audio continuously
    for i in tqdm(range(0, len(audio_data), processing_chunk_size), desc="Processing audio"):
        chunk = audio_data[i:i+processing_chunk_size]
        is_last_chunk = (i // processing_chunk_size) == last_chunk_index
        
        # Pad with zeros if needed
        if len(chunk) < processing_chunk_size:
            chunk = np.pad(chunk, (0, processing_chunk_size - len(chunk)), 'constant')
        
        # Process the chunk with VAD iterator
        result = vad(chunk, return_seconds=True)
        
        if result:
            results.append(result)
        
        # If this is the last chunk and VAD is still triggered, force an end
        if is_last_chunk and vad.triggered:
            results.append({'end': len(audio_data) / sample_rate})
            print(f"Forced end at {len(audio_data) / sample_rate:.2f}s because audio ended while speech was active")
    
    return results

def compare_vad_methods(batch_results, chunked_results, continuous_results, streaming_results=None):
    """
    Compare different VAD processing methods
    
    Args:
        batch_results: Results from batch VAD processing
        chunked_results: Results from chunked VAD processing (仅用于计算重叠率，不在图中显示)
        continuous_results: Results from continuous VAD processing
        streaming_results: Results from streaming VAD processing (optional)
    """
    print("\n===== VAD 处理方法比较 =====")
    
    # Convert continuous results to segments
    continuous_segments = []
    start_time = None
    
    for result in continuous_results:
        if 'start' in result:
            start_time = result['start']
        elif 'end' in result and start_time is not None:
            continuous_segments.append({
                'start': start_time,
                'end': result['end'],
                'duration': result['end'] - start_time
            })
            start_time = None
    
    # Print comparison
    print(f"Batch VAD 检测到的语音段数量: {len(batch_results)}")
    print(f"Continuous VAD 检测到的语音段数量: {len(continuous_segments)}")
    if streaming_results:
        print(f"Streaming VAD 检测到的语音段数量: {len(streaming_results)}")
    
    # Print batch segments
    print("\nBatch VAD 语音段:")
    print(f"{'#':<5} {'Start (s)':<15} {'End (s)':<15} {'Duration (s)':<15}")
    print("-" * 50)
    
    for i, segment in enumerate(batch_results):
        duration = segment['end'] - segment['start']
        print(f"{i+1:<5} {segment['start']:<15.2f} {segment['end']:<15.2f} {duration:<15.2f}")
    
    # Print continuous segments
    print("\nContinuous VAD 语音段:")
    print(f"{'#':<5} {'Start (s)':<15} {'End (s)':<15} {'Duration (s)':<15}")
    print("-" * 50)
    
    for i, segment in enumerate(continuous_segments):
        print(f"{i+1:<5} {segment['start']:<15.2f} {segment['end']:<15.2f} {segment['duration']:<15.2f}")
    
    # Print streaming segments if available
    if streaming_results:
        print("\nStreaming VAD 语音段:")
        print(f"{'#':<5} {'Start (s)':<15} {'End (s)':<15} {'Duration (s)':<15}")
        print("-" * 50)
        
        for i, segment in enumerate(streaming_results):
            duration = segment['end'] - segment['start']
            print(f"{i+1:<5} {segment['start']:<15.2f} {segment['end']:<15.2f} {duration:<15.2f}")
    
    # Calculate overlap between batch and continuous
    batch_continuous_overlap = calculate_overlap(batch_results, continuous_segments)
    
    print(f"\nBatch vs Continuous 重叠比例: {batch_continuous_overlap:.2%}")
    
    # Calculate overlap with streaming results if available
    if streaming_results:
        batch_streaming_overlap = calculate_overlap(batch_results, streaming_results)
        continuous_streaming_overlap = calculate_overlap(continuous_segments, streaming_results)
        
        print(f"Batch vs Streaming 重叠比例: {batch_streaming_overlap:.2%}")
        print(f"Continuous vs Streaming 重叠比例: {continuous_streaming_overlap:.2%}")
    
    # Compare total duration
    batch_duration = sum(seg['end'] - seg['start'] for seg in batch_results)
    continuous_duration = sum(seg['duration'] for seg in continuous_segments)
    
    print(f"\nBatch VAD 总语音时长: {batch_duration:.2f} 秒")
    print(f"Continuous VAD 总语音时长: {continuous_duration:.2f} 秒")
    
    if streaming_results:
        streaming_duration = sum(seg['end'] - seg['start'] for seg in streaming_results)
        print(f"Streaming VAD 总语音时长: {streaming_duration:.2f} 秒")
    
    # Plot the segments for visual comparison
    plot_segments_comparison(batch_results, chunked_results, continuous_segments, streaming_results)
    
    return continuous_segments
    
    for i, segment in enumerate(chunked_results):
        duration = segment['end'] - segment['start']
        print(f"{i+1:<5} {segment['start']:<15.2f} {segment['end']:<15.2f} {duration:<15.2f}")
    
    # Print continuous segments
    print("\nContinuous VAD 语音段:")
    print(f"{'#':<5} {'Start (s)':<15} {'End (s)':<15} {'Duration (s)':<15}")
    print("-" * 50)
    
    for i, segment in enumerate(continuous_segments):
        print(f"{i+1:<5} {segment['start']:<15.2f} {segment['end']:<15.2f} {segment['duration']:<15.2f}")
    
    # Print streaming segments if available
    if streaming_results:
        print("\nStreaming VAD 语音段:")
        print(f"{'#':<5} {'Start (s)':<15} {'End (s)':<15} {'Duration (s)':<15}")
        print("-" * 50)
        
        for i, segment in enumerate(streaming_results):
            duration = segment['end'] - segment['start']
            print(f"{i+1:<5} {segment['start']:<15.2f} {segment['end']:<15.2f} {duration:<15.2f}")
    
    # Calculate overlap between batch and chunked
    batch_chunked_overlap = calculate_overlap(batch_results, chunked_results)
    
    # Calculate overlap between batch and continuous
    batch_continuous_overlap = calculate_overlap(batch_results, continuous_segments)
    
    # Calculate overlap between chunked and continuous
    chunked_continuous_overlap = calculate_overlap(chunked_results, continuous_segments)
    
    print(f"\nBatch vs Chunked 重叠比例: {batch_chunked_overlap:.2%}")
    print(f"Batch vs Continuous 重叠比例: {batch_continuous_overlap:.2%}")
    print(f"Chunked vs Continuous 重叠比例: {chunked_continuous_overlap:.2%}")
    
    # Calculate overlap with streaming results if available
    if streaming_results:
        batch_streaming_overlap = calculate_overlap(batch_results, streaming_results)
        chunked_streaming_overlap = calculate_overlap(chunked_results, streaming_results)
        continuous_streaming_overlap = calculate_overlap(continuous_segments, streaming_results)
        
        print(f"Batch vs Streaming 重叠比例: {batch_streaming_overlap:.2%}")
        print(f"Chunked vs Streaming 重叠比例: {chunked_streaming_overlap:.2%}")
        print(f"Continuous vs Streaming 重叠比例: {continuous_streaming_overlap:.2%}")
    
    # Compare total duration
    batch_duration = sum(seg['end'] - seg['start'] for seg in batch_results)
    chunked_duration = sum(seg['end'] - seg['start'] for seg in chunked_results)
    continuous_duration = sum(seg['duration'] for seg in continuous_segments)
    
    print(f"\nBatch VAD 总语音时长: {batch_duration:.2f} 秒")
    print(f"Chunked VAD 总语音时长: {chunked_duration:.2f} 秒")
    print(f"Continuous VAD 总语音时长: {continuous_duration:.2f} 秒")
    
    if streaming_results:
        streaming_duration = sum(seg['end'] - seg['start'] for seg in streaming_results)
        print(f"Streaming VAD 总语音时长: {streaming_duration:.2f} 秒")
    
    # Plot the segments for visual comparison
    plot_segments_comparison(batch_results, chunked_results, continuous_segments, streaming_results)
    
    return continuous_segments

def calculate_overlap(segments1, segments2):
    """
    Calculate overlap ratio between two sets of segments
    """
    overlap_count = 0
    
    for seg1 in segments1:
        seg1_start = seg1['start'] if 'start' in seg1 else seg1['start']
        seg1_end = seg1['end'] if 'end' in seg1 else seg1['end']
        
        for seg2 in segments2:
            seg2_start = seg2['start'] if 'start' in seg2 else seg2['start']
            seg2_end = seg2['end'] if 'end' in seg2 else seg2['end']
            
            # Check if there is overlap
            if (seg2_start <= seg1_end and seg2_end >= seg1_start):
                overlap_count += 1
                break
    
    overlap_ratio = overlap_count / len(segments1) if segments1 else 0
    return overlap_ratio

def plot_segments_comparison(batch_segments, chunked_segments, continuous_segments, streaming_segments=None):
    """
    Plot segments from different VAD methods for visual comparison
    
    Args:
        batch_segments: Segments from batch VAD processing
        chunked_segments: Segments from chunked VAD processing (不在图中显示)
        continuous_segments: Segments from continuous VAD processing
        streaming_segments: Segments from streaming VAD processing (optional)
    """
    plt.figure(figsize=(20, 10 if streaming_segments else 8))
    
    # Plot audio waveform as background
    try:
        audio_data, sample_rate = sf.read("chinese_180s.wav")
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]  # Convert to mono
        
        # Normalize and scale the waveform
        audio_data = audio_data / np.max(np.abs(audio_data)) * 0.3
        
        # Calculate audio duration
        audio_duration = len(audio_data) / sample_rate
        
        # Plot waveform with transparency
        time_axis = np.arange(len(audio_data)) / sample_rate
        plt.plot(time_axis, audio_data + (2 if streaming_segments else 1.5), color='gray', alpha=0.3, linewidth=0.5)
        
        print(f"音频时长: {audio_duration:.2f} 秒")
    except Exception as e:
        print(f"无法绘制波形: {e}")
        audio_duration = 180.0  # 如果无法读取音频文件，默认为180秒
    
    # 计算每种方法的y位置 (移除chunked)
    y_positions = {}
    if streaming_segments:
        y_positions = {'batch': 3, 'continuous': 2, 'streaming': 1}
    else:
        y_positions = {'batch': 2, 'continuous': 1}
    
    # 为每种方法创建垂直偏移字典，用于避免重叠
    vertical_offsets = {'batch': {}, 'continuous': {}, 'streaming': {}}
    
    # 找出所有段的最大结束时间，用于设置x轴范围
    max_end_time = 0
    
    # 创建更小的时间区间以更好地处理重叠
    bin_size = 3  # 3秒的区间
    
    # 计算批处理段的垂直偏移
    for segment in batch_segments:
        start_bin = int(segment['start'] / bin_size)
        if start_bin not in vertical_offsets['batch']:
            vertical_offsets['batch'][start_bin] = 0
        else:
            vertical_offsets['batch'][start_bin] += 0.15  # 增加垂直偏移
    
    # 绘制批处理段
    for i, segment in enumerate(batch_segments):
        start_bin = int(segment['start'] / bin_size)
        offset = vertical_offsets['batch'][start_bin]
        y_pos = y_positions['batch'] + offset
        
        plt.plot([segment['start'], segment['end']], [y_pos, y_pos], 'b-', linewidth=3)
        
        # 只为较长的段添加时长标签，避免拥挤
        duration = segment['end'] - segment['start']
        if duration > 1.0:  # 只为超过1秒的段添加标签
            plt.text((segment['start'] + segment['end']) / 2, y_pos + 0.1, 
                    f"{duration:.1f}s", 
                    ha='center', fontsize=7)
        
        max_end_time = max(max_end_time, segment['end'])
        
        # 减少该区间的偏移量，为下一个段做准备
        vertical_offsets['batch'][start_bin] -= 0.15
    
    # 计算连续处理段的垂直偏移
    for segment in continuous_segments:
        if 'start' in segment and 'end' in segment:
            start_bin = int(segment['start'] / bin_size)
            if start_bin not in vertical_offsets['continuous']:
                vertical_offsets['continuous'][start_bin] = 0
            else:
                vertical_offsets['continuous'][start_bin] += 0.15
    
    # 绘制连续处理段
    for i, segment in enumerate(continuous_segments):
        if 'start' in segment and 'end' in segment:
            start_bin = int(segment['start'] / bin_size)
            offset = vertical_offsets['continuous'][start_bin]
            y_pos = y_positions['continuous'] + offset
            
            plt.plot([segment['start'], segment['end']], [y_pos, y_pos], 'g-', linewidth=3)
            
            # 只为较长的段添加时长标签
            duration = segment['end'] - segment['start']
            if duration > 1.0:
                plt.text((segment['start'] + segment['end']) / 2, y_pos + 0.1, 
                        f"{duration:.1f}s", 
                        ha='center', fontsize=7)
            
            max_end_time = max(max_end_time, segment['end'])
            
            # 减少该区间的偏移量
            vertical_offsets['continuous'][start_bin] -= 0.15
    
    # 如果有流式处理段，则绘制
    if streaming_segments:
        # 检查流式段的格式
        print(f"流式段数量: {len(streaming_segments)}")
        for i, segment in enumerate(streaming_segments[:5]):  # 打印前5个段的信息，用于调试
            print(f"流式段 {i+1}: {segment}")
        
        # 计算流式处理段的垂直偏移
        for segment in streaming_segments:
            if 'start' in segment and 'end' in segment:
                start_bin = int(segment['start'] / bin_size)
                if start_bin not in vertical_offsets['streaming']:
                    vertical_offsets['streaming'][start_bin] = 0
                else:
                    vertical_offsets['streaming'][start_bin] += 0.15
        
        for i, segment in enumerate(streaming_segments):
            # 确保流式段有正确的开始和结束时间
            if 'start' in segment and 'end' in segment:
                # 确保时间戳是有效的数值
                start = float(segment['start'])
                end = float(segment['end'])
                
                start_bin = int(start / bin_size)
                offset = vertical_offsets['streaming'][start_bin]
                y_pos = y_positions['streaming'] + offset
                
                # 绘制段
                plt.plot([start, end], [y_pos, y_pos], 'y-', linewidth=3)
                
                # 添加时长标签
                duration = end - start
                plt.text((start + end) / 2, y_pos + 0.1, 
                        f"{duration:.1f}s", 
                        ha='center', fontsize=7)
                
                # 更新最大结束时间
                max_end_time = max(max_end_time, end)
                
                # 减少该区间的偏移量
                vertical_offsets['streaming'][start_bin] -= 0.15
    
    # 确保max_end_time不为零，并添加一些填充
    max_end_time = max(max_end_time, audio_duration)
    max_end_time = max_end_time * 1.05  # 添加5%的填充
    
    # 每10秒添加一条垂直网格线
    grid_interval = 10  # 秒
    for i in range(0, int(max_end_time) + grid_interval, grid_interval):
        plt.axvline(x=i, color='gray', linestyle='--', alpha=0.3)
        plt.text(i, 0.5, f"{i}s", ha='center', fontsize=8)
    
    # 根据可用方法设置y刻度 (移除chunked)
    if streaming_segments:
        plt.yticks([1, 2, 3], ['Streaming', 'Continuous', 'Batch'])
    else:
        plt.yticks([1, 2], ['Continuous', 'Batch'])
    
    # 添加图例
    plt.plot([], [], 'b-', linewidth=3, label='Batch VAD')
    plt.plot([], [], 'g-', linewidth=3, label='Continuous VAD')
    if streaming_segments:
        plt.plot([], [], 'y-', linewidth=3, label='Streaming VAD')
    plt.legend(loc='upper right')
    
    plt.xlabel('Time (seconds)')
    plt.title('VAD Segment Comparison')
    plt.grid(True, axis='x', alpha=0.3)
    plt.xlim(0, max_end_time)  # 根据实际数据设置x轴范围
    plt.ylim(0.5, 4.0 if streaming_segments else 3.0)  # 设置y轴范围，增加空间以容纳垂直偏移
    plt.tight_layout()
    plt.savefig('vad_comparison.png', dpi=150)
    print(f"VAD比较图已保存到vad_comparison.png")

def test_different_chunk_sizes(audio_data, sample_rate=16000, batch_results=None):
    """
    Test different chunk sizes for VAD processing
    """
    print("\n===== 测试不同的分块大小 =====")
    
    # Define chunk sizes to test (in seconds)
    chunk_sizes = [3]
    
    best_chunk_size = None
    best_overlap = 0
    best_results = None
    
    # Fixed VAD parameters - using the specified values
    threshold = 0.2  # 指定参数
    min_silence_duration_ms = 300  # 指定参数
    speech_pad_ms = 100  # 指定参数
    
    for chunk_size in chunk_sizes:
        print(f"\n分块大小: {chunk_size} 秒")
        chunked_results = analyze_with_fixed_vad_chunks(
            audio_data, 
            sample_rate=sample_rate,
            chunk_duration=chunk_size,
            threshold=threshold,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms
        )
        
        # Calculate overlap with batch results
        if batch_results:
            overlap_ratio = calculate_overlap(batch_results, chunked_results)
            print(f"与批处理的重叠比例: {overlap_ratio:.2%}")
            
            # Update best chunk size if this one has better overlap
            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_chunk_size = chunk_size
                best_results = chunked_results
    
    # Print best chunk size
    if best_chunk_size:
        print(f"\n===== 最佳分块大小 =====")
        print(f"分块大小: {best_chunk_size} 秒")
        print(f"重叠比例: {best_overlap:.2%}")
    
    return best_chunk_size, best_results

def main():
    """Main function"""
    # Audio file path
    audio_file = "chinese_180s.wav"
    
    # Load audio file
    audio_data, sample_rate = load_audio_file(audio_file)
    
    # Fixed VAD parameters - using the specified values
    threshold = 0.3  # 指定参数
    min_silence_duration_ms = 300  # 指定参数
    speech_pad_ms = 100  # 指定参数
    
    print(f"\n===== 使用固定参数 =====")
    print(f"threshold={threshold}, min_silence_duration_ms={min_silence_duration_ms}, speech_pad_ms={speech_pad_ms}")
    
    # Run batch VAD with specified parameters
    batch_results = test_batch_vad(audio_file)
    
    # Test different chunk sizes with specified parameters
    best_chunk_size, best_chunked_results = test_different_chunk_sizes(audio_data, sample_rate, batch_results)
    
    # Process with continuous VAD using specified parameters
    continuous_results = analyze_with_fixed_vad_continuous(
        audio_data, 
        sample_rate=sample_rate,
        threshold=threshold,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms
    )
    
    print("\n===== test_streaming_vad =====")
    streaming_results = test_streaming_vad(audio_file, 0.128)
    
    # Compare VAD methods and get processed continuous segments
    continuous_segments = compare_vad_methods(batch_results, best_chunked_results, continuous_results, streaming_results)
    
    # TODO 把这边的节点 对比 batch_results/continuous_results/streaming_results
    print("\n" + "="*80)
    print("📊 详细VAD方法对比分析")
    print("="*80)
    
    detailed_vad_comparison(batch_results, continuous_segments, streaming_results, len(audio_data)/sample_rate)


def detailed_vad_comparison(batch_results, continuous_results, streaming_results, audio_duration):
    """
    详细对比三种VAD方法的关键指标
    
    Args:
        batch_results: 批处理VAD结果
        continuous_results: 连续VAD结果  
        streaming_results: 流式VAD结果
        audio_duration: 音频总时长
    """
    
    # 1. 基础统计对比
    print("\n🔢 基础统计对比")
    print("-" * 60)
    
    methods = {
        'Batch': batch_results,
        'Continuous': continuous_results, 
        'Streaming': streaming_results
    }
    
    stats = {}
    for method_name, results in methods.items():
        if results:
            # 计算总语音时长
            total_speech_duration = sum(seg['end'] - seg['start'] for seg in results)
            
            # 计算平均语音段长度
            avg_segment_duration = total_speech_duration / len(results)
            
            # 计算最长和最短语音段
            durations = [seg['end'] - seg['start'] for seg in results]
            max_duration = max(durations)
            min_duration = min(durations)
            
            # 计算语音占比
            speech_ratio = total_speech_duration / audio_duration * 100
            
            stats[method_name] = {
                'segment_count': len(results),
                'total_speech_duration': total_speech_duration,
                'avg_segment_duration': avg_segment_duration,
                'max_segment_duration': max_duration,
                'min_segment_duration': min_duration,
                'speech_ratio': speech_ratio
            }
            
            print(f"{method_name:12} | 段数: {len(results):2d} | 总时长: {total_speech_duration:6.1f}s | "
                  f"平均: {avg_segment_duration:4.1f}s | 最长: {max_duration:5.1f}s | "
                  f"最短: {min_duration:4.1f}s | 占比: {speech_ratio:4.1f}%")
    
    # 2. 时间精度对比
    print(f"\n⏱️  时间精度对比")
    print("-" * 60)
    
    if streaming_results and batch_results:
        # 计算时间戳差异
        timestamp_diffs = []
        for i, (batch_seg, stream_seg) in enumerate(zip(batch_results, streaming_results)):
            start_diff = abs(batch_seg['start'] - stream_seg['start'])
            end_diff = abs(batch_seg['end'] - stream_seg['end'])
            timestamp_diffs.extend([start_diff, end_diff])
            
            if i < 5:  # 只显示前5个段的详细对比
                print(f"段{i+1:2d} | Batch: {batch_seg['start']:6.2f}-{batch_seg['end']:6.2f}s | "
                      f"Stream: {stream_seg['start']:6.2f}-{stream_seg['end']:6.2f}s | "
                      f"差异: {start_diff:4.2f}s/{end_diff:4.2f}s")
        
        avg_timestamp_diff = sum(timestamp_diffs) / len(timestamp_diffs)
        max_timestamp_diff = max(timestamp_diffs)
        print(f"\n时间戳差异统计: 平均 {avg_timestamp_diff:.3f}s, 最大 {max_timestamp_diff:.3f}s")
    
    # 3. 语音段分布分析
    print(f"\n📈 语音段时长分布分析")
    print("-" * 60)
    
    duration_ranges = [(0, 1), (1, 3), (3, 5), (5, 10), (10, float('inf'))]
    range_labels = ['<1s', '1-3s', '3-5s', '5-10s', '>10s']
    
    for method_name, results in methods.items():
        if results:
            durations = [seg['end'] - seg['start'] for seg in results]
            distribution = []
            
            for min_dur, max_dur in duration_ranges:
                count = sum(1 for d in durations if min_dur <= d < max_dur)
                percentage = count / len(durations) * 100
                distribution.append(f"{count:2d}({percentage:4.1f}%)")
            
            print(f"{method_name:12} | " + " | ".join(f"{label:>5}: {dist}" 
                  for label, dist in zip(range_labels, distribution)))
    
    # 4. 一致性分析
    print(f"\n🎯 方法一致性分析")
    print("-" * 60)
    
    if len(methods) >= 2:
        method_pairs = [
            ('Batch', 'Continuous'),
            ('Batch', 'Streaming'), 
            ('Continuous', 'Streaming')
        ]
        
        for method1, method2 in method_pairs:
            if method1 in stats and method2 in stats:
                # 段数差异
                count_diff = abs(stats[method1]['segment_count'] - stats[method2]['segment_count'])
                count_diff_pct = count_diff / stats[method1]['segment_count'] * 100
                
                # 总时长差异
                duration_diff = abs(stats[method1]['total_speech_duration'] - stats[method2]['total_speech_duration'])
                duration_diff_pct = duration_diff / stats[method1]['total_speech_duration'] * 100
                
                # 重叠率计算
                overlap_rate = calculate_overlap(methods[method1], methods[method2]) * 100
                
                print(f"{method1:12} vs {method2:12} | 段数差异: {count_diff:2d}({count_diff_pct:4.1f}%) | "
                      f"时长差异: {duration_diff:4.1f}s({duration_diff_pct:4.1f}%) | 重叠率: {overlap_rate:5.1f}%")
    
    # 5. 性能评估
    print(f"\n⚡ 性能特征评估")
    print("-" * 60)
    
    performance_analysis = {
        'Batch': {
            'accuracy': '★★★★★',
            'latency': '★☆☆☆☆', 
            'memory': '★★★☆☆',
            'realtime': '❌',
            'use_case': '离线高精度处理'
        },
        'Continuous': {
            'accuracy': '★★★★☆',
            'latency': '★★★★☆',
            'memory': '★★★★☆', 
            'realtime': '✅',
            'use_case': '实时处理平衡方案'
        },
        'Streaming': {
            'accuracy': '★★★★★',
            'latency': '★★★★★',
            'memory': '★★★★★',
            'realtime': '✅', 
            'use_case': '实时流式处理'
        }
    }
    
    print(f"{'方法':12} | {'精度':8} | {'延迟':8} | {'内存':8} | {'实时':6} | 适用场景")
    print("-" * 80)
    for method, perf in performance_analysis.items():
        if method.lower().replace(' ', '') in [m.lower().replace(' ', '') for m in methods.keys()]:
            print(f"{method:12} | {perf['accuracy']:8} | {perf['latency']:8} | "
                  f"{perf['memory']:8} | {perf['realtime']:6} | {perf['use_case']}")
    
    # 6. 关键发现总结
    print(f"\n🎯 关键发现总结")
    print("-" * 60)
    
    findings = []
    
    # 检查是否所有方法检测到相同数量的语音段
    segment_counts = [stats[method]['segment_count'] for method in stats.keys()]
    if len(set(segment_counts)) == 1:
        findings.append(f"✅ 所有方法检测到相同数量的语音段 ({segment_counts[0]}段)")
    else:
        findings.append(f"⚠️  不同方法检测到的语音段数量不一致: {dict(zip(stats.keys(), segment_counts))}")
    
    # 检查语音时长一致性
    speech_durations = [stats[method]['total_speech_duration'] for method in stats.keys()]
    duration_variance = max(speech_durations) - min(speech_durations)
    if duration_variance < 1.0:  # 差异小于1秒
        findings.append(f"✅ 语音时长检测高度一致 (差异 {duration_variance:.1f}s)")
    else:
        findings.append(f"⚠️  语音时长检测存在差异 (差异 {duration_variance:.1f}s)")
    
    # 检查流式处理修复效果
    if 'Streaming' in stats and 'Batch' in stats:
        if stats['Streaming']['segment_count'] == stats['Batch']['segment_count']:
            findings.append("✅ 流式VAD最后一段缺失问题已修复")
        else:
            findings.append("❌ 流式VAD仍存在段数不一致问题")
    
    # 推荐使用场景
    findings.append("💡 推荐使用场景:")
    findings.append("   • 离线高精度处理 → Batch VAD")
    findings.append("   • 实时字幕生成 → Streaming VAD") 
    findings.append("   • 平衡方案 → Continuous VAD")
    
    for finding in findings:
        print(f"   {finding}")
    
    print("\n" + "="*80)

    
def analyze_with_fixed_vad_streaming(audio_stream, sample_rate=16000, 
                                    threshold=0.3, min_silence_duration_ms=300, speech_pad_ms=100,
                                    no_audio_input_threshold=0.5):
    """
    分析流式音频数据，使用面向对象的VACProcessor
    
    Args:
        audio_stream: 音频流迭代器
        sample_rate: 采样率
        threshold: 语音阈值 (0.0-1.0)
        min_silence_duration_ms: 最小静音持续时间(ms)
        speech_pad_ms: 语音段填充时间(ms)
        no_audio_input_threshold: 无音频输入阈值(秒)，超过此时间无新数据则强制结束
    """
    # 导入VAC处理器
    from subtitle_genius.stream.vac_processor import VACProcessor
    
    print(f"\n===== 使用面向对象的VACProcessor =====")
    
    # 创建VAC处理器
    vac_processor = VACProcessor(
        threshold=threshold,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        sample_rate=sample_rate,
        processing_chunk_size=512,
        no_audio_input_threshold=no_audio_input_threshold
    )
    
    # 处理流式音频，返回原始结果（不转换为segments）
    results = vac_processor.process_streaming_audio(audio_stream, return_segments=False, end_stream_flag = True)
    
    return results


def test_streaming_vad(audio_file, chunk_duration=0.128, sample_rate=16000):
    """
    测试流式VAD处理
    
    Args:
        audio_file: 音频文件路径
        chunk_duration: 每个块的持续时间（秒）
        sample_rate: 采样率
        
    Returns:
        list: 包含语音段信息的列表，每个段都有start、end和duration字段
    """
    import time
    
    # 加载音频文件
    audio_data, file_sample_rate = sf.read(audio_file)
    
    # 确保音频是单声道
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]
    
    # 确保音频是float32
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    # 如果需要，重采样
    if file_sample_rate != sample_rate:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=file_sample_rate, target_sr=sample_rate)
    
    # 计算每个块的样本数
    # 将chunk_size设置为512的整数倍，例如chunk_size = 1536（3*512）或chunk_size = 2048（4*512）
    chunk_size = int(chunk_duration * sample_rate)
    
    # 强制检查：确保chunk_size是512的整数倍
    VAD_CHUNK_SIZE = 512  # Silero VAD要求的块大小
    if chunk_size % VAD_CHUNK_SIZE != 0:
        raise ValueError(
            f"chunk_size必须是{VAD_CHUNK_SIZE}的整数倍！当前值为{chunk_size}。"
            f"请调整chunk_duration为{VAD_CHUNK_SIZE/sample_rate}的整数倍，"
            f"例如{VAD_CHUNK_SIZE/sample_rate*3}秒或{VAD_CHUNK_SIZE/sample_rate*4}秒。"
        )
    
    
    # 创建一个生成器，模拟流式输入
    def audio_stream_generator():
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:min(i+chunk_size, len(audio_data))]
            yield chunk
            # 模拟处理延迟
            # time.sleep(0.1)
    
    # 创建音频流
    audio_stream = audio_stream_generator()
    
    # 使用流式VAD处理
    streaming_results = analyze_with_fixed_vad_streaming(
        audio_stream,
        sample_rate=sample_rate,
        threshold=0.3,
        min_silence_duration_ms=300,
        speech_pad_ms=100,
        no_audio_input_threshold=0.5
    )
    
    # 转换结果为语音段
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
    
    return streaming_segments


def test_speech_segment_events(audio_file="chinese_180s.wav", chunk_duration=0.128, sample_rate=16000):
    """
    测试语音段事件订阅功能
    
    Args:
        audio_file: 音频文件路径
        chunk_duration: 每个块的持续时间（秒）
        sample_rate: 采样率
    """
    print("\n" + "="*80)
    print("🎤 测试语音段事件订阅功能")
    print("="*80)
    
    # 事件收集器
    detected_segments = []
    
    def on_speech_segment_detected(speech_segment):
        """
        语音段检测事件回调函数
        
        Args:
            speech_segment: 包含start、end、duration、audio_bytes、sample_rate的字典
        """
        print(f"🎯 检测到语音段:")
        print(f"   时间范围: {speech_segment['start']:.2f}s - {speech_segment['end']:.2f}s")
        print(f"   持续时间: {speech_segment['duration']:.2f}s")
        print(f"   音频数据: {len(speech_segment['audio_bytes'])} bytes")
        print(f"   采样率: {speech_segment['sample_rate']} Hz")
        
        # 计算音频数据的一些统计信息
        audio_array = np.frombuffer(speech_segment['audio_bytes'], dtype=np.float32)
        print(f"   音频样本数: {len(audio_array)}")
        print(f"   音频RMS: {np.sqrt(np.mean(audio_array**2)):.4f}")
        print(f"   音频峰值: {np.max(np.abs(audio_array)):.4f}")
        
        # 验证音频数据长度是否与时长匹配
        expected_samples = int(speech_segment['duration'] * speech_segment['sample_rate'])
        actual_samples = len(audio_array)
        sample_diff = abs(expected_samples - actual_samples)
        print(f"   样本数验证: 期望{expected_samples}, 实际{actual_samples}, 差异{sample_diff}")
        
        # 保存到收集器
        detected_segments.append({
            'start': speech_segment['start'],
            'end': speech_segment['end'],
            'duration': speech_segment['duration'],
            'audio_bytes_length': len(speech_segment['audio_bytes']),
            'audio_samples': len(audio_array),
            'audio_rms': np.sqrt(np.mean(audio_array**2)),
            'audio_peak': np.max(np.abs(audio_array)),
            'sample_rate': speech_segment['sample_rate']
        })
        
        print(f"   ✅ 事件处理完成 (总计已检测 {len(detected_segments)} 个语音段)")
        print("-" * 60)
    
    # 加载音频文件
    print(f"📁 加载音频文件: {audio_file}")
    audio_data, file_sample_rate = sf.read(audio_file)
    
    # 确保音频是单声道
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]
        print("🔄 转换为单声道")
    
    # 确保音频是float32
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
        print("🔄 转换为float32格式")
    
    # 如果需要，重采样
    if file_sample_rate != sample_rate:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=file_sample_rate, target_sr=sample_rate)
        print(f"🔄 重采样: {file_sample_rate}Hz → {sample_rate}Hz")
    
    print(f"📊 音频信息:")
    print(f"   时长: {len(audio_data)/sample_rate:.2f}s")
    print(f"   样本数: {len(audio_data)}")
    print(f"   采样率: {sample_rate}Hz")
    
    # 计算每个块的样本数
    chunk_size = int(chunk_duration * sample_rate)
    
    # 确保chunk_size是512的整数倍
    VAD_CHUNK_SIZE = 512
    if chunk_size % VAD_CHUNK_SIZE != 0:
        # 自动调整到最近的512的整数倍
        chunk_size = ((chunk_size // VAD_CHUNK_SIZE) + 1) * VAD_CHUNK_SIZE
        chunk_duration = chunk_size / sample_rate
        print(f"⚠️  自动调整chunk_size到512的整数倍: {chunk_size} (时长: {chunk_duration:.3f}s)")
    
    print(f"🔧 处理参数:")
    print(f"   块大小: {chunk_size} 样本 ({chunk_duration:.3f}s)")
    print(f"   VAD阈值: 0.3")
    print(f"   最小静音时长: 300ms")
    print(f"   语音填充: 100ms")
    
    # 创建一个生成器，模拟流式输入
    def audio_stream_generator():
        print(f"🚀 开始流式处理...")
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:min(i+chunk_size, len(audio_data))]
            current_time = i / sample_rate
            print(f"📦 处理音频块: {current_time:.2f}s - {(i+len(chunk))/sample_rate:.2f}s ({len(chunk)} 样本)")
            yield chunk
    
    # 导入VAC处理器
    from subtitle_genius.stream.vac_processor import VACProcessor
    
    # 创建带事件回调的VAC处理器
    print(f"🏗️  创建VACProcessor...")
    vac_processor = VACProcessor(
        threshold=0.3,
        min_silence_duration_ms=300,
        speech_pad_ms=100,
        sample_rate=sample_rate,
        processing_chunk_size=512,
        no_audio_input_threshold=0.5,
        on_speech_segment=on_speech_segment_detected  # 🎯 关键：设置事件回调
    )
    
    # 创建音频流
    audio_stream = audio_stream_generator()
    
    # 开始处理
    print(f"▶️  开始VAD处理...")
    start_time = time.time()
    
    # 处理流式音频（这会触发事件回调）
    vad_results = vac_processor.process_streaming_audio(audio_stream, return_segments=False, end_stream_flag=True)
    
    processing_time = time.time() - start_time
    print(f"⏱️  处理完成，耗时: {processing_time:.2f}s")
    
    # 统计结果
    print(f"\n📈 处理结果统计:")
    print(f"   VAD原始事件数: {len(vad_results)}")
    print(f"   检测到的语音段数: {len(detected_segments)}")
    
    if detected_segments:
        total_speech_duration = sum(seg['duration'] for seg in detected_segments)
        avg_duration = total_speech_duration / len(detected_segments)
        max_duration = max(seg['duration'] for seg in detected_segments)
        min_duration = min(seg['duration'] for seg in detected_segments)
        
        print(f"   总语音时长: {total_speech_duration:.2f}s")
        print(f"   平均段长: {avg_duration:.2f}s")
        print(f"   最长段: {max_duration:.2f}s")
        print(f"   最短段: {min_duration:.2f}s")
        print(f"   语音占比: {total_speech_duration/(len(audio_data)/sample_rate)*100:.1f}%")
        
        # 显示前几个检测到的语音段
        print(f"\n📋 检测到的语音段详情 (前5个):")
        print(f"{'#':<3} {'开始(s)':<8} {'结束(s)':<8} {'时长(s)':<8} {'音频(KB)':<10} {'RMS':<8} {'峰值':<8}")
        print("-" * 70)
        
        for i, seg in enumerate(detected_segments[:5]):
            audio_kb = seg['audio_bytes_length'] / 1024
            print(f"{i+1:<3} {seg['start']:<8.2f} {seg['end']:<8.2f} {seg['duration']:<8.2f} "
                  f"{audio_kb:<10.1f} {seg['audio_rms']:<8.4f} {seg['audio_peak']:<8.4f}")
    
    # 验证事件系统是否正常工作
    print(f"\n✅ 事件系统验证:")
    
    # 计算预期的语音段数（从VAD原始结果）
    expected_segments = 0
    in_speech = False
    for result in vad_results:
        if 'start' in result:
            in_speech = True
        elif 'end' in result and in_speech:
            expected_segments += 1
            in_speech = False
    
    print(f"   预期语音段数: {expected_segments}")
    print(f"   实际触发事件数: {len(detected_segments)}")
    
    if expected_segments == len(detected_segments):
        print(f"   ✅ 事件系统工作正常！所有语音段都触发了事件")
    else:
        print(f"   ❌ 事件系统异常！预期{expected_segments}个事件，实际{len(detected_segments)}个")
    
    # 模拟whisper_sagemaker调用
    print(f"\n🤖 模拟Whisper SageMaker调用:")
    for i, seg in enumerate(detected_segments[:3]):  # 只模拟前3个
        print(f"   📞 调用whisper_sagemaker.transcribe():")
        print(f"      语音段 {i+1}: {seg['start']:.2f}s-{seg['end']:.2f}s")
        print(f"      音频数据: {seg['audio_bytes_length']} bytes")
        print(f"      → 这里会调用实际的转录服务")
    
    print(f"\n🎉 事件订阅测试完成！")
    return detected_segments


if __name__ == "__main__":
    # 运行原有的主测试
    main()
    
    # # 运行新的事件订阅测试
    # print("\n" + "="*100)
    # test_speech_segment_events()

