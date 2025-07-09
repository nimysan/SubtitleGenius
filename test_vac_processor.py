#!/usr/bin/env python
"""
Test script to analyze chinese_90s.wav using FixedVADIterator with chunk processing
This script processes a long WAV file in chunks and applies VAD to each chunk,
ensuring consistent VAD effects across the entire audio.
"""

import os
import sys
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
    
    # Print key findings
    print("\n===== 关键发现 =====")
    print("1. 分块大小的影响:")
    print(f"   - 最佳分块大小: {best_chunk_size} 秒")
    print("   - 较小的分块 (5-7秒) 可能会导致语音段被过度分割")
    print("   - 较大的分块 (15-20秒) 可能会导致处理延迟增加")
    print("   - 分块大小应根据语音特点和实时性要求进行平衡")
    
    print("\n2. 处理方法比较:")
    print("   - 批处理 (Batch): 全局最优，但不适用于实时处理")
    print("   - 分块处理 (Chunked): 平衡实时性和准确性，适合大多数场景")
    print("   - 连续处理 (Continuous): 最实时，但可能会导致语音段分割不准确")
    
    print("\n3. 最佳实践:")
    print(f"   - 对于中文解说，推荐使用 {best_chunk_size} 秒的分块大小")
    print("   - VAD参数: threshold=0.3, min_silence_duration_ms=300, speech_pad_ms=100")
    print("   - 在实际应用中，可以根据具体需求动态调整分块大小")
    
    print("\n4. 性能考虑:")
    print("   - 分块大小越小，处理延迟越低，但准确性可能降低")
    print("   - 分块大小越大，准确性越高，但处理延迟也越高")
    print("   - 在资源受限的环境中，可以考虑使用较小的分块大小")
    
    # Calculate detailed statistics
    batch_duration = sum(seg['end'] - seg['start'] for seg in batch_results)
    chunked_duration = sum(seg['end'] - seg['start'] for seg in best_chunked_results if 'end' in seg and 'start' in seg and seg['end'] > seg['start'])
    continuous_duration = sum(seg['duration'] for seg in continuous_segments)
    
    # Calculate average segment durations
    batch_avg_duration = batch_duration / len(batch_results) if batch_results else 0
    chunked_avg_duration = chunked_duration / len(best_chunked_results) if best_chunked_results else 0
    continuous_avg_duration = continuous_duration / len(continuous_segments) if continuous_segments else 0
    
    # Calculate the number of chunks
    num_chunks = int(np.ceil(len(audio_data) / (best_chunk_size * sample_rate)))
    
    print("\n===== 详细统计 =====")
    print(f"音频总长度: {len(audio_data)/sample_rate:.2f} 秒")
    print(f"语音占比: {batch_duration/(len(audio_data)/sample_rate)*100:.1f}%")
    
    print("\n语音段数量比较:")
    print(f"- Batch: {len(batch_results)} 段")
    print(f"- Chunked ({best_chunk_size}s): {len(best_chunked_results)} 段")
    print(f"- Continuous: {len(continuous_segments)} 段")
    
    print("\n平均语音段长度:")
    print(f"- Batch: {batch_avg_duration:.2f} 秒")
    print(f"- Chunked ({best_chunk_size}s): {chunked_avg_duration:.2f} 秒")
    print(f"- Continuous: {continuous_avg_duration:.2f} 秒")


    
def analyze_with_fixed_vad_streaming(audio_stream, sample_rate=16000, 
                                    threshold=0.3, min_silence_duration_ms=300, speech_pad_ms=100,
                                    no_audio_input_threshold=0.5):
    """
    分析流式音频数据，使用FixedVADIterator
    
    Args:
        audio_stream: 音频流迭代器
        sample_rate: 采样率
        threshold: 语音阈值 (0.0-1.0)
        min_silence_duration_ms: 最小静音持续时间(ms)
        speech_pad_ms: 语音段填充时间(ms)
        no_audio_input_threshold: 无音频输入阈值(秒)，超过此时间无新数据则强制结束
    """
    import time
    
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
    processing_chunk_size = 512  # Silero VAD要求的确切大小
    results = []
    last_audio_time = time.time()
    total_samples_processed = 0
    
    print(f"\n===== 流式处理音频 =====")
    print(f"参数: threshold={threshold}, min_silence_duration_ms={min_silence_duration_ms}, speech_pad_ms={speech_pad_ms}")
    print(f"无音频输入阈值: {no_audio_input_threshold} 秒")
    
    try:
        # 处理流式音频
        for audio_chunk in audio_stream:
            # 更新最后接收音频的时间
            last_audio_time = time.time()
            
            # 处理音频块
            for i in range(0, len(audio_chunk), processing_chunk_size):
                chunk = audio_chunk[i:i+processing_chunk_size]
                
                # 如果需要，用零填充
                print(f"------->len of chunk is {len(chunk)} and processing chunk size is {processing_chunk_size}")
                if len(chunk) < processing_chunk_size:
                    chunk = np.pad(chunk, (0, processing_chunk_size - len(chunk)), 'constant')
                
                # 使用VAD迭代器处理块
                result = vad(chunk, return_seconds=True)
                
                total_samples_processed += len(chunk)
                
                if result:
                    print(f"---vad result is {result}")
                    results.append(result)
            
            # 检查是否超过无音频输入阈值
            if time.time() - last_audio_time > no_audio_input_threshold:
                # 如果VAD仍处于触发状态，强制结束
                if vad.triggered:
                    # 使用当前处理的总样本数计算结束时间
                    end_time = total_samples_processed / sample_rate
                    results.append({'end': end_time})
                    print(f"由于超过无音频输入阈值 {no_audio_input_threshold}秒，强制结束于 {end_time:.2f}秒")
                break
    except Exception as e:
        print(f"流处理中断: {e}")
        # 如果VAD仍处于触发状态，强制结束
        if vad.triggered:
            end_time = total_samples_processed / sample_rate
            results.append({'end': end_time})
            print(f"由于异常中断，强制结束于 {end_time:.2f}秒")
    
    return results


def test_streaming_vad(audio_file, chunk_duration=0.128, sample_rate=16000):
    """
    测试流式VAD处理
    
    Args:
        audio_file: 音频文件路径
        chunk_duration: 每个块的持续时间（秒）
        sample_rate: 采样率
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
    chunk_size = int(chunk_duration * sample_rate) # 如果chunk_size不是512的整数倍，例如chunk_size = 3*512 + 64 = 1600 Pad会导致时间超出
    
    
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


if __name__ == "__main__":
    main()

