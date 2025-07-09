"""
Silero VAD wrapper functions for batch processing
"""

import torch
import numpy as np
import soundfile as sf
from typing import List, Dict, Any, Union, Optional

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
