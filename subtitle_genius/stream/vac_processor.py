"""
VAC (Voice Activity Chunking) Processor

This module provides a processor for detecting voice activity in audio streams
and chunking the audio into segments containing speech.

It uses the Silero VAD model to detect speech segments in audio streams.
"""

import os
import sys
import logging
import numpy as np
import torch
from collections import deque
from typing import Tuple, Optional, List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the whisper_streaming directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'whisper_streaming'))
from silero_vad_iterator import FixedVADIterator

class VACProcessor:
    """
    Voice Activity Chunking Processor
    
    This class processes audio chunks and detects voice activity using Silero VAD.
    It returns segments of audio that contain speech.
    """
    
    def __init__(self, 
                 threshold: float = 0.5,
                 sampling_rate: int = 16000,
                 min_silence_duration_ms: int = 500,
                 speech_pad_ms: int = 100,
                 max_segment_duration_sec: float = 10.0):  # Added max segment duration
        """
        Initialize the VAC processor
        
        Args:
            threshold: Speech threshold. Probabilities above this value are considered speech.
            sampling_rate: Audio sampling rate (8000 or 16000 Hz)
            min_silence_duration_ms: Minimum silence duration in ms before ending a speech segment
            speech_pad_ms: Padding in ms to add to each side of a speech segment
            max_segment_duration_sec: Maximum duration of a speech segment in seconds
        """
        self.sampling_rate = sampling_rate
        self.threshold = threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.max_segment_duration_sec = max_segment_duration_sec
        
        # Load the Silero VAD model
        self.model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad'
        )
        
        # Create the VAD iterator
        self.vad = FixedVADIterator(
            model=self.model,
            threshold=threshold,
            sampling_rate=sampling_rate,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms
        )
        
        # Initialize buffers
        self.audio_buffer = np.array([], dtype=np.float32)
        self.current_time = 0.0  # Current time in seconds
        self.segments_buffer = deque()  # Buffer for completed segments
        
        # Speech segment tracking
        self.current_segment_audio = np.array([], dtype=np.float32)
        self.current_segment_start = None
        self.current_segment_last_activity = 0.0  # Track last activity time
        self.in_speech = False  # Track if we're currently in a speech segment
        
        # Chunk tracking
        self.chunk_counter = 0
        self.chunk_samples = []  # Store samples for each chunk for debugging
        
        logger.info(f"VAC Processor initialized with sampling_rate={sampling_rate}Hz, "
                   f"threshold={threshold}, min_silence_duration_ms={min_silence_duration_ms}, "
                   f"speech_pad_ms={speech_pad_ms}, max_segment_duration_sec={max_segment_duration_sec}")
    
    def add_audio_chunk(self, audio_bytes: bytes) -> None:
        """
        Add an audio chunk to the processor
        
        Args:
            audio_bytes: Raw audio bytes (16-bit PCM)
        """
        # Convert bytes to numpy array
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32767.0
        
        # Calculate chunk duration
        chunk_duration = len(audio_int16) / self.sampling_rate
        chunk_samples = len(audio_int16)
        
        # Store chunk info for debugging
        self.chunk_counter += 1
        self.chunk_samples.append(chunk_samples)
        
        # Process the audio chunk
        self._process_audio_chunk(audio_float32, chunk_duration)
    
    def _process_audio_chunk(self, audio_chunk: np.ndarray, chunk_duration: float) -> None:
        """
        Process an audio chunk and detect speech segments
        
        Args:
            audio_chunk: Audio chunk as float32 numpy array
            chunk_duration: Duration of the chunk in seconds
        """
        # Process the audio in smaller windows to get more accurate VAD results
        window_size = 1600  # 100ms at 16kHz
        for i in range(0, len(audio_chunk), window_size):
            window = audio_chunk[i:i+window_size]
            if len(window) < 512:  # Minimum size required by Silero VAD
                # Pad with zeros if needed
                window = np.pad(window, (0, 512 - len(window)), 'constant')
            
            window_duration = len(window) / self.sampling_rate
            window_time = self.current_time + (i / self.sampling_rate)
            
            # Process the window with the VAD
            result = self.vad(window, return_seconds=True)
            
            # Handle speech start
            if result and 'start' in result:
                start_time = window_time + result['start']
                
                # If we're already in a speech segment, check if we need to split due to max duration
                if self.in_speech and self.current_segment_start is not None:
                    current_duration = start_time - self.current_segment_start
                    if current_duration > self.max_segment_duration_sec:
                        # End the current segment and start a new one
                        self._finalize_segment(start_time)
                        self.current_segment_start = start_time
                        self.current_segment_audio = np.array([], dtype=np.float32)
                else:
                    # Start a new segment
                    self.in_speech = True
                    self.current_segment_start = start_time
                    self.current_segment_audio = np.array([], dtype=np.float32)
                
                # Update last activity time
                self.current_segment_last_activity = window_time + window_duration
            
            # Handle speech end
            elif result and 'end' in result:
                end_time = window_time + result['end']
                
                # If we're in a speech segment, finalize it
                if self.in_speech:
                    self._finalize_segment(end_time)
            
            # If we're in a speech segment, add the window to the current segment
            if self.in_speech and self.current_segment_start is not None:
                self.current_segment_audio = np.append(self.current_segment_audio, window)
                
                # Check for timeout (no activity for a while)
                silence_duration = window_time + window_duration - self.current_segment_last_activity
                if silence_duration > (self.min_silence_duration_ms / 1000 * 2):  # Double the min silence duration
                    self._finalize_segment(self.current_segment_last_activity)
                
                # Check for max duration
                current_duration = window_time + window_duration - self.current_segment_start
                if current_duration > self.max_segment_duration_sec:
                    self._finalize_segment(window_time + window_duration)
        
        # Update the current time
        self.current_time += chunk_duration
    
    def _finalize_segment(self, end_time: float) -> None:
        """
        Finalize the current speech segment
        
        Args:
            end_time: End time of the segment in seconds
        """
        if not self.in_speech or self.current_segment_start is None:
            return
        
        # Create the segment
        segment = (
            self.current_segment_audio,
            self.current_segment_start,
            end_time
        )
        
        # Add the segment to the buffer
        self.segments_buffer.append(segment)
        
        # Reset the current segment
        self.in_speech = False
        self.current_segment_start = None
        self.current_segment_audio = np.array([], dtype=np.float32)
        self.current_segment_last_activity = 0.0
    
    def has_pending_segments(self) -> bool:
        """
        Check if there are pending segments to process
        
        Returns:
            True if there are pending segments, False otherwise
        """
        return len(self.segments_buffer) > 0
    
    def get_next_voice_segment(self) -> Optional[Tuple[np.ndarray, float, float]]:
        """
        Get the next voice segment
        
        Returns:
            Tuple of (audio_data, start_time, end_time) or None if no segments are available
        """
        if self.segments_buffer:
            return self.segments_buffer.popleft()
        return None
    
    def reset(self) -> None:
        """
        Reset the processor state
        """
        self.vad.reset_states()
        self.audio_buffer = np.array([], dtype=np.float32)
        self.current_time = 0.0
        self.segments_buffer.clear()
        self.current_segment_start = None
        self.current_segment_audio = np.array([], dtype=np.float32)
        self.current_segment_last_activity = 0.0
        self.in_speech = False
        self.chunk_counter = 0
        self.chunk_samples = []
        logger.info("VAC Processor reset")
