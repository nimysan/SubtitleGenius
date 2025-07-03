import boto3
import json
import base64
import io
import wave
import math
import time
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
from io import BytesIO

def check_ffmpeg():
    """Check if FFmpeg is available in the environment."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print(f"FFmpeg found: {result.stdout.splitlines()[0] if result.stdout else 'No version info'}")
            return True
        print(f"FFmpeg command failed with return code {result.returncode}")
        return False
    except Exception as e:
        print(f"Exception checking for FFmpeg: {str(e)}")
        return False

def detect_audio_format(audio_data):
    """Detect the audio/video format based on file signatures."""
    signatures = {
        b'RIFF': 'wav',  # WAV files
        b'\xff\xfb': 'mp3',  # MP3 files
        b'\x00\x00\x00': 'mp4',  # MP4/MOV files
        b'ftyp': 'mp4',  # MP4 files
        b'ID3': 'mp3',  # MP3 files with ID3 tag
        b'OggS': 'ogg'   # OGG files
    }
    
    for sig, fmt in signatures.items():
        if audio_data.startswith(sig):
            return fmt
        if fmt == 'mp4' and sig == b'ftyp' and b'ftyp' in audio_data[:50]:
            return fmt
    
    return 'unknown'

def is_wav_format(audio_data):
    """Check if the audio data is in WAV format (starts with RIFF header)."""
    return audio_data.startswith(b'RIFF')

def convert_mp4_to_wav(mp4_data):
    """Convert MP4 audio data to WAV format using FFmpeg."""
    print(f"Converting MP4 to WAV. Input data size: {len(mp4_data)} bytes")
    
    tmp_dir = tempfile.gettempdir()
    timestamp = int(time.time())
    mp4_path = os.path.join(tmp_dir, f"input_{timestamp}.mp4")
    wav_path = os.path.join(tmp_dir, f"output_{timestamp}.wav")
    
    try:
        # Write the MP4 data to a file
        with open(mp4_path, 'wb') as mp4_file:
            mp4_file.write(mp4_data)
        
        print(f"MP4 file written to {mp4_path}")
        
        # Check FFmpeg availability
        ffmpeg_available = check_ffmpeg()
        
        if ffmpeg_available:
            # Use FFmpeg for conversion
            cmd = ['ffmpeg', '-i', mp4_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', wav_path]
            print(f"Running command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                print("FFmpeg conversion successful")
                with open(wav_path, 'rb') as wav_file:
                    wav_data = wav_file.read()
                    print(f"WAV data size: {len(wav_data)} bytes")
                return wav_data
            else:
                print(f"FFmpeg error: {result.stderr}")
                raise Exception(f"FFmpeg conversion failed: {result.stderr}")
        else:
            # Fallback: create minimal WAV header
            print("Using fallback WAV header creation")
            sample_rate = 44100
            channels = 2
            bits_per_sample = 16
            
            header = BytesIO()
            header.write(b'RIFF')
            header.write((36 + len(mp4_data)).to_bytes(4, 'little'))
            header.write(b'WAVE')
            header.write(b'fmt ')
            header.write((16).to_bytes(4, 'little'))
            header.write((1).to_bytes(2, 'little'))
            header.write((channels).to_bytes(2, 'little'))
            header.write((sample_rate).to_bytes(4, 'little'))
            header.write((sample_rate * channels * bits_per_sample // 8).to_bytes(4, 'little'))
            header.write((channels * bits_per_sample // 8).to_bytes(2, 'little'))
            header.write((bits_per_sample).to_bytes(2, 'little'))
            header.write(b'data')
            header.write((len(mp4_data)).to_bytes(4, 'little'))
            
            # Extract audio data from MP4 if possible
            data_start = 0
            for i in range(len(mp4_data) - 4):
                if mp4_data[i:i+4] == b'mdat':
                    data_start = i + 8
                    break
            
            audio_data = mp4_data[data_start:] if data_start > 0 else mp4_data
            max_size = sample_rate * channels * bits_per_sample // 8 * 30  # 30 seconds max
            audio_data = audio_data[:max_size] if len(audio_data) > max_size else audio_data
            
            wav_data = header.getvalue() + audio_data
            print(f"Created WAV with manual header. Size: {len(wav_data)} bytes")
            return wav_data
            
    except Exception as e:
        print(f"Error in convert_mp4_to_wav: {str(e)}")
        raise
    finally:
        # Cleanup temporary files
        try:
            if os.path.exists(mp4_path):
                os.remove(mp4_path)
            if os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")

def chunk_audio(audio_data, chunk_duration_seconds=30):
    """Split wave audio from BytesIO into chunks."""
    try:
        print(f"Starting audio chunking. Input data size: {len(audio_data)} bytes")
        
        # Check audio format
        audio_format = detect_audio_format(audio_data)
        print(f"Detected audio format: {audio_format}")
        
        # Convert to WAV if needed
        if not is_wav_format(audio_data):
            print(f"Input is not WAV format (detected as {audio_format}), attempting conversion...")
            audio_data = convert_mp4_to_wav(audio_data)
            print(f"Conversion completed. WAV data size: {len(audio_data)} bytes")
        
        # Create a BytesIO object from the audio data
        wav_buffer = BytesIO(audio_data)
        
        with wave.open(wav_buffer, 'rb') as wav_file:
            # Get file properties
            n_channels = wav_file.getnchannels()
            sampwidth = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            
            print(f"WAV properties: channels={n_channels}, sampwidth={sampwidth}, framerate={framerate}, frames={n_frames}")
            
            # Calculate frames per chunk
            frames_per_chunk = chunk_duration_seconds * framerate
            
            # Adjust chunk size for SageMaker payload limits (2MB safe limit)
            max_payload_size = 2 * 1024 * 1024
            bytes_per_frame = n_channels * sampwidth
            estimated_chunk_size = frames_per_chunk * bytes_per_frame + 44
            
            if estimated_chunk_size > max_payload_size:
                size_ratio = max_payload_size / estimated_chunk_size
                adjusted_chunk_duration = int(chunk_duration_seconds * size_ratio * 0.8)
                frames_per_chunk = adjusted_chunk_duration * framerate
                print(f"Adjusted chunk duration to {adjusted_chunk_duration} seconds")
            
            n_chunks = math.ceil(n_frames / frames_per_chunk)
            print(f"Audio will be split into {n_chunks} chunks")
            
            chunks = []
            for i in range(n_chunks):
                try:
                    # Read chunk of frames
                    start_frame = i * frames_per_chunk
                    wav_file.setpos(int(start_frame))
                    chunk_frames = wav_file.readframes(int(frames_per_chunk))
                    
                    # Create a new wave file in memory for this chunk
                    chunk_buffer = BytesIO()
                    with wave.open(chunk_buffer, 'wb') as chunk_wav:
                        chunk_wav.setnchannels(n_channels)
                        chunk_wav.setsampwidth(sampwidth)
                        chunk_wav.setframerate(framerate)
                        chunk_wav.writeframes(chunk_frames)
                    
                    chunk_data = chunk_buffer.getvalue()
                    print(f"Chunk {i+1}/{n_chunks} created, size: {len(chunk_data)} bytes")
                    chunks.append(chunk_data)
                except Exception as chunk_error:
                    print(f"Error processing chunk {i+1}: {chunk_error}")
            
            # Filter out chunks that are too small
            valid_chunks = [chunk for chunk in chunks if len(chunk) > 44]
            print(f"Successfully created {len(valid_chunks)} valid chunks")
            return valid_chunks
            
    except Exception as e:
        print(f"Error in chunk_audio: {str(e)}")
        return []

def transcribe_chunk(sagemaker_client, chunk_data, endpoint_name, language="en", task="transcribe"):
    """Transcribe a single audio chunk using SageMaker runtime with Whisper endpoint."""
    try:
        print(f"Using SageMaker endpoint: {endpoint_name}")
        print(f"Sending request with audio size: {len(chunk_data)} bytes")
        
        # Convert audio to hex string (format expected by Whisper endpoints)
        hex_audio = chunk_data.hex()
        
        # Create payload for Whisper endpoint
        payload = {
            "audio_input": hex_audio,
            "language": language,
            "task": task,
            "top_p": 0.9
        }
        
        # Invoke the SageMaker endpoint
        response = sagemaker_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        # Parse the response
        response_body = json.loads(response['Body'].read().decode('utf-8'))
        print(f"Response received from SageMaker endpoint")
        
        return response_body
        
    except Exception as e:
        print(f"Error invoking SageMaker endpoint: {str(e)}")
        raise
    """打印messages的树形结构，隐藏音频数据"""
    def print_tree(obj, prefix="", is_last=True, indent="  "):
        # 打印当前节点
        branch = "└── " if is_last else "├── "
        print(f"{prefix}{branch}", end="")
        
        if isinstance(obj, dict):
            if "audio" in obj:
                print("<audio>")
            elif "source" in obj and "data" in obj["source"]:
                print("<audio_data>")
            else:
                print("{")
                new_prefix = prefix + (indent if is_last else "│   ")
                items = list(obj.items())
                for i, (key, value) in enumerate(items):
                    print(f"{new_prefix}├── {key}: ", end="")
                    if isinstance(value, (dict, list)):
                        print()
                        print_tree(value, new_prefix + indent, i == len(items) - 1)
                    else:
                        if isinstance(value, str) and len(value) > 50:
                            print(f"{value[:50]}...")
                        else:
                            print(value)
                if not items:
                    print("}")
        elif isinstance(obj, list):
            print("[")
            new_prefix = prefix + (indent if is_last else "│   ")
            for i, item in enumerate(obj):
                print_tree(item, new_prefix, i == len(obj) - 1)
            if not obj:
                print(f"{prefix}{indent}]")
        else:
            if isinstance(obj, str) and len(obj) > 50:
                print(f"{obj[:50]}...")
            else:
                print(obj)

    print("\nMessages Tree Structure:")
    print_tree(messages)

class WhisperSageMakerClient:
    """使用SageMaker Runtime调用自定义Whisper Turbo端点的客户端"""
    
    def __init__(self, 
                 endpoint_name: str,
                 region_name: str = "us-east-1"):
        """
        初始化Whisper SageMaker客户端
        
        Args:
            endpoint_name: SageMaker端点名称
            region_name: AWS区域
        """
        self.endpoint_name = endpoint_name
        self.region_name = region_name
        
        try:
            self.sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=region_name)
            print(f"✅ 成功初始化SageMaker Runtime客户端，区域: {region_name}")
            print(f"🎯 端点名称: {endpoint_name}")
        except Exception as e:
            print(f"❌ 初始化SageMaker Runtime客户端时出错: {e}")
            print("请确保您已配置AWS凭证")
            raise
    
    def load_audio_file(self, audio_path: str) -> bytes:
        """
        加载音频文件并转换为字节数据
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            音频文件的字节数据
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        
        print(f"📁 加载音频文件: {audio_path}")
        print(f"📊 文件大小: {len(audio_data)} bytes")
        
        return audio_data
    
    def get_audio_info(self, audio_data: bytes) -> Dict[str, Any]:
        """
        获取音频文件信息
        
        Args:
            audio_data: 音频字节数据
            
        Returns:
            音频信息字典
        """
        try:
            # 尝试解析WAV文件头
            audio_io = io.BytesIO(audio_data)
            with wave.open(audio_io, 'rb') as wav_file:
                info = {
                    "format": "WAV",
                    "channels": wav_file.getnchannels(),
                    "sample_rate": wav_file.getframerate(),
                    "duration": wav_file.getnframes() / wav_file.getframerate(),
                    "bit_depth": wav_file.getsampwidth() * 8
                }
                return info
        except:
            # 如果不是WAV文件，返回基本信息
            return {
                "format": "Unknown",
                "size_bytes": len(audio_data)
            }
    
    def transcribe_audio(self, 
                        audio_path: str,
                        language: str = "ar",
                        task: str = "transcribe",
                        chunk_duration: int = 30) -> Dict[str, Any]:
        """
        使用SageMaker Runtime转录音频
        
        Args:
            audio_path: 音频文件路径
            language: 语言代码 (默认: ar - Arabic)
            task: 任务类型 ("transcribe" 或 "translate")
            chunk_duration: 音频分块时长（秒）
            
        Returns:
            转录结果字典
        """
        try:
            start_time = time.time()
            
            # 加载音频文件
            audio_data = self.load_audio_file(audio_path)
            audio_info = self.get_audio_info(audio_data)
            
            print(f"🎵 音频信息: {audio_info}")
            
            # 将音频分块
            print(f"🔪 开始音频分块，块时长: {chunk_duration}秒")
            chunks = chunk_audio(audio_data, chunk_duration)
            
            if not chunks:
                raise Exception("音频分块失败，无法处理")
            
            print(f"📦 成功创建 {len(chunks)} 个音频块")
            
            # 处理每个音频块
            all_transcriptions = []
            chunk_timings = []
            cumulative_duration = 0
            
            for i, chunk_data in enumerate(chunks, 1):
                print(f"🎯 处理音频块 {i}/{len(chunks)}")
                
                try:
                    # 计算时间戳
                    chunk_start = cumulative_duration
                    chunk_end = chunk_start + chunk_duration
                    cumulative_duration = chunk_end
                    
                    # 转录单个块
                    result = transcribe_chunk(
                        self.sagemaker_runtime, 
                        chunk_data, 
                        self.endpoint_name,
                        language=self._convert_language_code(language),
                        task=task
                    )
                    
                    all_transcriptions.append(result)
                    chunk_timings.append((chunk_start, chunk_end))
                    
                    print(f"✅ 音频块 {i} 转录完成")
                    
                except Exception as e:
                    print(f"❌ 处理音频块 {i} 时出错: {str(e)}")
                    # 继续处理下一个块
                    all_transcriptions.append({"text": f"[块 {i} 转录失败]"})
                    chunk_timings.append((chunk_start, chunk_start + chunk_duration))
            
            # 合并转录结果
            full_transcription = self._combine_transcriptions(all_transcriptions, chunk_timings)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            result = {
                "transcription": full_transcription,
                "language": language,
                "task": task,
                "audio_info": audio_info,
                "chunks_processed": len(chunks),
                "chunk_timings": chunk_timings,
                "metrics": {
                    "processing_time_seconds": round(processing_time, 2),
                    "chunks_count": len(chunks),
                    "average_chunk_time": round(processing_time / len(chunks), 2) if chunks else 0
                }
            }
            
            return result
            
        except Exception as e:
            print(f"❌ 转录音频时出错: {str(e)}")
            return {
                "error": str(e),
                "transcription": None
            }
    
    def _convert_language_code(self, language: str) -> str:
        """转换语言代码为Whisper支持的格式"""
        language_map = {
            "ar": "arabic",
            "ar-SA": "arabic", 
            "ar-AE": "arabic",
            "en": "english",
            "en-US": "english",
            "en-GB": "english",
            "zh": "chinese",
            "zh-CN": "chinese",
            "ja": "japanese",
            "ko": "korean",
            "fr": "french",
            "de": "german",
            "es": "spanish",
            "ru": "russian"
        }
        return language_map.get(language, "english")
    
    def _combine_transcriptions(self, transcriptions: List[Dict], timings: List[tuple]) -> str:
        """合并多个转录结果"""
        combined_text = []
        
        for i, (result, (start_time, end_time)) in enumerate(zip(transcriptions, timings)):
            if isinstance(result, dict) and 'text' in result:
                text = result['text'] if isinstance(result['text'], str) else ' '.join(result['text'])
            elif isinstance(result, str):
                text = result
            else:
                text = f"[块 {i+1} 无法解析]"
            
            # 清理文本
            text = text.strip()
            if text:
                combined_text.append(text)
        
        return ' '.join(combined_text)
    
    def batch_transcribe(self, 
                        audio_files: List[str],
                        language: str = "ar",
                        task: str = "transcribe",
                        chunk_duration: int = 30) -> List[Dict[str, Any]]:
        """
        批量转录音频文件
        
        Args:
            audio_files: 音频文件路径列表
            language: 语言代码
            task: 任务类型
            chunk_duration: 音频分块时长
            
        Returns:
            转录结果列表
        """
        results = []
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n📝 处理文件 {i}/{len(audio_files)}: {audio_file}")
            result = self.transcribe_audio(audio_file, language, task, chunk_duration)
            result["file_path"] = audio_file
            results.append(result)
        
        return results

def main():
    """示例用法"""
    # 配置参数
    ENDPOINT_NAME = "endpoint-quick-start-z9afg"  # 替换为你的SageMaker端点名称
    REGION_NAME = "us-east-1"
    
    # 初始化客户端
    client = WhisperSageMakerClient(
        endpoint_name=ENDPOINT_NAME,
        region_name=REGION_NAME
    )
    
    # 示例音频文件路径
    audio_files = [
        # "/Users/yexw/PycharmProjects/SubtitleGenius/test.wav",
        "/Users/yexw/PycharmProjects/SubtitleGenius/ar_football_mono.wav"
    ]
    
    # 单个文件转录示例
    print("=" * 60)
    print("🎤 单个文件转录示例 (SageMaker)")
    print("=" * 60)
    
    if Path(audio_files[0]).exists():
        result = client.transcribe_audio(
            audio_path=audio_files[0],
            language="ar",  # Arabic
            task="transcribe",
            chunk_duration=30
        )
        
        if result.get("transcription"):
            print(f"✅ 转录成功:")
            print(f"📝 文本: {result['transcription']}")
            print(f"⏱️  处理时间: {result['metrics']['processing_time_seconds']}秒")
            print(f"📦 处理块数: {result['metrics']['chunks_count']}")
        else:
            print(f"❌ 转录失败: {result.get('error')}")
    
    # 批量转录示例
    print("\n" + "=" * 60)
    print("📚 批量转录示例")
    print("=" * 60)
    
    existing_files = [f for f in audio_files if Path(f).exists()]
    if existing_files:
        results = client.batch_transcribe(
            audio_files=existing_files,
            language="ar",
            task="transcribe",
            chunk_duration=30
        )
        
        for i, result in enumerate(results, 1):
            print(f"\n📄 文件 {i}: {Path(result['file_path']).name}")
            if result.get("transcription"):
                print(f"✅ 转录: {result['transcription'][:100]}...")
                print(f"📦 块数: {result.get('chunks_processed', 'N/A')}")
            else:
                print(f"❌ 错误: {result.get('error')}")
    
    # 翻译示例
    print("\n" + "=" * 60)
    print("🌐 翻译示例")
    print("=" * 60)
    
    if Path(audio_files[0]).exists():
        result = client.transcribe_audio(
            audio_path=audio_files[0],
            language="ar",
            task="translate",  # 翻译任务
            chunk_duration=30
        )
        
        if result.get("transcription"):
            print(f"✅ 翻译成功:")
            print(f"🔄 结果: {result['transcription']}")

# 为了向后兼容，保留旧的类名作为别名
WhisperConverseClient = WhisperSageMakerClient

if __name__ == "__main__":
    main()
