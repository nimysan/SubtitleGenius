#!/usr/bin/env python3
"""
测试SageMaker Whisper endpoint的segment-level timestamps功能
"""

import json
import time
from pathlib import Path
from sagemaker_whisper import WhisperSageMakerClient

def test_segment_timestamps():
    """测试segment-level timestamps功能"""
    
    # 配置参数
    ENDPOINT_NAME = "endpoint-quick-start-z9afg"  # 替换为你的SageMaker端点名称
    REGION_NAME = "us-east-1"
    
    # 测试音频文件
    audio_file = "/Users/yexw/PycharmProjects/SubtitleGenius/ar_football_mono.wav"
    
    if not Path(audio_file).exists():
        print(f"❌ 音频文件不存在: {audio_file}")
        return
    
    print("=" * 80)
    print("🎯 测试SageMaker Whisper Segment-Level Timestamps")
    print("=" * 80)
    
    # 初始化客户端
    try:
        client = WhisperSageMakerClient(
            endpoint_name=ENDPOINT_NAME,
            region_name=REGION_NAME
        )
        print("✅ SageMaker客户端初始化成功")
    except Exception as e:
        print(f"❌ 客户端初始化失败: {e}")
        return
    
    # 测试转录
    print(f"\n🎵 开始转录音频文件: {Path(audio_file).name}")
    print(f"📊 使用segment-level timestamps")
    
    start_time = time.time()
    
    try:
        result = client.transcribe_audio(
            audio_path=audio_file,
            language="ar",  # Arabic
            task="transcribe",
            chunk_duration=10  # 使用较短的chunk来测试timestamps
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n⏱️  总处理时间: {processing_time:.2f}秒")
        
        if result.get("error"):
            print(f"❌ 转录失败: {result['error']}")
            return
        
        # 显示结果
        print("\n" + "=" * 60)
        print("📝 转录结果分析")
        print("=" * 60)
        
        print(f"🎯 语言: {result.get('language', 'N/A')}")
        print(f"📦 处理的音频块数: {result.get('chunks_processed', 'N/A')}")
        print(f"⏱️  平均每块处理时间: {result['metrics'].get('average_chunk_time', 'N/A')}秒")
        
        # 显示完整转录文本
        transcription = result.get("transcription", "")
        print(f"\n📄 完整转录文本:")
        print("-" * 40)
        print(transcription)
        print("-" * 40)
        
        # 显示chunk时间信息
        chunk_timings = result.get("chunk_timings", [])
        print(f"\n⏰ Chunk时间分布:")
        for i, (start, end) in enumerate(chunk_timings, 1):
            print(f"  Chunk {i}: {start:.1f}s - {end:.1f}s ({end-start:.1f}s)")
        
        # 保存详细结果到文件
        output_file = "segment_timestamps_test_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            # 创建可序列化的结果副本
            serializable_result = {
                "transcription": result.get("transcription"),
                "language": result.get("language"),
                "task": result.get("task"),
                "audio_info": result.get("audio_info"),
                "chunks_processed": result.get("chunks_processed"),
                "chunk_timings": result.get("chunk_timings"),
                "metrics": result.get("metrics"),
                "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "test_parameters": {
                    "endpoint_name": ENDPOINT_NAME,
                    "chunk_duration": 10,
                    "return_timestamps": True,
                    "timestamp_granularities": ["segment"]
                }
            }
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 详细结果已保存到: {output_file}")
        
        # 分析结果质量
        print(f"\n📊 结果质量分析:")
        text_length = len(transcription)
        words_count = len(transcription.split()) if transcription else 0
        print(f"  - 文本长度: {text_length} 字符")
        print(f"  - 单词数量: {words_count} 个")
        print(f"  - 平均每块字符数: {text_length / result.get('chunks_processed', 1):.1f}")
        
        if words_count > 0:
            print("✅ 转录成功，包含有效内容")
        else:
            print("⚠️  转录结果为空，可能需要检查音频质量或参数设置")
            
    except Exception as e:
        print(f"❌ 转录过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_segment_timestamps()
