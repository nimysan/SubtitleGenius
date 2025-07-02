#!/usr/bin/env python3
"""
简单的Whisper Turbo SageMaker示例
基于AWS样例代码，使用SageMaker Runtime调用自定义Whisper端点
"""

from whisper_converse import WhisperSageMakerClient
from pathlib import Path
import sys

def test_whisper_sagemaker():
    """测试Whisper SageMaker API"""
    
    print("🎤 Whisper Turbo SageMaker API 测试")
    print("=" * 50)
    
    # 配置你的端点信息
    ENDPOINT_NAME = "endpoint-quick-start-z9afg"  # 🔧 替换为你的实际SageMaker端点名称
    REGION_NAME = "us-east-1"  # 🔧 替换为你的AWS区域
    
    try:
        # 初始化客户端
        print(f"🚀 初始化SageMaker客户端...")
        print(f"   端点: {ENDPOINT_NAME}")
        print(f"   区域: {REGION_NAME}")
        
        client = WhisperSageMakerClient(
            endpoint_name=ENDPOINT_NAME,
            region_name=REGION_NAME
        )
        
        # 查找可用的音频文件
        audio_files = [
            # "/Users/yexw/PycharmProjects/SubtitleGenius/test.wav",
            "/Users/yexw/PycharmProjects/SubtitleGenius/output_16k_mono.wav",
            # "/Users/yexw/PycharmProjects/SubtitleGenius/input.aac"
        ]
        
        available_files = [f for f in audio_files if Path(f).exists()]
        
        if not available_files:
            print("❌ 未找到可用的音频文件")
            print("请确保以下文件之一存在:")
            for f in audio_files:
                print(f"   - {f}")
            return
        
        test_file = available_files[0]
        print(f"📁 使用测试文件: {Path(test_file).name}")
        
        # 测试1: Arabic转录 (分块处理)
        print(f"\n📝 测试1: Arabic语音转录 (分块处理)")
        print("-" * 30)
        
        result = client.transcribe_audio(
            audio_path=test_file,
            language="ar",
            task="transcribe",
            chunk_duration=30  # 30秒分块
        )
        
        if result.get("transcription"):
            print(f"✅ 转录成功!")
            print(f"🎯 结果: {result['transcription']}")
            print(f"⏱️  处理时间: {result['metrics']['processing_time_seconds']}秒")
            print(f"📦 处理块数: {result['metrics']['chunks_count']}")
            print(f"⚡ 平均每块时间: {result['metrics']['average_chunk_time']}秒")
        else:
            print(f"❌ 转录失败: {result.get('error')}")
        
        # 测试2: 翻译到英语
        print(f"\n🌐 测试2: 翻译到英语")
        print("-" * 30)
        
        result = client.transcribe_audio(
            audio_path=test_file,
            language="ar",
            task="translate",
            chunk_duration=20  # 较短的分块用于翻译
        )
        
        if result.get("transcription"):
            print(f"✅ 翻译成功!")
            print(f"🔄 结果: {result['transcription']}")
            print(f"📦 处理块数: {result['metrics']['chunks_count']}")
        else:
            print(f"❌ 翻译失败: {result.get('error')}")
        
        # 测试3: 批量处理
        if len(available_files) > 1:
            print(f"\n📚 测试3: 批量处理")
            print("-" * 30)
            
            batch_results = client.batch_transcribe(
                audio_files=available_files[:2],  # 只处理前两个文件
                language="ar",
                task="transcribe",
                chunk_duration=30
            )
            
            for i, result in enumerate(batch_results, 1):
                file_name = Path(result['file_path']).name
                print(f"📄 文件{i}: {file_name}")
                if result.get("transcription"):
                    # 只显示前100个字符
                    text = result['transcription']
                    display_text = text[:100] + "..." if len(text) > 100 else text
                    print(f"   ✅ {display_text}")
                    print(f"   📦 块数: {result.get('chunks_processed', 'N/A')}")
                else:
                    print(f"   ❌ {result.get('error')}")
        
        # 测试4: 不同音频格式处理
        print(f"\n🎵 测试4: 音频格式处理")
        print("-" * 30)
        
        # 查找不同格式的文件
        format_files = {
            "WAV": [f for f in available_files if f.endswith('.wav')],
            "AAC": [f for f in available_files if f.endswith('.aac')],
            "MP3": [f for f in available_files if f.endswith('.mp3')]
        }
        
        for format_name, files in format_files.items():
            if files:
                print(f"🔊 处理 {format_name} 格式: {Path(files[0]).name}")
                result = client.transcribe_audio(
                    audio_path=files[0],
                    language="ar",
                    task="transcribe",
                    chunk_duration=15  # 较短分块用于测试
                )
                
                if result.get("transcription"):
                    print(f"   ✅ 成功: {result['transcription'][:50]}...")
                else:
                    print(f"   ❌ 失败: {result.get('error')}")
        
        print(f"\n🎉 测试完成!")
        
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        print(f"💡 请检查:")
        print(f"   1. AWS凭证是否正确配置")
        print(f"   2. SageMaker端点名称是否正确")
        print(f"   3. 区域设置是否正确")
        print(f"   4. SageMaker端点是否已部署并运行")
        print(f"   5. IAM权限是否包含SageMaker:InvokeEndpoint")

def main():
    """主函数"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("使用方法:")
            print("  python example_whisper_converse.py")
            print("")
            print("配置:")
            print("  1. 编辑 example_whisper_converse.py")
            print("  2. 设置 ENDPOINT_NAME 为你的SageMaker Whisper端点名称")
            print("  3. 设置 REGION_NAME 为你的AWS区域")
            print("  4. 确保AWS凭证已正确配置")
            print("  5. 确保有SageMaker:InvokeEndpoint权限")
            print("")
            print("注意:")
            print("  - 现在使用SageMaker Runtime而不是Converse API")
            print("  - 支持音频分块处理，适合长音频文件")
            print("  - 支持多种音频格式 (WAV, MP3, AAC等)")
            return
    
    test_whisper_sagemaker()

if __name__ == "__main__":
    main()
