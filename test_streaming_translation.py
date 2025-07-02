#!/usr/bin/env python3
"""
测试流式字幕翻译系统
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

async def test_translation_service():
    """测试翻译服务"""
    print("🧪 测试翻译服务")
    print("-" * 50)
    
    try:
        from translation_service import translation_manager
        
        # 测试文本
        test_texts = [
            "Hello, how are you today?",
            "Good morning, welcome to our service.",
            "Thank you for using our application.",
            "The weather is very nice today.",
            "I hope you have a great day!"
        ]
        
        print(f"可用翻译服务: {translation_manager.get_available_services()}")
        print(f"默认翻译服务: {translation_manager.default_translator}")
        print()
        
        for i, text in enumerate(test_texts, 1):
            print(f"测试 {i}: {text}")
            
            try:
                result = await translation_manager.translate(text, target_lang="zh")
                print(f"翻译结果: {result.translated_text}")
                print(f"使用服务: {result.service}")
                print(f"置信度: {result.confidence}")
                print("-" * 30)
                
            except Exception as e:
                print(f"翻译失败: {e}")
                print("-" * 30)
        
        print("✅ 翻译服务测试完成")
        
    except ImportError as e:
        print(f"❌ 翻译服务导入失败: {e}")
        return False
    
    return True

def test_audio_preprocessing():
    """测试音频预处理"""
    print("\n🎵 测试音频预处理")
    print("-" * 50)
    
    # 检查是否有测试音频文件
    test_files = ["output.wav", "output_16k_mono.wav"]
    available_files = []
    
    for file in test_files:
        if Path(file).exists():
            available_files.append(file)
            print(f"✅ 找到测试文件: {file}")
    
    if not available_files:
        print("⚠️ 没有找到测试音频文件")
        print("请确保项目目录中有 output.wav 或 output_16k_mono.wav")
        return False
    
    # 测试音频格式检查
    try:
        import subprocess
        
        for file in available_files:
            print(f"\n📊 分析文件: {file}")
            
            cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', 
                   '-show_format', '-show_streams', file]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            import json
            info = json.loads(result.stdout)
            
            if 'streams' in info and len(info['streams']) > 0:
                stream = info['streams'][0]
                print(f"  采样率: {stream.get('sample_rate', 'N/A')} Hz")
                print(f"  声道数: {stream.get('channels', 'N/A')}")
                print(f"  编码: {stream.get('codec_name', 'N/A')}")
                print(f"  时长: {stream.get('duration', 'N/A')} 秒")
                
                # 检查是否符合 Transcribe 要求
                sample_rate = int(stream.get('sample_rate', 0))
                channels = int(stream.get('channels', 0))
                
                if sample_rate == 16000 and channels == 1:
                    print("  ✅ 格式符合 Amazon Transcribe 要求")
                else:
                    print("  ⚠️ 格式需要转换")
                    print(f"     建议: ffmpeg -i {file} -ar 16000 -ac 1 -sample_fmt s16 converted_{file} -y")
        
        print("\n✅ 音频预处理测试完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg 命令失败: {e}")
        return False
    except FileNotFoundError:
        print("❌ FFmpeg 未安装")
        print("请安装 FFmpeg: https://ffmpeg.org/download.html")
        return False
    except Exception as e:
        print(f"❌ 音频分析失败: {e}")
        return False

def test_transcribe_availability():
    """测试 Amazon Transcribe 可用性"""
    print("\n🎤 测试 Amazon Transcribe 可用性")
    print("-" * 50)
    
    try:
        import amazon_transcribe
        from amazon_transcribe.client import TranscribeStreamingClient
        print("✅ Amazon Transcribe SDK 已安装")
        
        # 测试 AWS 凭证
        import os
        aws_keys = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_REGION']
        missing_keys = []
        
        for key in aws_keys:
            if not os.getenv(key):
                missing_keys.append(key)
        
        if missing_keys:
            print(f"⚠️ 缺少 AWS 环境变量: {', '.join(missing_keys)}")
            print("请设置以下环境变量:")
            for key in missing_keys:
                print(f"  export {key}=your_value")
        else:
            print("✅ AWS 环境变量已配置")
        
        return len(missing_keys) == 0
        
    except ImportError:
        print("❌ Amazon Transcribe SDK 未安装")
        print("请运行: python install_streaming.py")
        return False

def test_gradio_availability():
    """测试 Gradio 可用性"""
    print("\n🌐 测试 Gradio 可用性")
    print("-" * 50)
    
    try:
        import gradio as gr
        print(f"✅ Gradio 已安装 (版本: {gr.__version__})")
        return True
    except ImportError:
        print("❌ Gradio 未安装")
        print("请运行: pip install gradio")
        return False

async def main():
    """主测试函数"""
    print("=" * 60)
    print("🧪 SubtitleGenius 流式字幕翻译系统测试")
    print("=" * 60)
    
    # 测试各个组件
    tests = [
        ("翻译服务", test_translation_service()),
        ("音频预处理", test_audio_preprocessing()),
        ("Amazon Transcribe", test_transcribe_availability()),
        ("Gradio 界面", test_gradio_availability())
    ]
    
    results = []
    
    for name, test in tests:
        if asyncio.iscoroutine(test):
            result = await test
        else:
            result = test
        results.append((name, result))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！系统可以正常运行")
        print("运行以下命令启动界面:")
        print("  python launch_streaming_translation.py")
        print("  或直接运行: python gradio_streaming_translation.py")
    else:
        print("⚠️ 部分测试失败，请解决上述问题后重试")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
