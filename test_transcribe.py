#!/usr/bin/env python3
"""
Amazon Transcribe 集成测试脚本 - 支持流式处理和Arabic语言
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from subtitle_genius.models.transcribe_model import TranscribeModel
from subtitle_genius.stream.processor import StreamProcessor
from subtitle_genius.core.config import config


async def test_transcribe_batch_mode():
    """测试 Amazon Transcribe 批处理模式"""
    
    print("🧪 测试 Amazon Transcribe 批处理模式")
    print("=" * 50)
    
    # 初始化模型 (禁用流式处理以测试批处理)
    try:
        model = TranscribeModel(region_name=config.aws_region, use_streaming=False)
        print(f"✅ TranscribeModel 初始化成功 (批处理模式)")
        print(f"   区域: {config.aws_region}")
        print(f"   S3存储桶: {config.aws_s3_bucket}")
    except Exception as e:
        print(f"❌ TranscribeModel 初始化失败: {e}")
        return
    
    # 检查模型可用性
    print("\n🔍 检查模型可用性...")
    if model.is_available():
        print("✅ Amazon Transcribe 可用")
    else:
        print("❌ Amazon Transcribe 不可用")
        print("请检查以下配置:")
        print("1. AWS_ACCESS_KEY_ID")
        print("2. AWS_SECRET_ACCESS_KEY") 
        print("3. AWS_REGION")
        print("4. 确保 AWS 账户有 Transcribe 和 S3 权限")
        return
    
    # 测试音频文件路径
    test_audio_files = [
        "input.mp4"
    ]
    
    test_file = None
    for file_path in test_audio_files:
        if Path(file_path).exists():
            test_file = file_path
            break
    
    if not test_file:
        print("\n⚠️  未找到测试音频文件")
        print("请确保以下文件之一存在:")
        for file_path in test_audio_files:
            print(f"   - {file_path}")
        return
    
    print(f"\n🎵 使用测试文件: {test_file}")
    
    # 测试转录 (使用 Arabic 作为默认语言)
    try:
        print("🚀 开始转录 (Arabic)...")
        subtitles = await model.transcribe(test_file, language="ar")
        
        print(f"✅ 转录完成！生成了 {len(subtitles)} 条字幕")
        
        # 显示前几条字幕
        print("\n📝 字幕预览:")
        for i, subtitle in enumerate(subtitles[:5]):
            print(f"   {i+1}. [{subtitle.start:.1f}s - {subtitle.end:.1f}s] {subtitle.text}")
        
        if len(subtitles) > 5:
            print(f"   ... 还有 {len(subtitles) - 5} 条字幕")
            
    except Exception as e:
        print(f"❌ 转录失败: {e}")
        import traceback
        traceback.print_exc()


async def test_transcribe_streaming_mode():
    """测试 Amazon Transcribe 流式处理模式"""
    
    print("\n🌊 测试 Amazon Transcribe 流式处理模式")
    print("=" * 50)
    
    # 初始化模型 (启用流式处理)
    try:
        model = TranscribeModel(region_name=config.aws_region, use_streaming=True)
        print(f"✅ TranscribeModel 初始化成功 (流式处理模式)")
        
        if not model.use_streaming:
            print("⚠️  流式处理不可用，请安装 amazon-transcribe 包:")
            print("   pip install amazon-transcribe")
            return
            
    except Exception as e:
        print(f"❌ TranscribeModel 初始化失败: {e}")
        return
    
    # 检查模型可用性
    if not model.is_available():
        print("❌ Amazon Transcribe 不可用")
        return
    
    # 测试音频文件路径
    test_file = "input.mp4"
    if not Path(test_file).exists():
        print(f"\n⚠️  未找到测试音频文件: {test_file}")
        return
    
    print(f"\n🎵 使用测试文件进行流式处理: {test_file}")
    
    try:
        # 创建流处理器
        stream_processor = StreamProcessor()
        
        print("🚀 开始流式转录 (Arabic)...")
        
        # 创建音频流
        audio_stream = stream_processor.process_file_stream(test_file)
        
        # 流式转录
        subtitle_count = 0
        async for subtitle in model.transcribe_stream(audio_stream, language="ar"):
            subtitle_count += 1
            print(f"📝 字幕 {subtitle_count}: [{subtitle.start:.1f}s - {subtitle.end:.1f}s] {subtitle.text}")
            
            # 限制显示数量以避免输出过多
            if subtitle_count >= 10:
                print("   ... (限制显示前10条字幕)")
                break
        
        print(f"✅ 流式转录完成！共处理了 {subtitle_count} 条字幕")
        
    except Exception as e:
        print(f"❌ 流式转录失败: {e}")
        import traceback
        traceback.print_exc()


def test_config():
    """测试配置"""
    print("⚙️  配置检查:")
    print(f"   AWS_REGION: {config.aws_region}")
    print(f"   AWS_S3_BUCKET: {config.aws_s3_bucket}")
    print(f"   默认语言: Arabic (ar)")
    
    # 检查环境变量
    import os
    aws_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    if aws_key:
        print(f"   AWS_ACCESS_KEY_ID: {aws_key[:8]}...")
    else:
        print("   AWS_ACCESS_KEY_ID: 未设置")
    
    if aws_secret:
        print(f"   AWS_SECRET_ACCESS_KEY: {aws_secret[:8]}...")
    else:
        print("   AWS_SECRET_ACCESS_KEY: 未设置")


async def main():
    """主函数"""
    print("🎬 SubtitleGenius - Amazon Transcribe 测试 (Arabic + Streaming)")
    print("=" * 70)
    
    # 测试配置
    test_config()
    print()
    
    # 测试批处理模式
    await test_transcribe_batch_mode()
    
    # 测试流式处理模式
    await test_transcribe_streaming_mode()
    
    print("\n" + "=" * 70)
    print("测试完成！")


if __name__ == "__main__":
    asyncio.run(main())
