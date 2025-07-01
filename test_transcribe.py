#!/usr/bin/env python3
"""
Amazon Transcribe 集成测试脚本
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from subtitle_genius.models.transcribe_model import TranscribeModel
from subtitle_genius.core.config import config


async def test_transcribe_model():
    """测试 Amazon Transcribe 模型"""
    
    print("🧪 测试 Amazon Transcribe 模型集成")
    print("=" * 50)
    
    # 初始化模型
    try:
        model = TranscribeModel(region_name=config.aws_region)
        print(f"✅ TranscribeModel 初始化成功")
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
        "test_arabic/arabic_test_audio.wav",
        "input.webm",
        "output.mp4"
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
    
    # 测试转录
    try:
        print("🚀 开始转录...")
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


def test_config():
    """测试配置"""
    print("⚙️  配置检查:")
    print(f"   AWS_REGION: {config.aws_region}")
    print(f"   AWS_S3_BUCKET: {config.aws_s3_bucket}")
    
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
    print("🎬 SubtitleGenius - Amazon Transcribe 测试")
    print("=" * 60)
    
    # 测试配置
    test_config()
    print()
    
    # 测试模型
    await test_transcribe_model()
    
    print("\n" + "=" * 60)
    print("测试完成！")


if __name__ == "__main__":
    asyncio.run(main())
