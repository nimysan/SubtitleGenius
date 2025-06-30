#!/usr/bin/env python3
"""
测试阿拉伯语字幕生成
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from subtitle_genius.subtitle.models import Subtitle
from subtitle_genius.subtitle.formatter import SubtitleFormatter


async def test_arabic_subtitles():
    """测试阿拉伯语字幕功能"""
    
    print("🎬 SubtitleGenius 阿拉伯语测试")
    print("=" * 40)
    
    # 创建示例阿拉伯语字幕
    arabic_subtitles = [
        Subtitle(start=0.0, end=3.0, text="السلام عليكم ورحمة الله وبركاته"),
        Subtitle(start=3.5, end=7.0, text="أهلاً وسهلاً بكم في نشرة الأخبار"),
        Subtitle(start=7.5, end=11.0, text="اليوم نتحدث عن التطورات الجديدة"),
        Subtitle(start=11.5, end=14.0, text="شكراً لكم على المتابعة"),
    ]
    
    # 测试字幕格式化
    formatter = SubtitleFormatter()
    
    print("\n📝 SRT格式字幕:")
    print("-" * 30)
    srt_content = formatter.to_srt(arabic_subtitles)
    print(srt_content)
    
    print("📝 WebVTT格式字幕:")
    print("-" * 30)
    vtt_content = formatter.to_vtt(arabic_subtitles)
    print(vtt_content)
    
    # 保存字幕文件
    test_dir = Path("test_arabic")
    test_dir.mkdir(exist_ok=True)
    
    srt_file = test_dir / "arabic_test.srt"
    vtt_file = test_dir / "arabic_test.vtt"
    
    with open(srt_file, "w", encoding="utf-8") as f:
        f.write(srt_content)
    
    with open(vtt_file, "w", encoding="utf-8") as f:
        f.write(vtt_content)
    
    print(f"✅ 字幕文件已保存:")
    print(f"   - {srt_file}")
    print(f"   - {vtt_file}")
    
    # 检查是否有音频文件
    audio_files = list(test_dir.glob("*.mp3")) + list(test_dir.glob("*.wav"))
    
    if audio_files:
        print(f"\n🎵 发现音频文件: {len(audio_files)} 个")
        for audio_file in audio_files:
            print(f"   - {audio_file.name}")
        
        print("\n💡 下一步: 配置API密钥后可以处理这些音频文件")
    else:
        print("\n💡 提示: 将阿拉伯语音频文件放入 test_arabic/ 目录进行测试")


if __name__ == "__main__":
    asyncio.run(test_arabic_subtitles())
