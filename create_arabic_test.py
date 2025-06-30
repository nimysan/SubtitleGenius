#!/usr/bin/env python3
"""
创建阿拉伯语测试音频和视频
"""

import os
import subprocess
import sys
from pathlib import Path


def create_test_audio():
    """创建阿拉伯语测试音频"""
    
    # 创建测试目录
    test_dir = Path("test_arabic")
    test_dir.mkdir(exist_ok=True)
    
    # 阿拉伯语测试文本
    arabic_texts = [
        "السلام عليكم ورحمة الله وبركاته",  # 问候语
        "أهلاً وسهلاً بكم في نشرة الأخبار",    # 欢迎收看新闻
        "اليوم نتحدث عن التطورات الجديدة",     # 今天我们谈论新发展
        "شكراً لكم على المتابعة",              # 感谢您的收看
    ]
    
    print("🎵 创建阿拉伯语测试音频文件...")
    
    # 创建文本文件
    text_file = test_dir / "arabic_text.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        for i, text in enumerate(arabic_texts, 1):
            f.write(f"{i}. {text}\n")
    
    print(f"✅ 创建了阿拉伯语文本文件: {text_file}")
    
    # 创建简单的音频文件说明
    readme_file = test_dir / "README.md"
    with open(readme_file, "w", encoding="utf-8") as f:
        f.write("""# 阿拉伯语测试音频

## 获取阿拉伯语测试音频的方法:

### 1. 在线文本转语音服务
- **Google Translate**: https://translate.google.com
  - 输入阿拉伯语文本
  - 点击播放按钮听发音
  - 使用浏览器录音工具录制

- **Microsoft Translator**: https://www.bing.com/translator
  - 支持阿拉伯语文本转语音

### 2. 免费阿拉伯语音频资源
- **Forvo**: https://forvo.com/languages/ar/
  - 阿拉伯语单词和短语发音
  
- **Common Voice**: https://commonvoice.mozilla.org/ar
  - Mozilla的开源语音数据集

### 3. YouTube阿拉伯语内容
- 搜索短的阿拉伯语教学视频
- 新闻片段
- 使用yt-dlp下载音频

### 4. 测试文本
```
السلام عليكم ورحمة الله وبركاته
أهلاً وسهلاً بكم في نشرة الأخبار  
اليوم نتحدث عن التطورات الجديدة
شكراً لكم على المتابعة
```

## 使用方法
1. 获取阿拉伯语音频文件 (MP3/WAV格式)
2. 放入此目录
3. 运行测试脚本
""")
    
    print(f"✅ 创建了说明文件: {readme_file}")
    
    return test_dir


def create_test_script():
    """创建阿拉伯语测试脚本"""
    
    test_script = '''#!/usr/bin/env python3
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
    
    print("\\n📝 SRT格式字幕:")
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
        print(f"\\n🎵 发现音频文件: {len(audio_files)} 个")
        for audio_file in audio_files:
            print(f"   - {audio_file.name}")
        
        print("\\n💡 下一步: 配置API密钥后可以处理这些音频文件")
    else:
        print("\\n💡 提示: 将阿拉伯语音频文件放入 test_arabic/ 目录进行测试")


if __name__ == "__main__":
    asyncio.run(test_arabic_subtitles())
'''
    
    with open("test_arabic_demo.py", "w", encoding="utf-8") as f:
        f.write(test_script)
    
    print("✅ 创建了阿拉伯语测试脚本: test_arabic_demo.py")


def main():
    """主函数"""
    print("🌍 SubtitleGenius 阿拉伯语测试环境设置")
    print("=" * 50)
    
    # 创建测试目录和文件
    test_dir = create_test_audio()
    create_test_script()
    
    print(f"\n🚀 设置完成!")
    print(f"📁 测试目录: {test_dir}")
    print(f"\n📋 下一步:")
    print("1. 运行: python test_arabic_demo.py (测试字幕格式化)")
    print("2. 获取阿拉伯语音频文件放入 test_arabic/ 目录")
    print("3. 配置 .env 文件中的API密钥")
    print("4. 使用真实音频测试语音识别功能")
    
    print(f"\n🎯 推荐的阿拉伯语音频来源:")
    print("- Google Translate 文本转语音")
    print("- YouTube 阿拉伯语新闻短片")
    print("- Mozilla Common Voice 阿拉伯语数据集")


if __name__ == "__main__":
    main()
