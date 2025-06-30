#!/usr/bin/env python3
"""
SubtitleGenius 使用示例
"""

import asyncio
from subtitle_genius.subtitle.models import Subtitle
from subtitle_genius.subtitle.formatter import SubtitleFormatter


async def main():
    """主函数"""
    print("🎬 SubtitleGenius 示例程序")
    print("=" * 50)
    
    # 创建示例字幕
    subtitles = [
        Subtitle(start=0.0, end=2.5, text="欢迎使用 SubtitleGenius"),
        Subtitle(start=2.5, end=5.0, text="基于GenAI的实时字幕生成工具"),
        Subtitle(start=5.0, end=8.0, text="支持多种AI模型和字幕格式"),
        Subtitle(start=8.0, end=10.5, text="让字幕生成变得简单高效")
    ]
    
    # 格式化字幕
    formatter = SubtitleFormatter()
    
    print("📝 生成的SRT格式字幕:")
    print("-" * 30)
    srt_content = formatter.to_srt(subtitles)
    print(srt_content)
    
    print("📝 生成的WebVTT格式字幕:")
    print("-" * 30)
    vtt_content = formatter.to_vtt(subtitles)
    print(vtt_content)
    
    print("✅ 示例运行完成!")
    print("\n💡 提示:")
    print("- 配置 .env 文件中的API密钥后可使用AI模型")
    print("- 使用 'uv run subtitle-genius --help' 查看CLI工具帮助")
    print("- 查看 README.md 了解更多使用方法")


if __name__ == "__main__":
    asyncio.run(main())
