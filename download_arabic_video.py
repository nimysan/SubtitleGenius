#!/usr/bin/env python3
"""
下载阿拉伯语新闻视频用于测试SubtitleGenius
"""

import os
import subprocess
import sys
from pathlib import Path


def install_yt_dlp():
    """安装yt-dlp如果没有安装"""
    try:
        subprocess.run(["yt-dlp", "--version"], check=True, capture_output=True)
        print("✅ yt-dlp 已安装")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("📦 正在安装 yt-dlp...")
        subprocess.run([sys.executable, "-m", "pip", "install", "yt-dlp"], check=True)


def download_arabic_news_video():
    """下载阿拉伯语新闻视频"""
    
    # 创建测试视频目录
    test_videos_dir = Path("test_videos")
    test_videos_dir.mkdir(exist_ok=True)
    
    # 一些公开的阿拉伯语新闻视频URL示例
    # 注意：这些是示例URL，实际使用时需要替换为有效的URL
    video_urls = [
        # Al Jazeera Arabic 短新闻片段
        "https://www.youtube.com/watch?v=EXAMPLE_ID_1",
        # BBC Arabic 新闻报道
        "https://www.youtube.com/watch?v=EXAMPLE_ID_2",
    ]
    
    # 实际可用的测试视频URL（需要手动更新）
    print("🔍 请手动从以下来源获取阿拉伯语新闻视频URL:")
    print("1. Al Jazeera Arabic: https://www.youtube.com/c/aljazeerachannel")
    print("2. BBC Arabic: https://www.youtube.com/c/BBCArabic")
    print("3. Sky News Arabia: https://www.youtube.com/c/skynewsarabia")
    print()
    
    # 让用户输入视频URL
    video_url = input("请输入阿拉伯语新闻视频的YouTube URL: ").strip()
    
    if not video_url:
        print("❌ 未提供视频URL")
        return
    
    try:
        print(f"📥 正在下载视频: {video_url}")
        
        # 下载视频，限制质量和时长
        cmd = [
            "yt-dlp",
            "--format", "best[height<=720][ext=mp4]",  # 限制为720p MP4格式
            "--output", str(test_videos_dir / "arabic_news_%(title)s.%(ext)s"),
            "--write-info-json",  # 保存视频信息
            "--write-subs",       # 下载字幕（如果有）
            "--sub-langs", "ar,en",  # 阿拉伯语和英语字幕
            video_url
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ 视频下载成功!")
        
        # 列出下载的文件
        print("\n📁 下载的文件:")
        for file in test_videos_dir.glob("*"):
            print(f"  - {file.name}")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 下载失败: {e}")
        print(f"错误输出: {e.stderr}")


def create_test_script():
    """创建测试脚本"""
    test_script = """#!/usr/bin/env python3
'''
测试SubtitleGenius处理阿拉伯语视频
'''

import asyncio
from pathlib import Path
from subtitle_genius import SubtitleGenerator

async def test_arabic_video():
    '''测试阿拉伯语视频字幕生成'''
    
    # 查找测试视频
    test_videos_dir = Path("test_videos")
    video_files = list(test_videos_dir.glob("*.mp4"))
    
    if not video_files:
        print("❌ 未找到测试视频文件")
        print("请先运行 python download_arabic_video.py 下载测试视频")
        return
    
    video_file = video_files[0]
    print(f"🎬 处理视频: {video_file.name}")
    
    try:
        # 初始化字幕生成器，设置为阿拉伯语
        generator = SubtitleGenerator(
            model="openai-whisper",  # 使用Whisper模型
            language="ar",           # 阿拉伯语
            output_format="srt"      # SRT格式
        )
        
        # 生成字幕
        print("🔄 正在生成阿拉伯语字幕...")
        subtitles = await generator.process_video(
            video_file,
            video_file.with_suffix('.ar.srt')
        )
        
        print(f"✅ 成功生成 {len(subtitles)} 条字幕!")
        print(f"📄 字幕文件: {video_file.with_suffix('.ar.srt')}")
        
        # 显示前几条字幕
        print("\\n📝 前5条字幕预览:")
        for i, subtitle in enumerate(subtitles[:5]):
            print(f"{i+1}. [{subtitle.start:.1f}s-{subtitle.end:.1f}s] {subtitle.text}")
            
    except Exception as e:
        print(f"❌ 处理失败: {e}")

if __name__ == "__main__":
    asyncio.run(test_arabic_video())
"""
    
    with open("test_arabic_subtitles.py", "w", encoding="utf-8") as f:
        f.write(test_script)
    
    print("✅ 创建了测试脚本: test_arabic_subtitles.py")


def main():
    """主函数"""
    print("🎬 SubtitleGenius 阿拉伯语视频下载工具")
    print("=" * 50)
    
    # 安装依赖
    install_yt_dlp()
    
    # 下载视频
    download_arabic_news_video()
    
    # 创建测试脚本
    create_test_script()
    
    print("\n🚀 下一步:")
    print("1. 配置 .env 文件中的API密钥")
    print("2. 运行: python test_arabic_subtitles.py")


if __name__ == "__main__":
    main()
