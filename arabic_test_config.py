#!/usr/bin/env python3
"""
阿拉伯语字幕生成测试配置
"""

# 阿拉伯语新闻视频资源
ARABIC_NEWS_SOURCES = {
    "al_jazeera": {
        "name": "الجزيرة (Al Jazeera Arabic)",
        "website": "https://www.aljazeera.net",
        "youtube": "https://www.youtube.com/c/aljazeerachannel",
        "description": "卡塔尔半岛电视台阿拉伯语频道"
    },
    "bbc_arabic": {
        "name": "بي بي سي عربي (BBC Arabic)",
        "website": "https://www.bbc.com/arabic",
        "youtube": "https://www.youtube.com/c/BBCArabic",
        "description": "英国广播公司阿拉伯语服务"
    },
    "sky_news_arabia": {
        "name": "سكاي نيوز عربية (Sky News Arabia)",
        "website": "https://www.skynewsarabia.com",
        "youtube": "https://www.youtube.com/c/skynewsarabia",
        "description": "天空新闻阿拉伯语频道"
    },
    "rt_arabic": {
        "name": "RT Arabic",
        "website": "https://arabic.rt.com",
        "youtube": "https://www.youtube.com/c/RTarabic",
        "description": "今日俄罗斯阿拉伯语频道"
    }
}

# 推荐的测试视频类型
RECOMMENDED_VIDEO_TYPES = [
    {
        "type": "news_bulletin",
        "description": "新闻简报",
        "duration": "2-5分钟",
        "characteristics": "清晰发音，标准阿拉伯语"
    },
    {
        "type": "interview",
        "description": "新闻访谈",
        "duration": "5-10分钟",
        "characteristics": "对话形式，多人发言"
    },
    {
        "type": "field_report",
        "description": "现场报道",
        "duration": "3-8分钟",
        "characteristics": "可能有背景噪音，方言混合"
    }
]

# 阿拉伯语语音识别配置
ARABIC_ASR_CONFIG = {
    "whisper": {
        "language": "ar",
        "model_size": "base",  # 可选: tiny, base, small, medium, large
        "task": "transcribe"
    },
    "openai_api": {
        "language": "ar",
        "model": "whisper-1",
        "response_format": "verbose_json"
    }
}

# 字幕格式配置
SUBTITLE_CONFIG = {
    "formats": ["srt", "vtt", "ass"],
    "max_line_length": 80,
    "max_lines_per_subtitle": 2,
    "min_duration": 1.0,
    "max_duration": 7.0
}

def print_resources():
    """打印阿拉伯语新闻资源"""
    print("🌍 阿拉伯语新闻视频资源:")
    print("=" * 50)
    
    for key, source in ARABIC_NEWS_SOURCES.items():
        print(f"\n📺 {source['name']}")
        print(f"   网站: {source['website']}")
        print(f"   YouTube: {source['youtube']}")
        print(f"   描述: {source['description']}")
    
    print(f"\n🎯 推荐测试视频类型:")
    print("=" * 30)
    
    for video_type in RECOMMENDED_VIDEO_TYPES:
        print(f"\n📹 {video_type['description']} ({video_type['type']})")
        print(f"   时长: {video_type['duration']}")
        print(f"   特点: {video_type['characteristics']}")

if __name__ == "__main__":
    print_resources()
