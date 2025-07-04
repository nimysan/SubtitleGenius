#!/usr/bin/env python3
"""
翻译模块演示
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from subtitle_genius.translation import (
    BedrockTranslator,
    GoogleTranslator,
    OpenAITranslator,
    TranslationInput,
    translate_text,
    batch_translate
)


async def demo_bedrock_translator():
    """演示Bedrock翻译服务"""
    print("🤖 Bedrock翻译服务演示")
    print("=" * 50)
    
    translator = BedrockTranslator()
    print(f"服务名称: {translator.get_service_name()}")
    
    # 测试案例
    test_cases = [
        {
            "text": "مرحبا بكم في مباراة اليوم",
            "source": "ar",
            "target": "zh",
            "context": "足球比赛开场"
        },
        {
            "text": "الرئيس يلتقي بالوزراء",
            "source": "ar", 
            "target": "en",
            "context": "政治新闻"
        },
        {
            "text": "شكرا لكم جميعا",
            "source": "ar",
            "target": "zh",
            "context": "感谢语"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- 测试 {i} ---")
        
        input_data = TranslationInput(
            text=case["text"],
            source_language=case["source"],
            target_language=case["target"],
            context=case["context"]
        )
        
        result = await translator.translate(input_data)
        
        print(f"原文: {result.original_text}")
        print(f"译文: {result.translated_text}")
        print(f"语言: {result.source_language} -> {result.target_language}")
        print(f"置信度: {result.confidence:.2f}")
        print(f"上下文: {case['context']}")


async def demo_google_translator():
    """演示Google翻译服务"""
    print("\n🌐 Google翻译服务演示")
    print("=" * 50)
    
    translator = GoogleTranslator()
    print(f"服务名称: {translator.get_service_name()}")
    
    # 注意：Google翻译需要网络连接，这里只演示接口
    input_data = TranslationInput(
        text="مرحبا",
        source_language="ar",
        target_language="zh"
    )
    
    try:
        result = await translator.translate(input_data)
        print(f"原文: {result.original_text}")
        print(f"译文: {result.translated_text}")
        print(f"服务: {result.service_name}")
    except Exception as e:
        print(f"Google翻译测试失败（可能需要网络连接）: {e}")


async def demo_convenience_functions():
    """演示便捷函数"""
    print("\n⚡ 便捷函数演示")
    print("=" * 50)
    
    # 单个翻译
    print("1. 单个文本翻译:")
    result = await translate_text(
        text="كرة القدم لعبة جميلة",
        source_language="ar",
        target_language="zh",
        service="bedrock"
    )
    print(f"   原文: كرة القدم لعبة جميلة")
    print(f"   译文: {result}")
    
    # 批量翻译
    print("\n2. 批量翻译:")
    texts = [
        "مرحبا بكم",
        "شكرا لكم", 
        "مع السلامة",
        "كرة القدم",
        "هدف رائع"
    ]
    
    results = await batch_translate(
        texts=texts,
        source_language="ar",
        target_language="zh",
        service="bedrock"
    )
    
    for original, translated in zip(texts, results):
        print(f"   {original} -> {translated}")


async def demo_different_languages():
    """演示不同语言翻译"""
    print("\n🌍 多语言翻译演示")
    print("=" * 50)
    
    translator = BedrockTranslator()
    
    # 阿拉伯语原文
    arabic_text = "اللاعب سجل هدف رائع في المباراة"
    
    # 翻译到不同语言
    target_languages = [
        ("zh", "中文"),
        ("en", "English"),
        ("fr", "Français"),
        ("es", "Español")
    ]
    
    print(f"原文 (Arabic): {arabic_text}")
    print("\n翻译结果:")
    
    for lang_code, lang_name in target_languages:
        input_data = TranslationInput(
            text=arabic_text,
            source_language="ar",
            target_language=lang_code,
            context="足球比赛"
        )
        
        result = await translator.translate(input_data)
        print(f"   {lang_name}: {result.translated_text}")


async def demo_translation_pipeline():
    """演示翻译流水线"""
    print("\n🔄 翻译流水线演示")
    print("=" * 50)
    print("模拟: 字幕识别 -> 纠错 -> 翻译")
    
    # 模拟字幕处理流程
    subtitle_pipeline = [
        {
            "step": "1. 语音识别",
            "text": "اللة يبارك في هذا اليوم الجميل"  # 包含拼写错误
        },
        {
            "step": "2. 纠错后",
            "text": "الله يبارك في هذا اليوم الجميل"  # 纠正后
        }
    ]
    
    for stage in subtitle_pipeline:
        print(f"\n{stage['step']}: {stage['text']}")
    
    # 翻译纠错后的文本
    print("\n3. 翻译:")
    final_translation = await translate_text(
        text=subtitle_pipeline[1]["text"],
        source_language="ar",
        target_language="zh",
        context="祝福语",
        service="bedrock"
    )
    
    print(f"   中文: {final_translation}")
    
    english_translation = await translate_text(
        text=subtitle_pipeline[1]["text"],
        source_language="ar", 
        target_language="en",
        context="祝福语",
        service="bedrock"
    )
    
    print(f"   English: {english_translation}")


async def main():
    """主演示函数"""
    print("🚀 SubtitleGenius 翻译模块演示")
    print("=" * 60)
    print("新架构: 模块化、统一接口、多服务支持")
    
    try:
        await demo_bedrock_translator()
        await demo_google_translator()
        await demo_convenience_functions()
        await demo_different_languages()
        await demo_translation_pipeline()
        
        print(f"\n🎯 演示完成!")
        print("翻译模块已重构为与correction模块相同的架构")
        
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
