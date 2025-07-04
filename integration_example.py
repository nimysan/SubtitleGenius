#!/usr/bin/env python3
"""
字幕纠错服务集成示例

展示如何在 transcribe -> translate 流程中集成纠错服务
"""

import asyncio
import sys
import os
from pathlib import Path

# 导入纠错服务
correction_file = Path(__file__).parent / "subtitle_genius" / "correction_service.py"
sys.path.insert(0, str(correction_file.parent))
exec(open(correction_file).read())


class SubtitleProcessor:
    """字幕处理器 - 集成纠错服务"""
    
    def __init__(self):
        self.correction_service = BasicCorrectionService()
        self.subtitle_history = []  # 存储同一视频的历史字幕
    
    async def mock_transcribe(self, audio_segment):
        """模拟 Whisper transcribe"""
        # 这里模拟从音频转录出的字幕（可能包含错误）
        mock_transcriptions = [
            "مرحبا بكم في مباراة اليوم",
            "اللة يبارك في هذا اليوم الجميل",  # 包含拼写错误
            "اللاعب يسجل هدف رائع",
            "الفريق يلعب بشكل ممتاز .",  # 包含标点错误
            "المباراة انتهت بنتيجة جيدة"
        ]
        return mock_transcriptions[audio_segment % len(mock_transcriptions)]
    
    async def mock_translate(self, arabic_text):
        """模拟翻译服务"""
        # 简单的模拟翻译
        translations = {
            "مرحبا بكم في مباراة اليوم": "Welcome to today's match",
            "الله يبارك في هذا اليوم الجميل": "God bless this beautiful day",
            "اللاعب يسجل هدف رائع": "The player scores a wonderful goal",
            "الفريق يلعب بشكل ممتاز.": "The team plays excellently.",
            "المباراة انتهت بنتيجة جيدة": "The match ended with a good result"
        }
        return translations.get(arabic_text, f"[Translation of: {arabic_text}]")
    
    async def process_subtitle(self, audio_segment, scene_description="足球比赛"):
        """
        完整的字幕处理流程：transcribe -> 纠错 -> translate
        """
        print(f"\n--- 处理音频片段 {audio_segment + 1} ---")
        
        # 1. Transcribe (现有流程)
        transcribed_text = await self.mock_transcribe(audio_segment)
        print(f"1. Transcribe: {transcribed_text}")
        
        # 2. 纠错 (新增服务)
        correction_input = CorrectionInput(
            current_subtitle=transcribed_text,
            history_subtitles=self.subtitle_history.copy(),
            scene_description=scene_description
        )
        
        correction_result = await self.correction_service.correct(correction_input)
        corrected_text = correction_result.corrected_subtitle
        
        print(f"2. 纠错: {corrected_text}")
        if correction_result.has_correction:
            print(f"   ✏️  已纠正 (置信度: {correction_result.confidence:.2f})")
        else:
            print(f"   ✅ 无需纠正")
        
        # 3. Translate (现有流程)
        translated_text = await self.mock_translate(corrected_text)
        print(f"3. Translate: {translated_text}")
        
        # 4. 更新历史记录
        self.subtitle_history.append(transcribed_text)
        
        return {
            "original": transcribed_text,
            "corrected": corrected_text,
            "translated": translated_text,
            "has_correction": correction_result.has_correction
        }
    
    def clear_history(self):
        """清空历史记录（新视频开始时调用）"""
        self.subtitle_history.clear()


async def demo_integration():
    """演示集成效果"""
    print("🎬 字幕处理流程集成演示")
    print("=" * 60)
    print("流程: Audio -> Transcribe -> 纠错 -> Translate")
    
    processor = SubtitleProcessor()
    
    # 模拟处理一个足球比赛视频的5个音频片段
    results = []
    for i in range(5):
        result = await processor.process_subtitle(i, "足球比赛")
        results.append(result)
    
    # 统计纠错效果
    print(f"\n📊 处理统计:")
    total_segments = len(results)
    corrected_segments = sum(1 for r in results if r["has_correction"])
    
    print(f"   总片段数: {total_segments}")
    print(f"   纠错片段数: {corrected_segments}")
    print(f"   纠错率: {corrected_segments/total_segments*100:.1f}%")
    
    print(f"\n📋 最终结果:")
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result['translated']}")
        if result["has_correction"]:
            print(f"      (已纠错: {result['original']} -> {result['corrected']})")


async def demo_different_scenes():
    """演示不同场景的处理"""
    print(f"\n🎭 不同场景处理演示")
    print("=" * 60)
    
    processor = SubtitleProcessor()
    
    scenes = [
        {
            "name": "足球比赛",
            "subtitle": "اللة يبارك في هذا الفوز"
        },
        {
            "name": "新闻播报", 
            "subtitle": "الرئيس يتحدث في المؤتمر ."
        },
        {
            "name": "通用场景",
            "subtitle": "مرحبا وأهلا بكم جميعا"
        }
    ]
    
    for scene in scenes:
        print(f"\n场景: {scene['name']}")
        
        correction_input = CorrectionInput(
            current_subtitle=scene["subtitle"],
            history_subtitles=[],
            scene_description=scene["name"]
        )
        
        result = await processor.correction_service.correct(correction_input)
        translation = await processor.mock_translate(result.corrected_subtitle)
        
        print(f"   原始: {scene['subtitle']}")
        print(f"   纠错: {result.corrected_subtitle}")
        print(f"   翻译: {translation}")
        if result.has_correction:
            print(f"   状态: ✏️  已纠正")
        else:
            print(f"   状态: ✅ 无需纠正")


async def main():
    """主演示函数"""
    await demo_integration()
    await demo_different_scenes()
    
    print(f"\n🎯 集成完成!")
    print("现在你可以在现有的字幕处理流程中加入纠错服务了")


if __name__ == "__main__":
    asyncio.run(main())
