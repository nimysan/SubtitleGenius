#!/usr/bin/env python3
"""
VTT字幕处理流水线
从VTT文件提取字幕 -> Correction纠错 -> Translation翻译 -> 保存新VTT
"""

import asyncio
import re
import sys
import os
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from subtitle_genius.correction import (
    BedrockCorrectionService,
    CorrectionInput
)
from subtitle_genius.translation import (
    BedrockTranslator,
    TranslationInput
)


@dataclass
class SubtitleEntry:
    """字幕条目"""
    start_time: str
    end_time: str
    arabic_text: str
    chinese_text: str = ""
    corrected_arabic: str = ""
    final_translation: str = ""


class VTTProcessor:
    """VTT字幕处理器"""
    
    def __init__(self):
        self.corrector = BedrockCorrectionService()
        self.translator = BedrockTranslator()
        self.history_subtitles = []
    
    def parse_vtt(self, vtt_path: str) -> List[SubtitleEntry]:
        """解析VTT文件"""
        print(f"📖 解析VTT文件: {vtt_path}")
        
        with open(vtt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        entries = []
        
        # 分割字幕块
        blocks = content.split('\n\n')
        
        for block in blocks:
            if not block.strip() or block.strip() == 'WEBVTT':
                continue
            
            lines = block.strip().split('\n')
            if len(lines) < 2:
                continue
            
            # 解析时间戳
            time_line = lines[0]
            time_match = re.match(r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})', time_line)
            
            if not time_match:
                continue
            
            start_time = time_match.group(1)
            end_time = time_match.group(2)
            
            # 提取阿拉伯语和中文文本
            arabic_text = ""
            chinese_text = ""
            
            for i in range(1, len(lines)):
                line = lines[i].strip()
                if not line:
                    continue
                
                # 检查是否包含阿拉伯语字符
                if re.search(r'[\u0600-\u06FF]', line):
                    arabic_text = line
                elif arabic_text and not chinese_text:
                    # 如果已经有阿拉伯语，下一行可能是中文
                    chinese_text = line
                    break
            
            if arabic_text:
                entry = SubtitleEntry(
                    start_time=start_time,
                    end_time=end_time,
                    arabic_text=arabic_text,
                    chinese_text=chinese_text
                )
                entries.append(entry)
        
        print(f"✅ 解析完成，共 {len(entries)} 条字幕")
        return entries
    
    async def process_correction(self, entries: List[SubtitleEntry]) -> List[SubtitleEntry]:
        """处理字幕纠错"""
        print(f"✏️  开始字幕纠错处理...")
        
        for i, entry in enumerate(entries, 1):
            print(f"纠错 {i}/{len(entries)}: {entry.arabic_text[:50]}...")
            
            # 创建纠错输入
            correction_input = CorrectionInput(
                current_subtitle=entry.arabic_text,
                history_subtitles=self.history_subtitles.copy(),
                scene_description="足球比赛解说",  # 根据内容判断是足球相关
                language="ar"
            )
            
            try:
                # 执行纠错
                correction_result = await self.corrector.correct(correction_input)
                entry.corrected_arabic = correction_result.corrected_subtitle
                
                if correction_result.has_correction:
                    print(f"  ✏️  已纠错: {entry.arabic_text} -> {entry.corrected_arabic}")
                else:
                    print(f"  ✅ 无需纠错")
                
                # 更新历史
                self.history_subtitles.append(entry.arabic_text)
                if len(self.history_subtitles) > 5:  # 保持最近5条
                    self.history_subtitles.pop(0)
                
            except Exception as e:
                print(f"  ❌ 纠错失败: {e}")
                entry.corrected_arabic = entry.arabic_text
            
            # 避免过快调用API
            await asyncio.sleep(0.1)
        
        print(f"✅ 纠错处理完成")
        return entries
    
    async def process_translation(self, entries: List[SubtitleEntry]) -> List[SubtitleEntry]:
        """处理字幕翻译"""
        print(f"🌐 开始字幕翻译处理...")
        
        for i, entry in enumerate(entries, 1):
            print(f"翻译 {i}/{len(entries)}: {entry.corrected_arabic[:50]}...")
            
            # 创建翻译输入
            translation_input = TranslationInput(
                text=entry.corrected_arabic,
                source_language="ar",
                target_language="zh",
                context="足球比赛解说"
            )
            
            try:
                # 执行翻译
                translation_result = await self.translator.translate(translation_input)
                entry.final_translation = translation_result.translated_text
                
                print(f"  🌐 翻译: {entry.final_translation}")
                
            except Exception as e:
                print(f"  ❌ 翻译失败: {e}")
                entry.final_translation = entry.chinese_text or "[翻译失败]"
            
            # 避免过快调用API
            await asyncio.sleep(0.1)
        
        print(f"✅ 翻译处理完成")
        return entries
    
    def save_vtt(self, entries: List[SubtitleEntry], output_path: str):
        """保存处理后的VTT文件"""
        print(f"💾 保存VTT文件: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            
            for entry in entries:
                f.write(f"{entry.start_time} --> {entry.end_time}\n")
                f.write(f"{entry.corrected_arabic}\n")
                f.write(f"{entry.final_translation}\n\n")
        
        print(f"✅ VTT文件保存完成")
    
    def save_comparison_report(self, entries: List[SubtitleEntry], report_path: str):
        """保存对比报告"""
        print(f"📊 保存对比报告: {report_path}")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 字幕处理对比报告\n\n")
            f.write("| 序号 | 时间戳 | 原始阿拉伯语 | 纠错后阿拉伯语 | 原始中文 | 新翻译 |\n")
            f.write("|------|--------|--------------|----------------|----------|--------|\n")
            
            for i, entry in enumerate(entries, 1):
                f.write(f"| {i} | {entry.start_time} | {entry.arabic_text} | {entry.corrected_arabic} | {entry.chinese_text} | {entry.final_translation} |\n")
        
        print(f"✅ 对比报告保存完成")


async def main():
    """主处理函数"""
    print("🚀 VTT字幕处理流水线")
    print("=" * 60)
    print("流程: VTT解析 -> Correction纠错 -> Translation翻译 -> 保存VTT")
    
    # 文件路径
    input_vtt = "/Users/yexw/PycharmProjects/SubtitleGenius/subtitles/samples/chunk_6_auto.vtt"
    output_vtt = "/Users/yexw/PycharmProjects/SubtitleGenius/subtitles/samples/chunk_6_auto_corrected.vtt"
    report_path = "/Users/yexw/PycharmProjects/SubtitleGenius/subtitles/samples/chunk_6_processing_report.md"
    
    # 检查输入文件
    if not os.path.exists(input_vtt):
        print(f"❌ 输入文件不存在: {input_vtt}")
        return
    
    try:
        # 创建处理器
        processor = VTTProcessor()
        
        # 1. 解析VTT文件
        entries = processor.parse_vtt(input_vtt)
        
        if not entries:
            print("❌ 没有找到有效的字幕条目")
            return
        
        print(f"\n📋 处理概览:")
        print(f"   输入文件: {input_vtt}")
        print(f"   输出文件: {output_vtt}")
        print(f"   字幕条目: {len(entries)} 条")
        print(f"   处理流程: 解析 -> 纠错 -> 翻译 -> 保存")
        
        # 2. 字幕纠错
        print(f"\n" + "="*50)
        entries = await processor.process_correction(entries)
        
        # 3. 字幕翻译
        print(f"\n" + "="*50)
        entries = await processor.process_translation(entries)
        
        # 4. 保存结果
        print(f"\n" + "="*50)
        processor.save_vtt(entries, output_vtt)
        processor.save_comparison_report(entries, report_path)
        
        # 5. 统计信息
        corrected_count = sum(1 for e in entries if e.corrected_arabic != e.arabic_text)
        
        print(f"\n🎯 处理完成!")
        print(f"   总条目数: {len(entries)}")
        print(f"   纠错条目: {corrected_count}")
        print(f"   纠错率: {corrected_count/len(entries)*100:.1f}%")
        print(f"   输出文件: {output_vtt}")
        print(f"   对比报告: {report_path}")
        
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
