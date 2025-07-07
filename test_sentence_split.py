#!/usr/bin/env python3
"""
测试BedrockCorrectionService的长句拆分功能
"""

import asyncio
import json
from subtitle_genius.correction import BedrockCorrectionService, CorrectionInput

async def test_sentence_split():
    """测试长句拆分功能"""
    
    # 创建BedrockCorrectionService实例
    correction_service = BedrockCorrectionService(
        model_id="us.anthropic.claude-3-5-sonnet-20240620-v1:0"
    )
    
    # 测试长句
    long_sentence = """这是北京时间凌晨4点开球的西甲联赛第28轮比赛,巴塞罗那队将在主场诺坎普球场迎战皇家马德里队。巴塞罗那队目前在联赛中排名第一,积分65分,而皇家马德里队排在第二位,积分64分,落后一分。"""
    
    # 创建纠错输入
    correction_input = CorrectionInput(
        current_subtitle=long_sentence,
        history_subtitles=[],
        scene_description="足球比赛",
        language="zh"
    )
    
    print("原始字幕:", long_sentence)
    print("字幕长度:", len(long_sentence))
    print("-" * 80)
    
    # 执行纠错和拆分
    try:
        result = await correction_service.correct(correction_input)
        
        print("纠错后字幕:", result.corrected_subtitle)
        print("是否进行了纠错:", result.has_correction)
        print("纠错置信度:", result.confidence)
        print("纠错详情:", result.correction_details)
        print("-" * 80)
        
        print("是否进行了拆分:", result.has_split)
        if result.has_split and result.split_subtitles:
            print(f"拆分为 {len(result.split_subtitles)} 个子句:")
            for i, subtitle in enumerate(result.split_subtitles, 1):
                print(f"  {i}. {subtitle}")
                print(f"     长度: {len(subtitle)}")
        else:
            print("未进行拆分")
            
    except Exception as e:
        print(f"测试过程中出错: {e}")

if __name__ == "__main__":
    asyncio.run(test_sentence_split())
