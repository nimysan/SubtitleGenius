"""
基础字幕纠错服务实现
"""

import re
from typing import Dict, List
from .base import SubtitleCorrectionService, CorrectionInput, CorrectionOutput


class BasicCorrectionService(SubtitleCorrectionService):
    """基础字幕纠错服务实现"""
    
    def __init__(self):
        # 场景相关术语词典
        self.scene_terms = {
            "足球比赛": {
                "كرة القدم": "كرة القدم",
                "هدف": "هدف", 
                "لاعب": "لاعب",
                "مباراة": "مباراة",
                "فريق": "فريق"
            },
            "新闻播报": {
                "رئيس": "رئيس",
                "حكومة": "حكومة",
                "وزير": "وزير",
                "مؤتمر": "مؤتمر"
            }
        }
        
        # 常见拼写错误纠正
        self.spelling_fixes = {
            "اللة": "الله",
            "انشاء الله": "إن شاء الله"
        }
        
        # 标点符号纠正规则
        self.punctuation_rules = [
            (r'\s+([.!?])', r'\1'),  # 去除标点前空格
            (r'([.!?])\s*([.!?])', r'\1'),  # 去除重复标点
            (r'\s+,', ','),  # 去除逗号前空格
        ]
    
    async def correct(self, input_data: CorrectionInput) -> CorrectionOutput:
        """执行字幕纠错"""
        original = input_data.current_subtitle.strip()
        corrected = original
        has_correction = False
        correction_details = []
        
        # 1. 拼写纠错
        for wrong, right in self.spelling_fixes.items():
            if wrong in corrected:
                corrected = corrected.replace(wrong, right)
                has_correction = True
                correction_details.append(f"拼写纠正: {wrong} -> {right}")
        
        # 2. 标点符号纠错
        for pattern, replacement in self.punctuation_rules:
            new_text = re.sub(pattern, replacement, corrected)
            if new_text != corrected:
                corrected = new_text
                has_correction = True
                correction_details.append("标点符号纠正")
        
        # 3. 场景相关纠错
        scene_corrections = self._apply_scene_correction(
            corrected, input_data.scene_description
        )
        if scene_corrections != corrected:
            corrected = scene_corrections
            has_correction = True
            correction_details.append("场景术语纠正")
        
        # 4. 基于历史的一致性纠错
        if input_data.history_subtitles:
            consistency_corrections = self._apply_consistency_correction(
                corrected, input_data.history_subtitles
            )
            if consistency_corrections != corrected:
                corrected = consistency_corrections
                has_correction = True
                correction_details.append("一致性纠正")
        
        # 计算置信度
        confidence = self._calculate_confidence(original, corrected, has_correction)
        
        return CorrectionOutput(
            corrected_subtitle=corrected,
            has_correction=has_correction,
            confidence=confidence,
            correction_details="; ".join(correction_details) if correction_details else None
        )
    
    def _apply_scene_correction(self, text: str, scene: str) -> str:
        """应用场景相关纠错"""
        terms = self.scene_terms.get(scene, {})
        corrected = text
        for term, standard_form in terms.items():
            # 这里可以添加更复杂的术语标准化逻辑
            pass
        return corrected
    
    def _apply_consistency_correction(self, text: str, history: List[str]) -> str:
        """应用基于历史的一致性纠错"""
        corrected = text
        # 简单示例：确保人名、地名等专有名词的一致性
        return corrected
    
    def _calculate_confidence(self, original: str, corrected: str, has_correction: bool) -> float:
        """计算纠错置信度"""
        if not has_correction:
            return 1.0
        
        # 基于编辑距离计算相似度
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, original, corrected).ratio()
        
        # 相似度越高，置信度越高
        if similarity > 0.9:
            return 0.95
        elif similarity > 0.8:
            return 0.85
        elif similarity > 0.7:
            return 0.75
        else:
            return 0.6
