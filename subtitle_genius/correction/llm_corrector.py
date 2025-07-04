"""
基于LLM的字幕纠错服务实现
"""

import json
import asyncio
from typing import Optional, Dict, Any
from .base import SubtitleCorrectionService, CorrectionInput, CorrectionOutput


class LLMCorrectionService(SubtitleCorrectionService):
    """基于LLM的字幕纠错服务"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.client = None
        
        # 如果有API key，初始化客户端
        if api_key:
            try:
                import openai
                self.client = openai.AsyncOpenAI(api_key=api_key)
            except ImportError:
                print("警告: 未安装openai库，将使用模拟模式")
    
    async def correct(self, input_data: CorrectionInput) -> CorrectionOutput:
        """使用LLM进行字幕纠错"""
        
        # 如果没有配置LLM客户端，使用模拟纠错
        if not self.client:
            return await self._mock_llm_correction(input_data)
        
        try:
            # 构建提示词
            prompt = self._build_prompt(input_data)
            
            # 调用LLM
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的阿拉伯语字幕纠错助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            # 解析响应
            result = self._parse_llm_response(response.choices[0].message.content)
            
            return CorrectionOutput(
                corrected_subtitle=result.get("corrected_text", input_data.current_subtitle),
                has_correction=result.get("has_correction", False),
                confidence=result.get("confidence", 0.8),
                correction_details=result.get("details", None)
            )
            
        except Exception as e:
            print(f"LLM纠错失败，使用备用方案: {e}")
            return await self._mock_llm_correction(input_data)
    
    def _build_prompt(self, input_data: CorrectionInput) -> str:
        """构建LLM提示词"""
        
        history_context = ""
        if input_data.history_subtitles:
            recent_history = input_data.history_subtitles[-3:]  # 最近3条
            history_context = f"\n历史字幕上下文:\n" + "\n".join(f"- {h}" for h in recent_history)
        
        prompt = f"""
请纠正以下阿拉伯语字幕中的错误。

场景: {input_data.scene_description}
当前字幕: {input_data.current_subtitle}{history_context}

请检查并纠正以下类型的错误:
1. 拼写错误
2. 语法错误  
3. 标点符号错误
4. 术语标准化
5. 上下文一致性

请以JSON格式返回结果:
{{
    "corrected_text": "纠正后的字幕",
    "has_correction": true/false,
    "confidence": 0.0-1.0,
    "details": "纠正说明"
}}
"""
        return prompt
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """解析LLM响应"""
        try:
            # 尝试解析JSON
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            # 如果不是JSON格式，尝试简单解析
            return {
                "corrected_text": response.strip(),
                "has_correction": True,
                "confidence": 0.7,
                "details": "LLM纠错"
            }
    
    async def _mock_llm_correction(self, input_data: CorrectionInput) -> CorrectionOutput:
        """模拟LLM纠错（用于测试和备用）"""
        
        # 模拟一些智能纠错规则
        original = input_data.current_subtitle.strip()
        corrected = original
        has_correction = False
        details = []
        
        # 模拟LLM的智能纠错
        llm_rules = {
            # 更智能的拼写纠正
            "اللة": "الله",
            "انشاء الله": "إن شاء الله",
            "مع السلامة": "مع السلامة",
            
            # 场景相关的智能纠正
            "كورة": "كرة" if "足球" in input_data.scene_description else "كورة",
            "جول": "هدف" if "足球" in input_data.scene_description else "جول",
        }
        
        for wrong, right in llm_rules.items():
            if wrong in corrected:
                corrected = corrected.replace(wrong, right)
                has_correction = True
                details.append(f"智能纠正: {wrong} -> {right}")
        
        # 模拟基于上下文的智能纠正
        if input_data.history_subtitles and "足球" in input_data.scene_description:
            # 检查球队名称一致性
            for hist in input_data.history_subtitles[-2:]:
                if "الفريق الأول" in hist and "فريق اول" in corrected:
                    corrected = corrected.replace("فريق اول", "الفريق الأول")
                    has_correction = True
                    details.append("上下文一致性纠正")
        
        # 模拟处理延迟
        await asyncio.sleep(0.1)
        
        confidence = 0.9 if has_correction else 1.0
        
        return CorrectionOutput(
            corrected_subtitle=corrected,
            has_correction=has_correction,
            confidence=confidence,
            correction_details="; ".join(details) if details else "模拟LLM纠错"
        )
