"""
基于LLM的多语言字幕纠错服务实现
"""

import json
import asyncio
from typing import Optional, Dict, Any
from .base import SubtitleCorrectionService, CorrectionInput, CorrectionOutput


class LLMCorrectionService(SubtitleCorrectionService):
    """基于LLM的多语言字幕纠错服务"""
    
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
    
    # 语言配置映射
    LANGUAGE_CONFIGS = {
        "ar": {
            "name": "阿拉伯语",
            "system_prompt": "你是一个专业的阿拉伯语字幕纠错助手。",
            "smart_fixes": {
                "اللة": "الله",
                "انشاء الله": "إن شاء الله",
                "كورة": "كرة",
                "جول": "هدف"
            }
        },
        "zh": {
            "name": "中文",
            "system_prompt": "你是一个专业的中文字幕纠错助手。",
            "smart_fixes": {
                "的的": "的",
                "了了": "了",
                "进球了": "进球",
                "射门了": "射门"
            }
        },
        "en": {
            "name": "English",
            "system_prompt": "You are a professional English subtitle correction assistant.",
            "smart_fixes": {
                "  ": " ",
                "goal goal": "goal",
                "shoot shoot": "shoot"
            }
        },
        "es": {
            "name": "Español",
            "system_prompt": "Eres un asistente profesional de corrección de subtítulos en español.",
            "smart_fixes": {
                "  ": " ",
                "gol gol": "gol",
                "tiro tiro": "tiro"
            }
        }
    }
    
    async def correct(self, input_data: CorrectionInput) -> CorrectionOutput:
        """使用LLM进行多语言字幕纠错"""
        
        # 如果没有配置LLM客户端，使用模拟纠错
        if not self.client:
            return await self._mock_llm_correction(input_data)
        
        try:
            # 获取语言配置
            lang_config = self.LANGUAGE_CONFIGS.get(input_data.language, self.LANGUAGE_CONFIGS["ar"])
            
            # 构建提示词
            prompt = self._build_prompt(input_data, lang_config)
            
            # 调用LLM
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": lang_config["system_prompt"]},
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
    
    def _build_prompt(self, input_data: CorrectionInput, lang_config: Dict[str, Any]) -> str:
        """构建多语言LLM提示词"""
        
        language_name = lang_config["name"]
        
        history_context = ""
        if input_data.history_subtitles:
            recent_history = input_data.history_subtitles[-3:]  # 最近3条
            history_context = f"\n历史字幕上下文:\n" + "\n".join(f"- {h}" for h in recent_history)
        
        prompt = f"""
请纠正以下{language_name}字幕中的错误。

场景: {input_data.scene_description}
当前字幕: {input_data.current_subtitle}{history_context}

请检查并纠正以下类型的错误:
1. 拼写错误
2. 语法错误  
3. 标点符号错误
4. 术语标准化 (根据场景使用标准术语)
5. 上下文一致性 (与历史字幕保持一致)

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
        """多语言模拟LLM纠错（用于测试和备用）"""
        
        # 获取语言配置
        lang_config = self.LANGUAGE_CONFIGS.get(input_data.language, self.LANGUAGE_CONFIGS["ar"])
        
        # 模拟一些智能纠错规则
        original = input_data.current_subtitle.strip()
        corrected = original
        has_correction = False
        details = []
        
        # 应用语言特定的智能纠错规则
        smart_rules = lang_config["smart_fixes"]
        
        for wrong, right in smart_rules.items():
            if wrong in corrected:
                # 场景相关的智能纠正
                if input_data.scene_description == "足球比赛":
                    if wrong == "كورة" and input_data.language == "ar":
                        right = "كرة"
                    elif wrong == "جول" and input_data.language == "ar":
                        right = "هدف"
                
                corrected = corrected.replace(wrong, right)
                has_correction = True
                details.append(f"智能纠正: {wrong} -> {right}")
        
        # 模拟基于上下文的智能纠正
        if input_data.history_subtitles and "足球" in input_data.scene_description:
            # 检查术语一致性
            for hist in input_data.history_subtitles[-2:]:
                if input_data.language == "ar":
                    if "الفريق الأول" in hist and "فريق اول" in corrected:
                        corrected = corrected.replace("فريق اول", "الفريق الأول")
                        has_correction = True
                        details.append("上下文一致性纠正")
                elif input_data.language == "zh":
                    if "主队" in hist and "主场队伍" in corrected:
                        corrected = corrected.replace("主场队伍", "主队")
                        has_correction = True
                        details.append("上下文一致性纠正")
        
        # 通用标点符号纠正
        import re
        new_text = re.sub(r'\s+', ' ', corrected.strip())  # 合并多个空格
        if new_text != corrected:
            corrected = new_text
            has_correction = True
            details.append("标点符号纠正")
        
        # 模拟处理延迟
        await asyncio.sleep(0.1)
        
        confidence = 0.9 if has_correction else 1.0
        
        return CorrectionOutput(
            corrected_subtitle=corrected,
            has_correction=has_correction,
            confidence=confidence,
            correction_details="; ".join(details) if details else f"模拟{lang_config['name']}LLM纠错"
        )
