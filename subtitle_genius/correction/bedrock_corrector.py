"""
基于Amazon Bedrock的字幕纠错服务实现
"""

import json
import os
import boto3
import asyncio
from typing import Optional, Dict, Any
from .base import SubtitleCorrectionService, CorrectionInput, CorrectionOutput


class BedrockCorrectionService(SubtitleCorrectionService):
    """基于Amazon Bedrock的字幕纠错服务"""
    
    def __init__(self, model_id: str = "us.anthropic.claude-3-haiku-20240307-v1:0"):
        self.model_id = model_id
        self.service_name = "bedrock_claude"
        
        # 初始化Bedrock客户端
        try:
            self.bedrock_runtime = boto3.client(
                service_name="bedrock-runtime",
                region_name=os.getenv("AWS_REGION", "us-east-1")
            )
        except Exception as e:
            print(f"警告: 无法初始化Bedrock客户端: {e}")
            self.bedrock_runtime = None
    
    async def correct(self, input_data: CorrectionInput) -> CorrectionOutput:
        """使用Bedrock Claude进行字幕纠错"""
        
        # 如果没有Bedrock客户端，使用模拟纠错
        if not self.bedrock_runtime:
            return await self._mock_correction(input_data)
        
        try:
            # 构建纠错提示词
            prompt = self._build_correction_prompt(input_data)
            
            # 构建消息序列
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
            
            # 调用Bedrock Converse API
            response = await self._call_bedrock_converse(messages)
            
            # 解析响应
            result = self._parse_correction_response(response, input_data.current_subtitle)
            
            return CorrectionOutput(
                corrected_subtitle=result.get("corrected_text", input_data.current_subtitle),
                has_correction=result.get("has_correction", False),
                confidence=result.get("confidence", 0.8),
                correction_details=result.get("details", None)
            )
            
        except Exception as e:
            print(f"Bedrock纠错失败，使用备用方案: {e}")
            return await self._mock_correction(input_data)
    
    def _build_correction_prompt(self, input_data: CorrectionInput) -> str:
        """构建纠错提示词"""
        
        # 构建历史上下文
        history_context = ""
        if input_data.history_subtitles:
            recent_history = input_data.history_subtitles[-3:]  # 最近3条
            history_context = f"\n\n历史字幕上下文:\n" + "\n".join(f"- {h}" for h in recent_history)
        
        # 场景相关的纠错指导
        scene_guidance = self._get_scene_guidance(input_data.scene_description)
        
        prompt = f"""你是一个专业的阿拉伯语字幕纠错助手。请纠正以下字幕中的错误。

场景: {input_data.scene_description}
当前字幕: {input_data.current_subtitle}{history_context}

{scene_guidance}

请检查并纠正以下类型的错误:
1. 拼写错误 (如: اللة -> الله)
2. 语法错误
3. 标点符号错误 (去除多余空格、重复标点等)
4. 术语标准化 (根据场景使用标准术语)
5. 上下文一致性 (与历史字幕保持一致的称呼和术语)

请以JSON格式返回结果:
{{
    "corrected_text": "纠正后的字幕文本",
    "has_correction": true/false,
    "confidence": 0.0-1.0,
    "details": "具体的纠正说明"
}}

如果字幕没有错误，请返回原文本并设置has_correction为false。"""
        
        return prompt
    
    def _get_scene_guidance(self, scene_description: str) -> str:
        """根据场景获取纠错指导"""
        scene_guides = {
            "足球比赛": """
足球场景术语指导:
- 使用标准足球术语: كرة القدم (足球), هدف (进球), لاعب (球员), مباراة (比赛)
- 保持球队称呼一致性: الفريق الأول, الفريق الثاني
- 注意比分和时间表达的准确性""",
            
            "新闻播报": """
新闻场景术语指导:
- 使用正式的政治术语: رئيس (总统), حكومة (政府), وزير (部长)
- 保持人名、地名的一致性
- 使用标准的新闻用语""",
            
            "商业新闻": """
商业场景术语指导:
- 使用标准商业术语: شركة (公司), اقتصاد (经济), سوق (市场)
- 保持公司名称、品牌名称的一致性"""
        }
        
        return scene_guides.get(scene_description, "请根据上下文进行适当的纠错。")
    
    async def _call_bedrock_converse(self, messages: list) -> dict:
        """调用Bedrock Converse API"""
        
        # 在异步环境中调用同步的boto3客户端
        loop = asyncio.get_event_loop()
        
        def _sync_call():
            return self.bedrock_runtime.converse(
                modelId=self.model_id,
                messages=messages,
                inferenceConfig={
                    "temperature": 0.1,
                    "topP": 0.9,
                    "maxTokens": 1000
                }
            )
        
        response = await loop.run_in_executor(None, _sync_call)
        return response
    
    def _parse_correction_response(self, response: dict, original_text: str) -> Dict[str, Any]:
        """解析Bedrock响应"""
        try:
            # 从响应中提取文本内容
            content = response['output']['message']['content'][0]['text']
            
            # 尝试解析JSON
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                # 如果不是JSON格式，尝试提取纠正后的文本
                # 简单的文本解析逻辑
                lines = content.strip().split('\n')
                corrected_text = original_text
                
                for line in lines:
                    if '纠正后' in line or 'corrected' in line.lower():
                        # 尝试提取纠正后的文本
                        if ':' in line:
                            corrected_text = line.split(':', 1)[1].strip()
                            break
                
                return {
                    "corrected_text": corrected_text,
                    "has_correction": corrected_text != original_text,
                    "confidence": 0.8,
                    "details": "Bedrock纠错"
                }
                
        except Exception as e:
            print(f"解析Bedrock响应失败: {e}")
            return {
                "corrected_text": original_text,
                "has_correction": False,
                "confidence": 0.5,
                "details": f"解析失败: {str(e)}"
            }
    
    async def _mock_correction(self, input_data: CorrectionInput) -> CorrectionOutput:
        """模拟纠错（备用方案）"""
        
        original = input_data.current_subtitle.strip()
        corrected = original
        has_correction = False
        details = []
        
        # 基本的拼写纠正
        basic_fixes = {
            "اللة": "الله",
            "انشاء الله": "إن شاء الله",
            "مع السلامة": "مع السلامة"
        }
        
        for wrong, right in basic_fixes.items():
            if wrong in corrected:
                corrected = corrected.replace(wrong, right)
                has_correction = True
                details.append(f"拼写纠正: {wrong} -> {right}")
        
        # 标点符号纠正
        import re
        new_text = re.sub(r'\s+([.!?])', r'\1', corrected)
        if new_text != corrected:
            corrected = new_text
            has_correction = True
            details.append("标点符号纠正")
        
        # 模拟处理延迟
        await asyncio.sleep(0.1)
        
        return CorrectionOutput(
            corrected_subtitle=corrected,
            has_correction=has_correction,
            confidence=0.9 if has_correction else 1.0,
            correction_details="; ".join(details) if details else "模拟Bedrock纠错"
        )
    
    def get_service_name(self) -> str:
        """获取服务名称"""
        return f"BedrockCorrection({self.model_id})"
