"""
Amazon Bedrock翻译服务实现
"""

import os
import boto3
import json
import asyncio
from typing import Optional
from .base import TranslationService, TranslationInput, TranslationOutput


class BedrockTranslator(TranslationService):
    """Amazon Bedrock Claude翻译服务"""
    
    def __init__(self, model_id: str = "us.anthropic.claude-3-haiku-20240307-v1:0"):
        self.model_id = model_id
        self.service_name = "bedrock_claude"
        
        # 语言映射
        self.lang_map = {
            "zh": "中文",
            "en": "English",
            "ar": "Arabic", 
            "ja": "Japanese",
            "ko": "Korean",
            "fr": "French",
            "de": "German",
            "es": "Spanish",
            "ru": "Russian"
        }
        
        # 初始化Bedrock客户端
        try:
            self.bedrock_runtime = boto3.client(
                service_name="bedrock-runtime",
                region_name=os.getenv("AWS_REGION", "us-east-1")
            )
        except Exception as e:
            print(f"警告: 无法初始化Bedrock客户端: {e}")
            self.bedrock_runtime = None
    
    async def translate(self, input_data: TranslationInput) -> TranslationOutput:
        """使用Bedrock Claude翻译文本"""
        
        if not input_data.text.strip():
            return TranslationOutput(
                original_text=input_data.text,
                translated_text=input_data.text,
                source_language=input_data.source_language,
                target_language=input_data.target_language,
                confidence=1.0,
                service_name=self.get_service_name()
            )
        
        # 如果没有Bedrock客户端，使用模拟翻译
        if not self.bedrock_runtime:
            return await self._mock_translation(input_data)
        
        try:
            # 构建翻译提示词
            prompt = self._build_translation_prompt(input_data)
            
            # 构建消息序列
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
            
            # 调用Bedrock Converse API
            response = await self._call_bedrock_converse(messages)
            
            # 解析响应
            translated_text = self._parse_translation_response(response)
            
            return TranslationOutput(
                original_text=input_data.text,
                translated_text=translated_text,
                source_language=input_data.source_language,
                target_language=input_data.target_language,
                confidence=0.9,
                service_name=self.get_service_name(),
                translation_details=f"使用{self.model_id}模型翻译"
            )
            
        except Exception as e:
            print(f"Bedrock翻译失败，使用备用方案: {e}")
            return await self._mock_translation(input_data)
    
    def _build_translation_prompt(self, input_data: TranslationInput) -> str:
        """构建翻译提示词"""
        
        source_language = self.lang_map.get(input_data.source_language, input_data.source_language)
        target_language = self.lang_map.get(input_data.target_language, input_data.target_language)
        
        prompt = f"""请将以下{source_language}文本准确翻译成{target_language}。

要求:
1. 保持原文的意思和语调
2. 使用自然流畅的{target_language}表达
3. 如果是字幕文本，保持简洁易懂
4. 保留重要的专有名词"""
        
        if input_data.context:
            prompt += f"\n\n上下文信息: {input_data.context}"
        
        prompt += f"\n\n原文: {input_data.text}\n\n请直接提供翻译结果，不需要额外说明:"
        
        return prompt
    
    async def _call_bedrock_converse(self, messages: list) -> dict:
        """调用Bedrock Converse API"""
        
        # 在异步环境中调用同步的boto3客户端
        loop = asyncio.get_event_loop()
        
        def _sync_call():
            return self.bedrock_runtime.converse(
                modelId=self.model_id,
                messages=messages,
                inferenceConfig={
                    "temperature": 0.3,
                    "topP": 0.9,
                    "maxTokens": 1000
                }
            )
        
        response = await loop.run_in_executor(None, _sync_call)
        return response
    
    def _parse_translation_response(self, response: dict) -> str:
        """解析Bedrock翻译响应"""
        try:
            # 从响应中提取文本内容
            content = response['output']['message']['content'][0]['text']
            return content.strip()
            
        except Exception as e:
            print(f"解析Bedrock翻译响应失败: {e}")
            return "翻译失败"
    
    async def _mock_translation(self, input_data: TranslationInput) -> TranslationOutput:
        """模拟翻译（备用方案）"""
        
        # 简单的模拟翻译规则
        mock_translations = {
            # 阿拉伯语到中文
            ("ar", "zh"): {
                "مرحبا": "你好",
                "شكرا": "谢谢", 
                "نعم": "是的",
                "لا": "不",
                "مع السلامة": "再见",
                "كرة القدم": "足球",
                "هدف": "进球",
                "لاعب": "球员",
                "مباراة": "比赛",
                "الرئيس": "总统",
                "حكومة": "政府"
            },
            # 阿拉伯语到英语
            ("ar", "en"): {
                "مرحبا": "Hello",
                "شكرا": "Thank you",
                "نعم": "Yes", 
                "لا": "No",
                "مع السلامة": "Goodbye",
                "كرة القدم": "Football",
                "هدف": "Goal",
                "لاعب": "Player",
                "مباراة": "Match"
            }
        }
        
        # 模拟处理延迟
        await asyncio.sleep(0.1)
        
        # 获取对应的翻译字典
        translation_dict = mock_translations.get(
            (input_data.source_language, input_data.target_language), {}
        )
        
        # 简单的词汇替换翻译
        translated_text = input_data.text
        for arabic, translation in translation_dict.items():
            translated_text = translated_text.replace(arabic, translation)
        
        # 如果没有找到翻译，返回原文加标记
        if translated_text == input_data.text:
            translated_text = f"[模拟翻译: {input_data.text}]"
        
        return TranslationOutput(
            original_text=input_data.text,
            translated_text=translated_text,
            source_language=input_data.source_language,
            target_language=input_data.target_language,
            confidence=0.7,
            service_name=self.get_service_name(),
            translation_details="模拟Bedrock翻译"
        )
    
    def get_service_name(self) -> str:
        return f"BedrockTranslator({self.model_id})"
