#!/usr/bin/env python3
"""
翻译服务模块
支持多种翻译 API: OpenAI GPT, Google Translate, 百度翻译等
"""

import os
import asyncio
import aiohttp
import json
import boto3
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class TranslationResult:
    """翻译结果"""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    service: str
    confidence: float = 0.0

class TranslationService:
    """翻译服务基类"""
    
    def __init__(self):
        self.name = "base"
    
    async def translate(self, text: str, target_lang: str = "zh", source_lang: str = "auto") -> TranslationResult:
        """翻译文本"""
        raise NotImplementedError

class OpenAITranslator(TranslationService):
    """OpenAI GPT 翻译服务"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.name = "openai"
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1/chat/completions"
    
    async def translate(self, text: str, target_lang: str = "zh", source_lang: str = "auto") -> TranslationResult:
        """使用 OpenAI GPT 翻译"""
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        # 语言映射
        lang_map = {
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
        
        target_language = lang_map.get(target_lang, "中文")
        
        prompt = f"""请将以下文本翻译为{target_language}，只返回翻译结果，不要添加任何解释：

{text}"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.3
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        translated_text = result["choices"][0]["message"]["content"].strip()
                        
                        return TranslationResult(
                            original_text=text,
                            translated_text=translated_text,
                            source_language=source_lang,
                            target_language=target_lang,
                            service=self.name,
                            confidence=0.9
                        )
                    else:
                        error_text = await response.text()
                        raise Exception(f"OpenAI API error: {response.status} - {error_text}")
        
        except Exception as e:
            raise Exception(f"OpenAI translation failed: {str(e)}")

class GoogleTranslator(TranslationService):
    """Google Translate 服务 (免费版本)"""
    
    def __init__(self):
        super().__init__()
        self.name = "google"
        self.base_url = "https://translate.googleapis.com/translate_a/single"
    
    async def translate(self, text: str, target_lang: str = "zh", source_lang: str = "auto") -> TranslationResult:
        """使用 Google Translate 免费 API"""
        
        # 语言代码映射
        if target_lang == "zh":
            target_lang = "zh-cn"
        
        params = {
            "client": "gtx",
            "sl": source_lang,
            "tl": target_lang,
            "dt": "t",
            "q": text
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # 解析 Google Translate 响应
                        translated_text = ""
                        if result and len(result) > 0 and result[0]:
                            for item in result[0]:
                                if item and len(item) > 0:
                                    translated_text += item[0]
                        
                        return TranslationResult(
                            original_text=text,
                            translated_text=translated_text,
                            source_language=source_lang,
                            target_language=target_lang,
                            service=self.name,
                            confidence=0.8
                        )
                    else:
                        raise Exception(f"Google Translate error: {response.status}")
        
        except Exception as e:
            raise Exception(f"Google translation failed: {str(e)}")

class BedrockTranslator(TranslationService):
    """Amazon Bedrock Claude翻译服务"""
    
    def __init__(self, model_id: str = "us.anthropic.claude-3-haiku-20240307-v1:0"):
        super().__init__()
        self.name = "bedrock_claude"
        self.model_id = model_id
        
        # 初始化Bedrock客户端
        self.bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )
    
    async def translate(self, text: str, target_lang: str = "zh", source_lang: str = "auto") -> TranslationResult:
        """使用Bedrock Claude Haiku翻译文本"""
        if not text.strip():
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_lang,
                target_language=target_lang,
                service=self.name,
                confidence=1.0
            )
        
        # 语言映射
        lang_map = {
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
        
        target_language = lang_map.get(target_lang, "中文")
        source_language = lang_map.get(source_lang, source_lang)
        
        try:
            # 构建消息序列
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": f"请将以下{source_language}文本翻译为{target_language}，只返回翻译结果，不要添加任何解释或额外内容：\n\n{text}"
                        }
                    ]
                }
            ]
            
            # 调用Bedrock API使用converse方法
            response = self.bedrock_runtime.converse(
                modelId=self.model_id,
                messages=messages,
                inferenceConfig={
                    "temperature": 0.1,
                    "maxTokens": 1000,
                }
            )
            
            # 解析响应
            translated_text = response.get('output')["message"]["content"][0]["text"].strip()
            
            return TranslationResult(
                original_text=text,
                translated_text=translated_text,
                source_language=source_lang,
                target_language=target_lang,
                service=self.name,
                confidence=0.95
            )
            
        except Exception as e:
            print(f"Bedrock翻译失败: {e}")
            raise


class BaiduTranslator(TranslationService):
    """百度翻译服务"""
    
    def __init__(self, app_id: Optional[str] = None, secret_key: Optional[str] = None):
        super().__init__()
        self.name = "baidu"
        self.app_id = app_id or os.getenv("BAIDU_TRANSLATE_APP_ID")
        self.secret_key = secret_key or os.getenv("BAIDU_TRANSLATE_SECRET_KEY")
        self.base_url = "https://fanyi-api.baidu.com/api/trans/vip/translate"
    
    async def translate(self, text: str, target_lang: str = "zh", source_lang: str = "auto") -> TranslationResult:
        """使用百度翻译 API"""
        if not self.app_id or not self.secret_key:
            raise ValueError("Baidu Translate credentials not provided")
        
        import hashlib
        import random
        
        # 生成签名
        salt = str(random.randint(32768, 65536))
        sign_str = self.app_id + text + salt + self.secret_key
        sign = hashlib.md5(sign_str.encode('utf-8')).hexdigest()
        
        params = {
            "q": text,
            "from": source_lang,
            "to": target_lang,
            "appid": self.app_id,
            "salt": salt,
            "sign": sign
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, data=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        if "trans_result" in result:
                            translated_text = result["trans_result"][0]["dst"]
                            
                            return TranslationResult(
                                original_text=text,
                                translated_text=translated_text,
                                source_language=source_lang,
                                target_language=target_lang,
                                service=self.name,
                                confidence=0.85
                            )
                        else:
                            raise Exception(f"Baidu API error: {result}")
                    else:
                        raise Exception(f"Baidu Translate error: {response.status}")
        
        except Exception as e:
            raise Exception(f"Baidu translation failed: {str(e)}")

class MockTranslator(TranslationService):
    """模拟翻译服务 (用于测试)"""
    
    def __init__(self):
        super().__init__()
        self.name = "mock"
        
        # 简单的翻译映射
        self.translation_map = {
            "hello": "你好",
            "world": "世界",
            "good": "好的",
            "morning": "早上好",
            "afternoon": "下午好",
            "evening": "晚上好",
            "night": "晚安",
            "thank you": "谢谢",
            "thanks": "谢谢",
            "please": "请",
            "yes": "是的",
            "no": "不是",
            "welcome": "欢迎",
            "goodbye": "再见",
            "how are you": "你好吗",
            "fine": "很好",
            "okay": "好的",
            "sorry": "对不起",
            "excuse me": "打扰一下"
        }
    
    async def translate(self, text: str, target_lang: str = "zh", source_lang: str = "auto") -> TranslationResult:
        """模拟翻译"""
        await asyncio.sleep(0.1)  # 模拟网络延迟
        
        translated = text.lower()
        
        # 简单替换
        for en, zh in self.translation_map.items():
            translated = translated.replace(en, zh)
        
        # 如果没有匹配，添加前缀
        if translated == text.lower():
            translated = f"[译] {text}"
        
        return TranslationResult(
            original_text=text,
            translated_text=translated,
            source_language=source_lang,
            target_language=target_lang,
            service=self.name,
            confidence=0.5
        )

class TranslationManager:
    """翻译管理器"""
    
    def __init__(self):
        self.translators = {}
        self.default_translator = "mock"
        
        # 初始化翻译服务
        self._init_translators()
    
    def _init_translators(self):
        """初始化翻译服务"""
        # Mock 翻译器 (总是可用)
        self.translators["mock"] = MockTranslator()
        
        # OpenAI 翻译器
        if os.getenv("OPENAI_API_KEY"):
            try:
                self.translators["openai"] = OpenAITranslator()
                self.default_translator = "openai"
                print("✅ OpenAI 翻译服务已启用")
            except Exception as e:
                print(f"⚠️ OpenAI 翻译服务初始化失败: {e}")
        
        # Google 翻译器
        try:
            self.translators["google"] = GoogleTranslator()
            if self.default_translator == "mock":
                self.default_translator = "google"
            print("✅ Google 翻译服务已启用")
        except Exception as e:
            print(f"⚠️ Google 翻译服务初始化失败: {e}")
        
        # 百度翻译器
        if os.getenv("BAIDU_TRANSLATE_APP_ID") and os.getenv("BAIDU_TRANSLATE_SECRET_KEY"):
            try:
                self.translators["baidu"] = BaiduTranslator()
                print("✅ 百度翻译服务已启用")
            except Exception as e:
                print(f"⚠️ 百度翻译服务初始化失败: {e}")
        
        # Bedrock翻译器
        try:
            model_id = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-3-haiku-20240307-v1:0")
            self.translators["bedrock"] = BedrockTranslator(model_id=model_id)
            self.default_translator = "bedrock"  # 设为默认翻译器
            print(f"✅ Bedrock Claude翻译服务已启用 (模型: {model_id})")
        except Exception as e:
            print(f"⚠️ Bedrock Claude翻译服务初始化失败: {e}")
    
    async def translate(self, text: str, target_lang: str = "zh", 
                       service: Optional[str] = None) -> TranslationResult:
        """翻译文本"""
        if not text or not text.strip():
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language="auto",
                target_language=target_lang,
                service="none",
                confidence=1.0
            )
        
        # 选择翻译服务
        translator_name = service or self.default_translator
        
        if translator_name not in self.translators:
            translator_name = self.default_translator
        
        translator = self.translators[translator_name]
        
        try:
            result = await translator.translate(text, target_lang)
            return result
        except Exception as e:
            print(f"翻译失败 ({translator_name}): {e}")
            
            # 回退到 mock 翻译器
            if translator_name != "mock":
                try:
                    return await self.translators["mock"].translate(text, target_lang)
                except Exception as fallback_error:
                    print(f"回退翻译也失败: {fallback_error}")
            
            # 最后的回退
            return TranslationResult(
                original_text=text,
                translated_text=f"[翻译失败] {text}",
                source_language="auto",
                target_language=target_lang,
                service="error",
                confidence=0.0
            )
    
    def get_available_services(self) -> list:
        """获取可用的翻译服务"""
        return list(self.translators.keys())

# 全局翻译管理器实例
translation_manager = TranslationManager()

async def translate_text(text: str, target_lang: str = "zh", 
                        service: Optional[str] = None) -> str:
    """便捷的翻译函数"""
    result = await translation_manager.translate(text, target_lang, service)
    return result.translated_text

# 测试函数
async def test_translation():
    """测试翻译功能"""
    test_texts = [
        "Hello, how are you?",
        "Good morning, welcome to our service.",
        "Thank you for using our application.",
        "مرحبا، كيف حالك؟",  # Arabic
        "Bonjour, comment allez-vous?"  # French
    ]
    
    print("🧪 测试翻译功能")
    print("-" * 50)
    
    for text in test_texts:
        try:
            result = await translation_manager.translate(text)
            print(f"原文: {text}")
            print(f"译文: {result.translated_text}")
            print(f"服务: {result.service}")
            print("-" * 30)
        except Exception as e:
            print(f"翻译失败: {text} - {e}")

if __name__ == "__main__":
    asyncio.run(test_translation())
