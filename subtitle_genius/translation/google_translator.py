"""
Google翻译服务实现
"""

import aiohttp
from .base import TranslationService, TranslationInput, TranslationOutput


class GoogleTranslator(TranslationService):
    """Google Translate服务（免费版本）"""
    
    def __init__(self):
        self.base_url = "https://translate.googleapis.com/translate_a/single"
        
        # 语言代码映射
        self.lang_map = {
            "zh": "zh-cn",
            "en": "en",
            "ar": "ar", 
            "ja": "ja",
            "ko": "ko",
            "fr": "fr",
            "de": "de",
            "es": "es",
            "ru": "ru"
        }
    
    async def translate(self, input_data: TranslationInput) -> TranslationOutput:
        """使用Google Translate免费API"""
        
        if not input_data.text.strip():
            return TranslationOutput(
                original_text=input_data.text,
                translated_text=input_data.text,
                source_language=input_data.source_language,
                target_language=input_data.target_language,
                confidence=1.0,
                service_name=self.get_service_name()
            )
        
        # 映射语言代码
        source_lang = self.lang_map.get(input_data.source_language, input_data.source_language)
        target_lang = self.lang_map.get(input_data.target_language, input_data.target_language)
        
        params = {
            "client": "gtx",
            "sl": source_lang,
            "tl": target_lang,
            "dt": "t",
            "q": input_data.text
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # 解析Google Translate响应
                        translated_text = ""
                        if result and len(result) > 0 and result[0]:
                            for item in result[0]:
                                if item and len(item) > 0:
                                    translated_text += item[0]
                        
                        return TranslationOutput(
                            original_text=input_data.text,
                            translated_text=translated_text,
                            source_language=input_data.source_language,
                            target_language=input_data.target_language,
                            confidence=0.8,
                            service_name=self.get_service_name(),
                            translation_details="Google Translate免费API"
                        )
                    else:
                        raise Exception(f"Google Translate error: {response.status}")
        
        except Exception as e:
            raise Exception(f"Google translation failed: {str(e)}")
    
    def get_service_name(self) -> str:
        return "GoogleTranslate"
