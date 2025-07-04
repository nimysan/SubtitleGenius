"""
翻译模块的pytest测试
"""

import pytest
import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from subtitle_genius.translation import (
    BedrockTranslator,
    GoogleTranslator,
    OpenAITranslator,
    TranslationInput,
    TranslationOutput,
    translate_text,
    batch_translate
)


class TestTranslationBase:
    """翻译基础功能测试"""
    
    def test_translation_input_creation(self):
        """测试TranslationInput创建"""
        input_data = TranslationInput(
            text="مرحبا بكم",
            source_language="ar",
            target_language="zh",
            context="问候语"
        )
        
        assert input_data.text == "مرحبا بكم"
        assert input_data.source_language == "ar"
        assert input_data.target_language == "zh"
        assert input_data.context == "问候语"
    
    def test_translation_output_creation(self):
        """测试TranslationOutput创建"""
        output = TranslationOutput(
            original_text="مرحبا",
            translated_text="你好",
            source_language="ar",
            target_language="zh",
            confidence=0.9,
            service_name="TestService"
        )
        
        assert output.original_text == "مرحبا"
        assert output.translated_text == "你好"
        assert output.confidence == 0.9
        assert output.service_name == "TestService"


class TestBedrockTranslator:
    """Bedrock翻译服务测试"""
    
    @pytest.fixture
    def translator(self):
        """创建Bedrock翻译器实例"""
        return BedrockTranslator()
    
    @pytest.mark.asyncio
    async def test_bedrock_service_creation(self, translator):
        """测试Bedrock服务创建"""
        assert translator.model_id == "us.anthropic.claude-3-haiku-20240307-v1:0"
        assert translator.service_name == "bedrock_claude"
        assert "BedrockTranslator" in translator.get_service_name()
    
    @pytest.mark.asyncio
    async def test_bedrock_empty_text(self, translator):
        """测试空文本翻译"""
        input_data = TranslationInput(
            text="",
            source_language="ar",
            target_language="zh"
        )
        
        result = await translator.translate(input_data)
        
        assert result.original_text == ""
        assert result.translated_text == ""
        assert result.confidence == 1.0
    
    @pytest.mark.asyncio
    async def test_bedrock_mock_translation(self, translator):
        """测试Bedrock模拟翻译"""
        input_data = TranslationInput(
            text="مرحبا",
            source_language="ar",
            target_language="zh"
        )
        
        result = await translator.translate(input_data)
        
        assert isinstance(result, TranslationOutput)
        assert result.original_text == "مرحبا"
        assert result.translated_text == "你好"  # 模拟翻译结果
        assert result.source_language == "ar"
        assert result.target_language == "zh"
        assert result.confidence > 0
    
    @pytest.mark.asyncio
    async def test_bedrock_prompt_building(self, translator):
        """测试Bedrock提示词构建"""
        input_data = TranslationInput(
            text="مرحبا بكم",
            source_language="ar",
            target_language="zh",
            context="问候语场景"
        )
        
        prompt = translator._build_translation_prompt(input_data)
        
        assert "Arabic" in prompt
        assert "中文" in prompt
        assert "مرحبا بكم" in prompt
        assert "问候语场景" in prompt
        assert "翻译结果" in prompt


class TestGoogleTranslator:
    """Google翻译服务测试"""
    
    @pytest.fixture
    def translator(self):
        """创建Google翻译器实例"""
        return GoogleTranslator()
    
    @pytest.mark.asyncio
    async def test_google_service_creation(self, translator):
        """测试Google服务创建"""
        assert translator.base_url == "https://translate.googleapis.com/translate_a/single"
        assert translator.get_service_name() == "GoogleTranslate"
    
    @pytest.mark.asyncio
    async def test_google_empty_text(self, translator):
        """测试空文本翻译"""
        input_data = TranslationInput(
            text="",
            source_language="ar",
            target_language="zh"
        )
        
        result = await translator.translate(input_data)
        
        assert result.original_text == ""
        assert result.translated_text == ""
        assert result.confidence == 1.0
    
    def test_google_language_mapping(self, translator):
        """测试Google语言代码映射"""
        assert translator.lang_map["zh"] == "zh-cn"
        assert translator.lang_map["en"] == "en"
        assert translator.lang_map["ar"] == "ar"


class TestOpenAITranslator:
    """OpenAI翻译服务测试"""
    
    @pytest.fixture
    def translator(self):
        """创建OpenAI翻译器实例"""
        return OpenAITranslator()
    
    def test_openai_service_creation(self, translator):
        """测试OpenAI服务创建"""
        assert translator.model == "gpt-3.5-turbo"
        assert translator.base_url == "https://api.openai.com/v1/chat/completions"
        assert "OpenAI" in translator.get_service_name()
    
    @pytest.mark.asyncio
    async def test_openai_empty_text(self, translator):
        """测试空文本翻译"""
        input_data = TranslationInput(
            text="",
            source_language="ar",
            target_language="zh"
        )
        
        result = await translator.translate(input_data)
        
        assert result.original_text == ""
        assert result.translated_text == ""
        assert result.confidence == 1.0
    
    @pytest.mark.asyncio
    async def test_openai_no_api_key(self, translator):
        """测试没有API key的情况"""
        translator.api_key = None
        
        input_data = TranslationInput(
            text="مرحبا",
            source_language="ar",
            target_language="zh"
        )
        
        with pytest.raises(ValueError, match="OpenAI API key not provided"):
            await translator.translate(input_data)
    
    def test_openai_language_mapping(self, translator):
        """测试OpenAI语言映射"""
        assert translator.lang_map["zh"] == "中文"
        assert translator.lang_map["en"] == "English"
        assert translator.lang_map["ar"] == "Arabic"


class TestTranslationUtils:
    """翻译工具函数测试"""
    
    @pytest.mark.asyncio
    async def test_translate_text_function(self):
        """测试translate_text便捷函数"""
        result = await translate_text(
            text="مرحبا",
            source_language="ar",
            target_language="zh",
            service="bedrock"
        )
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_batch_translate_function(self):
        """测试batch_translate批量翻译函数"""
        texts = ["مرحبا", "شكرا", "مع السلامة"]
        
        results = await batch_translate(
            texts=texts,
            source_language="ar",
            target_language="zh",
            service="bedrock"
        )
        
        assert len(results) == len(texts)
        assert all(isinstance(result, str) for result in results)
        assert results[0] == "你好"  # 模拟翻译结果
        assert results[1] == "谢谢"
        assert results[2] == "再见"


class TestTranslationScenarios:
    """翻译场景测试"""
    
    @pytest.mark.asyncio
    async def test_arabic_to_chinese_translation(self):
        """测试阿拉伯语到中文翻译"""
        translator = BedrockTranslator()
        
        input_data = TranslationInput(
            text="كرة القدم لعبة جميلة",
            source_language="ar",
            target_language="zh"
        )
        
        result = await translator.translate(input_data)
        
        assert result.source_language == "ar"
        assert result.target_language == "zh"
        assert "足球" in result.translated_text
    
    @pytest.mark.asyncio
    async def test_arabic_to_english_translation(self):
        """测试阿拉伯语到英文翻译"""
        translator = BedrockTranslator()
        
        input_data = TranslationInput(
            text="مرحبا بكم في المباراة",
            source_language="ar",
            target_language="en"
        )
        
        result = await translator.translate(input_data)
        
        assert result.source_language == "ar"
        assert result.target_language == "en"
        assert isinstance(result.translated_text, str)
    
    @pytest.mark.asyncio
    async def test_translation_with_context(self):
        """测试带上下文的翻译"""
        translator = BedrockTranslator()
        
        input_data = TranslationInput(
            text="هدف",
            source_language="ar",
            target_language="zh",
            context="足球比赛场景"
        )
        
        result = await translator.translate(input_data)
        
        assert result.original_text == "هدف"
        assert isinstance(result.translated_text, str)


@pytest.mark.parametrize("service_class", [BedrockTranslator, GoogleTranslator])
@pytest.mark.asyncio
async def test_service_interface_compliance(service_class):
    """测试服务接口合规性"""
    translator = service_class()
    
    input_data = TranslationInput(
        text="مرحبا",
        source_language="ar",
        target_language="zh"
    )
    
    result = await translator.translate(input_data)
    
    # 检查返回类型
    assert isinstance(result, TranslationOutput)
    assert isinstance(result.original_text, str)
    assert isinstance(result.translated_text, str)
    assert isinstance(result.source_language, str)
    assert isinstance(result.target_language, str)
    assert isinstance(result.confidence, (int, float))
    assert 0 <= result.confidence <= 1
    assert isinstance(result.service_name, str)
    
    # 检查服务名称
    assert isinstance(translator.get_service_name(), str)


@pytest.mark.parametrize("text,source_lang,target_lang", [
    ("مرحبا", "ar", "zh"),
    ("شكرا", "ar", "en"),
    ("كرة القدم", "ar", "zh"),
    ("الرئيس", "ar", "en"),
])
@pytest.mark.asyncio
async def test_parametrized_translations(text, source_lang, target_lang):
    """参数化测试各种翻译情况"""
    translator = BedrockTranslator()
    
    input_data = TranslationInput(
        text=text,
        source_language=source_lang,
        target_language=target_lang
    )
    
    result = await translator.translate(input_data)
    
    assert result.original_text == text
    assert result.source_language == source_lang
    assert result.target_language == target_lang
    assert isinstance(result.translated_text, str)
    assert len(result.translated_text) > 0
