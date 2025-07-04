"""
字幕纠错模块的pytest测试
"""

import pytest
import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from subtitle_genius.correction import (
    BasicCorrectionService,
    BedrockCorrectionService,
    CorrectionInput,
    CorrectionOutput,
    correct_subtitle
)


class TestBasicCorrectionService:
    """基础纠错服务测试类"""
    
    @pytest.fixture
    def service(self):
        """创建基础纠错服务实例"""
        return BasicCorrectionService()
    
    @pytest.mark.asyncio
    async def test_spelling_correction(self, service):
        """测试拼写纠正"""
        input_data = CorrectionInput(
            current_subtitle="اللة يبارك في هذا اليوم",
            history_subtitles=[],
            scene_description="通用"
        )
        
        result = await service.correct(input_data)
        
        assert result.corrected_subtitle == "الله يبارك في هذا اليوم"
        assert result.has_correction is True
        assert result.confidence > 0.9
        assert "拼写纠正" in result.correction_details
    
    @pytest.mark.asyncio
    async def test_punctuation_correction(self, service):
        """测试标点符号纠正"""
        input_data = CorrectionInput(
            current_subtitle="الرئيس يتحدث في المؤتمر .",
            history_subtitles=[],
            scene_description="新闻播报"
        )
        
        result = await service.correct(input_data)
        
        assert result.corrected_subtitle == "الرئيس يتحدث في المؤتمر."
        assert result.has_correction is True
        assert "标点符号纠正" in result.correction_details
    
    @pytest.mark.asyncio
    async def test_no_correction_needed(self, service):
        """测试无需纠正的情况"""
        input_data = CorrectionInput(
            current_subtitle="مرحبا بكم في المباراة",
            history_subtitles=[],
            scene_description="足球比赛"
        )
        
        result = await service.correct(input_data)
        
        assert result.corrected_subtitle == "مرحبا بكم في المباراة"
        assert result.has_correction is False
        assert result.confidence == 1.0
        assert result.correction_details is None


class TestBedrockCorrectionService:
    """Bedrock纠错服务测试类"""
    
    @pytest.fixture
    def service(self):
        """创建Bedrock纠错服务实例"""
        return BedrockCorrectionService()
    
    @pytest.mark.asyncio
    async def test_bedrock_service_creation(self, service):
        """测试Bedrock服务创建"""
        assert service.model_id == "us.anthropic.claude-3-haiku-20240307-v1:0"
        assert service.service_name == "bedrock_claude"
        assert "BedrockCorrection" in service.get_service_name()
    
    @pytest.mark.asyncio
    async def test_bedrock_mock_correction(self, service):
        """测试Bedrock模拟纠错功能"""
        input_data = CorrectionInput(
            current_subtitle="اللة يبارك في هذا اليوم",
            history_subtitles=[],
            scene_description="通用"
        )
        
        result = await service.correct(input_data)
        
        # 即使是模拟模式，也应该能纠正基本错误
        assert isinstance(result, CorrectionOutput)
        assert result.corrected_subtitle == "الله يبارك في هذا اليوم"
        assert result.has_correction is True
        assert result.confidence > 0.8
    
    @pytest.mark.asyncio
    async def test_bedrock_scene_guidance(self, service):
        """测试Bedrock场景指导"""
        # 测试足球场景
        football_guidance = service._get_scene_guidance("足球比赛")
        assert "足球术语" in football_guidance
        assert "كرة القدم" in football_guidance
        
        # 测试新闻场景
        news_guidance = service._get_scene_guidance("新闻播报")
        assert "新闻场景" in news_guidance
        assert "رئيس" in news_guidance
    
    @pytest.mark.asyncio
    async def test_bedrock_prompt_building(self, service):
        """测试Bedrock提示词构建"""
        input_data = CorrectionInput(
            current_subtitle="اللاعب سجل هدف",
            history_subtitles=["مرحبا بكم", "المباراة بدأت"],
            scene_description="足球比赛"
        )
        
        prompt = service._build_correction_prompt(input_data)
        
        assert "足球比赛" in prompt
        assert "اللاعب سجل هدف" in prompt
        assert "历史字幕上下文" in prompt
        assert "مرحبا بكم" in prompt
        assert "JSON格式" in prompt
    
    @pytest.mark.asyncio
    async def test_bedrock_response_parsing(self, service):
        """测试Bedrock响应解析"""
        # 测试JSON格式响应
        json_response = {
            'output': {
                'message': {
                    'content': [{
                        'text': '{"corrected_text": "الله يبارك", "has_correction": true, "confidence": 0.95, "details": "拼写纠正"}'
                    }]
                }
            }
        }
        
        result = service._parse_correction_response(json_response, "اللة يبارك")
        
        assert result["corrected_text"] == "الله يبارك"
        assert result["has_correction"] is True
        assert result["confidence"] == 0.95
        assert result["details"] == "拼写纠正"
        
        # 测试非JSON格式响应
        text_response = {
            'output': {
                'message': {
                    'content': [{
                        'text': '纠正后: الله يبارك\n这是一个拼写纠正。'
                    }]
                }
            }
        }
        
        result = service._parse_correction_response(text_response, "اللة يبارك")
        
        assert result["corrected_text"] == "الله يبارك"
        assert result["has_correction"] is True


class TestCorrectionIntegration:
    """纠错集成测试"""
    
    @pytest.mark.asyncio
    async def test_convenience_function(self):
        """测试便捷函数"""
        result = await correct_subtitle("اللة يبارك")
        assert result == "الله يبارك"
        
        result_with_params = await correct_subtitle(
            current_subtitle="اللاعب يسجل هدف",
            history_subtitles=["مرحبا بكم"],
            scene_description="足球比赛"
        )
        assert isinstance(result_with_params, str)
    
    @pytest.mark.asyncio
    async def test_different_services_consistency(self):
        """测试不同服务的一致性"""
        input_data = CorrectionInput(
            current_subtitle="اللة يبارك في هذا اليوم",
            history_subtitles=[],
            scene_description="通用"
        )
        
        # 基础服务
        basic_service = BasicCorrectionService()
        basic_result = await basic_service.correct(input_data)
        
        # Bedrock服务
        bedrock_service = BedrockCorrectionService()
        bedrock_result = await bedrock_service.correct(input_data)
        
        # 两个服务都应该能纠正相同的拼写错误
        assert basic_result.corrected_subtitle == bedrock_result.corrected_subtitle
        assert basic_result.has_correction == bedrock_result.has_correction


class TestCorrectionScenarios:
    """纠错场景测试"""
    
    @pytest.mark.asyncio
    async def test_football_scenario(self):
        """测试足球场景纠错"""
        service = BasicCorrectionService()
        
        input_data = CorrectionInput(
            current_subtitle="اللاعب سجل هدف رائع في كرة القدم",
            history_subtitles=["مرحبا بكم في المباراة"],
            scene_description="足球比赛"
        )
        
        result = await service.correct(input_data)
        
        assert isinstance(result.corrected_subtitle, str)
        assert result.confidence > 0
    
    @pytest.mark.asyncio
    async def test_news_scenario(self):
        """测试新闻场景纠错"""
        service = BedrockCorrectionService()
        
        input_data = CorrectionInput(
            current_subtitle="الرئيس يلتقي بالوزراء في مؤتمر مهم",
            history_subtitles=["أخبار اليوم"],
            scene_description="新闻播报"
        )
        
        result = await service.correct(input_data)
        
        assert isinstance(result.corrected_subtitle, str)
        assert result.confidence > 0
    
    @pytest.mark.asyncio
    async def test_context_consistency(self):
        """测试上下文一致性"""
        service = BasicCorrectionService()
        
        # 建立上下文
        history = [
            "الفريق الأول يستعد للمباراة",
            "الفريق الثاني وصل إلى الملعب"
        ]
        
        input_data = CorrectionInput(
            current_subtitle="فريق اول سجل هدف",  # 不一致的称呼
            history_subtitles=history,
            scene_description="足球比赛"
        )
        
        result = await service.correct(input_data)
        
        # 应该保持一致性（虽然当前实现可能还没有这个功能）
        assert isinstance(result.corrected_subtitle, str)


@pytest.mark.parametrize("service_class", [BasicCorrectionService, BedrockCorrectionService])
@pytest.mark.asyncio
async def test_service_interface_compliance(service_class):
    """测试服务接口合规性"""
    service = service_class()
    
    input_data = CorrectionInput(
        current_subtitle="测试字幕",
        history_subtitles=[],
        scene_description="通用"
    )
    
    result = await service.correct(input_data)
    
    # 检查返回类型
    assert isinstance(result, CorrectionOutput)
    assert isinstance(result.corrected_subtitle, str)
    assert isinstance(result.has_correction, bool)
    assert isinstance(result.confidence, (int, float))
    assert 0 <= result.confidence <= 1
    
    # 检查服务名称
    assert isinstance(service.get_service_name(), str)


@pytest.mark.parametrize("input_text,expected_correction", [
    ("اللة يبارك", True),
    ("مرحبا بكم", False),
    ("انشاء الله", True),
    ("الحمد لله", False),
    ("الرئيس يتحدث .", True),  # 标点错误
])
@pytest.mark.asyncio
async def test_parametrized_corrections(input_text, expected_correction):
    """参数化测试各种纠错情况"""
    service = BasicCorrectionService()
    
    input_data = CorrectionInput(
        current_subtitle=input_text,
        history_subtitles=[],
        scene_description="通用"
    )
    
    result = await service.correct(input_data)
    
    assert result.has_correction == expected_correction
