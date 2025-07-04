"""
字幕纠错服务的pytest测试
"""

import pytest
import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入纠错服务
correction_file = project_root / "subtitle_genius" / "correction_service.py"
exec(open(correction_file).read())


class TestBasicCorrectionService:
    """基础纠错服务测试类"""
    
    @pytest.fixture
    def service(self):
        """创建纠错服务实例"""
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
    
    @pytest.mark.asyncio
    async def test_with_history(self, service):
        """测试带历史记录的纠错"""
        history = ["مرحبا بكم في المباراة", "الفريق الأول يستعد"]
        
        input_data = CorrectionInput(
            current_subtitle="اللاعب سجل هدف رائع",
            history_subtitles=history,
            scene_description="足球比赛"
        )
        
        result = await service.correct(input_data)
        
        assert isinstance(result.corrected_subtitle, str)
        assert isinstance(result.has_correction, bool)
        assert 0 <= result.confidence <= 1
    
    @pytest.mark.asyncio
    async def test_different_scenes(self, service):
        """测试不同场景"""
        test_cases = [
            {
                "scene": "足球比赛",
                "subtitle": "اللاعب يسجل هدف",
                "expected_type": str
            },
            {
                "scene": "新闻播报",
                "subtitle": "الرئيس يتحدث",
                "expected_type": str
            },
            {
                "scene": "通用",
                "subtitle": "مرحبا بكم",
                "expected_type": str
            }
        ]
        
        for case in test_cases:
            input_data = CorrectionInput(
                current_subtitle=case["subtitle"],
                history_subtitles=[],
                scene_description=case["scene"]
            )
            
            result = await service.correct(input_data)
            
            assert isinstance(result.corrected_subtitle, case["expected_type"])
            assert isinstance(result.has_correction, bool)
            assert 0 <= result.confidence <= 1
    
    @pytest.mark.asyncio
    async def test_empty_input(self, service):
        """测试空输入"""
        input_data = CorrectionInput(
            current_subtitle="",
            history_subtitles=[],
            scene_description="通用"
        )
        
        result = await service.correct(input_data)
        
        assert result.corrected_subtitle == ""
        assert result.has_correction is False
        assert result.confidence == 1.0
    
    @pytest.mark.asyncio
    async def test_multiple_corrections(self, service):
        """测试多重纠错"""
        input_data = CorrectionInput(
            current_subtitle="اللة يبارك في هذا اليوم .",  # 拼写+标点错误
            history_subtitles=[],
            scene_description="通用"
        )
        
        result = await service.correct(input_data)
        
        assert "الله" in result.corrected_subtitle  # 拼写已纠正
        assert not result.corrected_subtitle.endswith(" .")  # 标点已纠正
        assert result.has_correction is True


class TestConvenienceFunction:
    """便捷函数测试类"""
    
    @pytest.mark.asyncio
    async def test_correct_subtitle_simple(self):
        """测试简单的便捷函数"""
        result = await correct_subtitle("اللة يبارك")
        
        assert result == "الله يبارك"
    
    @pytest.mark.asyncio
    async def test_correct_subtitle_with_params(self):
        """测试带参数的便捷函数"""
        result = await correct_subtitle(
            current_subtitle="اللاعب يسجل هدف",
            history_subtitles=["مرحبا بكم"],
            scene_description="足球比赛"
        )
        
        assert isinstance(result, str)
        assert len(result) > 0


class TestDataClasses:
    """数据类测试"""
    
    def test_correction_input_creation(self):
        """测试CorrectionInput创建"""
        input_data = CorrectionInput(
            current_subtitle="test",
            history_subtitles=["history1", "history2"],
            scene_description="test_scene"
        )
        
        assert input_data.current_subtitle == "test"
        assert len(input_data.history_subtitles) == 2
        assert input_data.scene_description == "test_scene"
    
    def test_correction_output_creation(self):
        """测试CorrectionOutput创建"""
        output = CorrectionOutput(
            corrected_subtitle="corrected",
            has_correction=True,
            confidence=0.95
        )
        
        assert output.corrected_subtitle == "corrected"
        assert output.has_correction is True
        assert output.confidence == 0.95


@pytest.mark.parametrize("input_text,expected_correction", [
    ("اللة يبارك", True),
    ("مرحبا بكم", False),
    ("انشاء الله", True),
    ("الحمد لله", False),
])
@pytest.mark.asyncio
async def test_parametrized_spelling(input_text, expected_correction):
    """参数化测试拼写纠正"""
    service = BasicCorrectionService()
    
    input_data = CorrectionInput(
        current_subtitle=input_text,
        history_subtitles=[],
        scene_description="通用"
    )
    
    result = await service.correct(input_data)
    
    assert result.has_correction == expected_correction


@pytest.mark.parametrize("scene,subtitle", [
    ("足球比赛", "اللاعب يسجل هدف"),
    ("新闻播报", "الرئيس يتحدث"),
    ("通用", "مرحبا بكم"),
])
@pytest.mark.asyncio
async def test_parametrized_scenes(scene, subtitle):
    """参数化测试不同场景"""
    service = BasicCorrectionService()
    
    input_data = CorrectionInput(
        current_subtitle=subtitle,
        history_subtitles=[],
        scene_description=scene
    )
    
    result = await service.correct(input_data)
    
    assert isinstance(result.corrected_subtitle, str)
    assert isinstance(result.has_correction, bool)
    assert 0 <= result.confidence <= 1
