"""
字幕纠错服务工厂类
"""

from typing import Optional
from .base import SubtitleCorrectionService
from .bedrock_corrector import BedrockCorrectionService
from .llm_corrector import LLMCorrectionService
from .basic_corrector import BasicCorrectionService


class CorrectionServiceFactory:
    """字幕纠错服务工厂"""
    
    @staticmethod
    def create_service(
        service_type: str = "bedrock",
        **kwargs
    ) -> SubtitleCorrectionService:
        """
        创建纠错服务实例
        
        Args:
            service_type: 服务类型 ("bedrock", "llm", "basic")
            **kwargs: 服务特定的参数
            
        Returns:
            SubtitleCorrectionService: 纠错服务实例
        """
        
        if service_type.lower() == "bedrock":
            model_id = kwargs.get("model_id", "us.anthropic.claude-3-haiku-20240307-v1:0")
            return BedrockCorrectionService(model_id=model_id)
        
        elif service_type.lower() == "llm":
            api_key = kwargs.get("api_key")
            model = kwargs.get("model", "gpt-3.5-turbo")
            return LLMCorrectionService(api_key=api_key, model=model)
        
        elif service_type.lower() == "basic":
            return BasicCorrectionService()
        
        else:
            raise ValueError(f"不支持的服务类型: {service_type}")
    
    @staticmethod
    def get_available_services() -> list:
        """获取可用的服务类型列表"""
        return ["bedrock", "llm", "basic"]


# 便捷函数
def create_corrector(
    service_type: str = "bedrock",
    language: str = "ar",
    **kwargs
) -> SubtitleCorrectionService:
    """
    创建纠错服务的便捷函数
    
    Args:
        service_type: 服务类型
        language: 目标语言 (ar, zh, en, es等)
        **kwargs: 其他参数
        
    Returns:
        SubtitleCorrectionService: 配置好的纠错服务
    """
    return CorrectionServiceFactory.create_service(service_type, **kwargs)
