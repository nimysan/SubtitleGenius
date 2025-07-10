"""
基于Amazon Bedrock的多语言字幕纠错服务实现
"""

import json
import os
import boto3
import asyncio
import logging
from typing import Optional, Dict, Any
from .base import SubtitleCorrectionService, CorrectionInput, CorrectionOutput

# 配置日志
logger = logging.getLogger(__name__)


class BedrockCorrectionService(SubtitleCorrectionService):
    """基于Amazon Bedrock的多语言字幕纠错服务"""
    
    def __init__(self, model_id: str = "us.anthropic.claude-3-5-sonnet-20240620-v1:0"):
        self.model_id = model_id
        self.service_name = "bedrock_claude"
        
        logger.info(f"初始化BedrockCorrectionService，模型ID: {model_id}")
        
        # 初始化Bedrock客户端
        try:
            aws_region = os.getenv("AWS_REGION", "us-east-1")
            logger.info(f"正在初始化Bedrock客户端，区域: {aws_region}")
            
            self.bedrock_runtime = boto3.client(
                service_name="bedrock-runtime",
                region_name=aws_region
            )
            logger.info("✅ Bedrock客户端初始化成功")
            
        except Exception as e:
            logger.error(f"❌ 无法初始化Bedrock客户端: {e}")
            logger.warning("将使用模拟纠错作为备用方案")
            self.bedrock_runtime = None
    
    # 语言配置映射
    LANGUAGE_CONFIGS = {
        "ar": {
            "name": "阿拉伯语",
            "assistant_role": "你是一个专业的阿拉伯语字幕纠错助手",
            "common_fixes": {
                "اللة": "الله",
                "انشاء الله": "إن شاء الله",
                "مع السلامة": "مع السلامة"
            }
        },
        "zh": {
            "name": "中文",
            "assistant_role": "你是一个专业的中文字幕纠错助手",
            "common_fixes": {
                "的的": "的",
                "了了": "了",
                "。。": "。"
            }
        },
        "en": {
            "name": "英语",
            "assistant_role": "You are a professional English subtitle correction assistant",
            "common_fixes": {
                "  ": " ",  # 多余空格
                "..": ".",   # 重复句号
                "??": "?"    # 重复问号
            }
        },
        "es": {
            "name": "西班牙语",
            "assistant_role": "Eres un asistente profesional de corrección de subtítulos en español",
            "common_fixes": {
                "  ": " ",
                "..": ".",
                "??": "?"
            }
        }
    }
    
    async def correct(self, input_data: CorrectionInput) -> CorrectionOutput:
        """使用Bedrock Claude进行字幕纠错"""
        
        logger.info(f"🔧 开始字幕纠错处理")
        logger.info(f"  - 原始字幕: '{input_data.current_subtitle}'")
        logger.info(f"  - 语言: {input_data.language}")
        logger.info(f"  - 场景: {input_data.scene_description}")
        logger.info(f"  - 历史字幕数量: {len(input_data.history_subtitles) if input_data.history_subtitles else 0}")
        
        # 如果没有Bedrock客户端，使用模拟纠错
        if not self.bedrock_runtime:
            logger.warning("⚠️  Bedrock客户端不可用，使用模拟纠错")
            return await self._mock_correction(input_data)
        
        try:
            logger.debug("构建纠错提示词...")
            # 构建纠错提示词
            prompt = self._build_correction_prompt(input_data)
            logger.debug(f"提示词长度: {len(prompt)} 字符")
            
            # 构建消息序列 - 修正格式
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
            
            logger.info("📡 调用Bedrock Converse API...")
            # 调用Bedrock Converse API
            response = await self._call_bedrock_converse(messages)
            logger.info("✅ Bedrock API调用成功")
            
            logger.debug("解析纠错响应...")
            # 解析响应
            result = self._parse_correction_response(response, input_data.current_subtitle)
            
            # 记录纠错结果
            logger.info(f"🎯 纠错完成:")
            logger.info(f"  - 纠正后字幕: '{result.get('corrected_text', input_data.current_subtitle)}'")
            logger.info(f"  - 是否有纠正: {result.get('has_correction', False)}")
            logger.info(f"  - 置信度: {result.get('confidence', 0.8):.2f}")
            logger.info(f"  - 是否拆分: {result.get('has_split', False)}")
            
            if result.get('has_split', False):
                split_subtitles = result.get('split_subtitles', [])
                logger.info(f"  - 拆分结果 ({len(split_subtitles)} 个句子):")
                for i, subtitle in enumerate(split_subtitles, 1):
                    logger.info(f"    {i}. '{subtitle}'")
            
            if result.get('details'):
                logger.info(f"  - 纠正详情: {result.get('details')}")
            
            return CorrectionOutput(
                corrected_subtitle=result.get("corrected_text", input_data.current_subtitle),
                has_correction=result.get("has_correction", False),
                confidence=result.get("confidence", 0.8),
                correction_details=result.get("details", None),
                split_subtitles=result.get("split_subtitles", []),
                has_split=result.get("has_split", False)
            )
            
        except Exception as e:
            logger.error(f"❌ Bedrock纠错失败: {e}")
            logger.exception("详细错误信息:")
            logger.warning("🔄 切换到备用纠错方案")
            return await self._mock_correction(input_data)
    
    def _build_correction_prompt(self, input_data: CorrectionInput) -> str:
        """构建多语言纠错提示词"""
        
        logger.debug(f"构建纠错提示词，语言: {input_data.language}")
        
        # 获取语言配置
        lang_config = self.LANGUAGE_CONFIGS.get(input_data.language, self.LANGUAGE_CONFIGS["ar"])
        language_name = lang_config["name"]
        
        logger.debug(f"使用语言配置: {language_name}")
        
        # 构建历史上下文
        history_context = ""
        if input_data.history_subtitles:
            recent_history = input_data.history_subtitles[-3:]  # 最近3条
            history_context = f"\n\n历史字幕上下文:\n" + "\n".join(f"- {h}" for h in recent_history)
            logger.debug(f"添加历史上下文，包含 {len(recent_history)} 条历史字幕")
        else:
            logger.debug("无历史字幕上下文")
        
        # 场景相关的纠错指导
        logger.debug(f"获取场景指导，场景: {input_data.scene_description}")
        scene_guidance = self._get_scene_guidance(input_data.scene_description, input_data.language)
        
        prompt = f"""{lang_config["assistant_role"]}。请纠正以下{language_name}字幕中的错误。

场景: {input_data.scene_description}
当前字幕: {input_data.current_subtitle}{history_context}

{scene_guidance}

请检查并纠正以下类型的错误:
1. 拼写错误
2. 语法错误
3. 标点符号错误 (去除多余空格、重复标点等)
4. 术语标准化 (根据场景使用标准术语)
5. 上下文一致性 (与历史字幕保持一致的称呼和术语)

【长句拆分】
如果当前字幕是一个较长的句子（通常超过15-20个字），请将其拆分成2-3个较短的、语义完整的句子。
拆分时请遵循以下原则：
1. 每个拆分后的句子必须语义完整，表达清晰
2. 保持原始语义不变
3. 在自然的断句点进行拆分（如逗号、分号等位置）
4. 拆分后的每个句子长度应相对均衡

请以JSON格式返回结果:
{{
    "corrected_text": "纠正后的字幕文本",
    "has_correction": true/false,
    "confidence": 0.0-1.0,
    "details": "具体的纠正说明",
    "has_split": true/false,
    "split_subtitles": ["拆分后的第一个句子", "拆分后的第二个句子", ...]
}}

如果字幕没有错误，请返回原文本并设置has_correction为false。
如果字幕不需要拆分（句子较短或已经是简短句子），请设置has_split为false并保持split_subtitles为空数组。"""
        
        logger.debug(f"提示词构建完成，总长度: {len(prompt)} 字符")
        return prompt
    
    def _get_scene_guidance(self, scene_description: str, language: str = "ar") -> str:
        """根据场景和语言获取纠错指导"""
        
        # 多语言场景指导
        scene_guides = {
            "ar": {
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
            },
            "zh": {
                "足球比赛": """
足球场景术语指导:
- 使用标准足球术语: 足球、进球、球员、比赛、射门、传球
- 保持球队称呼一致性: 主队、客队，或具体队名
- 注意比分和时间表达的准确性""",
                
                "新闻播报": """
新闻场景术语指导:
- 使用正式的政治术语: 总统、政府、部长、国务院
- 保持人名、地名的一致性
- 使用标准的新闻用语""",
                
                "商业新闻": """
商业场景术语指导:
- 使用标准商业术语: 公司、经济、市场、股票、投资
- 保持公司名称、品牌名称的一致性"""
            },
            "en": {
                "足球比赛": """
Football scene terminology guidance:
- Use standard football terms: football, goal, player, match, shot, pass
- Maintain team name consistency: home team, away team, or specific team names
- Pay attention to score and time expressions""",
                
                "新闻播报": """
News scene terminology guidance:
- Use formal political terms: president, government, minister, administration
- Maintain consistency in person names and place names
- Use standard news language""",
                
                "商业新闻": """
Business scene terminology guidance:
- Use standard business terms: company, economy, market, stock, investment
- Maintain consistency in company names and brand names"""
            }
        }
        
        lang_guides = scene_guides.get(language, scene_guides["ar"])
        return lang_guides.get(scene_description, "请根据上下文进行适当的纠错。")
    
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
                # 确保结果包含所有必要的字段
                if "has_split" not in result:
                    result["has_split"] = False
                if "split_subtitles" not in result:
                    result["split_subtitles"] = []
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
                    "details": "Bedrock纠错",
                    "has_split": False,
                    "split_subtitles": []
                }
                
        except Exception as e:
            print(f"解析Bedrock响应失败: {e}")
            return {
                "corrected_text": original_text,
                "has_correction": False,
                "confidence": 0.5,
                "details": f"解析失败: {str(e)}",
                "has_split": False,
                "split_subtitles": []
            }
    
    async def _mock_correction(self, input_data: CorrectionInput) -> CorrectionOutput:
        """多语言模拟纠错（备用方案）"""
        
        original = input_data.current_subtitle.strip()
        corrected = original
        has_correction = False
        details = []
        
        # 获取语言特定的基本纠错规则
        lang_config = self.LANGUAGE_CONFIGS.get(input_data.language, self.LANGUAGE_CONFIGS["ar"])
        basic_fixes = lang_config["common_fixes"]
        
        # 应用基本纠错规则
        for wrong, right in basic_fixes.items():
            if wrong in corrected:
                corrected = corrected.replace(wrong, right)
                has_correction = True
                details.append(f"拼写纠正: {wrong} -> {right}")
        
        # 通用标点符号纠正
        import re
        new_text = re.sub(r'\s+([.!?])', r'\1', corrected)  # 移除标点前的空格
        new_text = re.sub(r'\s+', ' ', new_text)  # 合并多个空格
        if new_text != corrected:
            corrected = new_text
            has_correction = True
            details.append("标点符号纠正")
        
        # 模拟长句拆分
        has_split = False
        split_subtitles = []
        
        # 简单的长句拆分逻辑：按标点符号拆分
        if len(corrected) > 20:  # 假设超过20个字符的句子需要拆分
            # 按标点符号拆分
            potential_splits = re.split(r'([.!?;,，。！？；])', corrected)
            
            # 重组带标点的句子片段
            segments = []
            for i in range(0, len(potential_splits)-1, 2):
                if i+1 < len(potential_splits):
                    segments.append(potential_splits[i] + potential_splits[i+1])
                else:
                    segments.append(potential_splits[i])
            
            # 如果最后一个元素没有配对，添加它
            if len(potential_splits) % 2 == 1:
                segments.append(potential_splits[-1])
            
            # 合并短片段，形成2-3个均衡的句子
            if len(segments) >= 2:
                has_split = True
                
                # 简单策略：尽量平均分配
                if len(segments) == 2:
                    split_subtitles = segments
                elif len(segments) == 3:
                    split_subtitles = segments
                else:
                    # 如果有更多片段，尝试合并成2-3个句子
                    mid_point = len(segments) // 2
                    split_subtitles = [
                        ''.join(segments[:mid_point]),
                        ''.join(segments[mid_point:])
                    ]
        
        # 模拟处理延迟
        await asyncio.sleep(0.1)
        
        return CorrectionOutput(
            corrected_subtitle=corrected,
            has_correction=has_correction,
            confidence=0.9 if has_correction else 1.0,
            correction_details="; ".join(details) if details else f"模拟{lang_config['name']}纠错",
            split_subtitles=split_subtitles,
            has_split=has_split
        )
    
    def get_service_name(self) -> str:
        """获取服务名称"""
        return f"BedrockCorrection({self.model_id})"
