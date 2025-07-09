"""字幕处理模块，用于处理和管理字幕"""

import logging
import json
import os
import uuid
import asyncio
import time
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..subtitle.models import Subtitle
from ..correction import BedrockCorrectionService, CorrectionInput
from ..models.whisper_language_config import get_correction_scene_description
from translation_service import translation_manager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 字幕输出目录
subtitle_dir = Path("./subtitles")
os.makedirs(subtitle_dir, exist_ok=True)

class SubtitleProcessor:
    """字幕处理器，用于处理和管理字幕"""
    
    def __init__(self, correction_service=None):
        """初始化字幕处理器
        
        Args:
            correction_service: 纠错服务实例
        """
        self.correction_service = correction_service
        self.client_subtitles: Dict[str, List[Subtitle]] = {}
        self.client_timestamps: Dict[str, Dict] = {}
    
    def register_client(self, client_id: str):
        """注册新客户端
        
        Args:
            client_id: 客户端ID
        """
        self.client_subtitles[client_id] = []
        self.client_timestamps[client_id] = {}
    
    def unregister_client(self, client_id: str):
        """注销客户端
        
        Args:
            client_id: 客户端ID
        """
        if client_id in self.client_subtitles:
            del self.client_subtitles[client_id]
        if client_id in self.client_timestamps:
            del self.client_timestamps[client_id]
    
    async def process_timestamp_message(self, client_id: str, message_data: dict):
        """处理时间戳消息
        
        Args:
            client_id: 客户端ID
            message_data: 消息数据
            
        Returns:
            bool: 处理是否成功
        """
        try:
            timestamp_info = message_data.get('timestamp', {})
            
            # 存储时间戳信息
            if client_id not in self.client_timestamps:
                self.client_timestamps[client_id] = {}
            
            chunk_index = timestamp_info.get('chunk_index', 0)
            self.client_timestamps[client_id][chunk_index] = {
                'start_time': timestamp_info.get('start_time', 0.0),
                'end_time': timestamp_info.get('end_time', 0.0),
                'duration': timestamp_info.get('duration', 0.0),
                'chunk_index': chunk_index,
                'total_samples_processed': timestamp_info.get('total_samples_processed', 0),
                'audio_start_time': timestamp_info.get('audio_start_time', 0.0),
                'processing_start_time': timestamp_info.get('processing_start_time', 0.0),
                'current_time': timestamp_info.get('current_time', 0.0),
                'received_at': datetime.datetime.now().isoformat()
            }
            
            logger.info(f"客户端 {client_id} 时间戳信息已存储:")
            logger.info(f"  - Chunk {chunk_index}: {timestamp_info.get('start_time', 0.0):.2f}s - {timestamp_info.get('end_time', 0.0):.2f}s")
            
            return True
        except Exception as e:
            logger.error(f"处理时间戳消息失败: {e}")
            return False
    
    async def get_chunk_timestamp(self, client_id: str, chunk_index: int) -> Optional[Dict]:
        """获取指定chunk的时间戳信息
        
        Args:
            client_id: 客户端ID
            chunk_index: chunk索引
            
        Returns:
            Dict: 时间戳信息
            None: 如果不存在
        """
        if client_id in self.client_timestamps and chunk_index in self.client_timestamps[client_id]:
            return self.client_timestamps[client_id][chunk_index]
        return None
    
    async def apply_timestamp_to_subtitle(self, subtitle: Subtitle, timestamp_info: Dict) -> Subtitle:
        """将时间戳信息应用到字幕对象
        
        Args:
            subtitle: 字幕对象
            timestamp_info: 时间戳信息
            
        Returns:
            Subtitle: 更新后的字幕对象
        """
        if timestamp_info:
            # 获取相对时间戳（相对于视频开始的时间）
            # 如果时间戳是绝对时间（Unix时间戳），则转换为相对时间
            start_time = timestamp_info['start_time']
            end_time = timestamp_info['end_time']
            
            # 检查是否为绝对时间戳（Unix时间戳通常大于1000000000）
            if start_time > 1000000000:
                # 计算相对时间
                # 使用chunk_index作为基准，每个chunk的时长通常是固定的
                chunk_index = timestamp_info.get('chunk_index', 0)
                chunk_duration = timestamp_info.get('duration', 3.0)  # 默认3秒
                
                # 计算相对时间（从视频开始计时）
                relative_start = chunk_index * chunk_duration
                relative_end = relative_start + chunk_duration
                
                logger.info(f"转换绝对时间戳为相对时间: {start_time:.2f}s -> {relative_start:.2f}s, {end_time:.2f}s -> {relative_end:.2f}s")
                
                # 更新时间戳
                subtitle.start = relative_start
                subtitle.end = relative_end
            else:
                # 已经是相对时间，直接使用
                subtitle.start = start_time
                subtitle.end = end_time
            
            # 确保结束时间大于开始时间
            if subtitle.end <= subtitle.start:
                subtitle.end = subtitle.start + timestamp_info.get('duration', 3.0)
            
            # 记录详细的时间戳信息到日志
            logger.info(f"应用时间戳到字幕: {subtitle.start:.2f}s - {subtitle.end:.2f}s")
            logger.info(f"时间戳详情: chunk_index={timestamp_info.get('chunk_index')}, "
                       f"duration={timestamp_info.get('duration'):.2f}s, "
                       f"total_samples={timestamp_info.get('total_samples_processed')}")
            
            # 将字幕时间戳信息记录到segments.log文件
            with open('segments.log', 'a') as f:
                f.write(f"SUBTITLE_TIMESTAMP: text=\"{subtitle.text}\", start={subtitle.start:.2f}s, end={subtitle.end:.2f}s, chunk_index={timestamp_info.get('chunk_index')}, duration={timestamp_info.get('duration', 0):.2f}s, is_relative={timestamp_info.get('is_relative_time', False)}, original_start={start_time:.2f}s, original_end={end_time:.2f}s\n")
        
        return subtitle
    
    async def process_subtitle(
        self,
        subtitle: Subtitle,
        client_id: str,
        language: str = "ar",
        enable_correction: bool = True,
        enable_translation: bool = True,
        target_language: str = "en"
    ) -> Dict:
        """处理字幕，包括纠错和翻译
        
        Args:
            subtitle: 字幕对象
            client_id: 客户端ID
            language: 字幕语言
            enable_correction: 是否启用纠错
            enable_translation: 是否启用翻译
            target_language: 目标翻译语言
            
        Returns:
            Dict: 处理结果，包含字幕信息和处理状态
        """
        try:
            # 创建唯一ID
            subtitle_id = f"{client_id}_{uuid.uuid4()}"
            logger.info(f"处理字幕 - 原文: {subtitle.text}")
            logger.info(f"处理参数 - 纠错: {enable_correction}, 翻译: {enable_translation}, 目标语言: {target_language}")
            
            # 将字幕添加到客户端的字幕列表中
            if client_id in self.client_subtitles:
                self.client_subtitles[client_id].append(subtitle)
            
            # 步骤1: 字幕纠错 (如果启用)
            corrected_text = subtitle.text
            correction_applied = False
            split_subtitles = []
            has_split = False
            
            if enable_correction and subtitle.text.strip() and self.correction_service:
                try:
                    # 获取历史字幕作为上下文
                    history_subtitles = []
                    if client_id in self.client_subtitles and len(self.client_subtitles[client_id]) > 1:
                        # 获取最近的3条历史字幕
                        recent_subtitles = self.client_subtitles[client_id][-4:-1]  # 排除当前字幕
                        history_subtitles = [s.text for s in recent_subtitles if s.text.strip()]
                    
                    # 构建纠错输入
                    correction_input = CorrectionInput(
                        current_subtitle=subtitle.text,
                        history_subtitles=history_subtitles,
                        scene_description=get_correction_scene_description(language),
                        language=language
                    )
                    
                    # 执行纠错
                    correction_result = await self.correction_service.correct(correction_input)
                    
                    if correction_result.has_correction:
                        corrected_text = correction_result.corrected_subtitle
                        correction_applied = True
                        logger.info(f"字幕已纠错: '{subtitle.text}' -> '{corrected_text}' (置信度: {correction_result.confidence})")
                        
                        # 更新subtitle对象中的文本
                        subtitle.text = corrected_text
                    else:
                        logger.info(f"字幕无需纠错: '{subtitle.text}'")
                    
                    # 处理长句拆分结果
                    if correction_result.has_split and correction_result.split_subtitles:
                        has_split = True
                        split_subtitles = correction_result.split_subtitles
                        logger.info(f"字幕已拆分为 {len(split_subtitles)} 个子句: {split_subtitles}")
                        
                except Exception as e:
                    logger.error(f"字幕纠错失败: {e}")
                    # 纠错失败时使用原始文本
                    corrected_text = subtitle.text
            else:
                logger.info(f"跳过纠错 - 启用状态: {enable_correction}")
            
            # 如果没有拆分，按原流程处理单个字幕
            if not has_split:
                # 步骤2: 翻译字幕文本 (如果启用，使用纠错后的文本)
                if enable_translation and corrected_text.strip():
                    try:
                        # 使用翻译服务翻译文本
                        translation_result = await translation_manager.translate(
                            text=corrected_text,
                            target_lang=target_language,
                            service="bedrock"  # 优先使用Bedrock翻译服务
                        )
                        
                        # 设置翻译结果
                        subtitle.translated_text = translation_result.translated_text
                        logger.info(f"字幕已翻译: {corrected_text} -> {subtitle.translated_text}")
                    except Exception as e:
                        logger.error(f"翻译失败: {e}")
                        subtitle.translated_text = f"[翻译失败] {corrected_text}"
                else:
                    logger.info(f"跳过翻译 - 启用状态: {enable_translation}")
                    subtitle.translated_text = None
                
                # 返回单个字幕处理结果
                return {
                    "type": "subtitle",
                    "subtitle": {
                        "id": subtitle_id,
                        "start": subtitle.start,
                        "end": subtitle.end,
                        "text": subtitle.text,  # 纠错后的文本
                        "original_text": subtitle.text if not correction_applied else None,  # 原始文本(如果有纠错)
                        "translated_text": subtitle.translated_text,
                        "correction_applied": correction_applied,
                        "translation_applied": enable_translation and subtitle.translated_text is not None,
                        "target_language": target_language if enable_translation else None
                    }
                }
            else:
                # 处理拆分后的多个字幕
                # 计算每个子句的时间分配
                total_duration = subtitle.duration
                sub_duration = total_duration / len(split_subtitles)
                
                # 为每个拆分的字幕创建新的Subtitle对象并返回结果列表
                split_results = []
                for i, split_text in enumerate(split_subtitles):
                    # 计算子句的时间范围
                    sub_start = subtitle.start + i * sub_duration
                    sub_end = sub_start + sub_duration
                    
                    # 创建新的Subtitle对象
                    sub_subtitle = Subtitle(
                        start=sub_start,
                        end=sub_end,
                        text=split_text
                    )
                    
                    # 添加到客户端的字幕列表
                    if client_id in self.client_subtitles:
                        self.client_subtitles[client_id].append(sub_subtitle)
                    
                    # 翻译子句
                    if enable_translation and split_text.strip():
                        try:
                            # 使用翻译服务翻译文本
                            translation_result = await translation_manager.translate(
                                text=split_text,
                                target_lang=target_language,
                                service="bedrock"
                            )
                            
                            # 设置翻译结果
                            sub_subtitle.translated_text = translation_result.translated_text
                            logger.info(f"拆分字幕已翻译: {split_text} -> {sub_subtitle.translated_text}")
                        except Exception as e:
                            logger.error(f"拆分字幕翻译失败: {e}")
                            sub_subtitle.translated_text = f"[翻译失败] {split_text}"
                    
                    # 添加拆分后的字幕结果
                    sub_id = f"{client_id}_{uuid.uuid4()}"
                    split_results.append({
                        "type": "subtitle",
                        "subtitle": {
                            "id": sub_id,
                            "start": sub_subtitle.start,
                            "end": sub_subtitle.end,
                            "text": sub_subtitle.text,
                            "original_text": None,  # 已经是纠错后的文本
                            "translated_text": sub_subtitle.translated_text,
                            "correction_applied": True,  # 拆分的字幕都是经过纠错的
                            "translation_applied": enable_translation and sub_subtitle.translated_text is not None,
                            "target_language": target_language if enable_translation else None,
                            "is_split": True,  # 标记为拆分字幕
                            "split_index": i,  # 拆分序号
                            "split_total": len(split_subtitles),  # 拆分总数
                            "original_subtitle_id": subtitle_id  # 原始字幕ID
                        }
                    })
                
                # 返回拆分结果
                return {
                    "type": "split_subtitles",
                    "subtitles": split_results,
                    "original_id": subtitle_id,
                    "count": len(split_results)
                }
                
        except Exception as e:
            logger.error(f"处理字幕失败: {e}")
            return {
                "type": "error",
                "message": f"处理字幕失败: {str(e)}"
            }
    
    def save_subtitles(self, client_id: str, filename: str, language: str):
        """保存字幕到文件
        
        Args:
            client_id: 客户端ID
            filename: 文件名
            language: 语言代码
            
        Returns:
            bool: 保存是否成功
        """
        if client_id not in self.client_subtitles or not self.client_subtitles[client_id]:
            logger.warning("没有字幕可保存")
            return False
        
        subtitles = self.client_subtitles[client_id]
        
        # 确保文件名不包含非法字符
        filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
        
        try:
            # 保存SRT格式
            srt_path = subtitle_dir / f"{filename}_{language}.srt"
            with open(srt_path, "w", encoding="utf-8") as f:
                for i, subtitle in enumerate(subtitles, 1):
                    f.write(f"{i}\n")
                    f.write(f"{subtitle.format_time(subtitle.start, 'srt')} --> {subtitle.format_time(subtitle.end, 'srt')}\n")
                    f.write(f"{subtitle.text}\n")
                    if subtitle.translated_text:
                        f.write(f"{subtitle.translated_text}\n")
                    f.write("\n")
            
            # 保存VTT格式
            vtt_path = subtitle_dir / f"{filename}_{language}.vtt"
            with open(vtt_path, "w", encoding="utf-8") as f:
                f.write("WEBVTT\n\n")
                for subtitle in subtitles:
                    f.write(f"{subtitle.format_time(subtitle.start, 'vtt')} --> {subtitle.format_time(subtitle.end, 'vtt')}\n")
                    f.write(f"{subtitle.text}\n")
                    if subtitle.translated_text:
                        f.write(f"{subtitle.translated_text}\n")
                    f.write("\n")
            
            # 保存JSON格式（包含更多元数据）
            json_path = subtitle_dir / f"{filename}_{language}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "language": language,
                        "subtitles": [subtitle.to_dict() for subtitle in subtitles]
                    },
                    f,
                    ensure_ascii=False,
                    indent=2
                )
            
            logger.info(f"字幕已保存为SRT格式: {srt_path}")
            logger.info(f"字幕已保存为VTT格式: {vtt_path}")
            logger.info(f"字幕已保存为JSON格式: {json_path}")
            
            return True
        except Exception as e:
            logger.error(f"保存字幕文件失败: {e}")
            return False
