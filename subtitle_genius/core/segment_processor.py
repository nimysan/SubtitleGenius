"""音频段处理器模块"""

import asyncio
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Tuple, Union
from loguru import logger
import numpy as np

from ..subtitle.models import Subtitle
from ..correction.base import CorrectionInput, CorrectionOutput
from ..correction.factory import create_corrector
from ..translation.utils import translate_text, batch_translate
from .generator import SubtitleGenerator
from .segment_processor_config import AudioSegmentProcessorConfig, segment_processor_config


class AudioSegmentProcessor:
    """音频段处理器，实现transcribe-correction-translate处理流程"""
    
    def __init__(self, config: Optional[AudioSegmentProcessorConfig] = None):
        """
        初始化音频段处理器
        
        Args:
            config: 处理器配置，如果为None则使用默认配置
        """
        self.config = config or segment_processor_config
        
        # 初始化转录组件
        self.subtitle_generator = SubtitleGenerator(
            model=self.config.transcribe_model,
            language=self.config.source_language,
            output_format=self.config.output_format
        )
        
        # 初始化校正组件(如果启用)
        self.correction_service = None
        if self.config.correction_enabled:
            self.correction_service = create_corrector(
                service_type=self.config.correction_service,
                language=self.config.source_language,
                **self.config.get_correction_kwargs()
            )
        
        # 历史字幕，用于校正
        self.history_subtitles = []
        
        logger.info(f"AudioSegmentProcessor initialized with config: {self.config}")
    
    async def process_segment(
        self, 
        audio_segment: np.ndarray, 
        timestamp: float
    ) -> Subtitle:
        """
        处理单个音频段
        
        Args:
            audio_segment: 音频段数据
            timestamp: 音频段开始时间戳(秒)
            
        Returns:
            Subtitle: 处理后的字幕
        """
        # 1. 转录
        subtitle = await self.transcribe_segment(audio_segment, timestamp)
        
        # 2. 校正(如果启用)
        if self.config.correction_enabled and self.correction_service:
            subtitle = await self._correct_single_subtitle(subtitle)
        # 3. 翻译(如果启用)
        if self.config.translation_enabled:
            subtitle = await self._translate_single_subtitle(subtitle)
            
        return subtitle
    
    async def transcribe_segment(
        self, 
        audio_segment: np.ndarray, 
        timestamp: float
    ) -> Subtitle:
        """
        转录单个音频段
        
        Args:
            audio_segment: 音频段数据
            timestamp: 音频段开始时间戳(秒)
            
        Returns:
            Subtitle: 转录后的字幕
        """
        try:
            # 计算结束时间
            duration = len(audio_segment) / self.subtitle_generator.audio_processor.sample_rate
            end_time = timestamp + duration
            
            # 使用模型转录音频段
            # 注意：WhisperSageMakerStreamingModel使用transcribe_audio方法
            text = await self.subtitle_generator.model.transcribe(
                audio_segment, 
                self.config.source_language
            )
            
            # 创建字幕对象列表
            subtitles = [Subtitle(start=timestamp, end=timestamp+duration, text=text)] if text else []
            
            # 如果有转录结果，使用第一个字幕的文本
            if subtitles and len(subtitles) > 0:
                text = subtitles[0].text
            else:
                text = ""
            
            # 创建字幕对象
            subtitle = Subtitle(
                start=timestamp,
                end=end_time,
                text=text
            )
            
            return subtitle
            
        except Exception as e:
            logger.error(f"Error in segment transcription: {e}")
            # 创建空字幕对象
            return Subtitle(
                start=timestamp,
                end=timestamp + 1.0,  # 默认1秒
                text=""
            )
    
    async def process_file(
        self, 
        file_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> List[Subtitle]:
        """
        处理音频/视频文件
        
        Args:
            file_path: 文件路径
            output_path: 输出路径
            
        Returns:
            List[Subtitle]: 处理后的字幕列表
        """
        file_path = Path(file_path)
        
        logger.info(f"Processing file: {file_path}")
        
        # 加载音频文件
        audio_data = await self.subtitle_generator.audio_processor.process_file(file_path)
        
        # 分段处理
        segments = self._segment_audio(audio_data)
        
        # 处理每个段
        subtitles = []
        for segment, timestamp in segments:
            subtitle = await self.process_segment(segment, timestamp)
            if subtitle.text.strip():  # 只添加非空字幕
                subtitles.append(subtitle)
        
        logger.info(f"Generated {len(subtitles)} subtitles")
        
        # 保存字幕
        if output_path:
            self.subtitle_generator.save_subtitles(subtitles, output_path)
            logger.info(f"Subtitles saved to: {output_path}")
        
        return subtitles
    
    async def process_realtime_stream(
        self, 
        audio_segment_stream: AsyncGenerator[Tuple[np.ndarray, float], None]
    ) -> AsyncGenerator[Subtitle, None]:
        """
        实时处理音频段流
        
        Args:
            audio_segment_stream: 音频段流，每个元素是(音频段数据, 时间戳)的元组
            
        Yields:
            Subtitle: 处理后的字幕
        """
        logger.info("Starting real-time audio segment processing")
        
        async for segment, timestamp in audio_segment_stream:
            # 处理音频段
            subtitle = await self.process_segment(segment, timestamp)
            
            # 只输出非空字幕
            if subtitle.text.strip():
                yield subtitle
    
    async def _correct_single_subtitle(self, subtitle: Subtitle) -> Subtitle:
        """
        校正单个字幕
        
        Args:
            subtitle: 原始字幕
            
        Returns:
            Subtitle: 校正后的字幕
        """
        # 如果字幕为空，直接返回
        if not subtitle.text.strip():
            return subtitle
            
        # 准备校正输入
        input_data = CorrectionInput(
            current_subtitle=subtitle.text,
            history_subtitles=self.history_subtitles[-5:] if self.history_subtitles else [],
            scene_description=self.config.scene_description,
            language=self.config.source_language
        )
        
        try:
            # 执行校正
            correction_result = await self.correction_service.correct(input_data)
            
            # 更新字幕文本
            subtitle.text = correction_result.corrected_subtitle
            
            # 添加到历史
            self.history_subtitles.append(subtitle.text)
            
        except Exception as e:
            logger.error(f"Error in subtitle correction: {e}")
        
        return subtitle
    
    async def _translate_single_subtitle(self, subtitle: Subtitle) -> Subtitle:
        """
        翻译单个字幕
        
        Args:
            subtitle: 字幕
            
        Returns:
            Subtitle: 翻译后的字幕
        """
        # 如果字幕为空，直接返回
        if not subtitle.text.strip():
            return subtitle
            
        try:
            # 翻译
            translated_text = await translate_text(
                text=subtitle.text,
                source_language=self.config.source_language,
                target_language=self.config.target_language,
                context=self.config.scene_description,
                service=self.config.translation_service
            )
            
            # 更新字幕的翻译文本
            subtitle.translated_text = translated_text
            
        except Exception as e:
            logger.error(f"Error in subtitle translation: {e}")
        
        return subtitle
    
    def _segment_audio(self, audio_data: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """
        将音频数据分段
        
        Args:
            audio_data: 音频数据
            
        Returns:
            List[Tuple[np.ndarray, float]]: 音频段列表，每个元素是(音频段数据, 时间戳)的元组
        """
        sample_rate = self.subtitle_generator.audio_processor.sample_rate
        segment_size = int(self.config.segment_duration_ms * sample_rate / 1000)
        overlap_size = int(self.config.segment_overlap_ms * sample_rate / 1000)
        min_segment_size = int(self.config.min_segment_duration_ms * sample_rate / 1000)
        
        segments = []
        pos = 0
        
        while pos < len(audio_data):
            # 计算段结束位置
            end_pos = min(pos + segment_size, len(audio_data))
            
            # 如果剩余音频长度小于最小段长度，合并到上一段
            if end_pos - pos < min_segment_size and segments:
                last_segment, last_timestamp = segments[-1]
                combined = np.concatenate([last_segment, audio_data[pos:end_pos]])
                segments[-1] = (combined, last_timestamp)
            else:
                # 创建新段
                timestamp = pos / sample_rate
                segments.append((audio_data[pos:end_pos], timestamp))
            
            # 移动位置，考虑重叠
            pos = end_pos - overlap_size
            if pos >= len(audio_data):
                break
        
        return segments
    
    async def process_video(
        self, 
        video_path: Union[str, Path], 
        output_path: Optional[Union[str, Path]] = None
    ) -> List[Subtitle]:
        """
        处理视频文件
        
        Args:
            video_path: 视频文件路径
            output_path: 输出路径
            
        Returns:
            List[Subtitle]: 处理后的字幕列表
        """
        video_path = Path(video_path)
        
        if output_path is None:
            output_path = video_path.with_suffix(f'.{self.config.output_format}')
        
        logger.info(f"Processing video: {video_path}")
        
        # 从视频中提取音频
        audio_data = await self.subtitle_generator.audio_processor.extract_from_video(video_path)
        
        # 分段处理
        segments = self._segment_audio(audio_data)
        
        # 处理每个段
        subtitles = []
        for segment, timestamp in segments:
            subtitle = await self.process_segment(segment, timestamp)
            if subtitle.text.strip():  # 只添加非空字幕
                subtitles.append(subtitle)
        
        logger.info(f"Generated {len(subtitles)} subtitles")
        
        # 保存字幕
        self.subtitle_generator.save_subtitles(subtitles, output_path)
        logger.info(f"Subtitles saved to: {output_path}")
        
        return subtitles
