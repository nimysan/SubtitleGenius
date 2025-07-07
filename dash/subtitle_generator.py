#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
字幕生成器 - 从音频生成WebVTT字幕
"""

import asyncio
import argparse
import os
import subprocess
import tempfile
import logging
import time
import sys
import signal
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('subtitle_generator')

# 导入SubtitleGenius相关模块
try:
    from subtitle_genius.audio.processor import AudioProcessor
    from subtitle_genius.models.transcribe_model import TranscribeModel
    from subtitle_genius.models.whisper_sagemaker_streaming import WhisperSageMakerStreamConfig
    from subtitle_genius.subtitle.models import Subtitle
    from translation_service import translation_manager
    SUBTITLE_GENIUS_AVAILABLE = True
except ImportError:
    logger.warning("无法导入SubtitleGenius模块，将使用模拟模式")
    SUBTITLE_GENIUS_AVAILABLE = False
    
    # 定义模拟类
    class Subtitle:
        def __init__(self, start, end, text):
            self.start = start
            self.end = end
            self.text = text
            self.translated_text = None
            
        def format_time(self, time_seconds, format_type="vtt"):
            hours = int(time_seconds // 3600)
            minutes = int((time_seconds % 3600) // 60)
            seconds = int(time_seconds % 60)
            milliseconds = int((time_seconds % 1) * 1000)
            
            if format_type == "vtt":
                return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
            else:
                return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

class MockTranslationManager:
    """模拟翻译管理器"""
    async def translate(self, text, target_lang, service=None):
        class Result:
            def __init__(self, text):
                self.translated_text = f"[翻译] {text}"
        return Result(text)

# 如果翻译服务不可用，使用模拟翻译
if 'translation_manager' not in globals():
    translation_manager = MockTranslationManager()

async def generate_subtitles(
    audio_input,
    subtitle_output,
    language="ar",
    target_language="zh",
    endpoint_name="your-sagemaker-endpoint",
    region="us-east-1",
    verbose=False
):
    """
    从音频生成WebVTT字幕
    
    参数:
        audio_input: 输入音频路径或管道
        subtitle_output: 输出字幕路径或管道
        language: 源语言
        target_language: 目标翻译语言
        endpoint_name: SageMaker端点名称
        region: AWS区域
        verbose: 是否输出详细日志
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
        
    logger.info(f"初始化字幕生成器，语言: {language} -> {target_language}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(os.path.abspath(subtitle_output))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"创建输出目录: {output_dir}")
    
    # 如果SubtitleGenius不可用，使用模拟模式
    if not SUBTITLE_GENIUS_AVAILABLE:
        return await mock_generate_subtitles(
            audio_input, subtitle_output, language, target_language, verbose
        )
    
    # 初始化音频处理器
    audio_processor = AudioProcessor()
    
    # 如果输入是管道，先保存到临时文件
    temp_file = None
    if audio_input.startswith('pipe:'):
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        pipe_path = audio_input[5:]
        
        logger.info(f"从管道读取音频: {pipe_path}")
        
        # 从管道读取音频并保存
        try:
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', pipe_path, 
                '-f', 'wav', '-ar', '16000', '-ac', '1',
                temp_file.name
            ]
            
            if not verbose:
                ffmpeg_cmd.insert(1, '-v')
                ffmpeg_cmd.insert(2, 'error')
                
            subprocess.run(ffmpeg_cmd, check=True)
            logger.debug(f"音频已保存到临时文件: {temp_file.name}")
            
            audio_path = temp_file.name
        except Exception as e:
            logger.error(f"处理音频管道失败: {e}")
            if temp_file:
                os.unlink(temp_file.name)
            return False
    else:
        audio_path = audio_input
    
    try:
        # 加载音频文件
        logger.info(f"加载音频文件: {audio_path}")
        audio_data = await audio_processor.process_file(audio_path)
        
        # 配置Whisper模型
        config = WhisperSageMakerStreamConfig(
            chunk_duration=30,
            overlap_duration=2,
            voice_threshold=0.01,
            sagemaker_chunk_duration=30
        )
        
        # 初始化转录模型
        logger.info(f"初始化转录模型: {endpoint_name} ({region})")
        model = TranscribeModel(
            backend="sagemaker_whisper",
            sagemaker_endpoint=endpoint_name,
            region_name=region,
            whisper_config=config
        )
        
        # 创建异步生成器
        async def audio_generator():
            yield audio_data
        
        # 打开字幕输出文件
        with open(subtitle_output, "w", encoding="utf-8") as f:
            # 写入WebVTT头
            f.write("WEBVTT\n\n")
            
            # 生成字幕
            subtitle_count = 0
            logger.info("开始生成字幕...")
            
            async for subtitle in model.transcribe_stream(audio_generator(), language=language):
                # 翻译字幕
                if target_language != language and subtitle.text.strip():
                    try:
                        logger.debug(f"翻译文本: {subtitle.text}")
                        translation_result = await translation_manager.translate(
                            text=subtitle.text,
                            target_lang=target_language,
                            service="bedrock"  # 或其他可用服务
                        )
                        subtitle.translated_text = translation_result.translated_text
                        logger.debug(f"翻译结果: {subtitle.translated_text}")
                    except Exception as e:
                        logger.error(f"翻译失败: {e}")
                        subtitle.translated_text = f"[翻译失败] {subtitle.text}"
                
                # 写入WebVTT格式
                f.write(f"{subtitle.format_time(subtitle.start, 'vtt')} --> {subtitle.format_time(subtitle.end, 'vtt')}\n")
                f.write(f"{subtitle.text}\n")
                if subtitle.translated_text:
                    f.write(f"{subtitle.translated_text}\n")
                f.write("\n")
                
                # 确保文件立即写入（对于流式处理很重要）
                f.flush()
                os.fsync(f.fileno())
                
                subtitle_count += 1
                if subtitle_count % 5 == 0:
                    logger.info(f"已生成 {subtitle_count} 条字幕")
        
        logger.info(f"字幕生成完成，共 {subtitle_count} 条字幕")
        return True
        
    except Exception as e:
        logger.error(f"生成字幕时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        # 清理临时文件
        if temp_file:
            try:
                os.unlink(temp_file.name)
                logger.debug("临时文件已删除")
            except:
                pass

async def mock_generate_subtitles(
    audio_input, 
    subtitle_output, 
    language="ar", 
    target_language="zh",
    verbose=False
):
    """
    模拟字幕生成（当SubtitleGenius不可用时使用）
    """
    logger.warning("使用模拟字幕生成模式")
    
    # 打开字幕输出文件
    with open(subtitle_output, "w", encoding="utf-8") as f:
        # 写入WebVTT头
        f.write("WEBVTT\n\n")
        
        # 生成模拟字幕
        for i in range(10):
            start_time = i * 5.0
            end_time = start_time + 4.5
            
            subtitle = Subtitle(
                start=start_time,
                end=end_time,
                text=f"这是第 {i+1} 条模拟字幕 ({language})"
            )
            
            # 模拟翻译
            if target_language != language:
                subtitle.translated_text = f"This is mock subtitle #{i+1} ({target_language})"
            
            # 写入WebVTT格式
            f.write(f"{subtitle.format_time(subtitle.start, 'vtt')} --> {subtitle.format_time(subtitle.end, 'vtt')}\n")
            f.write(f"{subtitle.text}\n")
            if subtitle.translated_text:
                f.write(f"{subtitle.translated_text}\n")
            f.write("\n")
            
            # 模拟处理时间
            await asyncio.sleep(0.5)
            
            if (i+1) % 5 == 0:
                logger.info(f"已生成 {i+1} 条模拟字幕")
    
    logger.info("模拟字幕生成完成")
    return True

def signal_handler(sig, frame):
    """处理信号"""
    logger.info("接收到中断信号，正在退出...")
    sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="音频字幕生成器")
    parser.add_argument("--audio", required=True, help="输入音频路径或管道")
    parser.add_argument("--output", required=True, help="输出字幕路径")
    parser.add_argument("--language", default="ar", help="源语言")
    parser.add_argument("--target", default="zh", help="目标语言")
    parser.add_argument("--endpoint", default="your-endpoint", help="SageMaker端点")
    parser.add_argument("--region", default="us-east-1", help="AWS区域")
    parser.add_argument("--verbose", action="store_true", help="输出详细日志")
    
    args = parser.parse_args()
    
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 运行字幕生成器
    asyncio.run(generate_subtitles(
        args.audio, 
        args.output, 
        args.language, 
        args.target, 
        args.endpoint, 
        args.region,
        args.verbose
    ))
