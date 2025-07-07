#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DASH字幕处理管道 - 主控制脚本
将webm视频转换为带字幕的DASH流
"""

import subprocess
import os
import asyncio
import argparse
import tempfile
import signal
import sys
import logging
import time
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dash_pipeline')

# 获取脚本目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

async def process_webm_to_dash(
    input_webm,
    output_dir,
    language="ar",
    target_language="zh",
    delay_seconds=20,
    segment_duration=4,
    endpoint_name="your-sagemaker-endpoint",
    region="us-east-1",
    verbose=False
):
    """
    处理webm文件，生成带字幕的DASH流
    
    参数:
        input_webm: 输入webm文件路径
        output_dir: 输出DASH目录
        language: 源语言
        target_language: 目标翻译语言
        delay_seconds: 视频延迟秒数
        segment_duration: DASH分段时长(秒)
        endpoint_name: SageMaker端点名称
        region: AWS区域
        verbose: 是否输出详细日志
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    # 验证输入文件
    if not os.path.exists(input_webm):
        logger.error(f"输入文件不存在: {input_webm}")
        return False
    
    # 创建临时目录
    temp_dir = Path(tempfile.mkdtemp())
    audio_pipe = temp_dir / "audio.pipe"
    video_pipe = temp_dir / "video.pipe"
    subtitle_path = temp_dir / "subtitles.vtt"
    
    logger.info(f"创建临时目录: {temp_dir}")
    
    # 创建命名管道
    try:
        os.mkfifo(audio_pipe)
        os.mkfifo(video_pipe)
        logger.debug("命名管道已创建")
    except Exception as e:
        logger.error(f"创建命名管道失败: {e}")
        return False
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    logger.debug(f"输出目录: {output_dir}")
    
    # 启动进程列表
    processes = []
    
    try:
        # 1. 启动音频提取进程
        audio_cmd = [
            "ffmpeg", 
            "-i", input_webm, 
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            "-f", "wav", str(audio_pipe)
        ]
        
        if not verbose:
            audio_cmd.insert(1, '-v')
            audio_cmd.insert(2, 'error')
            
        logger.info("启动音频提取进程...")
        logger.debug(f"音频提取命令: {' '.join(audio_cmd)}")
        
        audio_process = subprocess.Popen(audio_cmd)
        processes.append(audio_process)
        logger.info("音频提取进程已启动")
        
        # 2. 启动字幕生成进程
        subtitle_cmd = [
            sys.executable, os.path.join(SCRIPT_DIR, "subtitle_generator.py"),
            "--audio", f"pipe:{audio_pipe}",
            "--output", str(subtitle_path),
            "--language", language,
            "--target", target_language,
            "--endpoint", endpoint_name,
            "--region", region
        ]
        
        if verbose:
            subtitle_cmd.append("--verbose")
            
        logger.info("启动字幕生成进程...")
        logger.debug(f"字幕生成命令: {' '.join(subtitle_cmd)}")
        
        subtitle_process = subprocess.Popen(subtitle_cmd)
        processes.append(subtitle_process)
        logger.info("字幕生成进程已启动")
        
        # 3. 启动视频缓冲进程
        buffer_cmd = [
            sys.executable, os.path.join(SCRIPT_DIR, "video_buffer.py"),
            "--input", input_webm,
            "--output", f"pipe:{video_pipe}",
            "--delay", str(delay_seconds)
        ]
        
        if verbose:
            buffer_cmd.append("--verbose")
            
        logger.info("启动视频缓冲进程...")
        logger.debug(f"视频缓冲命令: {' '.join(buffer_cmd)}")
        
        buffer_process = subprocess.Popen(buffer_cmd)
        processes.append(buffer_process)
        logger.info(f"视频缓冲进程已启动，延迟设置为{delay_seconds}秒")
        
        # 等待字幕文件创建
        subtitle_wait_time = 0
        max_wait_time = 60  # 最多等待60秒
        
        while not os.path.exists(subtitle_path) and subtitle_wait_time < max_wait_time:
            await asyncio.sleep(1)
            subtitle_wait_time += 1
            if subtitle_wait_time % 5 == 0:
                logger.info(f"等待字幕文件创建... ({subtitle_wait_time}秒)")
        
        if not os.path.exists(subtitle_path):
            logger.warning(f"字幕文件未创建，将继续处理（可能没有字幕）")
            # 创建一个空的字幕文件
            with open(subtitle_path, "w", encoding="utf-8") as f:
                f.write("WEBVTT\n\n")
        else:
            logger.info("字幕文件已创建")
        
        # 4. 启动DASH封装进程
        dash_cmd = [
            "ffmpeg", 
            "-i", f"pipe:{video_pipe}",  # 延迟后的视频
            "-i", str(subtitle_path),    # 字幕文件
            "-map", "0:v", "-map", "0:a", "-map", "1",  # 映射流
            "-c:v", "copy",              # 复制视频编码
            "-c:a", "aac", "-b:a", "128k",  # 音频编码
            "-c:s", "webvtt",            # 字幕编码
            "-f", "dash",                # DASH格式
            "-seg_duration", str(segment_duration),  # 分段时长
            "-streaming", "1",           # 流式输出
            "-use_template", "1",        # 使用模板
            "-use_timeline", "1",        # 使用时间线
            "-window_size", "5",         # 窗口大小
            "-adaptation_sets", "id=0,streams=v id=1,streams=a id=2,streams=s",  # 自适应集
            f"{output_dir}/manifest.mpd"  # 输出文件
        ]
        
        if not verbose:
            dash_cmd.insert(1, '-v')
            dash_cmd.insert(2, 'error')
            
        logger.info("启动DASH封装进程...")
        logger.debug(f"DASH封装命令: {' '.join(dash_cmd)}")
        
        dash_process = subprocess.Popen(dash_cmd)
        processes.append(dash_process)
        logger.info(f"DASH封装进程已启动，输出到 {output_dir}/manifest.mpd")
        
        # 等待所有进程完成
        logger.info("等待所有进程完成...")
        
        # 监控进程状态
        while any(p.poll() is None for p in processes):
            await asyncio.sleep(1)
            
            # 检查是否有进程异常退出
            for i, p in enumerate(processes):
                if p.poll() is not None and p.returncode != 0:
                    process_names = ["音频提取", "字幕生成", "视频缓冲", "DASH封装"]
                    if i < len(process_names):
                        logger.error(f"{process_names[i]}进程异常退出，返回码: {p.returncode}")
                    else:
                        logger.error(f"进程 #{i} 异常退出，返回码: {p.returncode}")
        
        logger.info("所有进程已完成")
        
        # 检查输出文件
        manifest_path = os.path.join(output_dir, "manifest.mpd")
        if os.path.exists(manifest_path):
            logger.info(f"DASH清单文件已生成: {manifest_path}")
            return manifest_path
        else:
            logger.error("DASH清单文件未生成")
            return False
            
    except KeyboardInterrupt:
        logger.info("处理中断")
        return False
    except Exception as e:
        logger.error(f"处理过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        # 清理进程
        for p in processes:
            try:
                if p.poll() is None:  # 如果进程还在运行
                    p.terminate()
                    p.wait(timeout=5)
            except:
                pass
        
        # 清理管道
        try:
            if os.path.exists(audio_pipe):
                os.unlink(audio_pipe)
            if os.path.exists(video_pipe):
                os.unlink(video_pipe)
        except:
            pass
            
        logger.info("处理完成，资源已清理")

def signal_handler(sig, frame):
    """处理信号"""
    logger.info("接收到中断信号，正在退出...")
    sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DASH流媒体字幕处理系统")
    parser.add_argument("--input", required=True, help="输入webm文件路径")
    parser.add_argument("--output", required=True, help="输出DASH目录")
    parser.add_argument("--language", default="ar", help="源语言")
    parser.add_argument("--target", default="zh", help="目标语言")
    parser.add_argument("--delay", type=int, default=20, help="视频延迟秒数")
    parser.add_argument("--segment", type=int, default=4, help="DASH分段时长(秒)")
    parser.add_argument("--endpoint", default="your-endpoint", help="SageMaker端点")
    parser.add_argument("--region", default="us-east-1", help="AWS区域")
    parser.add_argument("--verbose", action="store_true", help="输出详细日志")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 运行处理流程
    result = asyncio.run(process_webm_to_dash(
        args.input, args.output, args.language, args.target,
        args.delay, args.segment, args.endpoint, args.region,
        args.verbose
    ))
    
    if result:
        logger.info(f"处理成功，DASH清单文件: {result}")
        sys.exit(0)
    else:
        logger.error("处理失败")
        sys.exit(1)
