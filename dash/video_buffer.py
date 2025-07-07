#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频缓冲器 - 实现固定延迟的视频流处理
"""

import subprocess
import numpy as np
import time
from collections import deque
import os
import argparse
import sys
import signal
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('video_buffer')

def create_buffer(input_path, output_path, delay_seconds=20, frame_rate=30, resolution="1280x720", verbose=False):
    """
    创建视频缓冲器，实现固定延迟
    
    参数:
        input_path: 输入视频路径
        output_path: 输出视频路径
        delay_seconds: 延迟秒数
        frame_rate: 视频帧率
        resolution: 视频分辨率
        verbose: 是否输出详细日志
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    # 解析分辨率
    width, height = map(int, resolution.split('x'))
    
    # 计算帧大小和缓冲区容量
    frame_size = width * height * 3  # RGB24格式
    buffer_capacity = delay_seconds * frame_rate
    
    logger.info(f"初始化视频缓冲器 - 延迟: {delay_seconds}秒, 分辨率: {resolution}, 帧率: {frame_rate}")
    
    # 创建帧缓冲区
    frame_buffer = deque(maxlen=buffer_capacity)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"创建输出目录: {output_dir}")
    
    # 创建命名管道（如果输出路径是管道）
    if output_path.startswith('pipe:'):
        pipe_path = output_path[5:]
        if not os.path.exists(pipe_path):
            try:
                os.mkfifo(pipe_path)
                logger.debug(f"创建命名管道: {pipe_path}")
            except Exception as e:
                logger.error(f"创建命名管道失败: {e}")
                return False
    
    # 获取输入视频信息
    try:
        probe_cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate',
            '-of', 'csv=p=0', input_path
        ]
        probe_result = subprocess.check_output(probe_cmd).decode('utf-8').strip().split(',')
        
        if len(probe_result) >= 3:
            actual_width, actual_height, actual_fps = probe_result
            # 解析帧率（可能是分数形式）
            if '/' in actual_fps:
                num, den = map(int, actual_fps.split('/'))
                actual_fps = num / den if den != 0 else 30
            else:
                actual_fps = float(actual_fps)
                
            logger.info(f"检测到视频信息 - 分辨率: {actual_width}x{actual_height}, 帧率: {actual_fps}")
            
            # 更新参数
            if actual_width and actual_height:
                width, height = int(actual_width), int(actual_height)
                resolution = f"{width}x{height}"
            
            if actual_fps:
                frame_rate = actual_fps
                buffer_capacity = int(delay_seconds * frame_rate)
                
            # 重新计算帧大小
            frame_size = width * height * 3
    except Exception as e:
        logger.warning(f"无法获取视频信息，使用默认参数: {e}")
    
    # 从输入读取视频帧
    try:
        input_cmd = [
            'ffmpeg', '-i', input_path, 
            '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-vf', f'fps={frame_rate}', '-'
        ]
        
        if verbose:
            logger.debug(f"输入命令: {' '.join(input_cmd)}")
        else:
            input_cmd.insert(1, '-v')
            input_cmd.insert(2, 'error')
            
        input_process = subprocess.Popen(
            input_cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if not verbose else None
        )
        
        logger.debug("输入进程已启动")
    except Exception as e:
        logger.error(f"启动输入进程失败: {e}")
        return False
    
    # 向输出写入延迟后的视频帧
    try:
        output_cmd = [
            'ffmpeg', 
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', 
            '-s', resolution, '-r', str(frame_rate), 
            '-i', '-', 
            '-c:v', 'libvpx-vp9', '-b:v', '2M',
            '-f', 'webm', output_path
        ]
        
        if verbose:
            logger.debug(f"输出命令: {' '.join(output_cmd)}")
        else:
            output_cmd.insert(1, '-v')
            output_cmd.insert(2, 'error')
            
        output_process = subprocess.Popen(
            output_cmd, 
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE if not verbose else None
        )
        
        logger.debug("输出进程已启动")
    except Exception as e:
        logger.error(f"启动输出进程失败: {e}")
        if 'input_process' in locals():
            input_process.terminate()
        return False
    
    logger.info(f"开始填充缓冲区，延迟设置为{delay_seconds}秒...")
    
    # 填充缓冲区
    frames_read = 0
    try:
        while frames_read < buffer_capacity:
            frame = input_process.stdout.read(frame_size)
            if not frame or len(frame) < frame_size:
                logger.warning("输入视频结束或读取错误，无法完全填充缓冲区")
                break
            frame_buffer.append(frame)
            frames_read += 1
            if frames_read % frame_rate == 0:
                logger.info(f"缓冲区填充进度: {frames_read}/{buffer_capacity} 帧 ({frames_read/buffer_capacity*100:.1f}%)")
    except Exception as e:
        logger.error(f"填充缓冲区时出错: {e}")
        input_process.terminate()
        output_process.terminate()
        return False
    
    logger.info(f"缓冲区已填充，开始输出（延迟{delay_seconds}秒）")
    
    # 持续处理
    try:
        while True:
            # 读取新帧
            new_frame = input_process.stdout.read(frame_size)
            if not new_frame or len(new_frame) < frame_size:
                logger.info("输入视频结束")
                break
                
            # 将新帧添加到缓冲区（自动移除最旧的帧）
            frame_buffer.append(new_frame)
            
            # 输出最旧的帧
            output_process.stdin.write(frame_buffer[0])
            
    except KeyboardInterrupt:
        logger.info("处理中断")
    except Exception as e:
        logger.error(f"处理视频帧时出错: {e}")
    finally:
        # 清理
        logger.info("处理完成，清理资源...")
        if 'input_process' in locals():
            input_process.terminate()
        if 'output_process' in locals():
            output_process.stdin.close()
            output_process.wait()
        logger.info("缓冲器已关闭")
        
    return True

def signal_handler(sig, frame):
    """处理信号"""
    logger.info("接收到中断信号，正在退出...")
    sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="视频缓冲器，实现固定延迟")
    parser.add_argument("--input", required=True, help="输入视频路径")
    parser.add_argument("--output", required=True, help="输出视频路径")
    parser.add_argument("--delay", type=int, default=20, help="延迟秒数")
    parser.add_argument("--fps", type=int, default=30, help="视频帧率")
    parser.add_argument("--resolution", default="1280x720", help="视频分辨率")
    parser.add_argument("--verbose", action="store_true", help="输出详细日志")
    
    args = parser.parse_args()
    
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 运行缓冲器
    create_buffer(
        args.input, 
        args.output, 
        args.delay, 
        args.fps, 
        args.resolution,
        args.verbose
    )
