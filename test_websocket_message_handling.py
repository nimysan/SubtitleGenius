#!/usr/bin/env python3
"""
WebSocket消息处理测试脚本
用于验证时间戳和音频数据的正确处理
"""

import asyncio
import json
import websockets
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_websocket_messages():
    """测试WebSocket消息发送和接收"""
    
    # WebSocket服务器地址
    uri = "ws://localhost:8000/ws/whisper?language=ar&correction=true&translation=true&target_language=en"
    
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("WebSocket连接已建立")
            
            # 1. 发送时间戳消息
            timestamp_message = {
                "type": "audio_with_timestamp",
                "timestamp": {
                    "start_time": 6.0,
                    "end_time": 9.0,
                    "duration": 3.0,
                    "chunk_index": 2,
                    "total_samples_processed": 48000,
                    "audio_start_time": 1234567890.123,
                    "processing_start_time": 1234567890.123,
                    "current_time": 1234567890.456
                }
            }
            
            logger.info("发送时间戳消息...")
            await websocket.send(json.dumps(timestamp_message))
            
            # 2. 创建测试音频数据（简单的WAV文件头 + 数据）
            # 这是一个最小的WAV文件结构
            wav_header = b'RIFF'
            wav_header += (36 + 1000).to_bytes(4, 'little')  # 文件大小
            wav_header += b'WAVE'
            wav_header += b'fmt '
            wav_header += (16).to_bytes(4, 'little')  # fmt chunk大小
            wav_header += (1).to_bytes(2, 'little')   # 音频格式 (PCM)
            wav_header += (1).to_bytes(2, 'little')   # 声道数
            wav_header += (16000).to_bytes(4, 'little')  # 采样率
            wav_header += (32000).to_bytes(4, 'little')  # 字节率
            wav_header += (2).to_bytes(2, 'little')   # 块对齐
            wav_header += (16).to_bytes(2, 'little')  # 位深度
            wav_header += b'data'
            wav_header += (1000).to_bytes(4, 'little')  # 数据大小
            
            # 添加一些测试音频数据（静音）
            audio_data = b'\x00' * 1000
            test_wav_data = wav_header + audio_data
            
            logger.info(f"发送音频数据，大小: {len(test_wav_data)} bytes")
            await websocket.send(test_wav_data)
            
            # 3. 等待服务器响应
            logger.info("等待服务器响应...")
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                logger.info(f"收到服务器响应: {response}")
            except asyncio.TimeoutError:
                logger.warning("等待服务器响应超时")
            
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        logger.error(f"错误堆栈: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(test_websocket_messages())
