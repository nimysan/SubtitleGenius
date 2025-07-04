#!/usr/bin/env python3
"""
测试WebSocket参数传递
验证前端传递的参数是否正确到达后端
"""

import asyncio
import websockets
import json
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_websocket_connection():
    """测试WebSocket连接和参数传递"""
    
    # 测试参数
    test_params = {
        "language": "ar",
        "correction": True,
        "translation": True,
        "target_language": "zh",
        "filename": "test_subtitle"
    }
    
    # 构建WebSocket URL
    base_url = "ws://localhost:8000/ws/whisper"
    params_str = "&".join([f"{k}={v}" for k, v in test_params.items()])
    ws_url = f"{base_url}?{params_str}"
    
    logger.info(f"测试WebSocket连接: {ws_url}")
    
    try:
        async with websockets.connect(ws_url) as websocket:
            logger.info("WebSocket连接已建立")
            
            # 等待连接确认消息
            response = await websocket.recv()
            data = json.loads(response)
            
            logger.info("收到服务器响应:")
            logger.info(json.dumps(data, indent=2, ensure_ascii=False))
            
            # 验证参数是否正确传递
            expected_params = {
                "language": test_params["language"],
                "correction_enabled": test_params["correction"],
                "translation_enabled": test_params["translation"],
                "target_language": test_params["target_language"]
            }
            
            success = True
            for key, expected_value in expected_params.items():
                if key in data and data[key] == expected_value:
                    logger.info(f"✓ 参数 {key}: {data[key]} (正确)")
                else:
                    logger.error(f"✗ 参数 {key}: 期望 {expected_value}, 实际 {data.get(key, '未找到')}")
                    success = False
            
            if success:
                logger.info("🎉 所有参数传递正确!")
            else:
                logger.error("❌ 参数传递存在问题")
            
            # 发送测试音频数据（空数据，仅测试连接）
            logger.info("发送测试数据...")
            test_audio_data = b"test_audio_data"
            await websocket.send(test_audio_data)
            
            # 等待一段时间看是否有响应
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                logger.info(f"收到音频处理响应: {response}")
            except asyncio.TimeoutError:
                logger.info("未收到音频处理响应（正常，因为发送的是测试数据）")
            
    except Exception as e:
        logger.error(f"WebSocket连接测试失败: {e}")

async def test_all_endpoints():
    """测试所有WebSocket端点"""
    endpoints = [
        "ws://localhost:8000/ws/whisper",
        "ws://localhost:8000/ws/transcribe", 
        "ws://localhost:8000/ws/claude"
    ]
    
    test_params = {
        "language": "ar",
        "correction": "true",
        "translation": "true", 
        "target_language": "en"
    }
    
    for endpoint in endpoints:
        logger.info(f"\n{'='*50}")
        logger.info(f"测试端点: {endpoint}")
        logger.info(f"{'='*50}")
        
        params_str = "&".join([f"{k}={v}" for k, v in test_params.items()])
        ws_url = f"{endpoint}?{params_str}"
        
        try:
            async with websockets.connect(ws_url) as websocket:
                logger.info("连接成功")
                
                # 等待连接确认
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                data = json.loads(response)
                
                logger.info("服务器响应:")
                logger.info(json.dumps(data, indent=2, ensure_ascii=False))
                
        except Exception as e:
            logger.error(f"连接失败: {e}")

if __name__ == "__main__":
    print("WebSocket参数传递测试")
    print("请确保WebSocket服务器正在运行 (python start_websocket_server.py)")
    print()
    
    # 运行测试
    asyncio.run(test_all_endpoints())
