#!/usr/bin/env python3
"""
简化的WebSocket调试端点
用于诊断前端发送的消息格式和数据类型
"""

import asyncio
import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WebSocket Debug Server")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/debug")
async def debug_websocket_endpoint(websocket: WebSocket):
    """调试WebSocket端点"""
    await websocket.accept()
    client_id = "debug-client"
    
    logger.info(f"调试客户端 {client_id} 已连接")
    
    # 发送连接确认
    await websocket.send_json({
        "type": "connection",
        "status": "connected",
        "client_id": client_id,
        "message": "调试模式已启用"
    })
    
    message_count = 0
    
    try:
        while True:
            # 接收消息
            message = await websocket.receive()
            message_count += 1
            
            logger.info(f"=== 消息 #{message_count} ===")
            logger.info(f"消息类型: {message.get('type')}")
            logger.info(f"消息键: {list(message.keys())}")
            
            if message["type"] == "websocket.receive":
                if "text" in message:
                    text_data = message["text"]
                    logger.info(f"文本消息长度: {len(text_data)}")
                    logger.info(f"文本消息前100字符: {text_data[:100]}")
                    
                    try:
                        json_data = json.loads(text_data)
                        logger.info(f"JSON解析成功，类型: {json_data.get('type')}")
                        if json_data.get('type') == 'audio_with_timestamp':
                            timestamp = json_data.get('timestamp', {})
                            logger.info(f"时间戳信息: chunk_index={timestamp.get('chunk_index')}, "
                                      f"start_time={timestamp.get('start_time')}, "
                                      f"end_time={timestamp.get('end_time')}")
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON解析失败: {e}")
                        
                elif "bytes" in message:
                    bytes_data = message["bytes"]
                    logger.info(f"二进制消息长度: {len(bytes_data)}")
                    logger.info(f"二进制数据类型: {type(bytes_data)}")
                    logger.info(f"前16字节: {bytes_data[:16]}")
                    
                    # 检查是否是WAV文件
                    if bytes_data.startswith(b'RIFF') and b'WAVE' in bytes_data[:20]:
                        logger.info("检测到WAV文件格式")
                    else:
                        logger.warning("未检测到WAV文件格式")
                        
                else:
                    logger.warning(f"未知消息格式，键: {list(message.keys())}")
                    
            elif message["type"] == "websocket.disconnect":
                logger.info("客户端断开连接")
                break
            else:
                logger.warning(f"未知WebSocket消息类型: {message['type']}")
                
            # 发送确认消息
            await websocket.send_json({
                "type": "debug_response",
                "message_number": message_count,
                "received": True,
                "timestamp": asyncio.get_event_loop().time()
            })
            
    except WebSocketDisconnect:
        logger.info(f"调试客户端 {client_id} 断开连接")
    except Exception as e:
        logger.error(f"调试端点错误: {e}")
        import traceback
        logger.error(f"错误堆栈: {traceback.format_exc()}")

if __name__ == "__main__":
    logger.info("启动WebSocket调试服务器...")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
