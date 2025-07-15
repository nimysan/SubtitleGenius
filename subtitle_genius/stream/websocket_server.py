import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from urllib.parse import urlparse, parse_qs

import websockets
from websockets.server import WebSocketServerProtocol

# 导入独立的连续音频处理器
from subtitle_genius.stream.continuous_audio_processor import ContinuousAudioProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("websocket_server")


class WebSocketServer:
    """WebSocket服务器，处理前端连接"""
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.active_connections = {}
        # 不再在这里创建单一的音频处理器，而是为每个连接创建独立的处理器
    
    def _parse_websocket_path(self, path: str) -> Dict[str, Any]:
        """解析WebSocket路径和参数"""
        parsed_url = urlparse(path)
        path_parts = parsed_url.path.strip('/').split('/')
        query_params = parse_qs(parsed_url.query)
        
        # 提取路径信息
        config = {
            'path': parsed_url.path,
            'model': 'whisper',  # 默认模型
            'language': 'zh',    # 默认语言
            'target_language': 'en',  # 默认目标语言
            'correction': True,   # 默认启用纠错
            'translation': False, # 默认不启用翻译
            'scene_description': '足球比赛',  # 默认场景
            'client_id': str(uuid.uuid4())  # 生成客户端ID
        }
        
        # 根据路径确定模型类型
        if len(path_parts) >= 2 and path_parts[0] == 'ws':
            model_type = path_parts[1]
            if model_type in ['whisper', 'claude', 'transcribe']:
                config['model'] = model_type
        
        # 解析查询参数
        for key, values in query_params.items():
            if values:  # 确保有值
                value = values[0]  # 取第一个值
                
                if key == 'language':
                    config['language'] = value
                elif key == 'target_language':
                    config['target_language'] = value
                elif key == 'correction':
                    config['correction'] = value.lower() in ['true', '1', 'yes']
                elif key == 'translation':
                    config['translation'] = value.lower() in ['true', '1', 'yes']
                elif key == 'scene_description':
                    config['scene_description'] = value
                elif key == 'filename':
                    config['filename'] = value
        
        logger.info(f"解析WebSocket路径: {path} -> 配置: {config}")
        return config
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """处理新的WebSocket连接"""
        connection_id = str(uuid.uuid4())
        
        # 解析路径和参数
        config = self._parse_websocket_path(path)
        config['client_id'] = connection_id  # 使用连接ID作为客户端ID
        
        # 为这个连接创建独立的音频处理器
        audio_processor = ContinuousAudioProcessor(config)
        
        self.active_connections[connection_id] = {
            'websocket': websocket,
            'audio_processor': audio_processor,
            'config': config
        }
        
        try:
            logger.info(f"新连接建立: {connection_id}, 路径: {path}")
            
            # 发送连接确认消息，包含配置信息
            await websocket.send(json.dumps({
                "type": "connection",
                "status": "connected",
                "client_id": connection_id,
                "model": config['model'],
                "language": config['language'],
                "target_language": config['target_language'],
                "correction_enabled": config['correction'],
                "translation_enabled": config['translation'],
                "scene_description": config['scene_description'],
                "timestamp": datetime.now().isoformat()
            }))
            
            stream_id = None
            
            async for message in websocket:
                if isinstance(message, str):
                    # 处理文本消息
                    new_stream_id = await self._handle_text_message(connection_id, message, websocket)
                    if new_stream_id:
                        stream_id = new_stream_id
                else:
                    # 处理二进制消息（音频数据）
                    new_stream_id = await self._handle_binary_message(connection_id, message, websocket, stream_id)
                    if new_stream_id:
                        stream_id = new_stream_id
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"连接关闭: {connection_id}")
        except Exception as e:
            logger.error(f"处理连接时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # 清理资源
            if connection_id in self.active_connections:
                connection_info = self.active_connections[connection_id]
                audio_processor = connection_info['audio_processor']
                
                # 停止所有活跃的流
                for active_stream_id in list(audio_processor.active_connections.keys()):
                    await audio_processor.stop_stream(active_stream_id)
                
                del self.active_connections[connection_id]
    
    async def _handle_text_message(self, connection_id: str, message: str, websocket: WebSocketServerProtocol) -> Optional[str]:
        """处理文本消息"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            logger.debug(f"收到来自 {connection_id} 的文本消息: {message_type}")
            
            # 获取连接信息
            if connection_id not in self.active_connections:
                logger.error(f"连接 {connection_id} 不存在")
                return None
            
            connection_info = self.active_connections[connection_id]
            audio_processor = connection_info['audio_processor']
            config = connection_info['config']
            
            if message_type == "start_stream":
                # 开始新的音频流
                stream_id = str(uuid.uuid4())
                await audio_processor.start_stream(stream_id, websocket)
                await websocket.send(json.dumps({
                    "type": "stream_started",
                    "stream_id": stream_id,
                    "client_id": connection_id,
                    "config": config
                }))
                return stream_id
            
            elif message_type == "stop_stream":
                # 停止现有的音频流
                stream_id = data.get("stream_id")
                if stream_id:
                    await audio_processor.stop_stream(stream_id)
                    await websocket.send(json.dumps({
                        "type": "stream_stopped",
                        "stream_id": stream_id,
                        "client_id": connection_id
                    }))
            
            elif message_type == "ping":
                # 简单的ping-pong测试
                await websocket.send(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat(),
                    "client_id": connection_id
                }))
            
            elif message_type == "get_audio_data":
                # 处理音频数据请求
                segment_id = data.get("segment_id")
                if segment_id:
                    # 直接调用音频处理器的handle_audio_request方法
                    if hasattr(audio_processor, 'handle_audio_request'):
                        await audio_processor.handle_audio_request(websocket, segment_id)
                    else:
                        logger.error("音频处理器没有handle_audio_request方法")
                        await websocket.send(json.dumps({
                            "type": "error",
                            "error": "服务器不支持音频数据请求功能",
                            "client_id": connection_id
                        }))
                else:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "error": "请求音频数据时未提供segment_id",
                        "client_id": connection_id
                    }))
            
            elif message_type == "audio_with_timestamp":
                # 处理带时间戳的音频消息（前端发送的元数据）
                timestamp_info = data.get("timestamp", {})
                logger.debug(f"收到音频时间戳信息: {timestamp_info}")
                # 这里可以存储时间戳信息，等待后续的二进制音频数据
                # 暂时只记录日志
            
            else:
                logger.warning(f"未知消息类型: {message_type}")
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": f"未知消息类型: {message_type}",
                    "client_id": connection_id
                }))
            
            return None
              
        
        except json.JSONDecodeError:
            logger.error(f"收到无效的JSON: {message}")
            await websocket.send(json.dumps({
                "type": "error",
                "error": "无效的JSON格式",
                "client_id": connection_id
            }))
        except Exception as e:
            logger.error(f"处理文本消息时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            await websocket.send(json.dumps({
                "type": "error",
                "error": str(e),
                "client_id": connection_id
            }))
        
        return None
    
    async def _handle_binary_message(self, connection_id: str, message: bytes, websocket: WebSocketServerProtocol, stream_id: Optional[str]) -> str:
        """处理二进制消息（音频数据）"""
        try:
            # 获取连接信息
            if connection_id not in self.active_connections:
                logger.error(f"连接 {connection_id} 不存在")
                return stream_id if stream_id else ""
            
            connection_info = self.active_connections[connection_id]
            audio_processor = connection_info['audio_processor']
            config = connection_info['config']
            
            # 保存音频块到文件（如果需要调试）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            # 如果没有流ID，自动创建一个
            if not stream_id:
                stream_id = str(uuid.uuid4())
                logger.info(f"为音频数据自动创建流: {stream_id}")
                await audio_processor.start_stream(stream_id, websocket)
                
                # 通知客户端新流已创建
                await websocket.send(json.dumps({
                    "type": "stream_started",
                    "stream_id": stream_id,
                    "auto_created": True,
                    "client_id": connection_id,
                    "config": config
                }))
            
            # 处理音频数据
            result = await audio_processor.process_audio(stream_id, message)
            
            # 发送处理结果给客户端
            await websocket.send(json.dumps({
                "type": "audio_processing",
                "stream_id": stream_id,
                "client_id": connection_id,
                "result": result,
                "config": {
                    "model": config['model'],
                    "language": config['language'],
                    "correction_enabled": config['correction'],
                    "translation_enabled": config['translation']
                }
            }))
            
            return stream_id
        
        except Exception as e:
            logger.error(f"处理二进制消息时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            await websocket.send(json.dumps({
                "type": "error",
                "error": str(e),
                "client_id": connection_id
            }))
            return stream_id if stream_id else ""
    
    async def start_server(self):
        """启动WebSocket服务器"""
        server = await websockets.serve(
            self.handle_connection,
            self.host,
            self.port
        )
        
        logger.info(f"WebSocket服务器已启动，地址: ws://{self.host}:{self.port}")
        
        return server


async def main():
    """主入口点"""
    server = WebSocketServer()
    ws_server = await server.start_server()
    
    try:
        # 保持服务器运行直到被中断
        await asyncio.Future()
    except asyncio.CancelledError:
        logger.info("服务器被取消")
    finally:
        # 关闭所有活跃的连接和流
        for connection_id, connection_info in list(server.active_connections.items()):
            audio_processor = connection_info['audio_processor']
            # 停止所有活跃的流
            for stream_id in list(audio_processor.active_connections.keys()):
                await audio_processor.stop_stream(stream_id)
        
        # 关闭WebSocket服务器
        ws_server.close()
        await ws_server.wait_closed()
        logger.info("WebSocket服务器已关闭")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("服务器被用户停止")
