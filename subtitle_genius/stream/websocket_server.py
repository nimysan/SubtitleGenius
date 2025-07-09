"""WebSocket服务器，用于接收和处理音频流"""

import asyncio
import json
import logging
import os
import uuid
import datetime
import tempfile
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 导入自定义模块
from .vac_processor import VACProcessor
from .subtitle_processor import SubtitleProcessor
from .message_handler import MessageHandler
from ..audio.processor import AudioProcessor
from ..models.transcribe_model import TranscribeModel
from ..models.whisper_sagemaker_streaming import WhisperSageMakerStreamConfig
from ..correction import BedrockCorrectionService
from translation_service import translation_manager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(title="SubtitleGenius WebSocket API")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境中应该限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SageMaker Whisper 配置
SAGEMAKER_ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT", "endpoint-quick-start-z9afg")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
SAGEMAKER_CHUNK_DURATION = int(os.getenv("SAGEMAKER_CHUNK_DURATION", "10"))

# 临时文件目录
temp_dir = Path(tempfile.gettempdir()) / "subtitle_genius"
os.makedirs(temp_dir, exist_ok=True)

# 字幕输出目录
subtitle_dir = Path("./subtitles")
os.makedirs(subtitle_dir, exist_ok=True)

# 全局实例
audio_processor = AudioProcessor()
vac_processor = None
subtitle_processor = None
message_handler = None
sagemaker_whisper_model = None
correction_service = None

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化模型和处理器"""
    global sagemaker_whisper_model, correction_service, vac_processor, subtitle_processor, message_handler
    
    # 初始化VAC处理器
    try:
        vac_processor = VACProcessor()
        logger.info("VAC处理器已初始化")
    except Exception as e:
        logger.error(f"VAC处理器初始化失败: {e}")
        vac_processor = None
    
    # 初始化SageMaker Whisper模型
    try:
        # 配置参数
        config = WhisperSageMakerStreamConfig(
            chunk_duration=30,      # 每duration秒处理一次
            overlap_duration=0,    # 2秒重叠避免截断
            voice_threshold=0.01,    # 语音检测阈值
            sagemaker_chunk_duration=SAGEMAKER_CHUNK_DURATION  # SageMaker 端点处理的块大小
        )
        
        # 创建模型 (使用 SageMaker Whisper 后端)
        sagemaker_whisper_model = TranscribeModel(
            backend="sagemaker_whisper",
            sagemaker_endpoint=SAGEMAKER_ENDPOINT,
            region_name=AWS_REGION,
            whisper_config=config
        )
        logger.info(f"SageMaker Whisper模型已初始化 config is {config}")
        if sagemaker_whisper_model.is_available():
            logger.info("SageMaker Whisper模型已初始化")
        else:
            logger.error(f"SageMaker Whisper模型不可用，请检查端点配置: {SAGEMAKER_ENDPOINT}")
    except Exception as e:
        logger.error(f"SageMaker Whisper模型初始化失败: {e}")
    
    # 初始化Correction服务 (使用Sonnet模型)
    try:
        correction_service = BedrockCorrectionService(
            model_id="us.anthropic.claude-3-5-sonnet-20240620-v1:0"
        )
        logger.info("Bedrock Correction服务已初始化 (Claude 3.5 Sonnet)")
    except Exception as e:
        logger.error(f"Correction服务初始化失败: {e}")
        correction_service = None
    
    # 确保翻译服务可用
    if "bedrock" in translation_manager.get_available_services():
        logger.info("Bedrock翻译服务已初始化")
    else:
        logger.warning("Bedrock翻译服务不可用，将使用备用翻译服务")
    
    # 初始化字幕处理器
    subtitle_processor = SubtitleProcessor(correction_service=correction_service)
    logger.info("字幕处理器已初始化")
    
    # 初始化消息处理器
    message_handler = MessageHandler(
        subtitle_processor=subtitle_processor,
        vac_processor=vac_processor,
        whisper_model=sagemaker_whisper_model
    )
    logger.info("消息处理器已初始化")


@app.websocket("/ws/whisper")
async def websocket_whisper_endpoint(
    websocket: WebSocket, 
    language: str = Query("ar"),
    correction: bool = Query(True),
    translation: bool = Query(True),
    target_language: str = Query("en"),
    filename: str = Query(None)
):
    """Whisper模型WebSocket端点"""
    global message_handler, vac_processor
    
    client_id = str(uuid.uuid4())
    
    await websocket.accept()
    
    # 注册连接
    message_handler.register_connection(client_id, websocket)
    
    # 重置VAC处理器状态
    if vac_processor:
        vac_processor.reset()
    logger.info("VAC处理器已重置")
    
    # 如果没有提供文件名，使用时间戳
    if not filename:
        filename = f"subtitle_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"客户端 {client_id} 已连接到Whisper端点")
    logger.info(f"接收到的参数:")
    logger.info(f"  - 视频语言: {language}")
    logger.info(f"  - 启用纠错: {correction}")
    logger.info(f"  - 启用翻译: {translation}")
    logger.info(f"  - 翻译目标语言: {target_language}")
    logger.info(f"  - 文件名: {filename}")
    
    try:
        # 发送连接确认
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "client_id": client_id,
            "model": "whisper",
            "language": language,
            "correction_enabled": correction,
            "translation_enabled": translation,
            "target_language": target_language
        })
        
        # 初始化处理状态
        current_chunk_index = 0
        pending_timestamp = None
        
        # 处理接收到的消息
        while True:
            try:
                # 接收消息（可能是文本或二进制）
                logger.debug(f"等待接收客户端 {client_id} 的消息...")
                message = await websocket.receive()
                logger.debug(f"收到消息类型: {message.get('type')}, 消息键: {list(message.keys())}")
                
                # 检查消息类型
                if message["type"] == "websocket.receive":
                    if "text" in message:
                        # 处理文本消息（时间戳信息）
                        logger.info(f"收到文本消息，长度: {len(message['text'])}")
                        result = await message_handler.handle_text_message(client_id, message['text'])
                        
                        # 如果是时间戳消息，更新待处理时间戳
                        if result and result.get('type') == 'timestamp_received':
                            timestamp_data = json.loads(message['text']).get('timestamp', {})
                            pending_timestamp = timestamp_data
                            logger.info(f"更新待处理时间戳: chunk_index={pending_timestamp.get('chunk_index', 'unknown')}")
                        
                        # 如果需要响应，发送响应
                        if result:
                            await websocket.send_json(result)
                            
                    elif "bytes" in message:
                        # 处理二进制消息（音频数据）
                        data = message["bytes"]
                        logger.info(f"接收到音频数据，大小: {len(data)} bytes, 数据类型: {type(data)}")
                        
                        # 处理音频数据
                        result = await message_handler.handle_binary_message(
                            client_id=client_id,
                            binary_data=data,
                            language=language,
                            enable_correction=correction,
                            enable_translation=translation,
                            target_language=target_language,
                            pending_timestamp=pending_timestamp,
                            current_chunk_index=current_chunk_index
                        )
                        
                        # 处理结果
                        if result:
                            if result.get('type') == 'audio_processed':
                                # 音频处理成功，发送字幕结果
                                for subtitle_result in result.get('results', []):
                                    await websocket.send_json(subtitle_result)
                                
                                # 清除已使用的时间戳
                                pending_timestamp = None
                            elif result.get('type') == 'error':
                                # 处理错误
                                await websocket.send_json(result)
                        
                        # 无论处理结果如何，都增加chunk索引
                        current_chunk_index += 1
                    else:
                        logger.warning(f"未知的消息格式，消息键: {list(message.keys())}")
                        
                elif message["type"] == "websocket.disconnect":
                    logger.info(f"客户端 {client_id} 断开连接")
                    break
                else:
                    logger.warning(f"未知的WebSocket消息类型: {message['type']}")
                    
            except WebSocketDisconnect:
                logger.info(f"客户端 {client_id} WebSocket连接断开")
                break
            except Exception as e:
                logger.error(f"处理消息时出错: {e}")
                logger.error(f"错误详情: {type(e).__name__}: {str(e)}")
                import traceback
                logger.error(f"错误堆栈: {traceback.format_exc()}")
                
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"处理消息失败: {str(e)}"
                    })
                except Exception as send_error:
                    logger.error(f"发送错误消息失败: {send_error}")
                break
    
    except WebSocketDisconnect:
        logger.info(f"客户端 {client_id} 断开连接")
        # 保存字幕文件
        try:
            subtitle_processor.save_subtitles(client_id, filename, language)
            logger.info(f"已保存字幕文件: {filename}")
        except Exception as e:
            logger.error(f"保存字幕文件失败: {e}")
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
    finally:
        # 清理连接
        message_handler.unregister_connection(client_id)


@app.websocket("/ws/transcribe")
async def websocket_transcribe_endpoint(
    websocket: WebSocket, 
    language: str = Query("ar"),
    correction: bool = Query(True),
    translation: bool = Query(True),
    target_language: str = Query("en"),
    filename: str = Query(None)
):
    """Amazon Transcribe模型WebSocket端点 - 当前不支持"""
    client_id = str(uuid.uuid4())
    
    await websocket.accept()
    logger.info(f"客户端 {client_id} 已连接到Transcribe端点，但该功能当前不支持")
    
    try:
        # 发送不支持的功能消息
        await websocket.send_json({
            "type": "error",
            "status": "unsupported",
            "message": "Amazon Transcribe功能当前不支持，请使用Whisper端点",
            "client_id": client_id
        })
        
        # 等待客户端断开连接
        while True:
            try:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
            except WebSocketDisconnect:
                break
    
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
    finally:
        logger.info(f"客户端 {client_id} 断开连接")


@app.websocket("/ws/save_subtitles")
async def websocket_save_subtitles_endpoint(
    websocket: WebSocket,
    client_id: str = Query(...),
    filename: str = Query(None)
):
    """保存字幕WebSocket端点"""
    global subtitle_processor
    
    await websocket.accept()
    
    try:
        # 如果没有提供文件名，使用时间戳
        if not filename:
            filename = f"subtitle_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 保存字幕
        result = await message_handler.handle_save_subtitles(client_id, filename, "auto")
        
        # 发送结果
        await websocket.send_json(result)
    
    except WebSocketDisconnect:
        logger.info("保存字幕连接已断开")
    except Exception as e:
        logger.error(f"保存字幕WebSocket错误: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"处理请求失败: {str(e)}"
        })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
