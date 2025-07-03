"""WebSocket服务器，用于接收和处理音频流"""

import asyncio
import json
import logging
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Any
import uuid
from pathlib import Path
import tempfile
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from ..audio.processor import AudioProcessor
from ..models.transcribe_model import TranscribeModel
from ..models.whisper_sagemaker_streaming import WhisperSageMakerStreamConfig
from ..models.claude_model import ClaudeModel
from ..subtitle.models import Subtitle
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

# 活跃连接管理
active_connections: Dict[str, WebSocket] = {}

# 音频处理器
audio_processor = AudioProcessor()

# 模型实例
transcribe_model = None
sagemaker_whisper_model = None
claude_model = None

# SageMaker Whisper 配置
SAGEMAKER_ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT", "endpoint-quick-start-z9afg")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
SAGEMAKER_CHUNK_DURATION = int(os.getenv("SAGEMAKER_CHUNK_DURATION", "30"))

# 临时文件目录
temp_dir = Path(tempfile.gettempdir()) / "subtitle_genius"
os.makedirs(temp_dir, exist_ok=True)


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化模型"""
    global sagemaker_whisper_model, transcribe_model, claude_model
    
    # 初始化SageMaker Whisper模型
    try:
        # 配置参数
        config = WhisperSageMakerStreamConfig(
            chunk_duration=10,      # 每3秒处理一次
            overlap_duration=2,    # 0.5秒重叠避免截断
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
        
        if sagemaker_whisper_model.is_available():
            logger.info("SageMaker Whisper模型已初始化")
        else:
            logger.error(f"SageMaker Whisper模型不可用，请检查端点配置: {SAGEMAKER_ENDPOINT}")
    except Exception as e:
        logger.error(f"SageMaker Whisper模型初始化失败: {e}")
    
    # 初始化Transcribe模型
    try:
        transcribe_model = TranscribeModel()
        logger.info("Transcribe模型已初始化")
    except Exception as e:
        logger.error(f"Transcribe模型初始化失败: {e}")
    
    # 初始化Claude模型
    try:
        claude_model = ClaudeModel()
        logger.info("Claude模型已初始化")
    except Exception as e:
        logger.error(f"Claude模型初始化失败: {e}")
    
    # 确保翻译服务可用
    if "bedrock" in translation_manager.get_available_services():
        logger.info("Bedrock翻译服务已初始化")
    else:
        logger.warning("Bedrock翻译服务不可用，将使用备用翻译服务")


@app.websocket("/ws/whisper")
async def websocket_whisper_endpoint(
    websocket: WebSocket, 
    language: str = Query("ar")
):
    """Whisper模型WebSocket端点"""
    client_id = str(uuid.uuid4())
    
    await websocket.accept()
    active_connections[client_id] = websocket
    
    logger.info(f"客户端 {client_id} 已连接到Whisper端点，语言: {language}")
    
    try:
        # 发送连接确认
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "client_id": client_id,
            "model": "whisper",
            "language": language
        })
        
        # 创建临时文件存储音频块
        audio_chunks: List[np.ndarray] = []
        
        # 处理接收到的音频数据
        async for data in websocket.iter_bytes():
            try:
                # 将WAV数据转换为numpy数组
                audio_data = await process_wav_data(data)
                
                if audio_data is not None:
                    # 添加到音频块列表
                    audio_chunks.append(audio_data)
                    
                    # 创建异步生成器
                    async def audio_generator():
                        yield audio_data
                    
                    # 使用SageMaker Whisper模型处理音频
                    if sagemaker_whisper_model and sagemaker_whisper_model.is_available():
                        async for subtitle in sagemaker_whisper_model.transcribe_stream(
                            audio_generator(), language=language
                        ):
                            # 发送字幕回客户端
                            await send_subtitle(websocket, subtitle, client_id, language)
            
            except Exception as e:
                logger.error(f"处理音频数据失败: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"处理音频数据失败: {str(e)}"
                })
    
    except WebSocketDisconnect:
        logger.info(f"客户端 {client_id} 断开连接")
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
    finally:
        # 清理连接
        if client_id in active_connections:
            del active_connections[client_id]


@app.websocket("/ws/transcribe")
async def websocket_transcribe_endpoint(
    websocket: WebSocket, 
    language: str = Query("ar")
):
    """Amazon Transcribe模型WebSocket端点"""
    client_id = str(uuid.uuid4())
    
    await websocket.accept()
    active_connections[client_id] = websocket
    
    logger.info(f"客户端 {client_id} 已连接到Transcribe端点，语言: {language}")
    
    try:
        # 发送连接确认
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "client_id": client_id,
            "model": "transcribe",
            "language": language
        })
        
        # 创建临时文件存储音频块
        audio_chunks: List[np.ndarray] = []
        
        # 处理接收到的音频数据
        async for data in websocket.iter_bytes():
            try:
                # 将WAV数据转换为numpy数组
                audio_data = await process_wav_data(data)
                
                if audio_data is not None:
                    # 添加到音频块列表
                    audio_chunks.append(audio_data)
                    
                    # 使用Transcribe模型处理音频
                    if transcribe_model:
                        # 保存临时WAV文件
                        temp_file = temp_dir / f"{client_id}_{len(audio_chunks)}.wav"
                        await audio_processor.save_audio(audio_data, temp_file)
                        
                        # 处理音频文件
                        subtitle = await transcribe_model.transcribe_chunk(
                            str(temp_file), language=language
                        )
                        
                        # 发送字幕回客户端
                        if subtitle:
                            await send_subtitle(websocket, subtitle, client_id, language)
                        
                        # 删除临时文件
                        try:
                            os.remove(temp_file)
                        except:
                            pass
            
            except Exception as e:
                logger.error(f"处理音频数据失败: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"处理音频数据失败: {str(e)}"
                })
    
    except WebSocketDisconnect:
        logger.info(f"客户端 {client_id} 断开连接")
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
    finally:
        # 清理连接
        if client_id in active_connections:
            del active_connections[client_id]


@app.websocket("/ws/claude")
async def websocket_claude_endpoint(
    websocket: WebSocket, 
    language: str = Query("ar")
):
    """Claude模型WebSocket端点"""
    client_id = str(uuid.uuid4())
    
    await websocket.accept()
    active_connections[client_id] = websocket
    
    logger.info(f"客户端 {client_id} 已连接到Claude端点，语言: {language}")
    
    try:
        # 发送连接确认
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "client_id": client_id,
            "model": "claude",
            "language": language
        })
        
        # 创建临时文件存储音频块
        audio_chunks: List[np.ndarray] = []
        
        # 处理接收到的音频数据
        async for data in websocket.iter_bytes():
            try:
                # 将WAV数据转换为numpy数组
                audio_data = await process_wav_data(data)
                
                if audio_data is not None:
                    # 添加到音频块列表
                    audio_chunks.append(audio_data)
                    
                    # 使用Claude模型处理音频
                    if claude_model:
                        # 保存临时WAV文件
                        temp_file = temp_dir / f"{client_id}_{len(audio_chunks)}.wav"
                        await audio_processor.save_audio(audio_data, temp_file)
                        
                        # 处理音频文件
                        subtitle = await claude_model.transcribe_audio(
                            str(temp_file), language=language
                        )
                        
                        # 发送字幕回客户端
                        if subtitle:
                            await send_subtitle(websocket, subtitle, client_id, language)
                        
                        # 删除临时文件
                        try:
                            os.remove(temp_file)
                        except:
                            pass
            
            except Exception as e:
                logger.error(f"处理音频数据失败: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"处理音频数据失败: {str(e)}"
                })
    
    except WebSocketDisconnect:
        logger.info(f"客户端 {client_id} 断开连接")
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
    finally:
        # 清理连接
        if client_id in active_connections:
            del active_connections[client_id]


async def process_wav_data(data: bytes) -> Optional[np.ndarray]:
    """处理WAV数据"""
    try:
        # 保存为临时文件
        temp_file = temp_dir / f"temp_{uuid.uuid4()}.wav"
        with open(temp_file, "wb") as f:
            f.write(data)
        
        # 使用音频处理器加载文件
        audio_data = await audio_processor.process_file(temp_file)
        
        # 删除临时文件
        try:
            os.remove(temp_file)
        except:
            pass
        
        return audio_data
    
    except Exception as e:
        logger.error(f"处理WAV数据失败: {e}")
        return None


async def send_subtitle(websocket: WebSocket, subtitle: Subtitle, client_id: str, language: str = "ar"):
    """发送字幕到客户端"""
    try:
        # 创建唯一ID
        subtitle_id = f"{client_id}_{uuid.uuid4()}"
        
        # 翻译字幕文本
        if subtitle.text.strip():
            try:
                # 使用翻译服务翻译文本
                translation_result = await translation_manager.translate(
                    text=subtitle.text,
                    target_lang="zh",
                    service="bedrock"  # 优先使用Bedrock翻译服务
                )
                
                # 设置翻译结果
                subtitle.translated_text = translation_result.translated_text
                logger.info(f"字幕已翻译: {subtitle.text} -> {subtitle.translated_text}")
            except Exception as e:
                logger.error(f"翻译失败: {e}")
                subtitle.translated_text = f"[翻译失败] {subtitle.text}"
        
        # 发送字幕
        await websocket.send_json({
            "type": "subtitle",
            "subtitle": {
                "id": subtitle_id,
                "start": subtitle.start,
                "end": subtitle.end,
                "text": subtitle.text,
                "translated_text": subtitle.translated_text
            }
        })
    except Exception as e:
        logger.error(f"发送字幕失败: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
