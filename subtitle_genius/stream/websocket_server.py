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
import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from ..audio.processor import AudioProcessor
from ..models.transcribe_model import TranscribeModel
from ..models.whisper_sagemaker_streaming import WhisperSageMakerStreamConfig
from ..models.whisper_language_config import (
    create_whisper_config, 
    get_sagemaker_whisper_params,
    get_correction_scene_description
)
from ..models.claude_model import ClaudeModel
from ..subtitle.models import Subtitle
from ..correction import BedrockCorrectionService, CorrectionInput
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

# 字幕存储 - 为每个客户端存储字幕列表
client_subtitles: Dict[str, List[Subtitle]] = {}

# 时间戳存储 - 为每个客户端存储时间戳信息
client_timestamps: Dict[str, Dict] = {}

# 音频处理器
audio_processor = AudioProcessor()

# 模型实例
sagemaker_whisper_model = None
correction_service = None


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


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化模型"""
    global sagemaker_whisper_model, correction_service
    
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
    
    # 初始化Correction服务 (使用Haiku模型)
    try:
        correction_service = BedrockCorrectionService(
            model_id="us.anthropic.claude-3-haiku-20240307-v1:0"
        )
        logger.info("Bedrock Correction服务已初始化 (Claude 3 Haiku)")
    except Exception as e:
        logger.error(f"Correction服务初始化失败: {e}")
        correction_service = None
    
    # 确保翻译服务可用
    if "bedrock" in translation_manager.get_available_services():
        logger.info("Bedrock翻译服务已初始化")
    else:
        logger.warning("Bedrock翻译服务不可用，将使用备用翻译服务")


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
    client_id = str(uuid.uuid4())
    
    await websocket.accept()
    active_connections[client_id] = websocket
    
    # 初始化字幕列表
    client_subtitles[client_id] = []
    
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
        
        # 创建临时文件存储音频块
        audio_chunks: List[np.ndarray] = []
        current_chunk_index = 0
        pending_timestamp = None
        
        # 处理接收到的消息（可能是时间戳信息或音频数据）
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
                        try:
                            message_data = json.loads(message["text"])
                            logger.info(f"解析JSON成功，消息类型: {message_data.get('type')}")
                            
                            if message_data.get('type') == 'audio_with_timestamp':
                                # 处理时间戳信息
                                await process_timestamp_message(client_id, message_data)
                                pending_timestamp = message_data.get('timestamp')
                                logger.info(f"接收到时间戳信息，chunk_index: {pending_timestamp.get('chunk_index', 'unknown')}")
                                continue
                            else:
                                logger.warning(f"未知的文本消息类型: {message_data.get('type')}")
                                continue
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON解析失败: {e}, 消息内容: {message['text'][:100]}...")
                            continue
                            
                    elif "bytes" in message:
                        # 处理二进制消息（音频数据）
                        data = message["bytes"]
                        logger.info(f"接收到音频数据，大小: {len(data)} bytes, 数据类型: {type(data)}")
                        
                        # 将WAV数据转换为numpy数组
                        try:
                            audio_data = await process_wav_data(data)
                            logger.info(f"音频数据处理结果: {type(audio_data) if audio_data is not None else 'None'}")
                        except Exception as wav_error:
                            logger.error(f"处理WAV数据失败: {wav_error}")
                            logger.error(f"WAV数据前32字节: {data[:32] if len(data) >= 32 else data}")
                            continue
                          
                        if audio_data is not None:
                            # 添加到音频块列表
                            audio_chunks.append(audio_data)
                            logger.info(f"音频数据添加到chunks，当前chunks数量: {len(audio_chunks)}")
                            
                            # 创建异步生成器
                            async def audio_generator():
                                yield audio_data
                            
                            # 使用SageMaker Whisper模型处理音频
                            if sagemaker_whisper_model and sagemaker_whisper_model.is_available():
                                logger.info("开始使用SageMaker Whisper处理音频...")
                                # 为当前语言更新模型配置
                                await update_whisper_model_language(language)
                                
                                async for subtitle in sagemaker_whisper_model.transcribe_stream(
                                    audio_generator(), language=language
                                ):
                                    # 如果有待处理的时间戳，应用到字幕
                                    if pending_timestamp:
                                        subtitle = await apply_timestamp_to_subtitle(subtitle, pending_timestamp)
                                        logger.info(f"应用时间戳到字幕: chunk_index={pending_timestamp.get('chunk_index')}")
                                        pending_timestamp = None  # 清除已使用的时间戳
                                    
                                    # 发送字幕回客户端（传递处理参数）
                                    logging.info(f"Received subtitle: {subtitle}")
                                    await send_subtitle(
                                        websocket, subtitle, client_id, 
                                        language=language,
                                        enable_correction=correction,
                                        enable_translation=translation,
                                        target_language=target_language
                                    )
                            else:
                                logger.warning("SageMaker Whisper模型不可用")
                                    
                            current_chunk_index += 1
                        else:
                            logger.warning("音频数据处理失败，跳过此chunk")
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
            
            except Exception as e:
                logger.error(f"处理音频数据失败: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"处理音频数据失败: {str(e)}"
                })
    
    except WebSocketDisconnect:
        logger.info(f"客户端 {client_id} 断开连接")
        # 保存字幕文件
        if client_id in client_subtitles and client_subtitles[client_id]:
            try:
                save_subtitles(client_subtitles[client_id], filename, language)
                logger.info(f"已保存字幕文件: {filename}")
            except Exception as e:
                logger.error(f"保存字幕文件失败: {e}")
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
    finally:
        # 清理连接
        if client_id in active_connections:
            del active_connections[client_id]
        # 清理字幕数据
        if client_id in client_subtitles:
            del client_subtitles[client_id]


@app.websocket("/ws/transcribe")
async def websocket_transcribe_endpoint(
    websocket: WebSocket, 
    language: str = Query("ar"),
    correction: bool = Query(True),
    translation: bool = Query(True),
    target_language: str = Query("en"),
    filename: str = Query(None)
):
    """Amazon Transcribe模型WebSocket端点"""
    client_id = str(uuid.uuid4())
    
    await websocket.accept()
    active_connections[client_id] = websocket
    
    # 初始化字幕列表
    client_subtitles[client_id] = []
    
    # 如果没有提供文件名，使用时间戳
    if not filename:
        filename = f"subtitle_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"客户端 {client_id} 已连接到Transcribe端点")
    logger.info(f"参数 - 语言: {language}, 纠错: {correction}, 翻译: {translation}, 目标语言: {target_language}, 文件名: {filename}")
    
    try:
        # 发送连接确认
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "client_id": client_id,
            "model": "transcribe",
            "language": language,
            "correction_enabled": correction,
            "translation_enabled": translation,
            "target_language": target_language
        })
        
        # 创建Transcribe模型实例
        transcribe_model = TranscribeModel(
            backend="transcribe",
            region_name=AWS_REGION
        )
        
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
                    
                    # 使用Amazon Transcribe模型处理音频
                    if transcribe_model and transcribe_model.is_available():
                        async for subtitle in transcribe_model.transcribe_stream(
                            audio_generator(), language=language
                        ):
                            # 发送字幕回客户端（传递处理参数）
                            logging.info(f"Received subtitle: {subtitle}")
                            await send_subtitle(
                                websocket, subtitle, client_id, 
                                language=language,
                                enable_correction=correction,
                                enable_translation=translation,
                                target_language=target_language
                            )
            
            except Exception as e:
                logger.error(f"处理音频数据失败: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"处理音频数据失败: {str(e)}"
                })
    
    except WebSocketDisconnect:
        logger.info(f"客户端 {client_id} 断开连接")
        # 保存字幕文件
        if client_id in client_subtitles and client_subtitles[client_id]:
            try:
                save_subtitles(client_subtitles[client_id], filename, language)
                logger.info(f"已保存字幕文件: {filename}")
            except Exception as e:
                logger.error(f"保存字幕文件失败: {e}")
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
    finally:
        # 清理连接
        if client_id in active_connections:
            del active_connections[client_id]
        # 清理字幕数据
        if client_id in client_subtitles:
            del client_subtitles[client_id]


async def process_timestamp_message(client_id: str, message_data: dict):
    """处理时间戳消息"""
    try:
        timestamp_info = message_data.get('timestamp', {})
        
        # 存储时间戳信息
        if client_id not in client_timestamps:
            client_timestamps[client_id] = {}
        
        chunk_index = timestamp_info.get('chunk_index', 0)
        client_timestamps[client_id][chunk_index] = {
            'start_time': timestamp_info.get('start_time', 0.0),
            'end_time': timestamp_info.get('end_time', 0.0),
            'duration': timestamp_info.get('duration', 0.0),
            'chunk_index': chunk_index,
            'total_samples_processed': timestamp_info.get('total_samples_processed', 0),
            'audio_start_time': timestamp_info.get('audio_start_time', 0.0),
            'processing_start_time': timestamp_info.get('processing_start_time', 0.0),
            'current_time': timestamp_info.get('current_time', 0.0),
            'received_at': datetime.datetime.now().isoformat()
        }
        
        logger.info(f"客户端 {client_id} 时间戳信息已存储:")
        logger.info(f"  - Chunk {chunk_index}: {timestamp_info.get('start_time', 0.0):.2f}s - {timestamp_info.get('end_time', 0.0):.2f}s")
        
        return True
    except Exception as e:
        logger.error(f"处理时间戳消息失败: {e}")
        return False


async def get_chunk_timestamp(client_id: str, chunk_index: int) -> Optional[Dict]:
    """获取指定chunk的时间戳信息"""
    if client_id in client_timestamps and chunk_index in client_timestamps[client_id]:
        return client_timestamps[client_id][chunk_index]
    return None


async def apply_timestamp_to_subtitle(subtitle: Subtitle, timestamp_info: Dict) -> Subtitle:
    """将时间戳信息应用到字幕对象"""
    if timestamp_info:
        # 使用前端提供的绝对时间戳
        subtitle.start = timestamp_info['start_time']
        subtitle.end = timestamp_info['end_time']
        
        # 如果字幕有相对时间戳，需要进行映射
        # 这里假设Whisper返回的是chunk内的相对时间戳
        # 实际实现可能需要更复杂的时间戳映射逻辑
        
        logger.info(f"应用时间戳到字幕: {subtitle.start:.2f}s - {subtitle.end:.2f}s")
    
    return subtitle


async def update_whisper_model_language(language: str):
    """动态更新Whisper模型的语言配置"""
    global sagemaker_whisper_model
    
    try:
        if sagemaker_whisper_model and hasattr(sagemaker_whisper_model, 'set_language'):
            # 如果模型支持动态语言设置
            await sagemaker_whisper_model.set_language(language)
            logger.info(f"Whisper模型语言已更新为: {language}")
        else:
            # 使用语言特定配置
            logger.info(f"为语言 {language} 重新配置Whisper模型")
            
            # 获取语言特定配置
            config = create_whisper_config(language)
            sagemaker_params = get_sagemaker_whisper_params(language)
            
            logger.info(f"语言 {language} 的Whisper配置: {config}")
            logger.info(f"语言 {language} 的SageMaker参数: {sagemaker_params}")
            
            # 更新模型配置
            if hasattr(sagemaker_whisper_model, 'whisper_config'):
                sagemaker_whisper_model.whisper_config = config
                logger.info(f"Whisper配置已更新")
            
            # 如果模型支持设置SageMaker参数
            if hasattr(sagemaker_whisper_model, 'set_sagemaker_params'):
                sagemaker_whisper_model.set_sagemaker_params(sagemaker_params)
                logger.info(f"SageMaker参数已更新")
            
    except Exception as e:
        logger.error(f"更新Whisper模型语言配置失败: {e}")


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


def save_subtitles(subtitles: List[Subtitle], filename: str, language: str):
    """保存字幕到文件"""
    if not subtitles:
        logger.warning("没有字幕可保存")
        return
    
    # 确保文件名不包含非法字符
    filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
    
    # 保存SRT格式
    srt_path = subtitle_dir / f"{filename}_{language}.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, subtitle in enumerate(subtitles, 1):
            f.write(f"{i}\n")
            f.write(f"{subtitle.format_time(subtitle.start, 'srt')} --> {subtitle.format_time(subtitle.end, 'srt')}\n")
            f.write(f"{subtitle.text}\n")
            if subtitle.translated_text:
                f.write(f"{subtitle.translated_text}\n")
            f.write("\n")
    
    # 保存VTT格式
    vtt_path = subtitle_dir / f"{filename}_{language}.vtt"
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for subtitle in subtitles:
            f.write(f"{subtitle.format_time(subtitle.start, 'vtt')} --> {subtitle.format_time(subtitle.end, 'vtt')}\n")
            f.write(f"{subtitle.text}\n")
            if subtitle.translated_text:
                f.write(f"{subtitle.translated_text}\n")
            f.write("\n")
    
    # 保存JSON格式（包含更多元数据）
    json_path = subtitle_dir / f"{filename}_{language}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "language": language,
                "subtitles": [subtitle.to_dict() for subtitle in subtitles]
            },
            f,
            ensure_ascii=False,
            indent=2
        )
    
    logger.info(f"字幕已保存为SRT格式: {srt_path}")
    logger.info(f"字幕已保存为VTT格式: {vtt_path}")
    logger.info(f"字幕已保存为JSON格式: {json_path}")


async def send_subtitle(
    websocket: WebSocket, 
    subtitle: Subtitle, 
    client_id: str, 
    language: str = "ar",
    enable_correction: bool = True,
    enable_translation: bool = True,
    target_language: str = "en"
):
    """发送字幕到客户端"""
    try:
        # 创建唯一ID
        subtitle_id = f"{client_id}_{uuid.uuid4()}"
        logger.info(f"处理字幕 - 原文: {subtitle.text}")
        logger.info(f"处理参数 - 纠错: {enable_correction}, 翻译: {enable_translation}, 目标语言: {target_language}")
        
        # 将字幕添加到客户端的字幕列表中
        if client_id in client_subtitles:
            client_subtitles[client_id].append(subtitle)
        
        # 步骤1: 字幕纠错 (如果启用)
        corrected_text = subtitle.text
        correction_applied = False
        
        if enable_correction and subtitle.text.strip() and correction_service:
            try:
                # 获取历史字幕作为上下文
                history_subtitles = []
                if client_id in client_subtitles and len(client_subtitles[client_id]) > 1:
                    # 获取最近的3条历史字幕
                    recent_subtitles = client_subtitles[client_id][-4:-1]  # 排除当前字幕
                    history_subtitles = [s.text for s in recent_subtitles if s.text.strip()]
                
                # 构建纠错输入
                correction_input = CorrectionInput(
                    current_subtitle=subtitle.text,
                    history_subtitles=history_subtitles,
                    scene_description=get_correction_scene_description(language),
                    language=language
                )
                
                # 执行纠错
                correction_result = await correction_service.correct(correction_input)
                
                if correction_result.has_correction:
                    corrected_text = correction_result.corrected_subtitle
                    correction_applied = True
                    logger.info(f"字幕已纠错: '{subtitle.text}' -> '{corrected_text}' (置信度: {correction_result.confidence})")
                    
                    # 更新subtitle对象中的文本
                    subtitle.text = corrected_text
                else:
                    logger.info(f"字幕无需纠错: '{subtitle.text}'")
                    
            except Exception as e:
                logger.error(f"字幕纠错失败: {e}")
                # 纠错失败时使用原始文本
                corrected_text = subtitle.text
        else:
            logger.info(f"跳过纠错 - 启用状态: {enable_correction}")
        
        # 步骤2: 翻译字幕文本 (如果启用，使用纠错后的文本)
        if enable_translation and corrected_text.strip():
            try:
                # 使用翻译服务翻译文本
                translation_result = await translation_manager.translate(
                    text=corrected_text,
                    target_lang=target_language,
                    service="bedrock"  # 优先使用Bedrock翻译服务
                )
                
                # 设置翻译结果
                subtitle.translated_text = translation_result.translated_text
                logger.info(f"字幕已翻译: {corrected_text} -> {subtitle.translated_text}")
            except Exception as e:
                logger.error(f"翻译失败: {e}")
                subtitle.translated_text = f"[翻译失败] {corrected_text}"
        else:
            logger.info(f"跳过翻译 - 启用状态: {enable_translation}")
            subtitle.translated_text = None
        
        # 发送字幕 (包含纠错和翻译信息)
        await websocket.send_json({
            "type": "subtitle",
            "subtitle": {
                "id": subtitle_id,
                "start": subtitle.start,
                "end": subtitle.end,
                "text": subtitle.text,  # 纠错后的文本
                "original_text": subtitle.text if not correction_applied else None,  # 原始文本(如果有纠错)
                "translated_text": subtitle.translated_text,
                "correction_applied": correction_applied,
                "translation_applied": enable_translation and subtitle.translated_text is not None,
                "target_language": target_language if enable_translation else None
            }
        })
    except Exception as e:
        logger.error(f"发送字幕失败: {e}")


@app.websocket("/ws/save_subtitles")
async def websocket_save_subtitles_endpoint(
    websocket: WebSocket,
    client_id: str = Query(...),
    filename: str = Query(None)
):
    """保存字幕WebSocket端点"""
    await websocket.accept()
    
    try:
        # 检查客户端ID是否存在
        if client_id not in client_subtitles:
            await websocket.send_json({
                "type": "error",
                "message": "客户端ID不存在或已过期"
            })
            return
        
        # 如果没有提供文件名，使用时间戳
        if not filename:
            filename = f"subtitle_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 获取字幕列表
        subtitles = client_subtitles[client_id]
        
        if not subtitles:
            await websocket.send_json({
                "type": "error",
                "message": "没有可用的字幕"
            })
            return
        
        # 保存字幕
        try:
            save_subtitles(subtitles, filename, "auto")
            
            # 发送成功消息
            await websocket.send_json({
                "type": "success",
                "message": "字幕已保存",
                "files": [
                    f"{filename}_auto.srt",
                    f"{filename}_auto.vtt",
                    f"{filename}_auto.json"
                ]
            })
        except Exception as e:
            logger.error(f"保存字幕失败: {e}")
            await websocket.send_json({
                "type": "error",
                "message": f"保存字幕失败: {str(e)}"
            })
    
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
