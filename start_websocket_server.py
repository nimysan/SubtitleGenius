#!/usr/bin/env python3
"""
WebSocket服务器启动脚本
支持从前端获取参数设定并进行字幕回显
"""

import asyncio
import logging
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from subtitle_genius.stream.websocket_server import main

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("websocket_server_launcher")

def print_banner():
    """打印启动横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    SubtitleGenius WebSocket服务器             ║
║                     支持参数配置和字幕回显                     ║
╠══════════════════════════════════════════════════════════════╣
║ 功能特性:                                                    ║
║ • 支持多种AI模型 (Whisper, Claude, Transcribe)              ║
║ • 支持多语言识别和翻译                                        ║
║ • 智能纠错和场景感知                                          ║
║ • 实时字幕生成和回显                                          ║
║ • URL参数配置                                                ║
╠══════════════════════════════════════════════════════════════╣
║ 连接示例:                                                    ║
║ ws://localhost:8000/ws/whisper?language=zh&translation=true  ║
║ ws://localhost:8000/ws/claude?language=ar&correction=true    ║
║ ws://localhost:8000/ws/transcribe?language=en               ║
╠══════════════════════════════════════════════════════════════╣
║ 支持的参数:                                                  ║
║ • language: 源语言 (zh, ar, en, fr, es等)                   ║
║ • target_language: 目标语言 (用于翻译)                       ║
║ • correction: 启用纠错 (true/false)                         ║
║ • translation: 启用翻译 (true/false)                        ║
║ • scene_description: 场景描述                               ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_environment():
    """检查环境配置"""
    logger.info("检查环境配置...")
    
    # 检查必要的环境变量
    required_env_vars = [
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY',
        'AWS_REGION'
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"缺少环境变量: {', '.join(missing_vars)}")
        logger.warning("某些功能可能无法正常工作")
    else:
        logger.info("✅ 环境变量配置完整")
    
    # 检查SageMaker端点配置
    sagemaker_endpoint = os.environ.get('SAGEMAKER_WHISPER_ENDPOINT')
    if sagemaker_endpoint:
        logger.info(f"✅ SageMaker Whisper端点: {sagemaker_endpoint}")
    else:
        logger.warning("⚠️  未配置SAGEMAKER_WHISPER_ENDPOINT，将使用默认端点")
    
    return len(missing_vars) == 0

async def main_with_error_handling():
    """带错误处理的主函数"""
    try:
        print_banner()
        
        # 检查环境
        env_ok = check_environment()
        if not env_ok:
            logger.warning("环境配置不完整，但服务器仍将启动")
        
        logger.info("启动WebSocket服务器...")
        logger.info("服务器地址: ws://localhost:8000")
        logger.info("按 Ctrl+C 停止服务器")
        logger.info("-" * 60)
        
        # 启动服务器
        await main()
        
    except KeyboardInterrupt:
        logger.info("\n服务器被用户停止")
    except Exception as e:
        logger.error(f"服务器启动失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main_with_error_handling())
    except KeyboardInterrupt:
        print("\n再见! 👋")
    except Exception as e:
        logger.error(f"启动脚本出错: {e}")
        sys.exit(1)
