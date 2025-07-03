#!/usr/bin/env python
"""
启动WebSocket服务器
用于接收前端音频流并进行字幕识别
"""

import uvicorn
import logging
import argparse
from pathlib import Path
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.absolute()))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/websocket_server.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='启动SubtitleGenius WebSocket服务器')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器主机地址')
    parser.add_argument('--port', type=int, default=8000, help='服务器端口')
    parser.add_argument('--reload', action='store_true', help='启用自动重载')
    
    args = parser.parse_args()
    
    # 确保日志目录存在
    Path('logs').mkdir(exist_ok=True)
    
    logger.info(f"启动WebSocket服务器 - 主机: {args.host}, 端口: {args.port}")
    
    # 启动FastAPI应用
    uvicorn.run(
        "subtitle_genius.stream.websocket_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("服务器已停止")
    except Exception as e:
        logger.error(f"服务器启动失败: {e}")
