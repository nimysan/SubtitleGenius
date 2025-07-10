#!/usr/bin/env python
"""
启动WebSocket服务器的脚本
用于启动subtitle_genius.stream.websocket_server模块中的WebSocket服务器
"""

import asyncio
import logging
import argparse
import os
import sys
from logging.handlers import RotatingFileHandler

# 确保logs目录存在
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# 配置主日志文件
main_log_file = os.path.join(logs_dir, 'server.log')
# 配置VAC处理器日志文件
vac_log_file = os.path.join(logs_dir, 'vac_processor.log')

# 创建主日志处理器
main_file_handler = RotatingFileHandler(
    main_log_file,
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,          # 保留5个备份文件
    encoding='utf-8'
)
console_handler = logging.StreamHandler()

# 创建VAC处理器日志处理器
vac_file_handler = RotatingFileHandler(
    vac_log_file,
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,          # 保留5个备份文件
    encoding='utf-8'
)



# 设置日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
main_file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
vac_file_handler.setFormatter(formatter)

# 配置根日志记录器
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(main_file_handler)
root_logger.addHandler(console_handler)

# 配置VAC处理器的日志记录器
vac_logger = logging.getLogger("subtitle_genius.stream.vac_processor")
# 移除从根日志记录器继承的处理器，避免重复日志
vac_logger.propagate = False
vac_logger.setLevel(logging.INFO)
vac_logger.addHandler(vac_file_handler)
vac_logger.addHandler(console_handler)  # 同时输出到控制台

logger = logging.getLogger("start_websocket_server")
logger.info(f"主日志将写入: {main_log_file}")
logger.info(f"VAC处理器日志将写入: {vac_log_file}")

# 确保可以导入subtitle_genius模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="启动WebSocket服务器")
    parser.add_argument("--host", default="localhost", help="服务器主机名")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    args = parser.parse_args()

    # 如果启用调试模式，设置日志级别为DEBUG
    if args.debug:
        root_logger.setLevel(logging.DEBUG)
        vac_logger.setLevel(logging.DEBUG)
        logger.info("调试模式已启用")

    try:
        # 导入WebSocketServer
        from subtitle_genius.stream.websocket_server import WebSocketServer

        # 创建并启动服务器
        logger.info(f"正在启动WebSocket服务器，地址: {args.host}:{args.port}")
        server = WebSocketServer(host=args.host, port=args.port)
        ws_server = await server.start_server()

        logger.info(f"WebSocket服务器已启动，地址: ws://{args.host}:{args.port}")
        logger.info("按Ctrl+C停止服务器")

        # 保持服务器运行
        await asyncio.Future()
    except KeyboardInterrupt:
        logger.info("服务器被用户停止")
    except Exception as e:
        logger.error(f"启动服务器时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # 如果ws_server已定义，关闭它
        if 'ws_server' in locals():
            ws_server.close()
            await ws_server.wait_closed()
            logger.info("服务器已关闭")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("服务器被用户停止")
