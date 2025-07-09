#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DASH流媒体转换和服务脚本
"""

import os
import sys
import shutil
import subprocess
import time
import signal
import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dash_server')

def clean_directory(directory):
    """清理目录，如果不存在则创建"""
    logger.info(f"清理目录: {directory}")
    if os.path.exists(directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            try:
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                logger.error(f"清理文件失败: {item_path} - {e}")
    else:
        os.makedirs(directory)
        logger.info(f"创建目录: {directory}")

def transcode_to_dash(input_file, output_dir, segment_duration=4, window_size=5, extra_window_size=10):
    """使用ffmpeg将视频转码为DASH格式"""
    if not os.path.exists(input_file):
        logger.error(f"输入文件不存在: {input_file}")
        return False
    
    # 构建ffmpeg命令
    cmd = [
        "ffmpeg", "-re", "-i", input_file,
        "-c:v", "libx264", "-c:a", "aac",
        "-f", "dash",
        "-use_template", "1", "-use_timeline", "1",
        "-window_size", str(window_size),
        "-extra_window_size", str(extra_window_size),
        "-seg_duration", str(segment_duration),
        os.path.join(output_dir, "index.mpd")
    ]
    
    logger.info(f"开始转码: {' '.join(cmd)}")
    
    try:
        # 使用subprocess.Popen而不是run，这样可以在后台运行
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        logger.info(f"转码进程已启动，PID: {process.pid}")
        return process
    except Exception as e:
        logger.error(f"转码失败: {e}")
        return None

class CORSHTTPRequestHandler(SimpleHTTPRequestHandler):
    """支持CORS的HTTP请求处理器"""
    
    def end_headers(self):
        """添加CORS头"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Range')
        self.send_header('Access-Control-Expose-Headers', 'Content-Length, Content-Range')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()
    
    def do_OPTIONS(self):
        """处理OPTIONS请求"""
        self.send_response(200)
        self.end_headers()
    
    def log_message(self, format, *args):
        """自定义日志格式"""
        logger.info(f"{self.address_string()} - {format % args}")

def start_http_server(directory, port=8080):
    """启动HTTP服务器"""
    # 切换到指定目录
    os.chdir(directory)
    
    # 创建服务器
    handler = CORSHTTPRequestHandler
    server = HTTPServer(('', port), handler)
    
    logger.info(f"启动HTTP服务器 - http://localhost:{port}")
    logger.info(f"提供目录: {os.path.abspath(directory)}")
    
    # 在新线程中启动服务器
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    return server

def signal_handler(sig, frame):
    """处理信号"""
    logger.info("接收到中断信号，正在退出...")
    if 'transcode_process' in globals() and transcode_process:
        transcode_process.terminate()
    if 'http_server' in globals() and http_server:
        http_server.shutdown()
    sys.exit(0)

def create_html_player(directory):
    """创建简单的HTML播放器"""
    html_content = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DASH播放器</title>
    <script src="https://cdn.dashjs.org/latest/dash.all.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
        }
        video {
            width: 100%;
            max-width: 1080px;
            margin: 0 auto;
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>DASH流媒体播放器</h1>
        <video id="videoPlayer" controls></video>
    </div>
    
    <script>
        // 初始化播放器
        var player = dashjs.MediaPlayer().create();
        player.initialize(document.querySelector("#videoPlayer"), "index.mpd", true);
        
        // 设置缓冲参数
        player.updateSettings({
            'streaming': {
                'buffer': {
                    'stableBufferTime': 20,
                    'bufferTimeAtTopQuality': 30
                }
            }
        });
    </script>
</body>
</html>
"""
    
    with open(os.path.join(directory, "player.html"), "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info(f"已创建HTML播放器: {os.path.join(directory, 'player.html')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DASH流媒体转换和服务脚本")
    parser.add_argument("--input", required=True, help="输入视频文件路径")
    parser.add_argument("--output", default="dash", help="输出目录")
    parser.add_argument("--port", type=int, default=8080, help="HTTP服务器端口")
    parser.add_argument("--segment", type=int, default=4, help="分段时长(秒)")
    parser.add_argument("--window", type=int, default=5, help="窗口大小")
    parser.add_argument("--buffer", type=int, default=10, help="额外缓冲窗口大小")
    
    args = parser.parse_args()
    
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 清理输出目录
    clean_directory(args.output)
    
    # 创建HTML播放器
    create_html_player(args.output)
    
    # 启动转码进程
    transcode_process = transcode_to_dash(
        args.input, 
        args.output, 
        args.segment, 
        args.window, 
        args.buffer
    )
    
    if not transcode_process:
        logger.error("转码进程启动失败")
        sys.exit(1)
    
    # 等待一段时间，确保初始分片已生成
    time.sleep(5)
    
    # 启动HTTP服务器
    http_server = start_http_server(args.output, args.port)
    
    logger.info(f"服务已启动，请访问: http://localhost:{args.port}/player.html")
    logger.info("按Ctrl+C停止服务")
    
    try:
        # 保持主线程运行
        while transcode_process.poll() is None:
            time.sleep(1)
        
        # 如果转码进程结束
        return_code = transcode_process.poll()
        if return_code == 0:
            logger.info("转码完成")
        else:
            logger.error(f"转码进程异常退出，返回码: {return_code}")
            stderr = transcode_process.stderr.read()
            logger.error(f"错误输出: {stderr}")
    except KeyboardInterrupt:
        logger.info("用户中断")
    finally:
        # 清理资源
        if transcode_process and transcode_process.poll() is None:
            transcode_process.terminate()
            logger.info("已终止转码进程")
        
        if http_server:
            http_server.shutdown()
            logger.info("已关闭HTTP服务器")
