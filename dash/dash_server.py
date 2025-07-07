#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DASH静态服务器 - 用于分发DASH内容
提供CORS支持和正确的MIME类型
"""

import os
import sys
import argparse
import logging
import signal
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dash_server')

# DASH相关MIME类型
MIME_TYPES = {
    '.mpd': 'application/dash+xml',
    '.m4s': 'video/iso.segment',
    '.mp4': 'video/mp4',
    '.webm': 'video/webm',
    '.vtt': 'text/vtt',
    '.html': 'text/html',
    '.js': 'application/javascript',
    '.css': 'text/css',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon',
}

class CORSHTTPRequestHandler(SimpleHTTPRequestHandler):
    """支持CORS的HTTP请求处理器"""
    
    def __init__(self, *args, directory=None, cors_origins="*", **kwargs):
        self.cors_origins = cors_origins
        super().__init__(*args, directory=directory, **kwargs)
    
    def end_headers(self):
        """添加CORS头"""
        self.send_header('Access-Control-Allow-Origin', self.cors_origins)
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Range')
        self.send_header('Access-Control-Expose-Headers', 'Content-Length, Content-Range')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()
    
    def do_OPTIONS(self):
        """处理OPTIONS请求"""
        self.send_response(200)
        self.end_headers()
    
    def guess_type(self, path):
        """根据文件扩展名猜测MIME类型"""
        base, ext = os.path.splitext(path)
        if ext in MIME_TYPES:
            return MIME_TYPES[ext]
        return super().guess_type(path)
    
    def log_message(self, format, *args):
        """自定义日志格式"""
        logger.info(f"{self.address_string()} - {format % args}")

def run_server(directory, port=8000, cors_origins="*"):
    """运行HTTP服务器"""
    # 确保目录存在
    if not os.path.isdir(directory):
        logger.error(f"目录不存在: {directory}")
        return False
    
    # 切换到指定目录
    os.chdir(directory)
    
    # 创建服务器
    handler = partial(CORSHTTPRequestHandler, directory=directory, cors_origins=cors_origins)
    server = HTTPServer(('', port), handler)
    
    logger.info(f"启动DASH服务器 - http://localhost:{port}")
    logger.info(f"提供目录: {os.path.abspath(directory)}")
    logger.info(f"CORS策略: {cors_origins}")
    logger.info("按Ctrl+C停止服务器")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("服务器已停止")
    except Exception as e:
        logger.error(f"服务器错误: {e}")
        return False
    finally:
        server.server_close()
    
    return True

def signal_handler(sig, frame):
    """处理信号"""
    logger.info("接收到中断信号，正在退出...")
    sys.exit(0)

def create_index_html(directory, manifest_path=None):
    """创建简单的HTML播放器页面"""
    index_path = os.path.join(directory, "index.html")
    
    # 如果文件已存在，不覆盖
    if os.path.exists(index_path):
        logger.info(f"播放器页面已存在: {index_path}")
        return index_path
    
    # 如果没有指定manifest_path，尝试查找
    if not manifest_path:
        for file in os.listdir(directory):
            if file.endswith(".mpd"):
                manifest_path = file
                break
    
    # 如果仍然没有找到，使用默认值
    if not manifest_path:
        manifest_path = "manifest.mpd"
    
    # 创建HTML内容
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DASH播放器</title>
    <script src="https://cdn.dashjs.org/latest/dash.all.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
        }}
        .player-container {{
            width: 100%;
            margin: 20px 0;
        }}
        video {{
            width: 100%;
            max-width: 1080px;
            margin: 0 auto;
            display: block;
            background-color: #000;
        }}
        .controls {{
            margin: 20px 0;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }}
        .controls label {{
            margin-right: 10px;
            font-weight: bold;
        }}
        .info {{
            margin-top: 20px;
            padding: 10px;
            background-color: #e9f7fe;
            border-left: 4px solid #2196f3;
            border-radius: 3px;
        }}
        .footer {{
            margin-top: 30px;
            text-align: center;
            font-size: 0.9em;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>DASH流媒体播放器</h1>
        
        <div class="player-container">
            <video id="videoPlayer" controls></video>
        </div>
        
        <div class="controls">
            <label for="manifestInput">DASH清单URL:</label>
            <input type="text" id="manifestInput" value="{manifest_path}" style="width: 60%;">
            <button onclick="loadManifest()">加载</button>
            <button onclick="toggleStats()">显示/隐藏统计</button>
        </div>
        
        <div id="stats" class="info" style="display: none;">
            <h3>播放统计</h3>
            <div id="statsContent"></div>
        </div>
        
        <div class="info">
            <h3>使用说明</h3>
            <p>1. 确认DASH清单URL正确</p>
            <p>2. 点击"加载"按钮加载视频</p>
            <p>3. 使用视频播放器控件控制播放</p>
            <p>4. 点击"显示/隐藏统计"查看播放统计信息</p>
        </div>
        
        <div class="footer">
            <p>由SubtitleGenius DASH流媒体字幕处理系统提供支持</p>
        </div>
    </div>
    
    <script>
        let player = null;
        let statsInterval = null;
        
        function loadManifest() {{
            const manifestUrl = document.getElementById('manifestInput').value;
            
            if (player) {{
                player.destroy();
                clearInterval(statsInterval);
            }}
            
            player = dashjs.MediaPlayer().create();
            player.initialize(document.querySelector("#videoPlayer"), manifestUrl, true);
            player.updateSettings({{
                'streaming': {{
                    'abr': {{
                        'autoSwitchBitrate': {{
                            'video': true,
                            'audio': true
                        }}
                    }},
                    'buffer': {{
                        'stableBufferTime': 20,
                        'bufferTimeAtTopQuality': 30
                    }}
                }}
            }});
            
            // 显示初始统计信息
            updateStats();
            
            // 设置定期更新统计信息
            if (statsInterval) {{
                clearInterval(statsInterval);
            }}
            statsInterval = setInterval(updateStats, 1000);
            
            console.log("已加载清单:", manifestUrl);
        }}
        
        function toggleStats() {{
            const statsDiv = document.getElementById('stats');
            if (statsDiv.style.display === 'none') {{
                statsDiv.style.display = 'block';
                updateStats();
            }} else {{
                statsDiv.style.display = 'none';
            }}
        }}
        
        function updateStats() {{
            if (!player || !document.getElementById('stats').style.display === 'none') {{
                return;
            }}
            
            const video = document.querySelector("#videoPlayer");
            const statsContent = document.getElementById('statsContent');
            
            try {{
                const dashMetrics = player.getDashMetrics();
                const streamInfo = player.getActiveStream().getStreamInfo();
                const dashAdapter = player.getDashAdapter();
                
                if (dashMetrics && streamInfo) {{
                    const repSwitch = dashMetrics.getCurrentRepresentationSwitch('video', true);
                    const bufferLevel = dashMetrics.getCurrentBufferLevel('video');
                    const bitrate = repSwitch ? Math.round(dashAdapter.getBandwidthForRepresentation(repSwitch.to, streamInfo.index) / 1000) : 0;
                    
                    let stats = `
                        <p><strong>播放状态:</strong> ${{video.paused ? '暂停' : '播放中'}}</p>
                        <p><strong>当前时间:</strong> ${{video.currentTime.toFixed(2)}}秒</p>
                        <p><strong>视频时长:</strong> ${{video.duration.toFixed(2)}}秒</p>
                        <p><strong>缓冲区:</strong> ${{bufferLevel ? bufferLevel.toFixed(2) : 0}}秒</p>
                        <p><strong>当前码率:</strong> ${{bitrate}} kbps</p>
                    `;
                    
                    statsContent.innerHTML = stats;
                }}
            }} catch (e) {{
                console.error("获取统计信息时出错:", e);
            }}
        }}
        
        // 页面加载完成后自动加载清单
        window.addEventListener('DOMContentLoaded', loadManifest);
    </script>
</body>
</html>
"""
    
    # 写入文件
    try:
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"已创建播放器页面: {index_path}")
        return index_path
    except Exception as e:
        logger.error(f"创建播放器页面失败: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DASH静态服务器")
    parser.add_argument("--dir", default=".", help="要提供服务的目录")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--cors", default="*", help="CORS策略")
    parser.add_argument("--create-player", action="store_true", help="创建HTML播放器页面")
    parser.add_argument("--manifest", help="DASH清单文件名（用于播放器页面）")
    
    args = parser.parse_args()
    
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 创建播放器页面
    if args.create_player:
        create_index_html(args.dir, args.manifest)
    
    # 运行服务器
    run_server(args.dir, args.port, args.cors)
