#!/usr/bin/env python3
"""
DASH服务器 - 支持动态添加字幕轨道到MPD文件
"""

import os
import re
from pathlib import Path
from flask import Flask, send_file, request, abort, Response
from xml.etree import ElementTree as ET
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 配置
DASH_OUTPUT_DIR = "dash_output"
SUBTITLE_BASE_URL = "https://dash.akamaized.net/akamai/test/caption_test/ElephantsDream/ElephantsDream_en.vtt"

def add_subtitle_to_mpd(mpd_content, program_id):
    """
    向MPD文件添加字幕AdaptationSet
    
    Args:
        mpd_content (str): 原始MPD文件内容
        program_id (str): 节目ID (如 tv001)
    
    Returns:
        str: 修改后的MPD内容
    """
    try:
        # 解析XML
        root = ET.fromstring(mpd_content)
        
        # 查找MPD命名空间
        namespace = ""
        if root.tag.startswith('{'):
            namespace = root.tag.split('}')[0] + '}'
        
        # 查找Period元素
        period_tag = f"{namespace}Period" if namespace else "Period"
        period = root.find(period_tag)
        
        if period is None:
            logger.warning("未找到Period元素")
            return mpd_content
        
        # 创建字幕AdaptationSet
        subtitle_adaptation_set = ET.Element("AdaptationSet")
        subtitle_adaptation_set.set("mimeType", "text/vtt")
        subtitle_adaptation_set.set("lang", "en")
        
        # 创建Representation
        representation = ET.SubElement(subtitle_adaptation_set, "Representation")
        representation.set("id", "caption_en")
        representation.set("bandwidth", "256")
        
        # 创建BaseURL
        base_url = ET.SubElement(representation, "BaseURL")
        # 可以根据program_id动态设置字幕URL
        base_url.text = f"/tv002/{program_id}.vtt"
        
        # 将字幕AdaptationSet添加到Period
        period.append(subtitle_adaptation_set)
        
        # 转换回字符串
        ET.register_namespace('', namespace.strip('{}') if namespace else '')
        modified_content = ET.tostring(root, encoding='unicode', method='xml')
        
        # 添加XML声明
        if not modified_content.startswith('<?xml'):
            modified_content = '<?xml version="1.0" encoding="UTF-8"?>\n' + modified_content
        
        logger.info(f"成功为 {program_id} 添加字幕轨道")
        return modified_content
        
    except ET.ParseError as e:
        logger.error(f"XML解析错误: {e}")
        return mpd_content
    except Exception as e:
        logger.error(f"添加字幕时出错: {e}")
        return mpd_content

@app.route('/')
def index():
    """首页 - 显示可用的节目列表"""
    try:
        dash_dir = Path(DASH_OUTPUT_DIR)
        if not dash_dir.exists():
            return f"DASH目录不存在: {DASH_OUTPUT_DIR}", 404
        
        programs = []
        for item in dash_dir.iterdir():
            if item.is_dir():
                # 查找MPD文件
                mpd_files = list(item.glob("*.mpd"))
                if mpd_files:
                    programs.append({
                        'id': item.name,
                        'mpd_url': f"/{item.name}/{item.name}.mpd",
                        'mpd_files': [f.name for f in mpd_files]
                    })
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>DASH服务器</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .program { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }
                .mpd-link { color: #0066cc; text-decoration: none; }
                .mpd-link:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <h1>DASH服务器 - 可用节目</h1>
        """
        
        if programs:
            for program in programs:
                html += f"""
                <div class="program">
                    <h3>节目: {program['id']}</h3>
                    <p><strong>MPD URL:</strong> <a href="{program['mpd_url']}" class="mpd-link">{program['mpd_url']}</a></p>
                    <p><strong>原始文件:</strong> {', '.join(program['mpd_files'])}</p>
                </div>
                """
        else:
            html += "<p>未找到任何DASH节目</p>"
        
        html += """
            <hr>
            <h2>使用说明</h2>
            <ul>
                <li>访问 <code>/节目ID/节目ID.mpd</code> 获取带字幕的MPD文件</li>
                <li>访问 <code>/节目ID/文件名.m4s</code> 获取媒体片段</li>
                <li>字幕会自动添加到MPD文件中</li>
            </ul>
        </body>
        </html>
        """
        
        return html
        
    except Exception as e:
        logger.error(f"生成首页时出错: {e}")
        return f"服务器错误: {e}", 500

@app.route('/<path:file_path>')
def serve_file(file_path):
    """
    提供任何文件（MPD、媒体片段、字幕等）
    
    Args:
        file_path (str): 文件路径，相对于DASH_OUTPUT_DIR
    """
    try:
        # 构建完整的文件路径
        full_path = Path(DASH_OUTPUT_DIR) / file_path
        
        if not full_path.exists():
            logger.warning(f"文件不存在: {full_path}")
            abort(404)
        
        logger.info(f"提供文件: {full_path}")
        
        # 如果是MPD文件，添加字幕轨道
        if full_path.suffix.lower() == '.mpd':
            # 尝试从路径中提取program_id
            parts = file_path.split('/')
            if len(parts) > 0:
                program_id = parts[0]  # 假设第一部分是program_id
                
                # 读取原始MPD内容
                with open(full_path, 'r', encoding='utf-8') as f:
                    original_content = f.read()
                
                # 添加字幕轨道
                modified_content = add_subtitle_to_mpd(original_content, program_id)
                
                # 返回修改后的MPD
                return Response(
                    modified_content,
                    mimetype='application/dash+xml',
                    headers={
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Headers': 'Range',
                        'Cache-Control': 'no-cache'
                    }
                )
        
        # 根据文件扩展名设置MIME类型
        mime_types = {
            '.mpd': 'application/dash+xml',
            '.m4s': 'video/mp4',
            '.mp4': 'video/mp4',
            '.m4v': 'video/mp4',
            '.m4a': 'audio/mp4',
            '.webm': 'video/webm',
            '.vtt': 'text/vtt'
        }
        
        file_ext = full_path.suffix.lower()
        mime_type = mime_types.get(file_ext, 'application/octet-stream')
        
        return send_file(
            full_path,
            mimetype=mime_type,
            as_attachment=False,
            conditional=True  # 支持Range请求
        )
        
    except Exception as e:
        logger.error(f"提供文件时出错: {e}")
        abort(500)

@app.route('/subtitles/<program_id>.vtt')
def serve_subtitle(program_id):
    """
    提供字幕文件
    
    Args:
        program_id (str): 节目ID
    """
    try:
        # 查找字幕文件
        subtitle_paths = [
            Path(DASH_OUTPUT_DIR) / program_id / f"{program_id}.vtt",
            Path(DASH_OUTPUT_DIR) / program_id / "subtitle.vtt",
            Path(DASH_OUTPUT_DIR) / program_id / "subtitles.vtt"
        ]
        
        subtitle_file = None
        for path in subtitle_paths:
            if path.exists():
                subtitle_file = path
                break
        
        if subtitle_file:
            logger.info(f"提供字幕文件: {subtitle_file}")
            return send_file(
                subtitle_file,
                mimetype='text/vtt',
                as_attachment=False
            )
        else:
            # 如果没有找到字幕文件，返回示例字幕
            logger.info(f"未找到字幕文件，返回示例字幕: {program_id}")
            sample_vtt = f"""WEBVTT

00:00:00.000 --> 00:00:05.000
示例字幕 - {program_id}

00:00:05.000 --> 00:00:10.000
这是一个示例字幕文件

00:00:10.000 --> 00:00:15.000
请替换为实际的字幕内容
"""
            return Response(
                sample_vtt,
                mimetype='text/vtt',
                headers={'Access-Control-Allow-Origin': '*'}
            )
            
    except Exception as e:
        logger.error(f"提供字幕文件时出错: {e}")
        abort(500)

@app.errorhandler(404)
def not_found(error):
    return f"文件未找到: {request.path}", 404

@app.errorhandler(500)
def internal_error(error):
    return f"服务器内部错误: {error}", 500

if __name__ == '__main__':
    # 检查DASH目录
    if not os.path.exists(DASH_OUTPUT_DIR):
        print(f"警告: DASH目录不存在: {DASH_OUTPUT_DIR}")
        print("请确保dash_output目录存在并包含节目文件夹")
    
    print(f"启动DASH服务器...")
    print(f"DASH目录: {os.path.abspath(DASH_OUTPUT_DIR)}")
    print(f"访问 http://localhost:8080 查看可用节目")
    print(f"访问 http://localhost:8080/tv001/tv001.mpd 获取带字幕的MPD文件")
    print(f"访问 http://localhost:8080/tv001/tv001_1.m4s 获取媒体片段")
    
    app.run(
        host='0.0.0.0',
        port=8080,
        debug=True,
        threaded=True
    )
