#!/usr/bin/env python3
"""
DASH服务器测试脚本
"""

import requests
import time
import threading
from pathlib import Path
import xml.etree.ElementTree as ET

def test_server():
    """测试服务器功能"""
    base_url = "http://localhost:8080"
    
    print("🧪 开始测试DASH服务器...")
    
    # 等待服务器启动
    time.sleep(2)
    
    try:
        # 测试1: 首页
        print("\n📋 测试1: 访问首页")
        response = requests.get(base_url)
        if response.status_code == 200:
            print("✅ 首页访问成功")
        else:
            print(f"❌ 首页访问失败: {response.status_code}")
        
        # 测试2: MPD文件
        print("\n📺 测试2: 获取MPD文件")
        mpd_url = f"{base_url}/tv001.mpd"
        response = requests.get(mpd_url)
        if response.status_code == 200:
            print("✅ MPD文件获取成功")
            
            # 检查是否包含字幕轨道
            mpd_content = response.text
            if 'text/vtt' in mpd_content and 'caption_en' in mpd_content:
                print("✅ 字幕轨道已添加")
                
                # 解析XML验证结构
                try:
                    root = ET.fromstring(mpd_content)
                    print("✅ XML结构有效")
                except ET.ParseError:
                    print("❌ XML结构无效")
            else:
                print("❌ 字幕轨道未找到")
        else:
            print(f"❌ MPD文件获取失败: {response.status_code}")
        
        # 测试3: 字幕文件
        print("\n📝 测试3: 获取字幕文件")
        subtitle_url = f"{base_url}/subtitles/tv001.vtt"
        response = requests.get(subtitle_url)
        if response.status_code == 200:
            print("✅ 字幕文件获取成功")
            if response.text.startswith('WEBVTT'):
                print("✅ VTT格式正确")
            else:
                print("❌ VTT格式错误")
        else:
            print(f"❌ 字幕文件获取失败: {response.status_code}")
        
        print("\n🎉 测试完成!")
        
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器，请确保服务器正在运行")
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")

def start_server():
    """启动服务器"""
    try:
        from dash_server import app
        app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)
    except Exception as e:
        print(f"服务器启动失败: {e}")

if __name__ == '__main__':
    # 检查测试环境
    dash_dir = Path("dash_output")
    if not dash_dir.exists():
        print("❌ dash_output目录不存在，请先运行服务器创建示例文件")
        exit(1)
    
    print("🚀 启动测试服务器...")
    
    # 在后台线程启动服务器
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # 运行测试
    test_server()
