#!/usr/bin/env python3
"""
简化版DASH服务器启动脚本
"""

import os
import sys
from pathlib import Path

def main():
    # 检查dash_output目录
    dash_dir = Path("dash_output")
    if not dash_dir.exists():
        print("❌ dash_output目录不存在")
        print("请先创建dash_output目录并放入节目文件夹")
        return
    
    # 检查是否有节目目录
    programs = [d for d in dash_dir.iterdir() if d.is_dir()]
    if not programs:
        print("❌ dash_output目录中没有找到节目文件夹")
        return
    
    print("✅ 找到以下节目:")
    for program in programs:
        mpd_files = list(program.glob("*.mpd"))
        print(f"  📁 {program.name} - {len(mpd_files)} 个MPD文件")
    
    print("\n🚀 启动DASH服务器...")
    print("📍 访问地址:")
    print("   http://localhost:8080 - 查看所有节目")
    print("   http://localhost:8080/tv001.mpd - 获取带字幕的MPD")
    print("\n按 Ctrl+C 停止服务器")
    
    # 启动服务器
    try:
        from dash_server import app
        app.run(host='0.0.0.0', port=8080, debug=False)
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except ImportError:
        print("❌ 请先安装Flask: pip install flask")

if __name__ == '__main__':
    main()
