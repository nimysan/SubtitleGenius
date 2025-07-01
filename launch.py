#!/usr/bin/env python3
"""
SubtitleGenius 启动器
"""

import sys
import argparse
from pathlib import Path


def launch_simple():
    """启动简化版界面"""
    print("🚀 启动简化版界面...")
    try:
        from gradio_simple import main
        main()
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return False
    return True


def launch_full():
    """启动完整版界面"""
    print("🚀 启动完整版界面...")
    try:
        from gradio_app import main
        main()
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("\n💡 建议尝试简化版: python launch.py --simple")
        return False
    return True


def check_environment():
    """检查环境"""
    print("🔍 环境检查...")
    
    # 检查 Python 版本
    if sys.version_info < (3, 10):
        print(f"❌ Python 版本过低: {sys.version}")
        print("需要 Python 3.10+")
        return False
    
    print(f"✅ Python 版本: {sys.version_info.major}.{sys.version_info.minor}")
    
    # 检查关键依赖
    try:
        import gradio
        print(f"✅ Gradio: {gradio.__version__}")
    except ImportError:
        print("❌ Gradio 未安装")
        return False
    
    # 检查配置文件
    env_file = Path(".env")
    if env_file.exists():
        print("✅ 配置文件存在")
    else:
        print("⚠️  .env 文件不存在，将使用默认配置")
    
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SubtitleGenius 启动器")
    parser.add_argument(
        "--simple", 
        action="store_true", 
        help="启动简化版界面（推荐）"
    )
    parser.add_argument(
        "--full", 
        action="store_true", 
        help="启动完整版界面"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=7860, 
        help="服务器端口（默认: 7860）"
    )
    
    args = parser.parse_args()
    
    print("🎬 SubtitleGenius 启动器")
    print("=" * 50)
    
    # 环境检查
    if not check_environment():
        print("❌ 环境检查失败")
        sys.exit(1)
    
    print("=" * 50)
    
    # 选择启动模式
    if args.simple:
        success = launch_simple()
    elif args.full:
        success = launch_full()
    else:
        # 默认启动简化版
        print("🎯 默认启动简化版界面")
        print("💡 使用 --full 启动完整版，--simple 明确启动简化版")
        print("=" * 50)
        success = launch_simple()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
