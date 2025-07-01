#!/usr/bin/env python3
"""
SubtitleGenius Gradio 启动脚本
"""

import sys
import os
import subprocess
from pathlib import Path


def check_python_version():
    """检查 Python 版本"""
    if sys.version_info < (3, 10):
        print(f"❌ Python 版本过低: {sys.version}")
        print("需要 Python 3.10 或更高版本")
        return False
    
    print(f"✅ Python 版本: {sys.version}")
    return True


def check_dependencies():
    """检查依赖"""
    print("🔍 检查依赖...")
    
    required_packages = [
        'gradio',
        'boto3',
        'openai',
        'anthropic'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} 未安装")
            missing_packages.append(package)
    
    return len(missing_packages) == 0


def install_dependencies():
    """安装依赖"""
    print("📦 安装依赖...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
        print("✅ 依赖安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖安装失败: {e}")
        return False


def check_config():
    """检查配置"""
    print("⚙️  检查配置...")
    
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env 文件不存在")
        print("创建示例配置文件...")
        
        example_env = Path(".env.example")
        if example_env.exists():
            import shutil
            shutil.copy2(example_env, env_file)
            print("✅ 已创建 .env 文件，请编辑配置")
        else:
            print("❌ .env.example 文件不存在")
            return False
    
    print("✅ 配置文件存在")
    return True


def start_gradio():
    """启动 Gradio 应用"""
    print("🚀 启动 Gradio 应用...")
    
    try:
        # 导入并启动应用
        from gradio_app import main
        main()
        
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        
        print("\n🔧 故障排除:")
        print("1. 检查端口是否被占用: lsof -i :7860")
        print("2. 尝试不同端口: 修改 gradio_app.py 中的 server_port")
        print("3. 检查网络设置")
        print("4. 查看详细错误信息:")
        
        import traceback
        traceback.print_exc()
        
        return False
    
    return True


def main():
    """主函数"""
    print("🎬 SubtitleGenius - Gradio 启动器")
    print("=" * 50)
    
    # 检查 Python 版本
    if not check_python_version():
        sys.exit(1)
    
    # 检查依赖
    if not check_dependencies():
        print("\n📦 尝试安装依赖...")
        if not install_dependencies():
            print("❌ 无法安装依赖，请手动运行: uv sync")
            sys.exit(1)
    
    # 检查配置
    if not check_config():
        print("❌ 配置检查失败")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 所有检查通过，启动应用...")
    print("📱 访问地址: http://127.0.0.1:7860")
    print("⏹️  按 Ctrl+C 停止应用")
    print("=" * 50)
    
    # 启动应用
    if not start_gradio():
        sys.exit(1)


if __name__ == "__main__":
    main()
