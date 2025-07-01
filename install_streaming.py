#!/usr/bin/env python3
"""
安装 Amazon Transcribe 流式处理依赖
"""

import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """运行命令并显示结果"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} 成功")
        if result.stdout:
            print(f"   输出: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败")
        print(f"   错误: {e.stderr.strip()}")
        return False


def main():
    """主函数"""
    print("🎬 SubtitleGenius - 安装流式处理依赖")
    print("=" * 50)
    
    # 检查是否在虚拟环境中
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ 检测到虚拟环境")
    else:
        print("⚠️  建议在虚拟环境中安装依赖")
    
    # 安装核心依赖
    dependencies = [
        ("pip install --upgrade pip", "升级 pip"),
        ("pip install boto3 botocore", "安装 AWS SDK"),
        ("pip install amazon-transcribe", "安装 Amazon Transcribe 流式处理包"),
        ("pip install pydantic-settings", "安装 Pydantic Settings"),
        ("pip install pyaudio", "安装 PyAudio (音频处理)"),
        ("pip install numpy", "安装 NumPy"),
        ("pip install asyncio", "安装 AsyncIO"),
    ]
    
    success_count = 0
    total_count = len(dependencies)
    
    for command, description in dependencies:
        if run_command(command, description):
            success_count += 1
        print()
    
    # 安装结果总结
    print("=" * 50)
    print(f"📊 安装结果: {success_count}/{total_count} 成功")
    
    if success_count == total_count:
        print("🎉 所有依赖安装成功！")
        print("\n📝 接下来的步骤:")
        print("1. 配置 AWS 凭证 (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
        print("2. 设置 AWS 区域 (AWS_REGION)")
        print("3. 运行测试: python test_streaming_arabic.py")
    else:
        print("⚠️  部分依赖安装失败，请检查错误信息")
        
        # 常见问题解决方案
        print("\n🔧 常见问题解决方案:")
        print("1. PyAudio 安装失败:")
        print("   macOS: brew install portaudio && pip install pyaudio")
        print("   Ubuntu: sudo apt-get install portaudio19-dev && pip install pyaudio")
        print("   Windows: pip install pipwin && pipwin install pyaudio")
        
        print("\n2. 权限问题:")
        print("   使用 --user 标志: pip install --user <package>")
        
        print("\n3. 网络问题:")
        print("   使用国内镜像: pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package>")


if __name__ == "__main__":
    main()
