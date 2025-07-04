#!/usr/bin/env python3
"""
运行字幕纠错服务测试的便捷脚本
"""

import subprocess
import sys
import os
from pathlib import Path


def run_pytest():
    """运行pytest测试"""
    print("🧪 运行字幕纠错服务测试")
    print("=" * 50)
    
    # 切换到项目目录
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # 运行pytest
    cmd = ["uv", "run", "pytest", "tests/test_correction_service.py", "-v", "--tb=short"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n✅ 所有测试通过!")
        else:
            print(f"\n❌ 测试失败，退出码: {result.returncode}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ 运行测试时出错: {e}")
        return False


def run_coverage():
    """运行测试覆盖率"""
    print("\n📊 运行测试覆盖率分析")
    print("=" * 50)
    
    # 安装coverage
    install_cmd = ["uv", "add", "pytest-cov", "--dev"]
    subprocess.run(install_cmd, capture_output=True)
    
    # 运行覆盖率测试
    cmd = [
        "uv", "run", "pytest", 
        "tests/test_correction_service.py", 
        "--cov=subtitle_genius.correction_service",
        "--cov-report=term-missing",
        "-v"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ 运行覆盖率测试时出错: {e}")


def main():
    """主函数"""
    print("🚀 字幕纠错服务测试套件")
    print("=" * 60)
    
    # 运行基本测试
    success = run_pytest()
    
    if success:
        # 如果基本测试通过，运行覆盖率测试
        run_coverage()
    
    print(f"\n🎯 测试完成!")


if __name__ == "__main__":
    main()
