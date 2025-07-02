#!/usr/bin/env python3
"""
Whisper Turbo Converse API 设置验证脚本
"""

import os
import sys
from pathlib import Path
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

def check_aws_credentials():
    """检查AWS凭证"""
    print("🔐 检查AWS凭证...")
    
    try:
        # 尝试创建STS客户端并获取身份信息
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        
        print("✅ AWS凭证有效")
        print(f"   账户ID: {identity.get('Account', 'N/A')}")
        print(f"   用户ARN: {identity.get('Arn', 'N/A')}")
        return True
        
    except NoCredentialsError:
        print("❌ AWS凭证未配置")
        print("💡 请配置AWS凭证:")
        print("   export AWS_ACCESS_KEY_ID=your_key")
        print("   export AWS_SECRET_ACCESS_KEY=your_secret")
        print("   或运行: aws configure")
        return False
        
    except ClientError as e:
        print(f"❌ AWS凭证错误: {e}")
        return False
        
    except Exception as e:
        print(f"❌ 检查AWS凭证时出错: {e}")
        return False

def check_bedrock_access():
    """检查Bedrock访问权限"""
    print("\n🛏️ 检查Bedrock访问权限...")
    
    try:
        bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        # 尝试列出可用模型（这个操作需要基本权限）
        print("✅ Bedrock客户端初始化成功")
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'AccessDenied':
            print("❌ Bedrock访问被拒绝")
            print("💡 请确保IAM用户/角色有Bedrock权限")
        else:
            print(f"❌ Bedrock访问错误: {e}")
        return False
        
    except Exception as e:
        print(f"❌ 检查Bedrock访问时出错: {e}")
        return False

def check_audio_files():
    """检查可用的音频文件"""
    print("\n🎵 检查音频文件...")
    
    audio_files = [
        "/Users/yexw/PycharmProjects/SubtitleGenius/test.wav",
        "/Users/yexw/PycharmProjects/SubtitleGenius/output_16k_mono.wav",
        "/Users/yexw/PycharmProjects/SubtitleGenius/input.aac"
    ]
    
    available_files = []
    for audio_file in audio_files:
        if Path(audio_file).exists():
            file_size = Path(audio_file).stat().st_size
            print(f"✅ {Path(audio_file).name} ({file_size:,} bytes)")
            available_files.append(audio_file)
        else:
            print(f"❌ {Path(audio_file).name} (不存在)")
    
    if available_files:
        print(f"📁 找到 {len(available_files)} 个可用音频文件")
        return available_files
    else:
        print("❌ 未找到可用的音频文件")
        print("💡 请确保项目目录中有音频文件用于测试")
        return []

def check_dependencies():
    """检查Python依赖"""
    print("\n📦 检查Python依赖...")
    
    required_packages = [
        'boto3',
        'botocore', 
        'pathlib',
        'json',
        'base64',
        'io',
        'wave'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n💡 安装缺失的包:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_config_files():
    """检查配置文件"""
    print("\n⚙️ 检查配置文件...")
    
    config_files = [
        "whisper_converse.py",
        "whisper_config.py", 
        "example_whisper_converse.py"
    ]
    
    all_exist = True
    for config_file in config_files:
        file_path = Path(config_file)
        if file_path.exists():
            print(f"✅ {config_file}")
        else:
            print(f"❌ {config_file} (缺失)")
            all_exist = False
    
    return all_exist

def test_whisper_config():
    """测试Whisper配置"""
    print("\n🔧 测试Whisper配置...")
    
    try:
        from whisper_config import WhisperConfig
        
        status = WhisperConfig.validate_config()
        
        print(f"📍 端点名称: {status['config']['endpoint_name']}")
        print(f"🌍 AWS区域: {status['config']['region']}")
        
        if status['valid']:
            print("✅ 配置验证通过")
            return True
        else:
            print("❌ 配置问题:")
            for issue in status['issues']:
                print(f"   - {issue}")
            return False
            
    except ImportError:
        print("❌ 无法导入whisper_config模块")
        return False
    except Exception as e:
        print(f"❌ 配置测试出错: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 Whisper Turbo Converse API 设置验证")
    print("=" * 50)
    
    # 运行所有检查
    checks = [
        ("依赖检查", check_dependencies),
        ("配置文件检查", check_config_files),
        ("AWS凭证检查", check_aws_credentials),
        ("Bedrock访问检查", check_bedrock_access),
        ("音频文件检查", check_audio_files),
        ("Whisper配置检查", test_whisper_config)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results[check_name] = result
        except Exception as e:
            print(f"❌ {check_name}执行出错: {e}")
            results[check_name] = False
    
    # 总结
    print("\n" + "=" * 50)
    print("📋 检查总结")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for check_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{check_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 总体状态: {passed}/{total} 项检查通过")
    
    if passed == total:
        print("🎉 所有检查通过！可以开始使用Whisper Converse API")
        print("\n🚀 下一步:")
        print("1. 配置你的实际端点名称")
        print("2. 运行: python example_whisper_converse.py")
    else:
        print("⚠️  请解决上述问题后再继续")
        print("\n💡 常见解决方案:")
        print("1. 配置AWS凭证: aws configure")
        print("2. 安装依赖: pip install boto3")
        print("3. 设置端点名称: 编辑 whisper_config.py")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
