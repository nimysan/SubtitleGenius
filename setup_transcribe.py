#!/usr/bin/env python3
"""
Amazon Transcribe 快速设置脚本
"""

import os
import sys
import subprocess
from pathlib import Path


def check_dependencies():
    """检查依赖"""
    print("🔍 检查依赖...")
    
    try:
        import boto3
        print("✅ boto3 已安装")
    except ImportError:
        print("❌ boto3 未安装，正在安装...")
        subprocess.run([sys.executable, "-m", "pip", "install", "boto3", "botocore"], check=True)
        print("✅ boto3 安装完成")


def check_aws_config():
    """检查 AWS 配置"""
    print("\n🔧 检查 AWS 配置...")
    
    env_file = Path(".env")
    
    # 检查环境变量
    aws_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_REGION', 'us-east-1')
    
    if aws_key and aws_secret:
        print("✅ AWS 凭证已配置")
        print(f"   区域: {aws_region}")
        return True
    
    print("❌ AWS 凭证未配置")
    
    # 检查 .env 文件
    if not env_file.exists():
        print("📝 创建 .env 文件...")
        with open(env_file, 'w') as f:
            f.write("""# AWS 配置 (Amazon Transcribe)
AWS_ACCESS_KEY_ID=your_access_key_id_here
AWS_SECRET_ACCESS_KEY=your_secret_access_key_here
AWS_REGION=us-east-1
AWS_S3_BUCKET=subtitle-genius-temp

# 其他配置
SUBTITLE_LANGUAGE=zh-CN
AUDIO_SAMPLE_RATE=16000
""")
        print("✅ .env 文件已创建")
    
    print("\n请编辑 .env 文件，填入你的 AWS 凭证:")
    print("1. AWS_ACCESS_KEY_ID")
    print("2. AWS_SECRET_ACCESS_KEY")
    print("3. AWS_REGION (可选，默认 us-east-1)")
    print("4. AWS_S3_BUCKET (可选，默认 subtitle-genius-temp)")
    
    return False


def test_connection():
    """测试 AWS 连接"""
    print("\n🧪 测试 AWS 连接...")
    
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError, ClientError
        
        # 测试 Transcribe 连接
        client = boto3.client('transcribe')
        client.list_transcription_jobs(MaxResults=1)
        
        print("✅ Amazon Transcribe 连接成功")
        return True
        
    except NoCredentialsError:
        print("❌ AWS 凭证未配置或无效")
        return False
    except ClientError as e:
        if e.response['Error']['Code'] == 'UnauthorizedOperation':
            print("❌ AWS 权限不足，请确保有 Transcribe 权限")
        else:
            print(f"❌ AWS 连接失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 连接测试失败: {e}")
        return False


def create_iam_policy():
    """显示 IAM 策略示例"""
    print("\n📋 所需的 IAM 策略:")
    
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "transcribe:StartTranscriptionJob",
                    "transcribe:GetTranscriptionJob",
                    "transcribe:ListTranscriptionJobs"
                ],
                "Resource": "*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "s3:PutObject",
                    "s3:GetObject",
                    "s3:DeleteObject",
                    "s3:CreateBucket",
                    "s3:ListBucket"
                ],
                "Resource": [
                    "arn:aws:s3:::subtitle-genius-temp",
                    "arn:aws:s3:::subtitle-genius-temp/*"
                ]
            }
        ]
    }
    
    import json
    print(json.dumps(policy, indent=2))


def main():
    """主函数"""
    print("🎬 SubtitleGenius - Amazon Transcribe 设置")
    print("=" * 50)
    
    # 检查依赖
    check_dependencies()
    
    # 检查配置
    config_ok = check_aws_config()
    
    if config_ok:
        # 测试连接
        if test_connection():
            print("\n🎉 Amazon Transcribe 设置完成！")
            print("\n下一步:")
            print("1. 运行 Gradio 应用: uv run python gradio_app.py")
            print("2. 在 AI 模型中选择 'Amazon Transcribe'")
            print("3. 开始生成字幕！")
        else:
            print("\n❌ 连接测试失败")
            create_iam_policy()
    else:
        print("\n⚠️  请先配置 AWS 凭证，然后重新运行此脚本")
        create_iam_policy()
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
