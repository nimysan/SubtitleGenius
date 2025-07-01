#!/usr/bin/env python3
"""
验证 Transcribe 模型修改是否正确
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """测试导入"""
    print("🔍 测试导入...")
    
    try:
        from subtitle_genius.models.transcribe_model import TranscribeModel
        print("✅ TranscribeModel 导入成功")
    except Exception as e:
        print(f"❌ TranscribeModel 导入失败: {e}")
        return False
    
    try:
        from subtitle_genius.stream.processor import StreamProcessor
        print("✅ StreamProcessor 导入成功")
    except Exception as e:
        print(f"❌ StreamProcessor 导入失败: {e}")
        return False
    
    try:
        from subtitle_genius.core.config import config
        print("✅ Config 导入成功")
        print(f"   默认语言: {config.subtitle_language}")
    except Exception as e:
        print(f"❌ Config 导入失败: {e}")
        return False
    
    return True


def test_model_initialization():
    """测试模型初始化"""
    print("\n🔧 测试模型初始化...")
    
    try:
        from subtitle_genius.models.transcribe_model import TranscribeModel
        
        # 测试流式处理模式
        model_streaming = TranscribeModel(use_streaming=True)
        print("✅ 流式处理模式初始化成功")
        print(f"   使用流式处理: {model_streaming.use_streaming}")
        
        # 测试批处理模式
        model_batch = TranscribeModel(use_streaming=False)
        print("✅ 批处理模式初始化成功")
        print(f"   使用流式处理: {model_batch.use_streaming}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return False


def test_language_mapping():
    """测试语言映射"""
    print("\n🌐 测试语言映射...")
    
    try:
        from subtitle_genius.models.transcribe_model import TranscribeModel
        
        model = TranscribeModel()
        
        # 测试 Arabic 语言映射
        test_cases = [
            ("ar", "ar-SA"),
            ("ar-SA", "ar-SA"),
            ("ar-AE", "ar-AE"),
            ("en", "en-US"),
            ("zh", "zh-CN"),
            ("unknown", "ar-SA"),  # 默认应该是 Arabic
        ]
        
        for input_lang, expected in test_cases:
            result = model._convert_language_code(input_lang)
            if result == expected:
                print(f"✅ {input_lang} -> {result}")
            else:
                print(f"❌ {input_lang} -> {result} (期望: {expected})")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 语言映射测试失败: {e}")
        return False


def test_streaming_availability():
    """测试流式处理可用性"""
    print("\n🌊 测试流式处理可用性...")
    
    try:
        from amazon_transcribe.client import TranscribeStreamingClient
        print("✅ amazon-transcribe 包已安装")
        return True
    except ImportError:
        print("⚠️  amazon-transcribe 包未安装")
        print("   安装命令: pip install amazon-transcribe")
        return False


def main():
    """主函数"""
    print("🎬 SubtitleGenius - 验证修改")
    print("=" * 50)
    
    tests = [
        ("导入测试", test_imports),
        ("模型初始化测试", test_model_initialization),
        ("语言映射测试", test_language_mapping),
        ("流式处理可用性测试", test_streaming_availability),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        if test_func():
            passed += 1
            print(f"✅ {test_name} 通过")
        else:
            print(f"❌ {test_name} 失败")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！")
        print("\n📝 接下来可以:")
        print("1. 配置 AWS 凭证")
        print("2. 运行: python example_streaming_arabic.py")
        print("3. 运行: python test_streaming_arabic.py")
    else:
        print("⚠️  部分测试失败，请检查错误信息")


if __name__ == "__main__":
    main()
