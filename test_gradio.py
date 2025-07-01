#!/usr/bin/env python3
"""
简单的 Gradio 测试脚本
"""

import gradio as gr


def simple_test(text):
    """简单的测试函数"""
    return f"你输入了: {text}"


def create_simple_interface():
    """创建简单的测试界面"""
    
    with gr.Blocks(title="Gradio 测试") as interface:
        gr.Markdown("# 🧪 Gradio 测试界面")
        
        with gr.Row():
            input_text = gr.Textbox(
                label="输入文本",
                placeholder="请输入一些文本..."
            )
            
            output_text = gr.Textbox(
                label="输出结果",
                interactive=False
            )
        
        test_btn = gr.Button("测试", variant="primary")
        
        test_btn.click(
            fn=simple_test,
            inputs=[input_text],
            outputs=[output_text]
        )
    
    return interface


def main():
    """主函数"""
    print("🧪 启动 Gradio 测试界面...")
    
    try:
        interface = create_simple_interface()
        
        print("✅ Gradio 界面创建成功")
        
        # 启动服务
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            debug=False,
            show_error=True,
            quiet=False
        )
        
    except Exception as e:
        print(f"❌ Gradio 启动失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
