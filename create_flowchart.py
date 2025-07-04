#!/usr/bin/env python3
"""
创建字幕处理流程图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_subtitle_flowchart():
    """创建字幕处理流程图"""
    
    # 创建图形和轴
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 定义颜色
    colors = {
        'input': '#E3F2FD',      # 浅蓝色 - 输入
        'whisper': '#FFF3E0',    # 浅橙色 - Whisper
        'correction': '#E8F5E8', # 浅绿色 - 纠错
        'translation': '#F3E5F5', # 浅紫色 - 翻译
        'output': '#FFEBEE',     # 浅红色 - 输出
        'border': '#424242'      # 深灰色 - 边框
    }
    
    # 流程步骤定义
    steps = [
        {
            'name': '音频输入\n(Audio Input)',
            'pos': (2, 10.5),
            'size': (1.5, 0.8),
            'color': colors['input'],
            'icon': '🎵'
        },
        {
            'name': 'Whisper语音识别\n(Speech Recognition)',
            'pos': (2, 8.5),
            'size': (2, 0.8),
            'color': colors['whisper'],
            'icon': '🎤',
            'details': '• Amazon SageMaker\n• 实时流式处理\n• 阿拉伯语优化'
        },
        {
            'name': '原始字幕\n(Raw Subtitles)',
            'pos': (2, 7),
            'size': (1.5, 0.6),
            'color': colors['input'],
            'icon': '📝',
            'example': 'اللة يبارك في المباراة'
        },
        {
            'name': 'Correction纠错服务\n(Subtitle Correction)',
            'pos': (2, 5.5),
            'size': (2, 0.8),
            'color': colors['correction'],
            'icon': '✏️',
            'details': '• 拼写纠正\n• 语法纠正\n• 上下文一致性\n• Bedrock LLM'
        },
        {
            'name': '纠错后字幕\n(Corrected Subtitles)',
            'pos': (2, 4),
            'size': (1.5, 0.6),
            'color': colors['correction'],
            'icon': '✅',
            'example': 'الله يبارك في المباراة'
        },
        {
            'name': 'Translation翻译服务\n(Subtitle Translation)',
            'pos': (2, 2.5),
            'size': (2, 0.8),
            'color': colors['translation'],
            'icon': '🌐',
            'details': '• 多语言支持\n• 场景感知翻译\n• Bedrock Claude\n• OpenAI/Google'
        },
        {
            'name': '最终字幕\n(Final Subtitles)',
            'pos': (2, 1),
            'size': (1.5, 0.6),
            'color': colors['output'],
            'icon': '🎯',
            'example': '愿真主保佑这场比赛'
        }
    ]
    
    # 绘制流程步骤
    boxes = []
    for step in steps:
        x, y = step['pos']
        w, h = step['size']
        
        # 创建圆角矩形
        box = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.1",
            facecolor=step['color'],
            edgecolor=colors['border'],
            linewidth=2
        )
        ax.add_patch(box)
        boxes.append(box)
        
        # 添加图标和主标题
        ax.text(x - w/2 + 0.2, y + 0.1, step['icon'], 
                fontsize=20, ha='left', va='center')
        ax.text(x, y + 0.1, step['name'], 
                fontsize=11, ha='center', va='center', weight='bold')
        
        # 添加详细信息
        if 'details' in step:
            ax.text(x, y - 0.25, step['details'], 
                    fontsize=8, ha='center', va='center', 
                    style='italic', color='#666666')
        
        # 添加示例
        if 'example' in step:
            ax.text(x, y - 0.2, f"例: {step['example']}", 
                    fontsize=9, ha='center', va='center', 
                    style='italic', color='#333333',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # 绘制箭头连接
    arrow_props = dict(
        arrowstyle='->', 
        connectionstyle='arc3,rad=0',
        color=colors['border'],
        lw=2
    )
    
    # 连接各个步骤
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)
    ]
    
    for start_idx, end_idx in connections:
        start_step = steps[start_idx]
        end_step = steps[end_idx]
        
        start_pos = (start_step['pos'][0], start_step['pos'][1] - start_step['size'][1]/2)
        end_pos = (end_step['pos'][0], end_step['pos'][1] + end_step['size'][1]/2)
        
        ax.annotate('', xy=end_pos, xytext=start_pos, arrowprops=arrow_props)
    
    # 添加侧边说明
    side_info = [
        {
            'title': '🔧 新增模块',
            'content': '• Correction纠错服务\n• 基于LLM的智能纠错\n• 场景感知处理',
            'pos': (6, 5.5),
            'color': colors['correction']
        },
        {
            'title': '🔄 重构模块', 
            'content': '• Translation翻译服务\n• 统一接口设计\n• 多服务支持',
            'pos': (6, 2.5),
            'color': colors['translation']
        },
        {
            'title': '⚡ 技术特性',
            'content': '• 实时流式处理\n• Amazon Bedrock集成\n• 模块化架构\n• 完整测试覆盖',
            'pos': (6, 8.5),
            'color': colors['whisper']
        }
    ]
    
    for info in side_info:
        x, y = info['pos']
        
        # 创建信息框
        info_box = FancyBboxPatch(
            (x - 1, y - 0.8), 2, 1.6,
            boxstyle="round,pad=0.1",
            facecolor=info['color'],
            edgecolor=colors['border'],
            linewidth=1,
            alpha=0.7
        )
        ax.add_patch(info_box)
        
        # 添加标题和内容
        ax.text(x, y + 0.4, info['title'], 
                fontsize=10, ha='center', va='center', weight='bold')
        ax.text(x, y - 0.2, info['content'], 
                fontsize=8, ha='center', va='center')
    
    # 添加标题
    ax.text(5, 11.5, 'SubtitleGenius 字幕处理流程', 
            fontsize=18, ha='center', va='center', weight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#F5F5F5', alpha=0.8))
    
    # 添加版本信息
    ax.text(8.5, 0.5, 'v1.0.0 - 模块化架构', 
            fontsize=8, ha='center', va='center', 
            style='italic', color='#888888')
    
    # 保存图片
    plt.tight_layout()
    plt.savefig('/Users/yexw/PycharmProjects/SubtitleGenius/subtitle_processing_flowchart.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('/Users/yexw/PycharmProjects/SubtitleGenius/subtitle_processing_flowchart.pdf', 
                bbox_inches='tight', facecolor='white')
    
    print("✅ 流程图已保存:")
    print("   📄 PNG: subtitle_processing_flowchart.png")
    print("   📄 PDF: subtitle_processing_flowchart.pdf")
    
    return fig

if __name__ == "__main__":
    # 安装依赖提示
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("❌ 需要安装matplotlib:")
        print("   uv add matplotlib")
        exit(1)
    
    print("🎨 正在创建字幕处理流程图...")
    fig = create_subtitle_flowchart()
    
    # 显示图片（可选）
    try:
        plt.show()
    except:
        print("💡 图片已保存，无法显示（可能是无GUI环境）")
    
    print("🎯 流程图创建完成!")
