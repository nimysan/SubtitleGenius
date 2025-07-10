#!/usr/bin/env python3
"""
VAD灵敏度参数调优测试
测试不同参数组合对语音段分割的影响，并生成可视化结果
"""

import numpy as np
import soundfile as sf
from subtitle_genius.stream.vac_processor import VACProcessor
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import seaborn as sns

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def test_vad_sensitivity():
    """测试不同VAD参数的灵敏度"""
    
    # 加载音频文件
    audio_file = "arabic-long-3min.wav"
    logger.info(f"加载音频文件: {audio_file}")
    
    try:
        audio_data, sample_rate = sf.read(audio_file)
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        logger.info(f"音频时长: {len(audio_data)/sample_rate:.2f}秒")
    except Exception as e:
        logger.error(f"无法加载音频文件: {e}")
        return
    
    # 测试参数组合 - 专注于缩短最长段
    test_configs = [
        {
            "name": "当前最佳 (温和调整)",
            "threshold": 0.25,
            "min_silence_duration_ms": 200,
            "speech_pad_ms": 80,
            "color": "#FF6B6B"
        },
        {
            "name": "进一步优化1",
            "threshold": 0.22,
            "min_silence_duration_ms": 150,
            "speech_pad_ms": 60,
            "color": "#4ECDC4"
        },
        {
            "name": "进一步优化2", 
            "threshold": 0.20,
            "min_silence_duration_ms": 120,
            "speech_pad_ms": 50,
            "color": "#45B7D1"
        },
        {
            "name": "进一步优化3",
            "threshold": 0.18,
            "min_silence_duration_ms": 100,
            "speech_pad_ms": 40,
            "color": "#96CEB4"
        },
        {
            "name": "精细调优1",
            "threshold": 0.23,
            "min_silence_duration_ms": 180,
            "speech_pad_ms": 70,
            "color": "#FFEAA7"
        },
        {
            "name": "精细调优2",
            "threshold": 0.21,
            "min_silence_duration_ms": 160,
            "speech_pad_ms": 60,
            "color": "#DDA0DD"
        },
        {
            "name": "精细调优3",
            "threshold": 0.19,
            "min_silence_duration_ms": 140,
            "speech_pad_ms": 50,
            "color": "#98D8C8"
        }
    ]
    
    results = []
    
    for config in test_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"测试配置: {config['name']}")
        logger.info(f"参数: threshold={config['threshold']}, "
                   f"min_silence={config['min_silence_duration_ms']}ms, "
                   f"speech_pad={config['speech_pad_ms']}ms")
        logger.info(f"{'='*60}")
        
        try:
            # 创建VAC处理器
            processor = VACProcessor(
                threshold=config['threshold'],
                min_silence_duration_ms=config['min_silence_duration_ms'],
                speech_pad_ms=config['speech_pad_ms'],
                sample_rate=sample_rate,
                processing_chunk_size=512,
                no_audio_input_threshold=5.0
            )
            
            # 处理音频
            segments = processor.process_streaming_audio(
                audio_stream=iter([audio_data]),
                end_stream_flag=True,
                return_segments=True
            )
            
            # 分析结果
            total_speech_duration = sum(seg['duration'] for seg in segments)
            avg_segment_duration = total_speech_duration / len(segments) if segments else 0
            max_segment_duration = max(seg['duration'] for seg in segments) if segments else 0
            min_segment_duration = min(seg['duration'] for seg in segments) if segments else 0
            
            # 计算段时长分布
            long_segments = [seg for seg in segments if seg['duration'] > 30]
            medium_segments = [seg for seg in segments if 10 <= seg['duration'] <= 30]
            short_segments = [seg for seg in segments if seg['duration'] < 10]
            
            result = {
                'config': config,
                'segment_count': len(segments),
                'total_speech_duration': total_speech_duration,
                'avg_segment_duration': avg_segment_duration,
                'max_segment_duration': max_segment_duration,
                'min_segment_duration': min_segment_duration,
                'long_segments_count': len(long_segments),
                'medium_segments_count': len(medium_segments),
                'short_segments_count': len(short_segments),
                'segments': segments
            }
            results.append(result)
            
            # 打印结果
            logger.info(f"✅ 检测到 {len(segments)} 个语音段")
            logger.info(f"   总语音时长: {total_speech_duration:.1f}秒")
            logger.info(f"   平均段时长: {avg_segment_duration:.1f}秒")
            logger.info(f"   最长段时长: {max_segment_duration:.1f}秒")
            logger.info(f"   最短段时长: {min_segment_duration:.1f}秒")
            logger.info(f"   段时长分布: >30s({len(long_segments)}个), 10-30s({len(medium_segments)}个), <10s({len(short_segments)}个)")
                
        except Exception as e:
            logger.error(f"❌ 测试失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # 生成可视化结果
    create_visualization(results, len(audio_data)/sample_rate)
    
    # 对比分析
    logger.info(f"\n{'='*90}")
    logger.info("📊 参数调优对比分析 - 专注于缩短最长段")
    logger.info(f"{'='*90}")
    
    print(f"{'配置':<18} {'段数':<6} {'总时长':<8} {'平均':<8} {'最长':<8} {'最短':<8} {'>30s':<6} {'10-30s':<8} {'<10s':<6}")
    print("-" * 90)
    
    for result in results:
        config_name = result['config']['name']
        print(f"{config_name:<18} "
              f"{result['segment_count']:<6} "
              f"{result['total_speech_duration']:<8.1f} "
              f"{result['avg_segment_duration']:<8.1f} "
              f"{result['max_segment_duration']:<8.1f} "
              f"{result['min_segment_duration']:<8.1f} "
              f"{result['long_segments_count']:<6} "
              f"{result['medium_segments_count']:<8} "
              f"{result['short_segments_count']:<6}")
    
    # 推荐最佳配置
    logger.info(f"\n💡 推荐配置分析 (目标: 最长段<30s, 平均段<15s):")
    
    # 找到符合目标的配置
    good_configs = [r for r in results if r['max_segment_duration'] < 30 and r['avg_segment_duration'] < 15]
    
    if good_configs:
        logger.info(f"   🎯 符合目标的配置:")
        for config in good_configs:
            logger.info(f"     ✅ {config['config']['name']}: 最长{config['max_segment_duration']:.1f}s, 平均{config['avg_segment_duration']:.1f}s")
        
        # 选择最佳配置（最长段最短的）
        best_config = min(good_configs, key=lambda x: x['max_segment_duration'])
        logger.info(f"\n   🏆 推荐最佳配置: {best_config['config']['name']}")
        logger.info(f"     参数: threshold={best_config['config']['threshold']}, "
                   f"min_silence={best_config['config']['min_silence_duration_ms']}ms, "
                   f"speech_pad={best_config['config']['speech_pad_ms']}ms")
        logger.info(f"     效果: {best_config['segment_count']}个段, 最长{best_config['max_segment_duration']:.1f}s, 平均{best_config['avg_segment_duration']:.1f}s")
    else:
        logger.warning("   ⚠️  没有配置完全符合目标，显示最接近的配置:")
        
        # 找到最长段最短的配置
        best_max = min(results, key=lambda x: x['max_segment_duration'])
        logger.info(f"     🥇 最短最长段: {best_max['config']['name']} (最长段: {best_max['max_segment_duration']:.1f}秒)")
        
        # 找到平均段最短的配置
        best_avg = min(results, key=lambda x: x['avg_segment_duration'])
        logger.info(f"     🥈 最短平均段: {best_avg['config']['name']} (平均: {best_avg['avg_segment_duration']:.1f}秒)")

def create_visualization(results, audio_duration):
    """Create visualization charts"""
    
    # Create main figure
    fig = plt.figure(figsize=(20, 24))
    
    # Set overall title
    fig.suptitle('VAD Parameter Tuning Test Results Visualization Analysis', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # 1. Key metrics comparison bar chart
    ax1 = plt.subplot(4, 2, 1)
    config_names = [r['config']['name'] for r in results]
    max_durations = [r['max_segment_duration'] for r in results]
    avg_durations = [r['avg_segment_duration'] for r in results]
    colors = [r['config']['color'] for r in results]
    
    x = np.arange(len(config_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, max_durations, width, label='Max Segment Duration', color=colors, alpha=0.8)
    bars2 = ax1.bar(x + width/2, avg_durations, width, label='Avg Segment Duration', color=colors, alpha=0.5)
    
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Duration (seconds)')
    ax1.set_title('Key Metrics Comparison: Max vs Avg Segment Duration')
    ax1.set_xticks(x)
    ax1.set_xticklabels(config_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add target lines
    ax1.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='Target Max (30s)')
    ax1.axhline(y=15, color='orange', linestyle='--', alpha=0.7, label='Target Avg (15s)')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}s', ha='center', va='bottom', fontsize=8)
    
    # 2. Segment count comparison
    ax2 = plt.subplot(4, 2, 2)
    segment_counts = [r['segment_count'] for r in results]
    bars = ax2.bar(config_names, segment_counts, color=colors, alpha=0.8)
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Number of Segments')
    ax2.set_title('Segment Count Comparison')
    ax2.set_xticklabels(config_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    # 3. 段时长分布堆叠柱状图
    ax3 = plt.subplot(4, 2, 3)
    long_counts = [r['long_segments_count'] for r in results]
    medium_counts = [r['medium_segments_count'] for r in results]
    short_counts = [r['short_segments_count'] for r in results]
    
    ax3.bar(config_names, long_counts, label='>30s (Long)', color='#FF6B6B', alpha=0.8)
    ax3.bar(config_names, medium_counts, bottom=long_counts, label='10-30s (Medium)', color='#4ECDC4', alpha=0.8)
    ax3.bar(config_names, short_counts, bottom=[l+m for l,m in zip(long_counts, medium_counts)], 
            label='<10s (Short)', color='#45B7D1', alpha=0.8)
    
    ax3.set_xlabel('Configuration')
    ax3.set_ylabel('Number of Segments')
    ax3.set_title('Segment Duration Distribution')
    ax3.set_xticklabels(config_names, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 参数散点图
    ax4 = plt.subplot(4, 2, 4)
    thresholds = [r['config']['threshold'] for r in results]
    min_silences = [r['config']['min_silence_duration_ms'] for r in results]
    
    scatter = ax4.scatter(thresholds, min_silences, c=max_durations, s=200, 
                         cmap='RdYlGn_r', alpha=0.8, edgecolors='black')
    ax4.set_xlabel('Threshold')
    ax4.set_ylabel('Min Silence Duration (ms)')
    ax4.set_title('Parameter Space Distribution (Color = Max Segment Duration)')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Max Segment Duration (s)')
    
    # 添加配置名称标签
    for i, txt in enumerate([r['config']['name'].split()[0] for r in results]):
        ax4.annotate(txt, (thresholds[i], min_silences[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 5. 语音段时间轴可视化 (选择前3个配置)
    ax5 = plt.subplot(4, 1, 3)
    
    top_configs = sorted(results, key=lambda x: x['max_segment_duration'])[:3]
    
    y_pos = 0
    y_labels = []
    
    for i, result in enumerate(top_configs):
        segments = result['segments']
        config_name = result['config']['name']
        color = result['config']['color']
        
        # 绘制语音段
        for seg in segments:
            rect = patches.Rectangle((seg['start'], y_pos), seg['duration'], 0.8, 
                                   linewidth=1, edgecolor='black', facecolor=color, alpha=0.7)
            ax5.add_patch(rect)
            
            # 添加时长标签（只对长段）
            if seg['duration'] > 10:
                ax5.text(seg['start'] + seg['duration']/2, y_pos + 0.4, 
                        f"{seg['duration']:.1f}s", ha='center', va='center', 
                        fontsize=8, fontweight='bold')
        
        y_labels.append(f"{config_name}\n(Max:{result['max_segment_duration']:.1f}s)")
        y_pos += 1
    
    ax5.set_xlim(0, audio_duration)
    ax5.set_ylim(-0.5, len(top_configs) - 0.5)
    ax5.set_xlabel('Time (seconds)')
    ax5.set_ylabel('Configuration')
    ax5.set_title('Speech Segment Timeline Visualization (Top 3 Configurations)')
    ax5.set_yticks(range(len(top_configs)))
    ax5.set_yticklabels(y_labels)
    ax5.grid(True, alpha=0.3)
    
    # 6. 效果评分雷达图
    ax6 = plt.subplot(4, 2, 7, projection='polar')
    
    # 选择前5个配置进行雷达图对比
    top_5_configs = sorted(results, key=lambda x: x['max_segment_duration'])[:5]
    
    # 评分标准 (分数越高越好)
    categories = ['Segment Count', 'Short Max', 'Short Avg', 'Good Distribution', 'Overall']
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    for result in top_5_configs:
        # 计算各项评分 (0-10分)
        segment_score = min(10, result['segment_count'] * 1.2)  # 段数适中
        max_duration_score = max(0, 10 - result['max_segment_duration'] / 6)  # 最长段短
        avg_duration_score = max(0, 10 - result['avg_segment_duration'] / 3)  # 平均段短
        distribution_score = 10 - result['long_segments_count'] * 2  # 分布均匀
        overall_score = (segment_score + max_duration_score + avg_duration_score + distribution_score) / 4
        
        scores = [segment_score, max_duration_score, avg_duration_score, distribution_score, overall_score]
        scores += scores[:1]  # 闭合图形
        
        ax6.plot(angles, scores, 'o-', linewidth=2, label=result['config']['name'], 
                color=result['config']['color'])
        ax6.fill(angles, scores, alpha=0.25, color=result['config']['color'])
    
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(categories)
    ax6.set_ylim(0, 10)
    ax6.set_title('Configuration Performance Comparison', pad=20)
    ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax6.grid(True)
    
    # 7. 参数建议表格
    ax7 = plt.subplot(4, 2, 8)
    ax7.axis('off')
    
    # 找到最佳配置
    best_config = min(results, key=lambda x: x['max_segment_duration'])
    
    # 创建建议表格
    table_data = [
        ['Metric', 'Current Best', 'Target', 'Status'],
        ['Max Segment', f"{best_config['max_segment_duration']:.1f}s", '<30s', 
         '✅' if best_config['max_segment_duration'] < 30 else '❌'],
        ['Avg Segment', f"{best_config['avg_segment_duration']:.1f}s", '<15s',
         '✅' if best_config['avg_segment_duration'] < 15 else '❌'],
        ['Segment Count', f"{best_config['segment_count']}", '8-12', 
         '✅' if 8 <= best_config['segment_count'] <= 12 else '⚠️'],
        ['Long Segments', f"{best_config['long_segments_count']}", '0', 
         '✅' if best_config['long_segments_count'] == 0 else '❌']
    ]
    
    table = ax7.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表格样式
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i == 0:  # 标题行
                cell.set_facecolor('#4ECDC4')
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor('#F8F9FA')
    
    ax7.set_title('Best Configuration Performance Assessment', fontsize=14, fontweight='bold', pad=20)
    
    # 添加推荐配置信息
    recommendation_text = f"""
Recommended Best Configuration: {best_config['config']['name']}

Parameter Settings:
• threshold: {best_config['config']['threshold']}
• min_silence_duration_ms: {best_config['config']['min_silence_duration_ms']}
• speech_pad_ms: {best_config['config']['speech_pad_ms']}

Expected Results:
• Segment Count: {best_config['segment_count']}
• Max Segment: {best_config['max_segment_duration']:.1f}s
• Avg Segment: {best_config['avg_segment_duration']:.1f}s
"""
    
    ax7.text(0.02, 0.3, recommendation_text, transform=ax7.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"vad_parameter_tuning_analysis_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"📊 可视化结果已保存到: {filename}")
    
    # 显示图表
    # plt.show()

if __name__ == "__main__":
    test_vad_sensitivity()
