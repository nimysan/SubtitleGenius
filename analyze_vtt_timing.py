#!/usr/bin/env python3
"""
VTT字幕文件时间和文本分析脚本
分析字幕的时间分布、文本长度等特征
"""

import re
import statistics
from pathlib import Path

def parse_time(time_str):
    """解析时间字符串为秒数"""
    # 格式: 00:00:00.000
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds_parts = parts[2].split('.')
    seconds = int(seconds_parts[0])
    milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
    
    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
    return total_seconds

def analyze_vtt_file(file_path):
    """分析VTT文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 解析字幕条目
    subtitle_pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\n(.+?)(?=\n\n|\n\d{2}:\d{2}:\d{2}\.\d{3}|$)'
    matches = re.findall(subtitle_pattern, content, re.DOTALL)
    
    subtitles = []
    for start_time_str, end_time_str, text in matches:
        start_time = parse_time(start_time_str)
        end_time = parse_time(end_time_str)
        duration = end_time - start_time
        text = text.strip()
        
        if text:  # 忽略空文本
            subtitles.append({
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'text': text,
                'char_count': len(text),
                'word_count': len(text.split()) if text else 0
            })
    
    return subtitles

def analyze_statistics(subtitles):
    """分析统计数据"""
    if not subtitles:
        return {}
    
    durations = [s['duration'] for s in subtitles]
    char_counts = [s['char_count'] for s in subtitles]
    word_counts = [s['word_count'] for s in subtitles]
    
    # 计算字符/秒的比率
    chars_per_second = [s['char_count'] / s['duration'] if s['duration'] > 0 else 0 for s in subtitles]
    
    stats = {
        'total_subtitles': len(subtitles),
        'total_duration': max([s['end_time'] for s in subtitles]) - min([s['start_time'] for s in subtitles]),
        
        # 时长统计
        'duration_avg': statistics.mean(durations),
        'duration_median': statistics.median(durations),
        'duration_min': min(durations),
        'duration_max': max(durations),
        'duration_std': statistics.stdev(durations) if len(durations) > 1 else 0,
        
        # 字符数统计
        'chars_avg': statistics.mean(char_counts),
        'chars_median': statistics.median(char_counts),
        'chars_min': min(char_counts),
        'chars_max': max(char_counts),
        'chars_std': statistics.stdev(char_counts) if len(char_counts) > 1 else 0,
        
        # 词数统计
        'words_avg': statistics.mean(word_counts),
        'words_median': statistics.median(word_counts),
        'words_min': min(word_counts),
        'words_max': max(word_counts),
        
        # 字符/秒统计
        'chars_per_second_avg': statistics.mean(chars_per_second),
        'chars_per_second_median': statistics.median(chars_per_second),
        'chars_per_second_min': min(chars_per_second),
        'chars_per_second_max': max(chars_per_second),
    }
    
    return stats

def print_analysis_report(subtitles, stats):
    """打印分析报告"""
    print("=" * 60)
    print("VTT字幕文件分析报告")
    print("=" * 60)
    
    print(f"\n📊 基本信息:")
    print(f"  总字幕条数: {stats['total_subtitles']}")
    print(f"  总时长: {stats['total_duration']:.1f}秒 ({stats['total_duration']/60:.1f}分钟)")
    
    print(f"\n⏱️ 时长分析:")
    print(f"  平均时长: {stats['duration_avg']:.2f}秒")
    print(f"  中位数时长: {stats['duration_median']:.2f}秒")
    print(f"  最短时长: {stats['duration_min']:.2f}秒")
    print(f"  最长时长: {stats['duration_max']:.2f}秒")
    print(f"  标准差: {stats['duration_std']:.2f}秒")
    
    print(f"\n📝 文本长度分析:")
    print(f"  平均字符数: {stats['chars_avg']:.1f}字")
    print(f"  中位数字符数: {stats['chars_median']:.1f}字")
    print(f"  最少字符数: {stats['chars_min']}字")
    print(f"  最多字符数: {stats['chars_max']}字")
    print(f"  标准差: {stats['chars_std']:.1f}字")
    
    print(f"\n🗣️ 词数分析:")
    print(f"  平均词数: {stats['words_avg']:.1f}词")
    print(f"  中位数词数: {stats['words_median']:.1f}词")
    print(f"  最少词数: {stats['words_min']}词")
    print(f"  最多词数: {stats['words_max']}词")
    
    print(f"\n⚡ 语速分析:")
    print(f"  平均语速: {stats['chars_per_second_avg']:.2f}字/秒")
    print(f"  中位数语速: {stats['chars_per_second_median']:.2f}字/秒")
    print(f"  最慢语速: {stats['chars_per_second_min']:.2f}字/秒")
    print(f"  最快语速: {stats['chars_per_second_max']:.2f}字/秒")
    
    # 时长分布分析
    print(f"\n📈 时长分布:")
    duration_ranges = [
        (0, 5, "0-5秒"),
        (5, 10, "5-10秒"),
        (10, 15, "10-15秒"),
        (15, 20, "15-20秒"),
        (20, float('inf'), "20秒以上")
    ]
    
    for min_dur, max_dur, label in duration_ranges:
        count = len([s for s in subtitles if min_dur <= s['duration'] < max_dur])
        percentage = count / len(subtitles) * 100
        print(f"  {label}: {count}条 ({percentage:.1f}%)")
    
    # 字符数分布分析
    print(f"\n📈 字符数分布:")
    char_ranges = [
        (0, 10, "0-10字"),
        (10, 20, "10-20字"),
        (20, 30, "20-30字"),
        (30, 50, "30-50字"),
        (50, float('inf'), "50字以上")
    ]
    
    for min_chars, max_chars, label in char_ranges:
        count = len([s for s in subtitles if min_chars <= s['char_count'] < max_chars])
        percentage = count / len(subtitles) * 100
        print(f"  {label}: {count}条 ({percentage:.1f}%)")

def recommend_chunk_size(stats):
    """基于分析结果推荐chunk大小"""
    print(f"\n🎯 Chunk大小推荐:")
    
    avg_duration = stats['duration_avg']
    avg_chars = stats['chars_avg']
    chars_per_second = stats['chars_per_second_avg']
    
    print(f"\n基于当前数据分析:")
    print(f"  - 平均每句话时长: {avg_duration:.1f}秒")
    print(f"  - 平均每句话字符数: {avg_chars:.1f}字")
    print(f"  - 平均语速: {chars_per_second:.2f}字/秒")
    
    # 推荐不同的chunk大小
    chunk_recommendations = [
        (3, "3秒chunk"),
        (5, "5秒chunk"),
        (10, "10秒chunk"),
        (15, "15秒chunk")
    ]
    
    print(f"\n📋 不同chunk大小分析:")
    for chunk_size, label in chunk_recommendations:
        sentences_per_chunk = chunk_size / avg_duration
        chars_per_chunk = chunk_size * chars_per_second
        
        print(f"\n  {label}:")
        print(f"    - 预计包含句子数: {sentences_per_chunk:.1f}句")
        print(f"    - 预计字符数: {chars_per_chunk:.0f}字")
        
        if chunk_size < avg_duration:
            print(f"    - ⚠️  可能会截断句子")
        elif chunk_size > avg_duration * 3:
            print(f"    - ⚠️  可能包含过多内容，处理延迟较大")
        else:
            print(f"    - ✅ 大小适中")
    
    # 最终推荐
    print(f"\n🏆 最终推荐:")
    if avg_duration <= 5:
        recommended_size = 5
        reason = "句子较短，5秒chunk可以包含完整句子"
    elif avg_duration <= 8:
        recommended_size = 10
        reason = "句子中等长度，10秒chunk平衡了完整性和延迟"
    else:
        recommended_size = 15
        reason = "句子较长，需要更大的chunk来保证完整性"
    
    print(f"  推荐chunk大小: {recommended_size}秒")
    print(f"  推荐理由: {reason}")
    print(f"  预计效果: 每个chunk包含{recommended_size/avg_duration:.1f}句话，约{recommended_size*chars_per_second:.0f}字")

def main():
    file_path = Path("/Users/yexw/PycharmProjects/SubtitleGenius/subtitles/chinese_football_auto.vtt")
    
    if not file_path.exists():
        print(f"文件不存在: {file_path}")
        return
    
    print(f"分析文件: {file_path}")
    
    # 解析字幕
    subtitles = analyze_vtt_file(file_path)
    
    if not subtitles:
        print("未找到有效的字幕条目")
        return
    
    # 计算统计数据
    stats = analyze_statistics(subtitles)
    
    # 打印分析报告
    print_analysis_report(subtitles, stats)
    
    # 推荐chunk大小
    recommend_chunk_size(stats)
    
    print(f"\n" + "=" * 60)

if __name__ == "__main__":
    main()
