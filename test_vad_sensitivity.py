#!/usr/bin/env python3
"""
VAD Sensitivity Parameter Tuning Test
Test different parameter combinations for speech segment splitting and generate visualization results
"""

import numpy as np
import soundfile as sf
from subtitle_genius.stream.vac_processor import VACProcessor
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set font and style
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def test_vad_sensitivity():
    """Test VAD parameter sensitivity"""
    
    # Load audio file
    audio_file = "arabic_news_3min.wav"
    logger.info(f"Loading audio file: {audio_file}")
    
    try:
        audio_data, sample_rate = sf.read(audio_file)
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        logger.info(f"Audio duration: {len(audio_data)/sample_rate:.2f} seconds")
    except Exception as e:
        logger.error(f"Cannot load audio file: {e}")
        return
    
    # Test parameter combinations - Focus on shortening longest segments
    test_configs = [
        {
            "name": "Current Best",
            "threshold": 0.25,
            "min_silence_duration_ms": 200,
            "speech_pad_ms": 80,
            "color": "#FF6B6B"
        },
        {
            "name": "Optimized 1",
            "threshold": 0.22,
            "min_silence_duration_ms": 150,
            "speech_pad_ms": 60,
            "color": "#4ECDC4"
        },
        {
            "name": "Optimized 2", 
            "threshold": 0.20,
            "min_silence_duration_ms": 120,
            "speech_pad_ms": 50,
            "color": "#45B7D1"
        },
        {
            "name": "Optimized 3",
            "threshold": 0.18,
            "min_silence_duration_ms": 100,
            "speech_pad_ms": 40,
            "color": "#96CEB4"
        },
        {
            "name": "Fine-tuned 1",
            "threshold": 0.23,
            "min_silence_duration_ms": 180,
            "speech_pad_ms": 70,
            "color": "#FFEAA7"
        }
    ]
    
    results = []
    
    for config in test_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing configuration: {config['name']}")
        logger.info(f"Parameters: threshold={config['threshold']}, "
                   f"min_silence={config['min_silence_duration_ms']}ms, "
                   f"speech_pad={config['speech_pad_ms']}ms")
        logger.info(f"{'='*60}")
        
        try:
            # Create VAC processor
            processor = VACProcessor(
                threshold=config['threshold'],
                min_silence_duration_ms=config['min_silence_duration_ms'],
                speech_pad_ms=config['speech_pad_ms'],
                sample_rate=sample_rate,
                processing_chunk_size=512,
                no_audio_input_threshold=5.0
            )
            
            # Process audio
            segments = processor.process_streaming_audio(
                audio_stream=iter([audio_data]),
                end_stream_flag=True,
                return_segments=True
            )
            
            # Analyze results
            total_speech_duration = sum(seg['duration'] for seg in segments)
            avg_segment_duration = total_speech_duration / len(segments) if segments else 0
            max_segment_duration = max(seg['duration'] for seg in segments) if segments else 0
            min_segment_duration = min(seg['duration'] for seg in segments) if segments else 0
            
            # Calculate segment duration distribution
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
            
            # Print results
            logger.info(f"✅ Detected {len(segments)} speech segments")
            logger.info(f"   Total speech duration: {total_speech_duration:.1f}s")
            logger.info(f"   Average segment duration: {avg_segment_duration:.1f}s")
            logger.info(f"   Max segment duration: {max_segment_duration:.1f}s")
            logger.info(f"   Min segment duration: {min_segment_duration:.1f}s")
            logger.info(f"   Duration distribution: >30s({len(long_segments)}), 10-30s({len(medium_segments)}), <10s({len(short_segments)})")
                
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Generate visualization results
    create_visualization(results, len(audio_data)/sample_rate)
    
    # Comparative analysis
    logger.info(f"\n{'='*90}")
    logger.info("📊 Parameter Tuning Comparison Analysis - Focus on Shortening Max Segments")
    logger.info(f"{'='*90}")
    
    print(f"{'Configuration':<18} {'Segments':<8} {'Total':<8} {'Avg':<8} {'Max':<8} {'Min':<8} {'>30s':<6} {'10-30s':<8} {'<10s':<6}")
    print("-" * 90)
    
    for result in results:
        config_name = result['config']['name']
        print(f"{config_name:<18} "
              f"{result['segment_count']:<8} "
              f"{result['total_speech_duration']:<8.1f} "
              f"{result['avg_segment_duration']:<8.1f} "
              f"{result['max_segment_duration']:<8.1f} "
              f"{result['min_segment_duration']:<8.1f} "
              f"{result['long_segments_count']:<6} "
              f"{result['medium_segments_count']:<8} "
              f"{result['short_segments_count']:<6}")
    
    # Recommend best configuration
    logger.info(f"\n💡 Recommended Configuration Analysis (Target: Max<30s, Avg<15s):")
    
    # Find configurations that meet targets
    good_configs = [r for r in results if r['max_segment_duration'] < 30 and r['avg_segment_duration'] < 15]
    
    if good_configs:
        logger.info(f"   🎯 Configurations meeting targets:")
        for config in good_configs:
            logger.info(f"     ✅ {config['config']['name']}: Max{config['max_segment_duration']:.1f}s, Avg{config['avg_segment_duration']:.1f}s")
        
        # Select best configuration (shortest max segment)
        best_config = min(good_configs, key=lambda x: x['max_segment_duration'])
        logger.info(f"\n   🏆 Recommended Best Configuration: {best_config['config']['name']}")
        logger.info(f"     Parameters: threshold={best_config['config']['threshold']}, "
                   f"min_silence={best_config['config']['min_silence_duration_ms']}ms, "
                   f"speech_pad={best_config['config']['speech_pad_ms']}ms")
        logger.info(f"     Results: {best_config['segment_count']} segments, Max{best_config['max_segment_duration']:.1f}s, Avg{best_config['avg_segment_duration']:.1f}s")
    else:
        logger.warning("   ⚠️  No configuration fully meets targets, showing closest configurations:")
        
        # Find configuration with shortest max segment
        best_max = min(results, key=lambda x: x['max_segment_duration'])
        logger.info(f"     🥇 Shortest Max Segment: {best_max['config']['name']} (Max: {best_max['max_segment_duration']:.1f}s)")
        
        # Find configuration with shortest average segment
        best_avg = min(results, key=lambda x: x['avg_segment_duration'])
        logger.info(f"     🥈 Shortest Avg Segment: {best_avg['config']['name']} (Avg: {best_avg['avg_segment_duration']:.1f}s)")

def create_visualization(results, audio_duration):
    """Create simplified visualization charts"""
    
    # Create main figure with fewer subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Set overall title
    fig.suptitle('VAD Parameter Tuning Analysis Results', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # 1. Key metrics comparison - Max vs Avg duration
    ax1 = plt.subplot(2, 2, 1)
    config_names = [r['config']['name'] for r in results]
    max_durations = [r['max_segment_duration'] for r in results]
    avg_durations = [r['avg_segment_duration'] for r in results]
    colors = [r['config']['color'] for r in results]
    
    x = np.arange(len(config_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, max_durations, width, label='Max Duration', color=colors, alpha=0.8)
    bars2 = ax1.bar(x + width/2, avg_durations, width, label='Avg Duration', color=colors, alpha=0.5)
    
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Duration (seconds)')
    ax1.set_title('Segment Duration Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(config_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add target lines
    ax1.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='Target Max (30s)')
    ax1.axhline(y=15, color='orange', linestyle='--', alpha=0.7, label='Target Avg (15s)')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Segment count and distribution
    ax2 = plt.subplot(2, 2, 2)
    long_counts = [r['long_segments_count'] for r in results]
    medium_counts = [r['medium_segments_count'] for r in results]
    short_counts = [r['short_segments_count'] for r in results]
    
    ax2.bar(config_names, long_counts, label='>30s (Long)', color='#FF6B6B', alpha=0.8)
    ax2.bar(config_names, medium_counts, bottom=long_counts, label='10-30s (Medium)', color='#4ECDC4', alpha=0.8)
    ax2.bar(config_names, short_counts, bottom=[l+m for l,m in zip(long_counts, medium_counts)], 
            label='<10s (Short)', color='#45B7D1', alpha=0.8)
    
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Number of Segments')
    ax2.set_title('Segment Duration Distribution')
    ax2.set_xticklabels(config_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Speech segment timeline visualization (top 3 configurations)
    ax3 = plt.subplot(2, 1, 2)
    
    # Select top 3 configurations with shortest max duration
    top_configs = sorted(results, key=lambda x: x['max_segment_duration'])[:3]
    
    y_pos = 0
    y_labels = []
    
    for i, result in enumerate(top_configs):
        segments = result['segments']
        config_name = result['config']['name']
        color = result['config']['color']
        
        # Draw speech segments
        for seg in segments:
            rect = patches.Rectangle((seg['start'], y_pos), seg['duration'], 0.8, 
                                   linewidth=1, edgecolor='black', facecolor=color, alpha=0.7)
            ax3.add_patch(rect)
            
            # Add duration labels for long segments only
            if seg['duration'] > 15:
                ax3.text(seg['start'] + seg['duration']/2, y_pos + 0.4, 
                        f"{seg['duration']:.1f}s", ha='center', va='center', 
                        fontsize=8, fontweight='bold', color='white')
        
        y_labels.append(f"{config_name}\n(Max: {result['max_segment_duration']:.1f}s)")
        y_pos += 1
    
    ax3.set_xlim(0, audio_duration)
    ax3.set_ylim(-0.5, len(top_configs) - 0.5)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Configuration')
    ax3.set_title('Speech Segment Timeline - Top 3 Configurations')
    ax3.set_yticks(range(len(top_configs)))
    ax3.set_yticklabels(y_labels)
    ax3.grid(True, alpha=0.3)
    
    # Add time markers every 30 seconds
    for t in range(0, int(audio_duration), 30):
        ax3.axvline(x=t, color='gray', linestyle=':', alpha=0.5)
        ax3.text(t, len(top_configs) - 0.3, f'{t}s', ha='center', va='bottom', fontsize=8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"vad_tuning_simplified_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"📊 Visualization saved to: {filename}")
    
    # Show chart
    # plt.show()
    
    # Print summary table
    print(f"\n{'='*80}")
    print("📋 CONFIGURATION SUMMARY")
    print(f"{'='*80}")
    
    # Find best configuration
    best_config = min(results, key=lambda x: x['max_segment_duration'])
    
    print(f"\n🏆 RECOMMENDED CONFIGURATION: {best_config['config']['name']}")
    print(f"   Parameters:")
    print(f"   • Threshold: {best_config['config']['threshold']}")
    print(f"   • Min Silence: {best_config['config']['min_silence_duration_ms']}ms")
    print(f"   • Speech Pad: {best_config['config']['speech_pad_ms']}ms")
    print(f"   Results:")
    print(f"   • Total Segments: {best_config['segment_count']}")
    print(f"   • Max Duration: {best_config['max_segment_duration']:.1f}s")
    print(f"   • Avg Duration: {best_config['avg_segment_duration']:.1f}s")
    print(f"   • Long Segments (>30s): {best_config['long_segments_count']}")
    
    # Performance assessment
    meets_max_target = best_config['max_segment_duration'] < 30
    meets_avg_target = best_config['avg_segment_duration'] < 15
    no_long_segments = best_config['long_segments_count'] == 0
    
    print(f"\n📊 TARGET ASSESSMENT:")
    print(f"   • Max Duration < 30s: {'✅ PASS' if meets_max_target else '❌ FAIL'}")
    print(f"   • Avg Duration < 15s: {'✅ PASS' if meets_avg_target else '❌ FAIL'}")
    print(f"   • No Long Segments: {'✅ PASS' if no_long_segments else '❌ FAIL'}")
    
    overall_score = sum([meets_max_target, meets_avg_target, no_long_segments])
    print(f"   • Overall Score: {overall_score}/3 {'🎯 EXCELLENT' if overall_score == 3 else '⚠️ NEEDS IMPROVEMENT' if overall_score >= 2 else '❌ POOR'}")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    test_vad_sensitivity()
