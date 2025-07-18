# SubtitleGenius 指标模块实现

本文档描述了使用 prometheus-client 实现的指标收集模块，用于监控 VAC 处理器的分段处理指标。

## 实现概述

指标模块由以下几个部分组成：

1. **MetricsManager** - 核心指标管理器，负责注册、收集和推送指标
2. **VACMetrics** - VAC 处理器的专用指标收集类
3. **VAC 处理器集成** - 在 VAC 处理器中添加的指标收集代码

## 文件结构

```
subtitle_genius/
├── metrics/
│   ├── __init__.py           # 模块初始化文件
│   ├── metrics_manager.py    # 核心指标管理器
│   ├── vac_metrics.py        # VAC 处理器指标定义和收集
│   └── README.md             # 使用说明
├── stream/
│   └── vac_processor.py      # 已修改，添加指标收集
├── examples/
│   └── metrics_example.py    # 示例脚本
└── test_metrics.py           # 测试脚本
```

## 收集的指标

VAC 处理器的指标包括：

1. **计数器类指标**
   - `subtitle_genius_vac_speech_segments_total` - 检测到的语音段总数
   - `subtitle_genius_vac_vad_events_total` - VAD 事件总数

2. **直方图类指标**
   - `subtitle_genius_vac_speech_segment_duration_seconds` - 语音段时长
   - `subtitle_genius_vac_speech_segment_processing_time_seconds` - 处理语音段所需时间
   - `subtitle_genius_vac_audio_buffer_size_bytes` - 音频缓冲区大小

3. **仪表盘类指标**
   - `subtitle_genius_vac_audio_buffer_chunks` - 音频缓冲区中的块数量
   - `subtitle_genius_vac_audio_buffer_duration_seconds` - 音频缓冲区中的音频时长
   - `subtitle_genius_vac_audio_completeness_percent` - 语音段音频完整性百分比

4. **摘要类指标**
   - `subtitle_genius_vac_speech_segment_samples` - 语音段中的样本数

## 使用方法

### 基本用法

```python
from subtitle_genius.stream.vac_processor import VACProcessor
from subtitle_genius.metrics.vac_metrics import register_vac_metrics

# 创建 VAC 处理器
vac = VACProcessor(
    threshold=0.3,
    min_silence_duration_ms=300,
    speech_pad_ms=100,
    sample_rate=16000
)

# 注册指标收集
metrics = register_vac_metrics(vac)

# 处理音频流
vac.process_streaming_audio(audio_stream)
```

### 配置 Pushgateway

默认情况下，指标会推送到 `localhost:9529`。可以通过环境变量或参数修改：

```python
# 通过环境变量
import os
os.environ['PROMETHEUS_PUSHGATEWAY_URL'] = 'your-pushgateway:9529'

# 或通过参数
from subtitle_genius.metrics.metrics_manager import get_metrics_manager
metrics_manager = get_metrics_manager('your-pushgateway:9529')
```

### 手动推送指标

```python
from subtitle_genius.metrics.metrics_manager import get_metrics_manager

# 获取指标管理器
metrics_manager = get_metrics_manager()

# 手动推送指标
metrics_manager.push_metrics(job="subtitle_genius")
```

### 自动推送指标

默认情况下，指标会每 15 秒自动推送一次。可以通过环境变量修改：

```python
import os
os.environ['METRICS_PUSH_INTERVAL_SECONDS'] = '30'  # 每 30 秒推送一次
os.environ['METRICS_AUTO_PUSH'] = 'true'  # 启用自动推送
```

## 测试指标收集

运行测试脚本：

```bash
python test_metrics.py
```

## 在 Prometheus 中查询

示例查询：

```
# 每分钟检测到的语音段数
rate(subtitle_genius_vac_speech_segments_total[1m])

# 语音段平均时长
rate(subtitle_genius_vac_speech_segment_duration_seconds_sum[5m]) / rate(subtitle_genius_vac_speech_segment_duration_seconds_count[5m])

# 音频缓冲区大小
subtitle_genius_vac_audio_buffer_chunks
```

## 注意事项

1. 确保 Pushgateway 服务器可访问（默认：localhost:9529）
2. 指标收集会增加少量处理开销，但影响通常可忽略不计
3. 在生产环境中，建议设置适当的推送间隔，避免过于频繁的推送
