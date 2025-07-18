# SubtitleGenius 指标模块

使用 prometheus-client 监控系统关键指标

## 概述

指标模块使用 Prometheus 客户端库收集和导出系统关键指标，支持以下功能：

- 自动收集 VAC 处理器的语音分段指标
- 支持 Prometheus Pushgateway 导出
- 提供计数器、仪表盘、直方图和摘要类型的指标
- 自动推送指标到 Pushgateway

## 快速开始

### 安装依赖

```bash
pip install prometheus-client
```

### 基本用法

```python
from subtitle_genius.stream.vac_processor import VACProcessor
from subtitle_genius.metrics.vac_metrics import register_vac_metrics

# 创建VAC处理器
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

### 运行示例

```bash
# 设置Pushgateway URL (可选)
export PROMETHEUS_PUSHGATEWAY_URL=localhost:9529

# 运行示例脚本
python examples/metrics_example.py
```

## 配置选项

通过环境变量配置指标模块：

- `PROMETHEUS_PUSHGATEWAY_URL`: Pushgateway的URL (默认: localhost:9529)
- `APP_NAME`: 应用名称，用于在Pushgateway中区分不同应用 (默认: subtitle_genius)
- `METRICS_PUSH_INTERVAL_SECONDS`: 自动推送间隔，单位秒 (默认: 15)
- `METRICS_AUTO_PUSH`: 是否启用自动推送 (默认: true)

## 收集的指标

### VAC处理器指标

| 指标名称 | 类型 | 描述 | 标签 |
|---------|------|------|------|
| subtitle_genius_vac_speech_segments_total | Counter | 检测到的语音段总数 | status (success, error, incomplete) |
| subtitle_genius_vac_vad_events_total | Counter | VAD事件总数 | event_type (start, end, orphaned_end) |
| subtitle_genius_vac_speech_segment_duration_seconds | Histogram | 语音段时长（秒） | - |
| subtitle_genius_vac_speech_segment_processing_time_seconds | Histogram | 处理语音段所需时间（秒） | - |
| subtitle_genius_vac_audio_buffer_size_bytes | Histogram | 音频缓冲区大小（字节） | - |
| subtitle_genius_vac_audio_buffer_chunks | Gauge | 音频缓冲区中的块数量 | - |
| subtitle_genius_vac_audio_buffer_duration_seconds | Gauge | 音频缓冲区中的音频时长（秒） | - |
| subtitle_genius_vac_audio_completeness_percent | Gauge | 语音段音频完整性百分比 | segment_id |
| subtitle_genius_vac_speech_segment_samples | Summary | 语音段中的样本数 | - |

## 在Prometheus中查询

示例查询：

```
# 每分钟检测到的语音段数
rate(subtitle_genius_vac_speech_segments_total[1m])

# 语音段平均时长
rate(subtitle_genius_vac_speech_segment_duration_seconds_sum[5m]) / rate(subtitle_genius_vac_speech_segment_duration_seconds_count[5m])

# 音频缓冲区大小
subtitle_genius_vac_audio_buffer_chunks
```

## 与Grafana集成

1. 在Prometheus中配置Pushgateway作为目标
2. 在Grafana中添加Prometheus数据源
3. 创建仪表板，使用上述指标

## 自定义指标

要添加自定义指标，请参考`vac_metrics.py`中的示例，并使用`MetricsManager`类注册新指标。
