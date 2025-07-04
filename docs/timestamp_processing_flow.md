# 时间戳处理流程图

## 完整的时间戳处理流程

```mermaid
graph TD
    A[前端音频播放] --> B[AudioWorklet处理器]
    B --> C[计算chunk时间戳]
    C --> D[音频数据 + 时间戳信息]
    D --> E[WebSocket发送]
    
    E --> F[后端WebSocket服务器]
    F --> G{消息类型判断}
    
    G -->|时间戳消息| H[存储时间戳信息]
    G -->|音频数据| I[处理音频数据]
    
    H --> J[等待对应音频数据]
    I --> K[Whisper语音识别]
    
    J --> L[时间戳映射]
    K --> M[获取识别结果]
    
    L --> N[应用绝对时间戳]
    M --> N
    
    N --> O[Correction纠错]
    O --> P[Translation翻译]
    P --> Q[生成最终字幕]
    
    Q --> R[返回前端显示]
    
    style A fill:#e1f5fe
    style B fill:#e8f5e8
    style F fill:#fff3e0
    style K fill:#f3e5f5
    style Q fill:#e8f5e8
```

## 时间戳数据结构

```mermaid
classDiagram
    class TimestampInfo {
        +float start_time
        +float end_time
        +float duration
        +int chunk_index
        +int total_samples_processed
        +float audio_start_time
        +float processing_start_time
        +float current_time
    }
    
    class AudioChunk {
        +Float32Array audioData
        +int originalSampleRate
        +int targetSampleRate
        +TimestampInfo timestamp
    }
    
    class Subtitle {
        +float start
        +float end
        +string text
        +string language
    }
    
    AudioChunk --> TimestampInfo
    TimestampInfo --> Subtitle : "映射时间戳"
```

## 处理时序图

```mermaid
sequenceDiagram
    participant F as 前端
    participant AW as AudioWorklet
    participant WS as WebSocket
    participant BE as 后端
    participant W as Whisper
    participant C as Correction
    participant T as Translation
    
    F->>AW: 播放音频
    AW->>AW: 处理音频chunk (3s)
    AW->>AW: 计算时间戳 (0-3s, 3-6s, ...)
    
    AW->>WS: 发送时间戳信息
    WS->>BE: 转发时间戳
    BE->>BE: 存储时间戳信息
    
    AW->>WS: 发送音频数据
    WS->>BE: 转发音频数据
    
    BE->>W: 语音识别 (1.1s)
    W-->>BE: 返回识别结果
    
    BE->>BE: 应用绝对时间戳
    
    BE->>C: 纠错处理 (1.2s)
    C-->>BE: 返回纠错结果
    
    BE->>T: 翻译处理 (0.5s)
    T-->>BE: 返回翻译结果
    
    BE->>WS: 发送最终字幕
    WS->>F: 显示字幕
    
    Note over F,T: 总延迟: 3.5秒<br/>但时间戳基于原始音频时间
```

## 关键时间点说明

| 时间点 | 说明 | 示例值 |
|--------|------|--------|
| `start_time` | chunk在音频中的起始时间 | 6.0s |
| `end_time` | chunk在音频中的结束时间 | 9.0s |
| `duration` | chunk的时长 | 3.0s |
| `chunk_index` | chunk的序号 | 2 |
| `audio_start_time` | 音频开始处理的时间戳 | AudioContext.currentTime |
| `processing_start_time` | 处理开始的时间戳 | AudioContext.currentTime |
| `current_time` | 当前处理时间戳 | AudioContext.currentTime |

## 时间戳映射逻辑

```mermaid
graph LR
    A[Whisper相对时间戳] --> B[映射函数]
    C[前端绝对时间戳] --> B
    B --> D[最终字幕时间戳]
    
    subgraph "映射公式"
        E["final_start = chunk_start_time + whisper_relative_start"]
        F["final_end = chunk_start_time + whisper_relative_end"]
    end
    
    B --> E
    B --> F
```
