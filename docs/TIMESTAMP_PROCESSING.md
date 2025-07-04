# SubtitleGenius 时间戳处理技术文档

## 概述

本文档详细描述了SubtitleGenius系统中音频chunk时间戳的处理机制，确保生成的字幕文件（VTT/SRT）具有准确的时间戳，不受处理延迟影响。

## 问题背景

### 原始问题
在实时字幕生成系统中，存在以下时间戳不准确的问题：

```
音频chunk(3s) → 网络传输(0.1s) → Transcribe(1.1s) → Correction(1.2s) → Translation(0.5s) → 网络返回(0.6s)
总延迟：3.5秒
```

当字幕返回时，音频已经播放到了 `原始时间 + 3.5秒` 的位置，导致字幕时间戳与实际音频内容不匹配。

### 解决方案
**核心思路**：前端在发送音频chunk时，同时传递该chunk在整个音频中的准确时间信息，后端直接使用这个时间作为字幕的时间戳基准。

## 系统架构

### 数据流设计

```
前端 → WebSocket → 后端
{
  type: "audio_with_timestamp",
  timestamp: {
    start_time: 6.0,      // 该chunk在完整音频中的起始时间(秒)
    end_time: 9.0,        // 该chunk在完整音频中的结束时间(秒)
    duration: 3.0,        // chunk时长
    chunk_index: 2,       // 第几个chunk
    total_samples_processed: 48000, // 总处理样本数
    audio_start_time: 1234567890.123, // 音频开始处理时间戳
    processing_start_time: 1234567890.123, // 处理开始时间戳
    current_time: 1234567890.456 // 当前时间戳
  }
}
```

## 技术实现

### 1. 前端实现

#### AudioWorklet处理器 (`audioProcessor.js`)

```javascript
class AudioProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    // ... 其他初始化代码
    
    // 时间戳跟踪
    this.totalSamplesProcessed = 0; // 总处理的样本数
    this.chunkIndex = 0; // chunk索引
    this.audioStartTime = null; // 音频开始处理的时间戳
    this.processingStartTime = currentTime; // 处理开始的时间
  }

  sendAudioData() {
    // 计算当前chunk在音频中的时间位置
    const chunkStartSample = this.totalSamplesProcessed - this.bufferIndex;
    const chunkEndSample = this.totalSamplesProcessed;
    
    const chunkStartTime = chunkStartSample / this.actualSampleRate;
    const chunkEndTime = chunkEndSample / this.actualSampleRate;
    const chunkDuration = this.bufferIndex / this.actualSampleRate;
    
    // 发送数据到主线程，包含时间戳信息
    this.port.postMessage({
      audioData: processedAudioData,
      timestamp: {
        start_time: chunkStartTime,
        end_time: chunkEndTime,
        duration: chunkDuration,
        chunk_index: this.chunkIndex,
        total_samples_processed: this.totalSamplesProcessed,
        audio_start_time: this.audioStartTime,
        processing_start_time: this.processingStartTime,
        current_time: currentTime
      }
    });
    
    this.chunkIndex++;
  }
}
```

#### WebSocket消息发送 (`AudioUtils.js`)

```javascript
export const sendAudioData = (socket, audioData, saveToFile = false, timestamp = null) => {
  if (socket && socket.readyState === WebSocket.OPEN) {
    if (timestamp) {
      // 先发送时间戳信息
      const message = {
        type: 'audio_with_timestamp',
        timestamp: {
          start_time: timestamp.start_time,
          end_time: timestamp.end_time,
          duration: timestamp.duration,
          chunk_index: timestamp.chunk_index,
          // ... 其他时间戳字段
        }
      };
      socket.send(JSON.stringify(message));
      
      // 然后发送音频数据
      socket.send(audioData);
    } else {
      // 兼容旧版本，直接发送音频数据
      socket.send(audioData);
    }
    return true;
  }
  return false;
};
```

### 2. 后端实现

#### WebSocket服务器 (`websocket_server.py`)

```python
# 时间戳存储 - 为每个客户端存储时间戳信息
client_timestamps: Dict[str, Dict] = {}

async def process_timestamp_message(client_id: str, message_data: dict):
    """处理时间戳消息"""
    try:
        timestamp_info = message_data.get('timestamp', {})
        
        # 存储时间戳信息
        if client_id not in client_timestamps:
            client_timestamps[client_id] = {}
        
        chunk_index = timestamp_info.get('chunk_index', 0)
        client_timestamps[client_id][chunk_index] = {
            'start_time': timestamp_info.get('start_time', 0.0),
            'end_time': timestamp_info.get('end_time', 0.0),
            'duration': timestamp_info.get('duration', 0.0),
            'chunk_index': chunk_index,
            # ... 其他字段
            'received_at': datetime.datetime.now().isoformat()
        }
        
        logger.info(f"客户端 {client_id} 时间戳信息已存储:")
        logger.info(f"  - Chunk {chunk_index}: {timestamp_info.get('start_time', 0.0):.2f}s - {timestamp_info.get('end_time', 0.0):.2f}s")
        
        return True
    except Exception as e:
        logger.error(f"处理时间戳消息失败: {e}")
        return False

async def apply_timestamp_to_subtitle(subtitle: Subtitle, timestamp_info: Dict) -> Subtitle:
    """将时间戳信息应用到字幕对象"""
    if timestamp_info:
        # 使用前端提供的绝对时间戳
        subtitle.start = timestamp_info['start_time']
        subtitle.end = timestamp_info['end_time']
        
        logger.info(f"应用时间戳到字幕: {subtitle.start:.2f}s - {subtitle.end:.2f}s")
    
    return subtitle
```

#### WebSocket端点处理

```python
async def websocket_whisper_endpoint(websocket: WebSocket, ...):
    # ... 初始化代码
    
    current_chunk_index = 0
    pending_timestamp = None
    
    # 处理接收到的消息（可能是时间戳信息或音频数据）
    while True:
        try:
            # 尝试接收文本消息（时间戳信息）
            try:
                message = await websocket.receive_text()
                message_data = json.loads(message)
                
                if message_data.get('type') == 'audio_with_timestamp':
                    # 处理时间戳信息
                    await process_timestamp_message(client_id, message_data)
                    pending_timestamp = message_data.get('timestamp')
                    continue
                    
            except Exception:
                # 如果不是JSON消息，尝试接收二进制数据
                pass
            
            # 接收音频数据
            data = await websocket.receive_bytes()
            audio_data = await process_wav_data(data)
              
            if audio_data is not None:
                # 使用Whisper处理音频
                async for subtitle in sagemaker_whisper_model.transcribe_stream(...):
                    # 如果有待处理的时间戳，应用到字幕
                    if pending_timestamp:
                        subtitle = await apply_timestamp_to_subtitle(subtitle, pending_timestamp)
                        pending_timestamp = None  # 清除已使用的时间戳
                    
                    # 发送字幕回客户端
                    await send_subtitle(websocket, subtitle, client_id, ...)
                    
        except WebSocketDisconnect:
            break
```

## 时间戳映射逻辑

### 基本映射公式

```python
def map_whisper_timestamps(whisper_result, chunk_start_time):
    """将Whisper的相对时间戳映射到绝对时间戳"""
    for segment in whisper_result:
        # Whisper返回的是chunk内的相对时间 (0-3秒)
        relative_start = segment.start
        relative_end = segment.end
        
        # 映射到音频的绝对时间
        absolute_start = chunk_start_time + relative_start
        absolute_end = chunk_start_time + relative_end
        
        segment.start = absolute_start
        segment.end = absolute_end
    
    return whisper_result
```

### 时间戳数据结构

```python
@dataclass
class TimestampInfo:
    start_time: float          # chunk在音频中的起始时间
    end_time: float            # chunk在音频中的结束时间
    duration: float            # chunk时长
    chunk_index: int           # chunk索引
    total_samples_processed: int  # 总处理样本数
    audio_start_time: float    # 音频开始处理时间戳
    processing_start_time: float  # 处理开始时间戳
    current_time: float        # 当前时间戳
```

## 字幕格式输出

### VTT格式
```
WEBVTT

00:00:06.000 --> 00:00:09.000
这是第二个chunk的字幕内容

00:00:09.000 --> 00:00:12.000
这是第三个chunk的字幕内容
```

### SRT格式
```
1
00:00:06,000 --> 00:00:09,000
这是第二个chunk的字幕内容

2
00:00:09,000 --> 00:00:12,000
这是第三个chunk的字幕内容
```

## 性能考虑

### 延迟分析
- **处理延迟**: 3.5秒（不影响时间戳准确性）
- **时间戳精度**: 毫秒级别
- **内存开销**: 每个客户端存储时间戳信息，内存占用较小

### 优化策略
1. **时间戳缓存**: 定期清理过期的时间戳信息
2. **批量处理**: 支持批量时间戳映射
3. **错误恢复**: 时间戳丢失时的降级处理

## 错误处理

### 常见错误场景
1. **时间戳信息丢失**: 使用估算时间戳作为后备
2. **网络断连**: 清理客户端时间戳缓存
3. **时间戳不连续**: 检测并修正时间戳跳跃

### 错误恢复机制
```python
async def handle_missing_timestamp(client_id: str, chunk_index: int):
    """处理时间戳丢失的情况"""
    if client_id in client_timestamps:
        # 尝试从前一个chunk推算时间戳
        prev_chunk = chunk_index - 1
        if prev_chunk in client_timestamps[client_id]:
            prev_timestamp = client_timestamps[client_id][prev_chunk]
            estimated_start = prev_timestamp['end_time']
            estimated_end = estimated_start + 3.0  # 假设3秒chunk
            
            return {
                'start_time': estimated_start,
                'end_time': estimated_end,
                'duration': 3.0,
                'chunk_index': chunk_index,
                'estimated': True
            }
    
    return None
```

## 测试验证

### 单元测试
```python
def test_timestamp_mapping():
    """测试时间戳映射功能"""
    whisper_result = [
        {'start': 0.5, 'end': 2.0, 'text': 'Hello'},
        {'start': 2.0, 'end': 2.8, 'text': 'World'}
    ]
    
    chunk_start_time = 6.0
    mapped_result = map_whisper_timestamps(whisper_result, chunk_start_time)
    
    assert mapped_result[0]['start'] == 6.5
    assert mapped_result[0]['end'] == 8.0
    assert mapped_result[1]['start'] == 8.0
    assert mapped_result[1]['end'] == 8.8
```

### 集成测试
1. **端到端时间戳验证**: 验证从前端到后端的完整时间戳流程
2. **多客户端测试**: 验证多个客户端同时处理时的时间戳隔离
3. **网络异常测试**: 验证网络中断时的时间戳恢复

## 部署配置

### 环境变量
```bash
# 时间戳相关配置
TIMESTAMP_CACHE_TTL=3600  # 时间戳缓存过期时间（秒）
TIMESTAMP_PRECISION=3     # 时间戳精度（小数位数）
ENABLE_TIMESTAMP_VALIDATION=true  # 启用时间戳验证
```

### 监控指标
- 时间戳处理成功率
- 时间戳映射延迟
- 客户端时间戳缓存大小
- 时间戳不连续事件数量

## 总结

通过实现基于前端时间戳的处理机制，SubtitleGenius系统能够：

1. **确保时间戳准确性**: 字幕时间戳基于原始音频时间，不受处理延迟影响
2. **简化实现复杂度**: 避免复杂的时间同步算法
3. **提高系统可靠性**: 前端控制时间基准，确保一致性
4. **支持多种格式**: 生成准确的VTT和SRT格式字幕文件

该方案已在生产环境中验证，能够有效解决实时字幕系统中的时间戳同步问题。
