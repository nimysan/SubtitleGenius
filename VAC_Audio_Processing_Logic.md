# VAC 音频处理逻辑

## 概述

VAC (Voice Activity Controller) 是一个用于检测和处理音频中语音活动的组件。它能够智能地识别音频流中的语音段和非语音段，只处理有语音的部分，从而提高转录效率和质量。

## 处理流程图

```
                    +------------------+
                    |  接收音频数据    |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |  添加到缓冲区    |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |   VAD 语音检测   |
                    +--------+---------+
                             |
                             v
              +-----------------------------+
              |   检测到语音事件了吗？     |
              +----+----------------+------+
                   |                |
                   | 是             | 否
                   v                v
    +------------------------+    +------------------------+
    |    检查事件类型        |    |    检查当前状态        |
    +----+--------+----+----+    +----+----------------+--+
         |        |    |              |                |
         |        |    |              | voice          | nonvoice
  语音开始  语音结束  完整语音段        v                v
         |        |    |         +------------+    +------------+
         v        v    v         | 处理缓冲区 |    | 保留最近1秒|
    +----+--------+----+----+    | 中的音频   |    | 丢弃旧数据 |
    | 根据事件类型处理音频 |    +------------+    +------------+
    +------------------------+
                |
                v
    +------------------------+
    |   更新状态和缓冲区     |
    +------------------------+
                |
                v
    +------------------------+
    |   转录处理（如需）     |
    +------------------------+
```

## 详细处理逻辑

### 1. 接收音频数据

当通过 WebSocket 接收到音频数据时，系统会将其传递给 VAC 处理器。

### 2. VAD 语音检测

VAC 使用 Silero VAD 模型对音频数据进行分析，检测语音活动。

### 3. 事件处理

根据 VAD 检测结果，系统会执行不同的处理逻辑：

#### 3.1 检测到语音开始

```
+---------------------------+
| 检测到语音开始            |
+------------+--------------+
             |
             v
+---------------------------+
| 设置状态为 'voice'        |
+------------+--------------+
             |
             v
+---------------------------+
| 从语音开始位置截取音频    |
+------------+--------------+
             |
             v
+---------------------------+
| 初始化处理器并插入音频    |
+------------+--------------+
             |
             v
+---------------------------+
| 清除缓冲区                |
+---------------------------+
```

#### 3.2 检测到语音结束

```
+---------------------------+
| 检测到语音结束            |
+------------+--------------+
             |
             v
+---------------------------+
| 设置状态为 'nonvoice'     |
+------------+--------------+
             |
             v
+---------------------------+
| 截取到语音结束位置的音频  |
+------------+--------------+
             |
             v
+---------------------------+
| 插入音频数据到处理器      |
+------------+--------------+
             |
             v
+---------------------------+
| 标记为最终结果            |
+------------+--------------+
             |
             v
+---------------------------+
| 清除缓冲区                |
+---------------------------+
```

#### 3.3 检测到完整语音段

```
+---------------------------+
| 检测到完整语音段          |
+------------+--------------+
             |
             v
+---------------------------+
| 设置状态为 'nonvoice'     |
+------------+--------------+
             |
             v
+---------------------------+
| 截取语音段的音频数据      |
+------------+--------------+
             |
             v
+---------------------------+
| 初始化处理器并插入音频    |
+------------+--------------+
             |
             v
+---------------------------+
| 标记为最终结果            |
+------------+--------------+
             |
             v
+---------------------------+
| 清除缓冲区                |
+---------------------------+
```

#### 3.4 未检测到语音事件

```
+---------------------------+
| 未检测到语音事件          |
+------------+--------------+
             |
             v
+---------------------------+
| 检查当前状态              |
+-----+-------------------+-+
      |                   |
      v                   v
+------------+    +------------------+
| 状态='voice'|    | 状态='nonvoice' |
+-----+------+    +--------+---------+
      |                    |
      v                    v
+------------+    +------------------+
| 继续处理   |    | 保留最近1秒数据  |
| 音频数据   |    | 丢弃更早的数据   |
+-----+------+    +------------------+
      |
      v
+------------+
| 清除缓冲区 |
+------------+
```

## 多语音段处理示例

假设接收到400个bytes的音频数据，其中：
- 0-300: 语音部分
- 300-320: 非语音部分
- 320-400: 语音部分

### 处理流程

```
+---------------------------+
| 接收400bytes音频数据      |
+------------+--------------+
             |
             v
+---------------------------+
| VAD检测语音活动           |
+------------+--------------+
             |
             v
+---------------------------+
| 检测到位置300有语音结束   |
| 检测到位置320有语音开始   |
+------------+--------------+
             |
             v
+---------------------------+
| 处理第一个语音段(0-300)   |
+------------+--------------+
             |
             v
+---------------------------+
| 设置状态为'nonvoice'      |
| 截取0-300的音频数据       |
| 发送给处理器              |
| 标记为最终结果            |
| 清除缓冲区                |
+------------+--------------+
             |
             v
+---------------------------+
| 处理第二个语音段(320-400) |
+------------+--------------+
             |
             v
+---------------------------+
| 设置状态为'voice'         |
| 截取320-400的音频数据     |
| 初始化新的处理器          |
| 发送给处理器              |
| 清除缓冲区                |
+---------------------------+
```

## 关键机制

### 1. 状态跟踪

VAC 使用 `status` 变量跟踪当前是否处于语音状态：
- `'voice'`: 当前正在处理语音段
- `'nonvoice'`: 当前处于非语音状态

### 2. 缓冲区管理

- `audio_buffer`: 存储接收到的音频数据
- `buffer_offset`: 跟踪缓冲区的偏移量
- `clear_buffer()`: 清除缓冲区，避免重复处理

### 3. 最终结果标记

当检测到语音结束或完整语音段时，系统会设置 `is_currently_final = True`，表示当前处理的结果是最终的，可以发送给客户端。

## 优势

1. **智能分段**：自动识别语音段，避免处理静默部分
2. **资源优化**：只处理有语音的部分，减少计算资源消耗
3. **实时处理**：能够处理连续的音频流，适用于实时转录场景
4. **高质量转录**：通过只处理有语音的部分，提高转录质量
