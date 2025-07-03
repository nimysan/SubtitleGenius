// 音频处理器 Worklet

class AudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // 调整缓冲区大小为3秒的音频数据（假设采样率为16000Hz）
    this.bufferSize = 48000; // 16000 * 3 = 48000
    this.buffer = new Float32Array(this.bufferSize);
    this.bufferIndex = 0;
    this.sampleRate = 16000; // 采样率
    this.chunkDuration = 3; // 每个chunk的时长（秒）
    console.log(`AudioProcessor初始化，缓冲区大小: ${this.bufferSize}，对应时长: ${this.bufferSize/this.sampleRate}秒`);
  }

  process(inputs, outputs, parameters) {
    // 获取输入数据
    const input = inputs[0];
    if (!input || !input.length) return true;
    
    const inputChannel = input[0];
    
    // 将数据添加到缓冲区
    for (let i = 0; i < inputChannel.length; i++) {
      if (this.bufferIndex >= this.bufferSize) {
        // 缓冲区已满，发送数据
        this.sendAudioData();
        this.bufferIndex = 0;
      }
      
      this.buffer[this.bufferIndex++] = inputChannel[i];
    }
    
    // 只有当缓冲区完全填满时才发送数据，确保每个chunk都是完整的3秒
    // 不再在缓冲区达到一半时发送数据
    
    // 返回true以保持处理器活动
    return true;
  }
  
  sendAudioData() {
    // 创建一个新的Float32Array，只包含有效数据
    const audioData = this.buffer.slice(0, this.bufferIndex);
    
    // 发送数据到主线程
    this.port.postMessage({
      audioData: audioData,
      duration: audioData.length / this.sampleRate // 添加音频时长信息（秒）
    });
    
    console.log(`发送音频数据，样本数: ${audioData.length}，时长: ${audioData.length/this.sampleRate}秒`);
  }
}

// 注册处理器
registerProcessor('audio-processor', AudioProcessor);
