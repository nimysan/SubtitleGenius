// 音频处理器 Worklet

class AudioProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    
    // 从选项中获取实际的采样率，如果没有则使用默认值
    this.actualSampleRate = options.processorOptions?.sampleRate || sampleRate || 44100;
    this.targetSampleRate = 16000; // 目标采样率（用于后端处理）
    
    // 调整缓冲区大小为30秒的音频数据（基于实际采样率）
    this.chunkDuration = 30; // 每个chunk的时长（秒）
    this.bufferSize = this.actualSampleRate * this.chunkDuration;
    this.buffer = new Float32Array(this.bufferSize);
    this.bufferIndex = 0;
    
    console.log(`AudioProcessor初始化:`);
    console.log(`- 实际采样率: ${this.actualSampleRate}Hz`);
    console.log(`- 目标采样率: ${this.targetSampleRate}Hz`);
    console.log(`- 缓冲区大小: ${this.bufferSize}，对应时长: ${this.bufferSize/this.actualSampleRate}秒`);
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
    
    // 返回true以保持处理器活动
    return true;
  }
  
  sendAudioData() {
    // 创建一个新的Float32Array，只包含有效数据
    const audioData = this.buffer.slice(0, this.bufferIndex);
    
    // 如果需要重采样到目标采样率
    let processedAudioData = audioData;
    if (this.actualSampleRate !== this.targetSampleRate) {
      processedAudioData = this.resampleAudio(audioData, this.actualSampleRate, this.targetSampleRate);
      console.log(`重采样: ${this.actualSampleRate}Hz -> ${this.targetSampleRate}Hz`);
      console.log(`原始样本数: ${audioData.length}, 重采样后: ${processedAudioData.length}`);
    }
    
    // 发送数据到主线程
    this.port.postMessage({
      audioData: processedAudioData,
      originalSampleRate: this.actualSampleRate,
      targetSampleRate: this.targetSampleRate,
      duration: audioData.length / this.actualSampleRate, // 基于原始采样率的时长
      originalLength: audioData.length,
      processedLength: processedAudioData.length
    });
    
    console.log(`发送音频数据:`);
    console.log(`- 原始样本数: ${audioData.length}，时长: ${audioData.length/this.actualSampleRate}秒`);
    console.log(`- 处理后样本数: ${processedAudioData.length}`);
  }
  
  // 简单的线性插值重采样
  resampleAudio(audioData, originalSampleRate, targetSampleRate) {
    if (originalSampleRate === targetSampleRate) {
      return audioData;
    }
    
    const ratio = originalSampleRate / targetSampleRate;
    const newLength = Math.round(audioData.length / ratio);
    const result = new Float32Array(newLength);
    
    for (let i = 0; i < newLength; i++) {
      const position = i * ratio;
      const index = Math.floor(position);
      const fraction = position - index;
      
      // 线性插值
      if (index + 1 < audioData.length) {
        result[i] = audioData[index] * (1 - fraction) + audioData[index + 1] * fraction;
      } else {
        result[i] = audioData[index];
      }
    }
    
    return result;
  }
}

// 注册处理器
registerProcessor('audio-processor', AudioProcessor);
