/**
 * 音频工具类
 * 提供音频数据处理和转换功能
 */

/**
 * 将Float32Array音频数据转换为WAV格式的Blob对象
 * @param {Float32Array} audioData - 音频数据
 * @param {number} sampleRate - 采样率
 * @returns {Blob} - WAV格式的Blob对象
 */
export const convertToWAV = (audioData, sampleRate) => {
  // WAV文件头大小
  const headerSize = 44;
  
  // 音频数据大小（字节）
  const dataSize = audioData.length * 2; // 16位每样本 = 2字节
  
  // 创建缓冲区
  const buffer = new ArrayBuffer(headerSize + dataSize);
  const view = new DataView(buffer);
  
  // 写入WAV文件头
  // "RIFF"标识
  writeString(view, 0, 'RIFF');
  // 文件大小
  view.setUint32(4, 36 + dataSize, true);
  // "WAVE"标识
  writeString(view, 8, 'WAVE');
  // "fmt "子块
  writeString(view, 12, 'fmt ');
  // 子块大小
  view.setUint32(16, 16, true);
  // 音频格式（1表示PCM）
  view.setUint16(20, 1, true);
  // 声道数
  view.setUint16(22, 1, true);
  // 采样率
  view.setUint32(24, sampleRate, true);
  // 字节率
  view.setUint32(28, sampleRate * 2, true);
  // 块对齐
  view.setUint16(32, 2, true);
  // 位深度
  view.setUint16(34, 16, true);
  // "data"子块
  writeString(view, 36, 'data');
  // 数据大小
  view.setUint32(40, dataSize, true);
  
  // 写入音频数据
  floatTo16BitPCM(view, headerSize, audioData);
  
  // 创建Blob对象
  return new Blob([buffer], { type: 'audio/wav' });
};

/**
 * 将字符串写入DataView
 * @param {DataView} view - DataView对象
 * @param {number} offset - 偏移量
 * @param {string} string - 要写入的字符串
 */
const writeString = (view, offset, string) => {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
};

/**
 * 将Float32Array转换为16位PCM
 * @param {DataView} view - DataView对象
 * @param {number} offset - 偏移量
 * @param {Float32Array} input - 输入的Float32Array
 */
const floatTo16BitPCM = (view, offset, input) => {
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    view.setInt16(offset + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
};

/**
 * 将多个音频块合并为一个
 * @param {Array<Float32Array>} audioChunks - 音频块数组
 * @returns {Float32Array} - 合并后的音频数据
 */
export const combineAudioChunks = (audioChunks) => {
  // 计算总长度
  let totalLength = 0;
  for (const chunk of audioChunks) {
    totalLength += chunk.length;
  }
  
  // 创建新数组
  const result = new Float32Array(totalLength);
  
  // 复制数据
  let offset = 0;
  for (const chunk of audioChunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  
  return result;
};

/**
 * 对音频数据进行重采样
 * @param {Float32Array} audioData - 原始音频数据
 * @param {number} originalSampleRate - 原始采样率
 * @param {number} targetSampleRate - 目标采样率
 * @returns {Float32Array} - 重采样后的音频数据
 */
export const resampleAudio = (audioData, originalSampleRate, targetSampleRate) => {
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
};

/**
 * 创建WebSocket连接
 * @param {string} url - WebSocket URL
 * @param {Function} onMessage - 消息处理回调
 * @param {Function} onOpen - 连接打开回调
 * @param {Function} onClose - 连接关闭回调
 * @param {Function} onError - 错误处理回调
 * @returns {WebSocket} - WebSocket实例
 */
export const createWebSocketConnection = (url, onMessage, onOpen, onClose, onError) => {
  const socket = new WebSocket(url);
  
  socket.onopen = onOpen || (() => console.log('WebSocket连接已建立'));
  socket.onmessage = onMessage || ((event) => console.log('收到消息:', event.data));
  socket.onclose = onClose || (() => console.log('WebSocket连接已关闭'));
  socket.onerror = onError || ((error) => console.error('WebSocket错误:', error));
  
  return socket;
};

/**
 * 发送音频数据到WebSocket
 * @param {WebSocket} socket - WebSocket实例
 * @param {Blob|ArrayBuffer} audioData - 音频数据
 * @returns {boolean} - 是否发送成功
 */
export const sendAudioData = (socket, audioData) => {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(audioData);
    return true;
  }
  return false;
};
