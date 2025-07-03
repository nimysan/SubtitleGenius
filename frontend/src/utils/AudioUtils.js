/**
 * 音频工具类
 * 使用第三方库提供更可靠的音频数据处理和转换功能
 */

// 导入第三方音频处理库
import toWav from 'audiobuffer-to-wav';
import WavEncoder from 'wav-encoder';

/**
 * 将Float32Array音频数据转换为WAV格式的Blob对象
 * 使用第三方库 wav-encoder 提供更可靠的转换
 * @param {Float32Array} audioData - 音频数据
 * @param {number} sampleRate - 采样率
 * @returns {Promise<Blob>} - WAV格式的Blob对象
 */
export const convertToWAV = async (audioData, sampleRate) => {
  try {
    // 使用 wav-encoder 进行转换
    const audioBuffer = {
      sampleRate: sampleRate,
      channelData: [audioData] // 单声道
    };
    
    const wavArrayBuffer = await WavEncoder.encode(audioBuffer);
    return new Blob([wavArrayBuffer], { type: 'audio/wav' });
  } catch (error) {
    console.warn('第三方库转换失败，使用降级方案:', error);
    // 降级到原始实现
    return convertToWAVFallback(audioData, sampleRate);
  }
};

/**
 * 从AudioBuffer转换为WAV (用于处理AudioBuffer类型输入)
 * @param {AudioBuffer} audioBuffer - AudioBuffer对象
 * @returns {Blob} - WAV格式的Blob对象
 */
export const convertAudioBufferToWAV = (audioBuffer) => {
  const wavArrayBuffer = toWav(audioBuffer);
  return new Blob([wavArrayBuffer], { type: 'audio/wav' });
};

/**
 * 智能转换函数 - 自动检测输入类型并转换
 * @param {Float32Array|AudioBuffer} audioInput - 音频输入
 * @param {number} sampleRate - 采样率（Float32Array时需要）
 * @returns {Promise<Blob>} - WAV格式的Blob对象
 */
export const smartConvertToWAV = async (audioInput, sampleRate = 16000) => {
  if (audioInput instanceof AudioBuffer) {
    return convertAudioBufferToWAV(audioInput);
  } else if (audioInput instanceof Float32Array) {
    return await convertToWAV(audioInput, sampleRate);
  } else {
    throw new Error('不支持的音频输入类型');
  }
};

/**
 * 降级方案：原始WAV转换实现（作为后备）
 * @param {Float32Array} audioData - 音频数据
 * @param {number} sampleRate - 采样率
 * @returns {Blob} - WAV格式的Blob对象
 */
const convertToWAVFallback = (audioData, sampleRate) => {
  console.warn('使用降级WAV转换方案');
  
  const headerSize = 44;
  const dataSize = audioData.length * 2;
  const buffer = new ArrayBuffer(headerSize + dataSize);
  const view = new DataView(buffer);
  
  // 写入WAV文件头
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);
  
  floatTo16BitPCM(view, headerSize, audioData);
  
  return new Blob([buffer], { type: 'audio/wav' });
};

// 辅助函数（保留用于降级方案）
const writeString = (view, offset, string) => {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
};

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
 * @param {boolean} saveToFile - 是否保存到文件
 * @returns {boolean} - 是否发送成功
 */
export const sendAudioData = (socket, audioData, saveToFile = false) => {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(audioData);
    
    // 如果需要保存到文件
    if (saveToFile && audioData instanceof Blob) {
      const timestamp = new Date().getTime();
      const url = URL.createObjectURL(audioData);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = `audio_chunk_${timestamp}.wav`;
      document.body.appendChild(a);
      a.click();
      setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }, 100);
      console.log(`已保存音频文件: audio_chunk_${timestamp}.wav`);
    }
    
    return true;
  }
  return false;
};

/**
 * 创建保存字幕的WebSocket连接
 * @param {string} clientId - 客户端ID
 * @param {string} filename - 文件名
 * @param {Function} onSuccess - 成功回调
 * @param {Function} onError - 错误回调
 * @returns {WebSocket} - WebSocket实例
 */
export const createSaveSubtitlesConnection = (clientId, filename, onSuccess, onError) => {
  const url = `ws://localhost:8000/ws/save_subtitles?client_id=${clientId}${filename ? `&filename=${filename}` : ''}`;
  
  const socket = new WebSocket(url);
  
  socket.onopen = () => console.log('保存字幕WebSocket连接已建立');
  
  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === 'success') {
        console.log('字幕保存成功:', data.message);
        if (onSuccess) onSuccess(data);
      } else if (data.type === 'error') {
        console.error('字幕保存失败:', data.message);
        if (onError) onError(data);
      }
    } catch (error) {
      console.error('解析WebSocket消息失败:', error);
      if (onError) onError({ message: '解析WebSocket消息失败' });
    }
  };
  
  socket.onclose = () => console.log('保存字幕WebSocket连接已关闭');
  
  socket.onerror = (error) => {
    console.error('保存字幕WebSocket错误:', error);
    if (onError) onError({ message: '保存字幕WebSocket错误' });
  };
  
  return socket;
};
