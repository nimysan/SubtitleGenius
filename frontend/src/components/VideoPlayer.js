import React, { useRef, useEffect, forwardRef, useImperativeHandle, useState } from 'react';
import './VideoPlayer.css';

const VideoPlayer = forwardRef(({ videoFile, onTimeUpdate, onAudioData }, ref) => {
  const videoRef = useRef(null);
  const audioContextRef = useRef(null);
  const sourceNodeRef = useRef(null);
  const processorNodeRef = useRef(null);
  const analyserNodeRef = useRef(null);
  const [isExtracting, setIsExtracting] = useState(false);

  useImperativeHandle(ref, () => ({
    getCurrentTime: () => videoRef.current?.currentTime || 0,
    play: () => videoRef.current?.play(),
    pause: () => videoRef.current?.pause(),
    seekTo: (time) => {
      if (videoRef.current) {
        videoRef.current.currentTime = time;
      }
    },
    startAudioExtraction: () => startAudioExtraction(),
    stopAudioExtraction: () => stopAudioExtraction()
  }));

  useEffect(() => {
    if (videoFile && videoRef.current) {
      const videoUrl = URL.createObjectURL(videoFile);
      videoRef.current.src = videoUrl;
      
      return () => {
        URL.revokeObjectURL(videoUrl);
      };
    }
  }, [videoFile]);

  // 开始音频提取
  const startAudioExtraction = () => {
    if (!videoRef.current || isExtracting) return;
    
    try {
      setIsExtracting(true);
      
      // 创建音频上下文
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      audioContextRef.current = new AudioContext();
      
      // 创建媒体源节点
      sourceNodeRef.current = audioContextRef.current.createMediaElementSource(videoRef.current);
      
      // 创建分析器节点
      analyserNodeRef.current = audioContextRef.current.createAnalyser();
      analyserNodeRef.current.fftSize = 2048;
      
      // 创建脚本处理节点
      const bufferSize = 4096;
      
      // 检查是否支持AudioWorkletNode
      if (audioContextRef.current.audioWorklet) {
        // 使用现代的AudioWorklet API
        audioContextRef.current.audioWorklet.addModule('audioProcessor.js')
          .then(() => {
            processorNodeRef.current = new AudioWorkletNode(
              audioContextRef.current,
              'audio-processor'
            );
            
            // 连接节点
            connectAudioNodes();
            
            // 设置消息处理
            processorNodeRef.current.port.onmessage = (event) => {
              if (onAudioData && event.data.audioData) {
                onAudioData(event.data.audioData);
              }
            };
          })
          .catch(err => {
            console.error('无法加载AudioWorklet:', err);
            fallbackToScriptProcessor();
          });
      } else {
        // 回退到旧的ScriptProcessorNode
        fallbackToScriptProcessor();
      }
      
      console.log('开始音频提取');
      
    } catch (error) {
      console.error('音频提取初始化失败:', error);
      stopAudioExtraction();
    }
  };
  
  // 回退到ScriptProcessorNode
  const fallbackToScriptProcessor = () => {
    try {
      // 创建脚本处理节点 - 调整缓冲区大小为3秒的音频数据
      const sampleRate = audioContextRef.current.sampleRate;
      const chunkDuration = 3; // 3秒
      const bufferSize = Math.pow(2, Math.ceil(Math.log2(sampleRate * chunkDuration))); // 向上取最接近的2的幂
      console.log(`ScriptProcessor缓冲区大小: ${bufferSize}，对应时长: ${bufferSize/sampleRate}秒`);
      
      processorNodeRef.current = audioContextRef.current.createScriptProcessor(
        bufferSize, 
        1, // 单声道输入
        1  // 单声道输出
      );
      
      // 设置音频处理回调
      processorNodeRef.current.onaudioprocess = (audioProcessingEvent) => {
        const inputBuffer = audioProcessingEvent.inputBuffer;
        const inputData = inputBuffer.getChannelData(0);
        
        // 克隆数据，因为inputData是只读的
        const audioData = new Float32Array(inputData.length);
        audioData.set(inputData);
        
        // 发送音频数据
        if (onAudioData) {
          console.log(`ScriptProcessor发送音频数据，样本数: ${audioData.length}，时长: ${audioData.length/audioContextRef.current.sampleRate}秒`);
          onAudioData(audioData);
        }
      };
      
      // 连接节点
      connectAudioNodes();
      
    } catch (error) {
      console.error('ScriptProcessor回退失败:', error);
      stopAudioExtraction();
    }
  };
  
  // 连接音频节点
  const connectAudioNodes = () => {
    if (!sourceNodeRef.current || !processorNodeRef.current || !audioContextRef.current) return;
    
    // 连接: 源 -> 分析器 -> 处理器 -> 目标
    sourceNodeRef.current.connect(analyserNodeRef.current);
    analyserNodeRef.current.connect(processorNodeRef.current);
    
    // ScriptProcessorNode需要连接到目标，AudioWorkletNode不需要
    if (processorNodeRef.current.constructor.name === 'ScriptProcessorNode') {
      processorNodeRef.current.connect(audioContextRef.current.destination);
    }
    
    // 同时连接源到目标，确保音频可以播放
    sourceNodeRef.current.connect(audioContextRef.current.destination);
  };

  // 停止音频提取
  const stopAudioExtraction = () => {
    if (!isExtracting) return;
    
    try {
      // 断开所有连接
      if (sourceNodeRef.current) {
        sourceNodeRef.current.disconnect();
      }
      
      if (analyserNodeRef.current) {
        analyserNodeRef.current.disconnect();
      }
      
      if (processorNodeRef.current) {
        processorNodeRef.current.disconnect();
      }
      
      // 关闭音频上下文
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
      }
      
      // 重置引用
      sourceNodeRef.current = null;
      processorNodeRef.current = null;
      analyserNodeRef.current = null;
      audioContextRef.current = null;
      
      console.log('停止音频提取');
      
    } catch (error) {
      console.error('停止音频提取失败:', error);
    } finally {
      setIsExtracting(false);
    }
  };

  // 组件卸载时清理资源
  useEffect(() => {
    return () => {
      stopAudioExtraction();
    };
  }, []);

  const handleTimeUpdate = () => {
    if (videoRef.current && onTimeUpdate) {
      onTimeUpdate(videoRef.current.currentTime);
    }
  };

  return (
    <div className="video-player-container">
      {videoFile ? (
        <video
          ref={videoRef}
          className="video-player"
          controls
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={() => {
            console.log('Video loaded:', videoFile.name);
          }}
        >
          您的浏览器不支持视频播放。
        </video>
      ) : (
        <div className="video-placeholder">
          <div className="placeholder-content">
            <div className="placeholder-icon">🎬</div>
            <h3>请上传视频文件</h3>
            <p>支持 MP4, WebM, AVI 等格式</p>
          </div>
        </div>
      )}
    </div>
  );
});

VideoPlayer.displayName = 'VideoPlayer';

export default VideoPlayer;
