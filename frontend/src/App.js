import React, { useState, useRef, useEffect, useCallback } from 'react';
import './App.css';
import VideoPlayer from './components/VideoPlayer';
import SubtitleDisplay from './components/SubtitleDisplay';
import ControlPanel from './components/ControlPanel';
import { convertToWAV, createWebSocketConnection, sendAudioData } from './utils/AudioUtils';

function App() {
  const [videoFile, setVideoFile] = useState(null);
  const [subtitles, setSubtitles] = useState([]);
  const [currentTime, setCurrentTime] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [socket, setSocket] = useState(null);
  const [selectedLanguage, setSelectedLanguage] = useState('ar');
  const [selectedModel, setSelectedModel] = useState('transcribe');
  const [isRealtime, setIsRealtime] = useState(true);
  const [debugMode, setDebugMode] = useState(false); // 调试模式状态
  
  const videoRef = useRef(null);
  const audioChunksRef = useRef([]);
  const processingRef = useRef(false);

  // 处理视频文件上传
  const handleVideoUpload = (file) => {
    setVideoFile(file);
    // 清空之前的字幕
    setSubtitles([]);
    // 关闭之前的WebSocket连接
    if (socket) {
      socket.close();
      setSocket(null);
    }
  };

  // 处理视频时间更新
  const handleTimeUpdate = (time) => {
    setCurrentTime(time);
  };

  // 处理音频数据
  const handleAudioData = useCallback(async (audioData, audioInfo = {}) => {
    try {
      // 将音频数据添加到缓冲区
      audioChunksRef.current.push(audioData);
      
      // 如果不是实时处理，则只收集数据
      if (!isRealtime) return;
      
      // 获取采样率信息，优先使用音频处理器提供的信息
      const sampleRate = audioInfo.targetSampleRate || audioInfo.originalSampleRate || 16000;
      
      console.log('音频转换信息:', {
        originalSampleRate: audioInfo.originalSampleRate,
        targetSampleRate: audioInfo.targetSampleRate,
        usingSampleRate: sampleRate,
        audioDataLength: audioData.length,
        duration: audioInfo.duration
      });
      
      // 将音频数据转换为WAV格式（现在是异步的）
      const wavBlob = await convertToWAV(audioData, sampleRate);
      
      // 通过WebSocket发送音频数据
      if (socket) {
        console.log('发送音频数据，大小:', audioData.length, '采样率:', sampleRate);
        sendAudioData(socket, wavBlob, debugMode); // 第三个参数表示是否保存到文件
      }
    } catch (error) {
      console.error('处理音频数据失败:', error);
    }
  }, [socket, isRealtime]);

  // 连接WebSocket
  const connectWebSocket = useCallback(() => {
    // 根据选择的模型确定WebSocket URL
    let wsUrl;
    switch (selectedModel) {
      case 'whisper':
        wsUrl = 'ws://localhost:8000/ws/whisper';
        break;
      case 'claude':
        wsUrl = 'ws://localhost:8000/ws/claude';
        break;
      case 'transcribe':
      default:
        wsUrl = 'ws://localhost:8000/ws/transcribe';
        break;
    }
    
    // 添加语言参数
    wsUrl += `?language=${selectedLanguage}`;
    
    // 创建WebSocket连接
    const newSocket = createWebSocketConnection(
      wsUrl,
      // 消息处理
      (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'subtitle') {
            // 添加新字幕
            setSubtitles(prev => {
              // 检查是否已存在相同ID的字幕
              const exists = prev.some(sub => sub.id === data.subtitle.id);
              if (exists) {
                // 更新现有字幕
                return prev.map(sub => 
                  sub.id === data.subtitle.id ? data.subtitle : sub
                );
              } else {
                // 添加新字幕
                return [...prev, data.subtitle];
              }
            });
          }
        } catch (error) {
          console.error('解析WebSocket消息失败:', error);
        }
      },
      // 连接打开
      () => {
        console.log('WebSocket连接已建立');
      },
      // 连接关闭
      () => {
        console.log('WebSocket连接已关闭');
        setSocket(null);
        setIsProcessing(false);
        processingRef.current = false;
      },
      // 错误处理
      (error) => {
        console.error('WebSocket错误:', error);
        setIsProcessing(false);
        processingRef.current = false;
      }
    );
    
    setSocket(newSocket);
    return newSocket;
  }, [selectedLanguage, selectedModel]);

  // 生成字幕
  const handleGenerateSubtitles = () => {
    if (!videoFile || isProcessing) return;
    
    setIsProcessing(true);
    processingRef.current = true;
    
    // 清空之前的字幕和音频数据
    setSubtitles([]);
    audioChunksRef.current = [];
    
    // 连接WebSocket
    const newSocket = connectWebSocket();
    
    // 开始音频提取
    if (videoRef.current) {
      // 先暂停视频
      videoRef.current.pause();
      // 回到开始位置
      videoRef.current.seekTo(0);
      
      // 开始音频提取
      setTimeout(() => {
        if (videoRef.current) {
          console.log('开始音频提取和播放视频');
          videoRef.current.startAudioExtraction();
          // 开始播放视频
          videoRef.current.play();
        } else {
          console.error('视频引用不存在，无法开始音频提取');
          setIsProcessing(false);
          processingRef.current = false;
        }
      }, 500);
    }
  };

  // 处理设置变更
  const handleSettingsChange = (language, model, realtime, debug) => {
    setSelectedLanguage(language);
    setSelectedModel(model);
    setIsRealtime(realtime);
    if (debug !== undefined) {
      setDebugMode(debug);
    }
  };

  // 组件卸载时清理资源
  useEffect(() => {
    return () => {
      if (socket) {
        socket.close();
      }
      if (videoRef.current) {
        videoRef.current.stopAudioExtraction();
      }
      processingRef.current = false;
    };
  }, [socket]);

  return (
    <div className="App">
      <header className="App-header">
        <h1>🎬 SubtitleGenius</h1>
        <p>基于GenAI的实时MP4音频流字幕生成工具</p>
      </header>

      <main className="App-main">
        <div className="video-section">
          <VideoPlayer
            videoFile={videoFile}
            onTimeUpdate={handleTimeUpdate}
            onAudioData={handleAudioData}
            ref={videoRef}
          />
        </div>

        <div className="subtitle-section">
          <SubtitleDisplay
            subtitles={subtitles}
            currentTime={currentTime}
          />
        </div>

        <div className="app-control-section">
          <ControlPanel
            onVideoUpload={handleVideoUpload}
            onGenerateSubtitles={handleGenerateSubtitles}
            onSettingsChange={handleSettingsChange}
            isProcessing={isProcessing}
            hasVideo={!!videoFile}
            selectedLanguage={selectedLanguage}
            selectedModel={selectedModel}
            isRealtime={isRealtime}
            debugMode={debugMode}
          />
        </div>
      </main>
    </div>
  );
}

export default App;
