import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
import 'react-tabs/style/react-tabs.css';
import './App.css';
import VideoPlayer from './components/VideoPlayer';
import DashPlayer from './components/DashPlayer';
import SubtitleDisplay from './components/SubtitleDisplay';
import ControlPanel from './components/ControlPanel';
import { convertToWAV, createWebSocketConnection, sendAudioData, createSaveSubtitlesConnection } from './utils/AudioUtils';

function App() {
  const [videoFile, setVideoFile] = useState(null);
  const [dashUrl, setDashUrl] = useState(''); // DASH URL状态
  const [subtitles, setSubtitles] = useState([]);
  const [currentTime, setCurrentTime] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [socket, setSocket] = useState(null);
  const [selectedLanguage, setSelectedLanguage] = useState('ar');
  const [selectedModel, setSelectedModel] = useState('whisper');
  const [isRealtime, setIsRealtime] = useState(true);
  const [debugMode, setDebugMode] = useState(false); // 调试模式状态
  const [enableCorrection, setEnableCorrection] = useState(true); // 启用纠错
  const [enableTranslation, setEnableTranslation] = useState(true); // 启用翻译
  const [targetLanguage, setTargetLanguage] = useState('en'); // 翻译目标语言
  const [clientId, setClientId] = useState(null); // 客户端ID
  const [saveStatus, setSaveStatus] = useState({ saving: false, message: '', success: false });
  const [tabIndex, setTabIndex] = useState(0); // 当前选中的标签页索引
  
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

  // 格式化时间显示
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
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
      
      // 获取时间戳信息
      const timestamp = audioInfo.timestamp || {};
      
      console.log('音频转换信息:', {
        originalSampleRate: audioInfo.originalSampleRate,
        targetSampleRate: audioInfo.targetSampleRate,
        usingSampleRate: sampleRate,
        audioDataLength: audioData.length,
        duration: audioInfo.duration,
        timestamp: {
          start_time: timestamp.start_time,
          end_time: timestamp.end_time,
          chunk_index: timestamp.chunk_index,
          duration: timestamp.duration
        }
      });
      
      // 将音频数据转换为WAV格式（现在是异步的）
      const wavBlob = await convertToWAV(audioData, sampleRate);
      
      // 通过WebSocket发送音频数据，包含时间戳信息
      if (socket) {
        console.log('发送音频数据，大小:', audioData.length, '采样率:', sampleRate, '时间戳:', timestamp);
        sendAudioData(socket, wavBlob, debugMode, timestamp); // 添加时间戳参数
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
    
    // 添加语言参数和处理选项
    wsUrl += `?language=${selectedLanguage}&correction=${enableCorrection}&translation=${enableTranslation}&target_language=${targetLanguage}`;
    
    // 添加调试日志
    console.log('建立WebSocket连接:', {
      model: selectedModel,
      language: selectedLanguage,
      correction: enableCorrection,
      translation: enableTranslation,
      targetLanguage: targetLanguage,
      url: wsUrl
    });
    
    // 创建WebSocket连接
    const newSocket = createWebSocketConnection(
      wsUrl,
      // 消息处理
      (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('收到WebSocket消息:', data);
          
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
          } else if (data.type === 'connection' && data.status === 'connected') {
            // 保存客户端ID
            setClientId(data.client_id);
            console.log('已连接到服务器:', {
              clientId: data.client_id,
              model: data.model,
              language: data.language,
              correctionEnabled: data.correction_enabled,
              translationEnabled: data.translation_enabled,
              targetLanguage: data.target_language
            });
          }
        } catch (error) {
          console.error('解析WebSocket消息失败:', error);
        }
      },
      // 连接打开
      () => {
        console.log('WebSocket连接已建立，URL:', wsUrl);
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
  }, [selectedLanguage, selectedModel, enableCorrection, enableTranslation, targetLanguage]);

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
  const handleSettingsChange = (language, model, realtime, debug, correction, translation, translationTarget) => {
    console.log('设置变更:', {
      language, model, realtime, debug, correction, translation, translationTarget
    });
    
    // 检查是否有关键参数变更
    const hasKeyParameterChange = (
      language !== selectedLanguage ||
      model !== selectedModel ||
      correction !== enableCorrection ||
      translation !== enableTranslation ||
      translationTarget !== targetLanguage
    );
    
    // 更新状态
    setSelectedLanguage(language);
    setSelectedModel(model);
    setIsRealtime(realtime);
    if (debug !== undefined) {
      setDebugMode(debug);
    }
    if (correction !== undefined) {
      setEnableCorrection(correction);
    }
    if (translation !== undefined) {
      setEnableTranslation(translation);
    }
    if (translationTarget !== undefined) {
      setTargetLanguage(translationTarget);
    }
    
    // 如果有关键参数变更且当前有活跃连接，重新建立连接
    if (hasKeyParameterChange && socket && socket.readyState === WebSocket.OPEN) {
      console.log('关键参数变更，重新建立WebSocket连接');
      socket.close();
      setSocket(null);
      // 连接会在下次处理时自动重新建立
    }
  };
  
  // 保存字幕
  const handleSaveSubtitles = (filename) => {
    if (!clientId || subtitles.length === 0) {
      alert('没有可用的字幕或客户端ID无效');
      return;
    }
    
    setSaveStatus({ saving: true, message: '正在保存字幕...', success: false });
    
    // 创建保存字幕的WebSocket连接
    const saveSocket = createSaveSubtitlesConnection(
      clientId,
      filename,
      (data) => {
        // 成功回调
        setSaveStatus({ 
          saving: false, 
          message: `字幕已保存为: ${data.files.join(', ')}`, 
          success: true 
        });
        setTimeout(() => {
          setSaveStatus({ saving: false, message: '', success: false });
        }, 5000);
      },
      (error) => {
        // 错误回调
        setSaveStatus({ 
          saving: false, 
          message: `保存字幕失败: ${error.message}`, 
          success: false 
        });
        setTimeout(() => {
          setSaveStatus({ saving: false, message: '', success: false });
        }, 5000);
      }
    );
    
    // 5秒后关闭连接
    setTimeout(() => {
      if (saveSocket && saveSocket.readyState === WebSocket.OPEN) {
        saveSocket.close();
      }
    }, 5000);
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

  // 处理DASH URL输入
  const handleDashUrlChange = (e) => {
    setDashUrl(e.target.value);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🎬 SubtitleGenius</h1>
        <p>基于GenAI的实时MP4音频流字幕生成工具</p>
      </header>

      <main className="App-main">
        <Tabs selectedIndex={tabIndex} onSelect={index => setTabIndex(index)}>
          <TabList>
            <Tab>视频字幕</Tab>
            <Tab>DASH播放器</Tab>
          </TabList>

          <TabPanel>
            {/* 原有的视频字幕功能 */}
            <div className="video-subtitle-container">
              <div className="video-section">
                <VideoPlayer
                  videoFile={videoFile}
                  onTimeUpdate={handleTimeUpdate}
                  onAudioData={handleAudioData}
                  ref={videoRef}
                />
                
                {/* 最新字幕直接显示在视频下方 */}
                {subtitles.length > 0 && (
                  <div className="latest-subtitle-overlay">
                    <div className="latest-subtitle-content">
                      <div className="latest-subtitle-label">
                        <div className="label-left">
                          <span>最新字幕</span>
                          <div className="live-indicator">
                            <span className="live-dot"></span>
                            <span>LIVE</span>
                          </div>
                        </div>
                        <div className="subtitle-info-inline">
                          <span className="subtitle-time">
                            [{formatTime(subtitles[subtitles.length - 1].start)} - {formatTime(subtitles[subtitles.length - 1].end)}]
                          </span>
                        </div>
                      </div>
                      <div className="subtitle-text-container">
                        <div className={`original-text ${selectedLanguage === 'ar' ? 'text-direction-rtl' : 'text-direction-ltr'}`}>
                          {subtitles[subtitles.length - 1].text}
                        </div>
                        {subtitles[subtitles.length - 1].translated_text && (
                          <div className="translated-text">{subtitles[subtitles.length - 1].translated_text}</div>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <div className="subtitle-section">
                <SubtitleDisplay
                  subtitles={subtitles}
                  currentTime={currentTime}
                  onSaveSubtitles={handleSaveSubtitles}
                  saveStatus={saveStatus}
                  hasClientId={!!clientId}
                  defaultLanguage={selectedLanguage}
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
                  enableCorrection={enableCorrection}
                  enableTranslation={enableTranslation}
                  targetLanguage={targetLanguage}
                />
              </div>
            </div>
          </TabPanel>

          <TabPanel>
            {/* DASH播放器标签页 */}
            <div className="dash-tab-container">
              <div className="dash-input-container">
                <input
                  type="text"
                  className="dash-url-input"
                  placeholder="输入DASH流URL (例如: http://localhost:8080/tv002/tv002.mpd)"
                  value={dashUrl}
                  onChange={handleDashUrlChange}
                />
              </div>
              
              <div className="dash-player-section">
                <DashPlayer dashUrl={dashUrl} />
              </div>
            </div>
          </TabPanel>
        </Tabs>
      </main>
    </div>
  );
}

export default App;
