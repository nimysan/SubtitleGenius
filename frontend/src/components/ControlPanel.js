import React, { useRef, useState, useEffect } from 'react';
import './ControlPanel.css';

const ControlPanel = ({ 
  onVideoUpload, 
  onGenerateSubtitles, 
  onSettingsChange,
  isProcessing, 
  hasVideo,
  selectedLanguage = 'ar',
  selectedModel = 'whisper',
  isRealtime = true,
  debugMode = false
}) => {
  const fileInputRef = useRef(null);
  const [language, setLanguage] = useState(selectedLanguage);
  const [model, setModel] = useState(selectedModel);
  const [realtime, setRealtime] = useState(isRealtime);
  const [debug, setDebug] = useState(debugMode);

  // 当设置变更时通知父组件
  useEffect(() => {
    if (onSettingsChange) {
      onSettingsChange(language, model, realtime, debug);
    }
  }, [language, model, realtime, debug, onSettingsChange]);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      // 检查文件类型
      const validTypes = ['video/mp4', 'video/webm', 'video/avi', 'video/mov'];
      if (validTypes.includes(file.type)) {
        onVideoUpload(file);
      } else {
        alert('请选择有效的视频文件格式 (MP4, WebM, AVI, MOV)');
      }
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };

  const handleDrop = (event) => {
    event.preventDefault();
    event.stopPropagation();
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      const validTypes = ['video/mp4', 'video/webm', 'video/avi', 'video/mov'];
      if (validTypes.includes(file.type)) {
        onVideoUpload(file);
      } else {
        alert('请选择有效的视频文件格式 (MP4, WebM, AVI, MOV)');
      }
    }
  };

  const handleLanguageChange = (e) => {
    setLanguage(e.target.value);
  };

  const handleModelChange = (e) => {
    setModel(e.target.value);
  };

  const handleRealtimeChange = (e) => {
    setRealtime(e.target.checked);
  };

  const handleDebugModeChange = (e) => {
    setDebug(e.target.checked);
  };

  return (
    <div className="control-panel">
      <div className="control-section">
        <h4>文件上传</h4>
        <div 
          className="upload-area"
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileSelect}
            accept="video/*"
            style={{ display: 'none' }}
          />
          <button 
            className="upload-button"
            onClick={handleUploadClick}
          >
            <span className="upload-icon">📁</span>
            选择视频文件
          </button>
          <p className="upload-hint">或拖拽文件到此处</p>
        </div>
      </div>

      <div className="control-section">
        <h4>字幕生成</h4>
        <div className="generation-controls">
          <button
            className={`generate-button ${isProcessing ? 'processing' : ''}`}
            onClick={onGenerateSubtitles}
            disabled={!hasVideo || isProcessing}
          >
            {isProcessing ? (
              <>
                <span className="loading-spinner"></span>
                处理中...
              </>
            ) : (
              <>
                <span className="generate-icon">🤖</span>
                生成字幕
              </>
            )}
          </button>
          
          <div className="generation-info">
            {!hasVideo && (
              <p className="info-text">请先上传视频文件</p>
            )}
            {isProcessing && (
              <p className="info-text">正在使用AI模型处理音频...</p>
            )}
          </div>
        </div>
      </div>

      <div className="control-section">
        <h4>设置选项</h4>
        <div className="settings-controls">
          <div className="setting-item">
            <label htmlFor="language-select">语言:</label>
            <select 
              id="language-select" 
              value={language}
              onChange={handleLanguageChange}
              disabled={isProcessing}
            >
              <option value="ar">العربية (Arabic)</option>
              <option value="en">English</option>
              <option value="zh">中文</option>
              <option value="fr">Français</option>
              <option value="es">Español</option>
            </select>
          </div>
          
          <div className="setting-item">
            <label htmlFor="model-select">AI模型:</label>
            <select 
              id="model-select" 
              value={model}
              onChange={handleModelChange}
              disabled={isProcessing}
            >
              <option value="transcribe">Amazon Transcribe</option>
              <option value="whisper">OpenAI Whisper</option>
              <option value="claude">Claude</option>
            </select>
          </div>
          
          <div className="setting-item">
            <label>
              <input 
                type="checkbox" 
                checked={realtime}
                onChange={handleRealtimeChange}
                disabled={isProcessing}
              />
              实时处理
            </label>
          </div>
          
          <div className="setting-item">
            <label>
              <input 
                type="checkbox" 
                checked={debug}
                onChange={handleDebugModeChange}
                disabled={isProcessing}
              />
              调试模式 (保存WAV文件)
            </label>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ControlPanel;
