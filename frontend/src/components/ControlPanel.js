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
  debugMode = false,
  enableCorrection = true,
  enableTranslation = true,
  targetLanguage = 'en'
}) => {
  const fileInputRef = useRef(null);
  const [language, setLanguage] = useState(selectedLanguage);
  const [model, setModel] = useState(selectedModel);
  const [realtime, setRealtime] = useState(isRealtime);
  const [debug, setDebug] = useState(debugMode);
  const [correction, setCorrection] = useState(enableCorrection);
  const [translation, setTranslation] = useState(enableTranslation);
  const [translationTarget, setTranslationTarget] = useState(targetLanguage);

  // 当设置变更时通知父组件
  useEffect(() => {
    if (onSettingsChange) {
      onSettingsChange(language, model, realtime, debug, correction, translation, translationTarget);
    }
  }, [language, model, realtime, debug, correction, translation, translationTarget, onSettingsChange]);

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

  const handleCorrectionChange = (e) => {
    setCorrection(e.target.checked);
  };

  const handleTranslationChange = (e) => {
    setTranslation(e.target.checked);
  };

  const handleTranslationTargetChange = (e) => {
    setTranslationTarget(e.target.value);
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
            <label htmlFor="language-select">视频语言:</label>
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
             
              <option value="whisper">Whisper In SageMaker</option>
               <option value="transcribe">Amazon Transcribe</option>
              <option value="claude">Claude</option>
            </select>
          </div>
          
          <div className="setting-item">
            <label htmlFor="translation-target-select">翻译目标语言:</label>
            <select 
              id="translation-target-select" 
              value={translationTarget}
              onChange={handleTranslationTargetChange}
              disabled={isProcessing || !translation}
            >
              <option value="en">English</option>
              <option value="zh">中文 (Chinese)</option>
              <option value="ar">العربية (Arabic)</option>
              <option value="fr">Français (French)</option>
              <option value="es">Español (Spanish)</option>
              <option value="de">Deutsch (German)</option>
              <option value="ja">日本語 (Japanese)</option>
              <option value="ko">한국어 (Korean)</option>
              <option value="ru">Русский (Russian)</option>
              <option value="pt">Português (Portuguese)</option>
              <option value="it">Italiano (Italian)</option>
              <option value="nl">Nederlands (Dutch)</option>
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
          
          <div className="setting-item">
            <label>
              <input 
                type="checkbox" 
                checked={correction}
                onChange={handleCorrectionChange}
                disabled={isProcessing}
              />
              启用智能纠错
            </label>
          </div>
          
          <div className="setting-item">
            <label>
              <input 
                type="checkbox" 
                checked={translation}
                onChange={handleTranslationChange}
                disabled={isProcessing}
              />
              启用多语言翻译
            </label>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ControlPanel;
