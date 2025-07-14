import React, { useRef, useEffect, useState } from 'react';
import * as dashjs from 'dashjs';
import './DashPlayer.css';

const DashPlayer = ({ dashUrl }) => {
  const videoRef = useRef(null);
  const playerRef = useRef(null);
  const [playerState, setPlayerState] = useState({
    isPlaying: false,
    currentTime: 0,
    duration: 0,
    bufferLevel: 0,
    bitrate: 0,
    qualityIndex: 0
  });

  // 初始化 DASH 播放器
  useEffect(() => {
    if (!dashUrl || !videoRef.current) return;
    
    // 如果已经有播放器实例，先销毁
    if (playerRef.current) {
      playerRef.current.destroy();
      playerRef.current = null;
    }
    
    // 创建新的播放器实例
    const player = dashjs.MediaPlayer().create();
    playerRef.current = player;
    
    // 初始化播放器
    player.initialize(videoRef.current, dashUrl, true); // 自动播放
    player.updateSettings({
      'debug': {
        'logLevel': dashjs.Debug.LOG_LEVEL_INFO
      },
      'streaming': {
        'abr': {
          'autoSwitchBitrate': true
        }
      }
    });
    
    // 监听播放器事件
    player.on(dashjs.MediaPlayer.events.PLAYBACK_METADATA_LOADED, () => {
      console.log('DASH 元数据已加载');
    });
    
    player.on(dashjs.MediaPlayer.events.ERROR, (e) => {
      console.error('DASH 播放器错误:', e);
    });
    
    // 定期更新播放器状态
    const interval = setInterval(() => {
      if (player) {
        setPlayerState({
          isPlaying: !player.isPaused(),
          currentTime: player.time(),
          duration: player.duration(),
          bufferLevel: player.getBufferLength(),
          bitrate: Math.round(player.getAverageThroughput('video') / 1000),
          qualityIndex: player.getQualityFor('video')
        });
      }
    }, 1000);
    
    return () => {
      clearInterval(interval);
      if (playerRef.current) {
        playerRef.current.destroy();
        playerRef.current = null;
      }
    };
  }, [dashUrl]);

  // 格式化时间显示
  const formatTime = (seconds) => {
    if (isNaN(seconds) || seconds < 0) return '00:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="dash-player-container">
      {dashUrl ? (
        <div className="dash-player-wrapper">
          <video
            ref={videoRef}
            className="dash-video-player"
            controls
            autoPlay
          >
            您的浏览器不支持视频播放。
          </video>
          
          <div className="dash-player-info">
            <div className="dash-info-item">
              <span className="dash-info-label">状态:</span>
              <span className="dash-info-value">{playerState.isPlaying ? '播放中' : '已暂停'}</span>
            </div>
            <div className="dash-info-item">
              <span className="dash-info-label">时间:</span>
              <span className="dash-info-value">
                {formatTime(playerState.currentTime)} / {formatTime(playerState.duration)}
              </span>
            </div>
            <div className="dash-info-item">
              <span className="dash-info-label">缓冲:</span>
              <span className="dash-info-value">{playerState.bufferLevel.toFixed(1)}秒</span>
            </div>
            <div className="dash-info-item">
              <span className="dash-info-label">比特率:</span>
              <span className="dash-info-value">{playerState.bitrate} kbps</span>
            </div>
            <div className="dash-info-item">
              <span className="dash-info-label">清晰度:</span>
              <span className="dash-info-value">质量 {playerState.qualityIndex}</span>
            </div>
            <div className="dash-info-item">
              <span className="dash-info-label">URL:</span>
              <span className="dash-info-value dash-url">{dashUrl}</span>
            </div>
          </div>
        </div>
      ) : (
        <div className="dash-placeholder">
          <div className="dash-placeholder-content">
            <div className="dash-placeholder-icon">📺</div>
            <h3>请输入DASH流URL</h3>
            <p>例如: http://localhost:8080/tv002/tv002.mpd</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default DashPlayer;
