import React, { useEffect, useRef } from 'react';
import './SubtitleDisplay.css';

const SubtitleDisplay = ({ subtitles, currentTime }) => {
  const subtitleRef = useRef(null);

  // 获取当前时间应该显示的字幕
  const getCurrentSubtitle = () => {
    return subtitles.find(subtitle => 
      currentTime >= subtitle.start && currentTime <= subtitle.end
    );
  };

  // 格式化时间显示
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // 自动滚动到当前字幕
  useEffect(() => {
    const currentSubtitle = getCurrentSubtitle();
    if (currentSubtitle && subtitleRef.current) {
      const currentElement = subtitleRef.current.querySelector(`[data-id="${currentSubtitle.id}"]`);
      if (currentElement) {
        currentElement.scrollIntoView({ 
          behavior: 'smooth', 
          block: 'center' 
        });
      }
    }
  }, [currentTime, subtitles]);

  const currentSubtitle = getCurrentSubtitle();
  
  // 反向字幕列表（最新的在上面）
  const reversedSubtitles = [...subtitles].reverse();

  return (
    <div className="subtitle-display-container">
      <div className="subtitle-header">
        <h3>字幕历史</h3>
        <div className="subtitle-info">
          <span className="current-time">当前时间: {formatTime(currentTime)}</span>
          <span className="subtitle-count">字幕数量: {subtitles.length}</span>
        </div>
      </div>

      {/* 字幕列表 - 反向显示，最新的在上面 */}
      <div className="subtitle-list" ref={subtitleRef}>
        {subtitles.length > 0 ? (
          reversedSubtitles.map((subtitle) => (
            <div
              key={subtitle.id}
              data-id={subtitle.id}
              className={`subtitle-item ${
                currentSubtitle && currentSubtitle.id === subtitle.id ? 'active' : ''
              }`}
            >
              <div className="subtitle-timeline">
                <span className="subtitle-time-range">
                  {formatTime(subtitle.start)} - {formatTime(subtitle.end)}
                </span>
              </div>
              <div className="subtitle-content">
                <div className="original-text">{subtitle.text}</div>
                {subtitle.translated_text && (
                  <div className="translated-text">{subtitle.translated_text}</div>
                )}
              </div>
            </div>
          ))
        ) : (
          <div className="empty-subtitles">
            <div className="empty-icon">📝</div>
            <p>暂无字幕内容</p>
            <p className="empty-hint">上传视频后点击"生成字幕"开始处理</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default SubtitleDisplay;
