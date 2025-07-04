# 前端到后端参数传递实现总结

## 概述
实现了从前端ControlPanel到后端WebSocket服务器的完整参数传递链路，支持动态配置字幕处理流程。

## 参数列表

### 前端参数
1. **视频语言** (`selectedLanguage`): 视频原始语言，默认 'ar'
2. **AI模型** (`selectedModel`): 使用的AI模型，支持 'whisper', 'transcribe', 'claude'
3. **翻译目标语言** (`targetLanguage`): 翻译的目标语言，默认 'en'
4. **启用智能纠错** (`enableCorrection`): 是否启用纠错功能，默认 true
5. **启用多语言翻译** (`enableTranslation`): 是否启用翻译功能，默认 true

### 传递路径

#### 1. 前端ControlPanel组件
```javascript
// ControlPanel.js
const ControlPanel = ({ 
  selectedLanguage = 'ar',
  selectedModel = 'whisper', 
  enableCorrection = true,
  enableTranslation = true,
  targetLanguage = 'en',
  onSettingsChange,
  // ... 其他props
}) => {
  // 状态管理
  const [language, setLanguage] = useState(selectedLanguage);
  const [correction, setCorrection] = useState(enableCorrection);
  const [translation, setTranslation] = useState(enableTranslation);
  const [translationTarget, setTranslationTarget] = useState(targetLanguage);
  
  // 参数变更时通知父组件
  useEffect(() => {
    if (onSettingsChange) {
      onSettingsChange(language, model, realtime, debug, correction, translation, translationTarget);
    }
  }, [language, model, realtime, debug, correction, translation, translationTarget, onSettingsChange]);
}
```

#### 2. App.js主组件
```javascript
// App.js
const handleSettingsChange = (language, model, realtime, debug, correction, translation, translationTarget) => {
  // 更新所有状态
  setSelectedLanguage(language);
  setSelectedModel(model);
  setEnableCorrection(correction);
  setEnableTranslation(translation);
  setTargetLanguage(translationTarget);
  
  // 如果关键参数变更，重新建立WebSocket连接
  if (hasKeyParameterChange && socket && socket.readyState === WebSocket.OPEN) {
    socket.close();
    setSocket(null);
  }
};

// WebSocket连接构建
const connectWebSocket = useCallback(() => {
  let wsUrl = `ws://localhost:8000/ws/${selectedModel}`;
  wsUrl += `?language=${selectedLanguage}&correction=${enableCorrection}&translation=${enableTranslation}&target_language=${targetLanguage}`;
  
  const newSocket = createWebSocketConnection(wsUrl, ...);
  return newSocket;
}, [selectedLanguage, selectedModel, enableCorrection, enableTranslation, targetLanguage]);
```

#### 3. 后端WebSocket服务器
```python
# websocket_server.py
@app.websocket("/ws/whisper")
async def websocket_whisper_endpoint(
    websocket: WebSocket, 
    language: str = Query("ar"),
    correction: bool = Query(True),
    translation: bool = Query(True),
    target_language: str = Query("en"),
    filename: str = Query(None)
):
    # 接收并使用参数
    logger.info(f"接收到的参数:")
    logger.info(f"  - 视频语言: {language}")
    logger.info(f"  - 启用纠错: {correction}")
    logger.info(f"  - 启用翻译: {translation}")
    logger.info(f"  - 翻译目标语言: {target_language}")
    
    # 发送连接确认（包含参数信息）
    await websocket.send_json({
        "type": "connection",
        "status": "connected",
        "client_id": client_id,
        "model": "whisper",
        "language": language,
        "correction_enabled": correction,
        "translation_enabled": translation,
        "target_language": target_language
    })
```

#### 4. 处理流程配置
```python
# 动态语言配置
async def update_whisper_model_language(language: str):
    config = create_whisper_config(language)  # 根据语言创建配置
    sagemaker_params = get_sagemaker_whisper_params(language)  # 获取SageMaker参数
    
# 条件处理流程
async def send_subtitle(websocket, subtitle, client_id, 
                       language="ar", enable_correction=True, 
                       enable_translation=True, target_language="en"):
    # 步骤1: 条件纠错
    if enable_correction and subtitle.text.strip() and correction_service:
        correction_input = CorrectionInput(
            current_subtitle=subtitle.text,
            scene_description=get_correction_scene_description(language),
            language=language
        )
        correction_result = await correction_service.correct(correction_input)
        # 应用纠错结果...
    
    # 步骤2: 条件翻译
    if enable_translation and corrected_text.strip():
        translation_result = await translation_manager.translate(
            text=corrected_text,
            target_lang=target_language,
            service="bedrock"
        )
        # 应用翻译结果...
```

## 语言特定配置

### Whisper模型配置
```python
# whisper_language_config.py
LANGUAGE_CONFIGS = {
    "ar": LanguageConfig(  # 阿拉伯语优化
        voice_threshold=0.005,
        chunk_duration=25.0,
        initial_prompt="هذا نص باللغة العربية",
        no_speech_threshold=0.5
    ),
    "en": LanguageConfig(  # 英语优化
        voice_threshold=0.01,
        chunk_duration=30.0,
        initial_prompt="This is English text"
    ),
    "zh": LanguageConfig(  # 中文优化
        voice_threshold=0.008,
        chunk_duration=20.0,
        initial_prompt="这是中文文本"
    )
}
```

## 测试工具

### 1. Python测试脚本
```bash
python test_websocket_params.py
```

### 2. HTML测试页面
```bash
# 在浏览器中打开
frontend/test_params.html
```

## 使用流程

1. **前端设置**: 用户在ControlPanel中调整参数
2. **参数传递**: 参数通过props和state传递到App.js
3. **WebSocket连接**: 参数作为查询参数附加到WebSocket URL
4. **后端接收**: WebSocket服务器接收并解析参数
5. **流程配置**: 根据参数动态配置处理流程
6. **条件处理**: 根据参数决定是否执行纠错和翻译
7. **结果返回**: 处理结果返回前端显示

## 关键特性

- ✅ **动态参数传递**: 支持实时参数变更
- ✅ **语言特定优化**: 不同语言使用优化的模型参数
- ✅ **条件处理**: 根据用户选择启用/禁用功能
- ✅ **连接重建**: 关键参数变更时自动重新连接
- ✅ **调试支持**: 详细的日志记录和测试工具
- ✅ **多端点支持**: whisper, transcribe, claude三个端点
- ✅ **参数验证**: 前后端参数一致性验证

## 调试建议

1. 检查浏览器控制台的WebSocket连接日志
2. 查看后端服务器日志确认参数接收
3. 使用测试工具验证参数传递
4. 确认WebSocket URL格式正确
5. 验证参数类型匹配（boolean vs string）
