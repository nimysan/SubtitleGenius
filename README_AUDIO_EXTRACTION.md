# SubtitleGenius 视频音频提取与流式字幕识别

本文档介绍了如何使用SubtitleGenius从视频中提取音频并进行流式字幕识别。

## 功能概述

SubtitleGenius现已支持从前端视频播放器中直接提取音频数据，并通过WebSocket实时传输到后端进行字幕识别。主要功能包括：

1. 从HTML5 video元素中提取音频流
2. 将音频数据转换为WAV格式
3. 通过WebSocket将音频数据流式传输到后端
4. 使用多种AI模型进行实时字幕识别
5. 将生成的字幕实时显示在前端界面

## 技术架构

### 前端部分

- **音频提取**：使用Web Audio API从视频元素中提取音频数据
- **音频处理**：使用AudioWorklet或ScriptProcessorNode处理音频数据
- **数据传输**：通过WebSocket将音频数据流式传输到后端
- **字幕显示**：实时接收并显示生成的字幕

### 后端部分

- **WebSocket服务器**：使用FastAPI实现WebSocket服务
- **音频处理**：使用AudioProcessor处理接收到的音频数据
- **字幕生成**：支持多种模型进行字幕识别
  - SageMaker Whisper模型：使用AWS SageMaker托管的Whisper模型进行流式处理
  - Amazon Transcribe：实时转录
  - Claude：AI辅助转录

## 安装与配置

### 前端安装

1. 进入前端目录：
   ```bash
   cd frontend
   ```

2. 安装依赖：
   ```bash
   npm install
   ```

3. 启动开发服务器：
   ```bash
   ./start_dev_server.sh
   ```
   或
   ```bash
   npm start
   ```

### 后端安装

1. 安装Python依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 配置环境变量：
   ```bash
   # 复制示例配置文件
   cp .env.example .env
   
   # 编辑配置文件，设置SageMaker端点和AWS区域
   # 使用你喜欢的编辑器打开.env文件
   nano .env
   ```

3. 启动WebSocket服务器：
   ```bash
   python start_websocket_server.py
   ```

## 使用方法

1. **上传视频**：
   - 点击"选择视频文件"按钮或将视频文件拖拽到上传区域
   - 支持MP4、WebM、AVI等常见视频格式

2. **配置设置**：
   - 选择目标语言（阿拉伯语、英语、中文等）
   - 选择AI模型（Amazon Transcribe、OpenAI Whisper、Claude）
   - 选择是否启用实时处理

3. **生成字幕**：
   - 点击"生成字幕"按钮开始处理
   - 系统会自动从视频中提取音频并发送到后端
   - 字幕会实时显示在界面上

4. **查看字幕**：
   - 字幕会根据视频播放进度自动显示
   - 可以暂停视频查看当前字幕

## 技术细节

### 音频提取流程

1. 创建AudioContext并连接到video元素
2. 使用AudioWorklet或ScriptProcessorNode处理音频数据
3. 将音频数据转换为WAV格式
4. 通过WebSocket发送到后端
5. 后端使用SageMaker托管的Whisper模型进行流式处理

### WebSocket通信协议

客户端发送：
- 二进制WAV格式音频数据

服务器响应：
```json
{
  "type": "subtitle",
  "subtitle": {
    "id": "unique_id",
    "start": 0.0,
    "end": 3.0,
    "text": "字幕文本内容"
  }
}
```

### 音频处理参数

- 采样率：16000 Hz（与后端模型匹配）
- 格式：16位PCM WAV
- 缓冲区大小：4096样本

## 故障排除

1. **无法提取音频**：
   - 检查浏览器是否支持Web Audio API
   - 确保视频文件包含音轨
   - 尝试使用不同的视频格式

2. **WebSocket连接失败**：
   - 确保后端服务器正在运行
   - 检查网络连接
   - 查看浏览器控制台错误信息

3. **字幕不显示**：
   - 检查选择的语言是否与视频语言匹配
   - 确保音频质量良好
   - 尝试使用不同的AI模型

## 开发者说明

### SageMaker Whisper配置

本项目使用AWS SageMaker托管的Whisper模型进行字幕识别。要使用此功能，您需要：

1. 在AWS SageMaker中部署Whisper模型
2. 在`.env`文件中配置以下参数：
   ```
   SAGEMAKER_ENDPOINT=your_sagemaker_endpoint_name
   AWS_REGION=your_aws_region
   SAGEMAKER_CHUNK_DURATION=30
   ```
3. 确保您的AWS凭证已正确配置（通过AWS CLI或环境变量）

### 添加新的AI模型

1. 在`subtitle_genius/models/`目录下创建新的模型类
2. 实现`transcribe_stream`或`transcribe_chunk`方法
3. 在`websocket_server.py`中添加新的WebSocket端点
4. 在前端ControlPanel组件中添加新的模型选项

### 自定义音频处理

可以通过修改`AudioUtils.js`中的函数来自定义音频处理逻辑，例如：
- 调整采样率
- 添加降噪处理
- 修改音频格式

## 许可证

本项目采用MIT许可证。详见LICENSE文件。
