# DASH服务器使用说明

这个Python服务器可以动态修改DASH MPD文件，自动添加字幕轨道。

## 功能特性

- 🎬 自动为MPD文件添加字幕AdaptationSet
- 📁 支持多个节目目录
- 🌐 提供Web界面查看可用节目
- 📝 支持VTT字幕格式
- 🔄 动态XML处理，保持原始MPD结构

## 目录结构

```
dash_output/
├── tv001/
│   ├── manifest.mpd      # 原始MPD文件
│   ├── tv001.vtt         # 字幕文件
│   ├── video_1.mp4       # 视频文件
│   └── audio_1.mp4       # 音频文件
├── tv002/
│   └── ...
└── tv003/
    └── ...
```

## 安装依赖

```bash
pip install flask
# 或使用项目的requirements文件
pip install -r requirements_dash.txt
```

## 启动服务器

### 方法1: 使用启动脚本（推荐）
```bash
python start_dash_server.py
```

### 方法2: 直接启动
```bash
python dash_server.py
```

## 访问地址

- **首页**: http://localhost:8080
  - 显示所有可用节目列表
  - 提供MPD文件链接

- **MPD文件**: http://localhost:8080/tv001.mpd
  - 自动添加字幕轨道的MPD文件
  - 支持CORS跨域访问

- **媒体文件**: http://localhost:8080/tv001/video_1.mp4
  - 直接访问媒体片段

- **字幕文件**: http://localhost:8080/subtitles/tv001.vtt
  - VTT格式字幕文件

## 字幕轨道配置

服务器会自动在MPD文件中添加以下字幕轨道：

```xml
<AdaptationSet mimeType="text/vtt" lang="en">
    <Representation id="caption_en" bandwidth="256">
        <BaseURL>/subtitles/tv001.vtt</BaseURL> 
    </Representation>
</AdaptationSet>
```

## 字幕文件查找顺序

1. `dash_output/节目ID/节目ID.vtt`
2. `dash_output/节目ID/subtitle.vtt`
3. `dash_output/节目ID/subtitles.vtt`
4. 如果都没找到，返回示例字幕

## 自定义配置

在 `dash_server.py` 中可以修改：

```python
# DASH文件目录
DASH_OUTPUT_DIR = "dash_output"

# 字幕语言和属性
subtitle_adaptation_set.set("lang", "en")  # 改为其他语言
representation.set("bandwidth", "256")     # 调整带宽
```

## 测试示例

1. 创建测试目录：
```bash
mkdir -p dash_output/tv001
```

2. 放入MPD文件和字幕文件

3. 启动服务器：
```bash
python start_dash_server.py
```

4. 访问测试：
```bash
curl http://localhost:8080/tv001.mpd
```

## 播放器集成

### Dash.js 示例
```javascript
var player = dashjs.MediaPlayer().create();
player.initialize(document.querySelector("#videoPlayer"), 
                 "http://localhost:8080/tv001.mpd", true);
```

### Video.js 示例
```javascript
var player = videojs('video-player');
player.src({
    src: 'http://localhost:8080/tv001.mpd',
    type: 'application/dash+xml'
});
```

## 故障排除

### 1. 端口被占用
```bash
# 修改端口
app.run(host='0.0.0.0', port=8081, debug=True)
```

### 2. CORS问题
服务器已配置CORS头，支持跨域访问。

### 3. XML解析错误
检查原始MPD文件格式是否正确。

### 4. 字幕不显示
- 确认字幕文件存在
- 检查VTT文件格式
- 验证播放器是否支持字幕

## 日志调试

服务器会输出详细日志：
```
INFO:__main__:提供MPD文件: dash_output/tv001/manifest.mpd
INFO:__main__:成功为 tv001 添加字幕轨道
INFO:__main__:提供字幕文件: dash_output/tv001/tv001.vtt
```

## 扩展功能

可以进一步扩展：
- 支持多语言字幕
- 动态字幕生成
- 字幕样式配置
- 缓存优化
- 负载均衡
