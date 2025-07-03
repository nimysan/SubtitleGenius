# SubtitleGenius Frontend

基于React的SubtitleGenius前端界面，提供视频播放和字幕生成功能。

## 功能特性

- 🎬 **视频播放器** - 支持多种视频格式播放
- 📝 **字幕显示** - 实时显示和高亮当前字幕
- 🎛️ **控制面板** - 文件上传、字幕生成、设置选项
- 🌐 **多语言支持** - 特别优化阿拉伯语显示
- 📱 **响应式设计** - 适配桌面和移动设备
- 🎨 **现代UI** - 渐变背景、毛玻璃效果、流畅动画

## 快速开始

### 安装依赖

```bash
cd frontend
npm install
```

### 启动开发服务器

```bash
npm start
```

应用将在 http://localhost:3000 打开。

### 构建生产版本

```bash
npm run build
```

## 项目结构

```
frontend/
├── public/
│   ├── index.html          # HTML模板
│   └── manifest.json       # PWA配置
├── src/
│   ├── components/         # React组件
│   │   ├── VideoPlayer.js  # 视频播放器组件
│   │   ├── SubtitleDisplay.js # 字幕显示组件
│   │   └── ControlPanel.js # 控制面板组件
│   ├── App.js             # 主应用组件
│   ├── App.css            # 主应用样式
│   ├── index.js           # 应用入口
│   └── index.css          # 全局样式
└── package.json           # 项目配置
```

## 组件说明

### VideoPlayer 组件
- 支持多种视频格式 (MP4, WebM, AVI, MOV)
- 提供播放控制接口
- 实时时间更新回调
- 拖拽上传支持

### SubtitleDisplay 组件
- 实时字幕高亮显示
- 自动滚动到当前字幕
- 支持阿拉伯语从右到左显示
- 字幕列表浏览

### ControlPanel 组件
- 文件上传（支持拖拽）
- 字幕生成控制
- 语言和模型选择
- 实时处理开关

## 样式特性

- **渐变背景** - 现代化视觉效果
- **毛玻璃效果** - backdrop-filter模糊背景
- **响应式布局** - Grid布局适配不同屏幕
- **阿拉伯语支持** - RTL文本方向和专用字体
- **流畅动画** - CSS过渡和关键帧动画
- **自定义滚动条** - 统一的滚动条样式

## 浏览器支持

- Chrome 88+
- Firefox 94+
- Safari 15.4+
- Edge 88+

## 开发说明

### 添加新功能
1. 在 `src/components/` 创建新组件
2. 添加对应的CSS文件
3. 在 `App.js` 中引入和使用

### 样式规范
- 使用CSS变量定义主题色彩
- 遵循BEM命名规范
- 优先使用Flexbox和Grid布局
- 保持响应式设计

### 状态管理
当前使用React内置的useState，后续可考虑：
- Context API (中等复杂度)
- Redux Toolkit (高复杂度)
- Zustand (轻量级选择)

## 后续开发计划

- [ ] 连接后台API
- [ ] WebSocket实时通信
- [ ] 字幕编辑功能
- [ ] 导出字幕文件
- [ ] 主题切换
- [ ] 国际化支持
- [ ] PWA离线功能

## 故障排除

### 常见问题

1. **端口占用**
   ```bash
   # 使用不同端口启动
   PORT=3001 npm start
   ```

2. **依赖安装失败**
   ```bash
   # 清除缓存重新安装
   rm -rf node_modules package-lock.json
   npm install
   ```

3. **视频无法播放**
   - 检查视频格式是否支持
   - 确认浏览器支持HTML5视频
   - 检查文件路径和权限

## 许可证

MIT License
