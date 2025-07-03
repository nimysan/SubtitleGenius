#!/bin/bash
# 启动前端开发服务器

# 确保脚本在frontend目录下执行
cd "$(dirname "$0")"

echo "正在启动SubtitleGenius前端开发服务器..."

# 检查node_modules是否存在，如果不存在则安装依赖
if [ ! -d "node_modules" ]; then
  echo "正在安装依赖..."
  npm install
fi

# 启动开发服务器
echo "启动开发服务器..."
npm start
