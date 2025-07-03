#!/bin/bash

# SubtitleGenius Frontend 启动脚本

echo "🎬 启动 SubtitleGenius Frontend..."

# 检查Node.js是否安装
if ! command -v node &> /dev/null; then
    echo "❌ Node.js 未安装，请先安装 Node.js"
    echo "访问 https://nodejs.org/ 下载安装"
    exit 1
fi

# 检查npm是否安装
if ! command -v npm &> /dev/null; then
    echo "❌ npm 未安装，请先安装 npm"
    exit 1
fi

# 进入frontend目录
cd "$(dirname "$0")"

# 检查package.json是否存在
if [ ! -f "package.json" ]; then
    echo "❌ package.json 文件不存在"
    exit 1
fi

# 安装依赖（如果node_modules不存在）
if [ ! -d "node_modules" ]; then
    echo "📦 安装依赖..."
    npm install
    
    if [ $? -ne 0 ]; then
        echo "❌ 依赖安装失败"
        exit 1
    fi
fi

# 启动开发服务器
echo "🚀 启动开发服务器..."
echo "应用将在 http://localhost:3000 打开"
echo "按 Ctrl+C 停止服务器"

npm start
