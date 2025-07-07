#!/bin/bash
# 从webm文件提取wav音频

# 检查参数
if [ $# -lt 1 ]; then
    echo "用法: $0 <输入webm文件> [输出wav文件] [采样率] [声道数]"
    echo "示例: $0 input.webm output.wav 16000 1"
    exit 1
fi

# 参数设置
INPUT_FILE=$1
OUTPUT_FILE=${2:-"${INPUT_FILE%.*}.wav"}  # 如果未提供输出文件名，使用输入文件名.wav
SAMPLE_RATE=${3:-16000}  # 默认采样率16kHz
CHANNELS=${4:-1}  # 默认单声道

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "错误: 输入文件 '$INPUT_FILE' 不存在"
    exit 1
fi

# 提取音频
echo "从 '$INPUT_FILE' 提取音频到 '$OUTPUT_FILE'..."
echo "采样率: $SAMPLE_RATE Hz, 声道数: $CHANNELS"

ffmpeg -i "$INPUT_FILE" -vn -acodec pcm_s16le -ar $SAMPLE_RATE -ac $CHANNELS "$OUTPUT_FILE"

# 检查命令是否成功
if [ $? -eq 0 ]; then
    echo "提取成功: '$OUTPUT_FILE'"
    echo "文件信息:"
    ffprobe -v error -show_entries stream=codec_name,sample_rate,channels -of default=noprint_wrappers=1 "$OUTPUT_FILE"
else
    echo "提取失败"
    exit 1
fi
