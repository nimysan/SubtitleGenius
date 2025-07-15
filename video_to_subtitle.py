#!/usr/bin/env python3
"""
视频字幕提取器

这个脚本接收一个视频文件或流媒体URL作为输入，
使用ffmpeg提取音频，然后使用ContinuousAudioProcessor
处理音频并生成字幕，最后将字幕追加到目标字幕文件。

支持的输入:
- 本地视频文件 (MP4, WebM, AVI等)
- HTTP/HTTPS流媒体链接:
  - HLS流 (.m3u8文件)
  - DASH流 (.mpd文件)
  - 其他ffmpeg支持的流媒体格式

用法:
    python video_to_subtitle.py input_video.mp4 output_subtitle.vtt [--language zh]
    python video_to_subtitle.py https://example.com/video.m3u8 output_subtitle.vtt [--language zh]
    python video_to_subtitle.py https://example.com/video.mpd output_subtitle.vtt [--language en]

DASH流示例:
    - https://dash.akamaized.net/dash264/TestCases/1a/netflix/exMPD_BIP_TC1.mpd
    - https://dash.akamaized.net/envivio/EnvivioDash3/manifest.mpd
    - https://media.axprod.net/TestVectors/v7-Clear/Manifest.mpd
"""

import argparse
import asyncio
import logging
import os
import subprocess
import sys
import numpy as np
import re
import urllib.request
import urllib.error
from typing import Dict, Any, Tuple, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("video_to_subtitle")

# 导入ContinuousAudioProcessor
from subtitle_genius.stream.continuous_audio_processor import ContinuousAudioProcessor

class VideoSubtitleExtractor:
    """视频字幕提取器"""
    
    def __init__(self, input_file: str, output_file: str, language: str = 'zh'):
        """
        初始化视频字幕提取器
        
        Args:
            input_file: 输入视频文件路径或URL
            output_file: 输出字幕文件路径
            language: 语言代码 (默认: zh)
        """
        self.input_file = input_file
        self.output_file = output_file
        self.language = language
        self.is_url = self._is_url(input_file)
        
        # 验证输入文件是否存在（仅对本地文件）
        if not self.is_url and not os.path.exists(input_file):
            raise FileNotFoundError(f"输入文件不存在: {input_file}")
        
        # 创建输出目录（如果不存在）
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 初始化字幕处理器
        self.processor = None
        self.stream_id = None
        
        logger.info(f"初始化完成: 输入={input_file} ({'URL' if self.is_url else '本地文件'}), 输出={output_file}, 语言={language}")
    
    def _is_url(self, path: str) -> bool:
        """
        检查路径是否为URL
        
        Args:
            path: 要检查的路径
            
        Returns:
            bool: 如果是URL则返回True，否则返回False
        """
        url_pattern = re.compile(r'^https?://')
        return bool(url_pattern.match(path))
    
    def _test_url_accessibility(self) -> Tuple[bool, Optional[str]]:
        """
        测试URL的可访问性
        
        Returns:
            Tuple[bool, Optional[str]]: (是否可访问, 错误信息)
        """
        if not self.is_url:
            return True, None
            
        try:
            # 创建请求对象
            req = urllib.request.Request(
                self.input_file,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
                }
            )
            
            # 尝试打开URL
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.getcode() == 200:
                    return True, None
                else:
                    return False, f"HTTP错误: {response.getcode()}"
                    
        except urllib.error.HTTPError as e:
            return False, f"HTTP错误: {e.code} - {e.reason}"
        except urllib.error.URLError as e:
            return False, f"URL错误: {e.reason}"
        except Exception as e:
            return False, f"未知错误: {str(e)}"
    
    async def initialize_processor(self):
        """初始化音频处理器"""
        # 创建配置
        config = {
            'language': self.language,
            'correction': True,
            'translation': False,
            'model': 'whisper',
            'client_id': 'video_subtitle_extractor'
        }
        
        # 创建处理器
        self.processor = ContinuousAudioProcessor(
            config=config,
            result_callback=self._on_subtitle_result
        )
        
        # 修改WebVTT文件路径为输出文件
        self.processor.webvtt_file = self.output_file
        
        # 初始化WebVTT文件
        self.processor._init_webvtt_file()
        
        # 启动流处理
        self.stream_id = await self.processor.start_stream('video_stream')
        logger.info(f"音频处理器初始化完成，流ID: {self.stream_id}")
    
    def _on_subtitle_result(self, result: Dict[str, Any]):
        """字幕结果回调"""
        if 'subtitle' in result and 'text' in result['subtitle']:
            subtitle = result['subtitle']
            logger.info(f"收到字幕: {subtitle['start']:.2f}s - {subtitle['end']:.2f}s: {subtitle['text']}")
            
            # 字幕已经通过ContinuousAudioProcessor的_write_subtitle_to_webvtt方法写入文件
            # 所以这里不需要额外的操作
    
    async def process(self):
        """处理视频文件或流媒体URL"""
        try:
            # 初始化处理器
            await self.initialize_processor()
            
            # 构建ffmpeg命令
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', self.input_file,  # 输入文件或URL
            ]
            
            # 对于URL，添加一些额外的参数以提高流媒体处理的稳定性
            if self.is_url:
                # 测试URL可访问性
                accessible, error_msg = self._test_url_accessibility()
                if not accessible:
                    logger.warning(f"URL可能无法访问: {error_msg}")
                    logger.warning("尝试继续处理，但可能会失败...")
                
                # 添加通用的流媒体处理参数
                ffmpeg_cmd.extend([
                    '-reconnect', '1',              # 断开连接时重新连接
                    '-reconnect_streamed', '1',     # 对于流媒体重新连接
                    '-reconnect_delay_max', '5',    # 最大重连延迟（秒）
                    '-protocol_whitelist', 'file,http,https,tcp,tls,crypto,data',  # 允许的协议，对DASH流很重要
                ])
                
                # 针对DASH流的特殊处理
                if ".mpd" in self.input_file.lower():
                    ffmpeg_cmd.extend([
                        '-allowed_extensions', 'ALL',   # 允许所有扩展名
                        '-timeout', '5000000',          # 增加超时时间
                        '-seekable', '0',               # 不可寻址
                        '-re',                          # 实时读取
                    ])
            
            # 添加音频处理参数
            ffmpeg_cmd.extend([
                '-vn',                   # 不处理视频
                '-f', 'f32le',           # 输出格式：32位浮点PCM
                '-acodec', 'pcm_f32le',  # 音频编码
                '-ar', '16000',          # 采样率：16kHz
                '-ac', '1',              # 单声道
                '-'                      # 输出到stdout
            ])
            
            logger.info(f"启动ffmpeg提取音频: {' '.join(ffmpeg_cmd)}")
            if self.is_url:
                # 检测流媒体类型
                stream_type = "未知类型"
                if ".m3u8" in self.input_file.lower():
                    stream_type = "HLS流"
                elif ".mpd" in self.input_file.lower():
                    stream_type = "DASH流"
                logger.info(f"正在处理流媒体URL ({stream_type}): {self.input_file}")
                logger.info("如果处理失败，请尝试以下DASH流示例:")
                logger.info("- https://dash.akamaized.net/dash264/TestCases/1a/netflix/exMPD_BIP_TC1.mpd")
                logger.info("- https://dash.akamaized.net/envivio/EnvivioDash3/manifest.mpd")
                logger.info("- https://media.axprod.net/TestVectors/v7-Clear/Manifest.mpd")
            
            # 启动ffmpeg进程
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # 读取并处理音频数据
            chunk_size = 4096  # 每次读取的字节数
            chunks_processed = 0
            total_bytes_processed = 0
            
            logger.info("开始处理音频数据...")
            
            while True:
                # 读取音频块
                audio_data = process.stdout.read(chunk_size)
                
                # 检查是否到达文件末尾
                if not audio_data:
                    break
                
                # 转换为numpy数组
                audio_np = np.frombuffer(audio_data, dtype=np.float32)
                
                # 处理音频数据
                await self.processor.process_audio_from_numpy(self.stream_id, audio_np)
                
                # 更新统计信息
                chunks_processed += 1
                total_bytes_processed += len(audio_data)
                
                # 每处理100个块打印一次进度
                if chunks_processed % 100 == 0:
                    logger.info(f"已处理 {chunks_processed} 个音频块，共 {total_bytes_processed / 1024:.2f} KB")
            
            # 等待ffmpeg进程结束
            process.wait()
            
            # 检查ffmpeg是否成功
            if process.returncode != 0:
                stderr = process.stderr.read().decode('utf-8', errors='ignore')
                logger.error(f"ffmpeg处理失败，返回码: {process.returncode}")
                logger.error(f"错误信息: {stderr}")
                
                # 提供更详细的错误分析和建议
                if "HTTP error 404" in stderr:
                    logger.error("错误分析: 请求的资源不存在 (HTTP 404)")
                    logger.error("建议: 请检查URL是否正确，或尝试使用其他DASH流URL")
                    logger.error("可用的DASH流示例:")
                    logger.error("- https://dash.akamaized.net/dash264/TestCases/1a/netflix/exMPD_BIP_TC1.mpd")
                    logger.error("- https://dash.akamaized.net/envivio/EnvivioDash3/manifest.mpd")
                    logger.error("- https://media.axprod.net/TestVectors/v7-Clear/Manifest.mpd")
                elif "Protocol not found" in stderr:
                    logger.error("错误分析: 协议不支持")
                    logger.error("建议: 请检查ffmpeg是否支持该协议，可能需要重新编译ffmpeg以支持更多协议")
                elif "Invalid data found" in stderr:
                    logger.error("错误分析: 无效的数据格式")
                    logger.error("建议: 请检查输入文件或URL的格式是否正确")
                
                return False
            
            # 停止流处理
            await self.processor.stop_stream(self.stream_id)
            
            logger.info(f"视频处理完成，共处理 {chunks_processed} 个音频块，{total_bytes_processed / 1024:.2f} KB")
            logger.info(f"字幕已保存到: {self.output_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"处理视频时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 尝试停止流处理
            if self.processor and self.stream_id:
                try:
                    await self.processor.stop_stream(self.stream_id)
                except:
                    pass
            
            return False

async def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='从视频或流媒体URL生成字幕')
    parser.add_argument('input', help='输入视频文件路径或流媒体URL (如HLS .m3u8或DASH .mpd)')
    parser.add_argument('output', help='输出字幕文件路径')
    parser.add_argument('--language', default='zh', help='语言代码 (默认: zh)')
    parser.add_argument('--debug', action='store_true', help='启用调试模式，显示更多日志信息')
    
    args = parser.parse_args()
    
    # 如果启用了调试模式，设置日志级别为DEBUG
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("调试模式已启用")
    
    # 创建并运行提取器
    extractor = VideoSubtitleExtractor(
        input_file=args.input,
        output_file=args.output,
        language=args.language
    )
    
    success = await extractor.process()
    
    # 设置退出码
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    # 运行主函数
    asyncio.run(main())
