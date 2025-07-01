"""Amazon Transcribe 模型实现"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import List, Any, Optional
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import uuid

from .base import BaseModel
from ..subtitle.models import Subtitle
from ..core.config import config


class TranscribeModel(BaseModel):
    """Amazon Transcribe 模型"""
    
    def __init__(self, region_name: str = "us-east-1"):
        """初始化 Transcribe 客户端
        
        Args:
            region_name: AWS 区域名称
        """
        self.region_name = region_name
        self.transcribe_client = None
        self.s3_client = None
        self.bucket_name = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """初始化 AWS 客户端"""
        try:
            # 初始化 Transcribe 客户端
            self.transcribe_client = boto3.client(
                'transcribe',
                region_name=self.region_name
            )
            
            # 初始化 S3 客户端 (Transcribe 需要从 S3 读取音频文件)
            self.s3_client = boto3.client(
                's3',
                region_name=self.region_name
            )
            
            # 设置默认 S3 存储桶名称
            self.bucket_name = getattr(config, 'aws_s3_bucket', 'subtitle-genius-temp')
            
        except Exception as e:
            print(f"初始化 AWS 客户端失败: {e}")
    
    def is_available(self) -> bool:
        """检查模型是否可用"""
        try:
            if not self.transcribe_client:
                return False
            
            # 测试 AWS 凭证
            self.transcribe_client.list_transcription_jobs(MaxResults=1)
            return True
            
        except (NoCredentialsError, ClientError) as e:
            print(f"AWS Transcribe 不可用: {e}")
            return False
        except Exception as e:
            print(f"检查 Transcribe 可用性失败: {e}")
            return False
    
    async def transcribe(self, audio_data: Any, language: str = "zh-CN") -> List[Subtitle]:
        """使用 Amazon Transcribe 转录音频
        
        Args:
            audio_data: 音频文件路径或音频数据
            language: 语言代码
            
        Returns:
            字幕列表
        """
        if not self.is_available():
            raise RuntimeError("Amazon Transcribe 不可用，请检查 AWS 凭证配置")
        
        try:
            # 处理音频文件
            audio_path = self._prepare_audio_file(audio_data)
            
            # 上传到 S3
            s3_uri = await self._upload_to_s3(audio_path)
            
            # 启动转录任务
            job_name = f"subtitle-genius-{uuid.uuid4().hex[:8]}"
            transcription_job = await self._start_transcription_job(
                job_name, s3_uri, language
            )
            
            # 等待任务完成
            result = await self._wait_for_transcription_completion(job_name)
            
            # 解析结果
            subtitles = self._parse_transcription_result(result)
            
            # 清理临时文件
            await self._cleanup_s3_file(s3_uri)
            
            return subtitles
            
        except Exception as e:
            print(f"Amazon Transcribe 转录失败: {e}")
            raise
    
    def _prepare_audio_file(self, audio_data: Any) -> Path:
        """准备音频文件"""
        if isinstance(audio_data, (str, Path)):
            audio_path = Path(audio_data)
            if audio_path.exists():
                return audio_path
        
        # 如果是音频数据，保存为临时文件
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_path = Path(temp_file.name)
        
        # 这里需要根据实际的音频数据格式进行处理
        # 假设 audio_data 是音频文件路径
        if hasattr(audio_data, 'save'):
            audio_data.save(temp_path)
        else:
            # 复制文件
            import shutil
            shutil.copy2(audio_data, temp_path)
        
        return temp_path
    
    async def _upload_to_s3(self, audio_path: Path) -> str:
        """上传音频文件到 S3"""
        try:
            # 确保存储桶存在
            await self._ensure_bucket_exists()
            
            # 生成 S3 对象键
            object_key = f"audio/{uuid.uuid4().hex}.{audio_path.suffix[1:]}"
            
            # 上传文件
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.s3_client.upload_file,
                str(audio_path),
                self.bucket_name,
                object_key
            )
            
            # 返回 S3 URI
            s3_uri = f"s3://{self.bucket_name}/{object_key}"
            print(f"音频文件已上传到: {s3_uri}")
            return s3_uri
            
        except Exception as e:
            print(f"上传到 S3 失败: {e}")
            raise
    
    async def _ensure_bucket_exists(self):
        """确保 S3 存储桶存在"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.s3_client.head_bucket,
                Bucket=self.bucket_name
            )
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                # 存储桶不存在，创建它
                print(f"创建 S3 存储桶: {self.bucket_name}")
                await loop.run_in_executor(
                    None,
                    self.s3_client.create_bucket,
                    Bucket=self.bucket_name
                )
            else:
                raise
    
    async def _start_transcription_job(
        self, 
        job_name: str, 
        s3_uri: str, 
        language: str
    ) -> dict:
        """启动转录任务"""
        # 转换语言代码
        language_code = self._convert_language_code(language)
        
        job_params = {
            'TranscriptionJobName': job_name,
            'LanguageCode': language_code,
            'Media': {'MediaFileUri': s3_uri},
            'OutputBucketName': self.bucket_name,
            'Settings': {
                'ShowSpeakerLabels': False,  # 不需要说话人标识
                'MaxSpeakerLabels': 2,
                'ChannelIdentification': False,
            }
        }
        
        # 如果支持自动语言检测
        if language_code == 'auto':
            job_params.pop('LanguageCode')
            job_params['IdentifyLanguage'] = True
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            self.transcribe_client.start_transcription_job,
            **job_params
        )
        
        print(f"转录任务已启动: {job_name}")
        return response
    
    async def _wait_for_transcription_completion(self, job_name: str) -> dict:
        """等待转录任务完成"""
        max_wait_time = 300  # 最大等待时间 5 分钟
        wait_interval = 5    # 检查间隔 5 秒
        elapsed_time = 0
        
        loop = asyncio.get_event_loop()
        
        while elapsed_time < max_wait_time:
            response = await loop.run_in_executor(
                None,
                self.transcribe_client.get_transcription_job,
                TranscriptionJobName=job_name
            )
            
            status = response['TranscriptionJob']['TranscriptionJobStatus']
            
            if status == 'COMPLETED':
                print(f"转录任务完成: {job_name}")
                return response
            elif status == 'FAILED':
                failure_reason = response['TranscriptionJob'].get('FailureReason', '未知错误')
                raise RuntimeError(f"转录任务失败: {failure_reason}")
            
            print(f"转录任务进行中... ({elapsed_time}s)")
            await asyncio.sleep(wait_interval)
            elapsed_time += wait_interval
        
        raise TimeoutError(f"转录任务超时: {job_name}")
    
    def _parse_transcription_result(self, transcription_response: dict) -> List[Subtitle]:
        """解析转录结果"""
        try:
            # 获取结果文件 URI
            result_uri = transcription_response['TranscriptionJob']['Transcript']['TranscriptFileUri']
            
            # 下载结果文件
            import requests
            response = requests.get(result_uri)
            response.raise_for_status()
            
            result_data = response.json()
            
            # 解析字幕
            subtitles = []
            
            if 'results' in result_data and 'items' in result_data['results']:
                items = result_data['results']['items']
                
                # 按时间戳分组创建字幕
                current_subtitle = None
                current_words = []
                
                for item in items:
                    if item['type'] == 'pronunciation':
                        word = item['alternatives'][0]['content']
                        start_time = float(item['start_time'])
                        end_time = float(item['end_time'])
                        
                        if current_subtitle is None:
                            current_subtitle = {'start': start_time, 'words': [word]}
                        else:
                            # 如果时间间隔太大，创建新的字幕段
                            if start_time - current_subtitle.get('end', start_time) > 2.0:
                                # 完成当前字幕
                                if current_subtitle['words']:
                                    subtitles.append(Subtitle(
                                        start=current_subtitle['start'],
                                        end=current_subtitle.get('end', start_time),
                                        text=' '.join(current_subtitle['words'])
                                    ))
                                
                                # 开始新字幕
                                current_subtitle = {'start': start_time, 'words': [word]}
                            else:
                                current_subtitle['words'].append(word)
                        
                        current_subtitle['end'] = end_time
                
                # 添加最后一个字幕
                if current_subtitle and current_subtitle['words']:
                    subtitles.append(Subtitle(
                        start=current_subtitle['start'],
                        end=current_subtitle['end'],
                        text=' '.join(current_subtitle['words'])
                    ))
            
            print(f"解析得到 {len(subtitles)} 条字幕")
            return subtitles
            
        except Exception as e:
            print(f"解析转录结果失败: {e}")
            return []
    
    async def _cleanup_s3_file(self, s3_uri: str):
        """清理 S3 临时文件"""
        try:
            # 解析 S3 URI
            parts = s3_uri.replace('s3://', '').split('/', 1)
            bucket = parts[0]
            key = parts[1]
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.s3_client.delete_object,
                Bucket=bucket,
                Key=key
            )
            
            print(f"已清理 S3 临时文件: {s3_uri}")
            
        except Exception as e:
            print(f"清理 S3 文件失败: {e}")
    
    def _convert_language_code(self, language: str) -> str:
        """转换语言代码为 Transcribe 支持的格式"""
        language_mapping = {
            'zh-CN': 'zh-CN',
            'zh': 'zh-CN',
            'en': 'en-US',
            'en-US': 'en-US',
            'ar': 'ar-SA',
            'ja': 'ja-JP',
            'ko': 'ko-KR',
            'fr': 'fr-FR',
            'de': 'de-DE',
            'es': 'es-ES',
            'ru': 'ru-RU',
            'auto': 'auto'
        }
        
        return language_mapping.get(language, 'en-US')
