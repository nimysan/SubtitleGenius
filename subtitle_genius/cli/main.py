"""命令行主程序"""

import asyncio
import click
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from loguru import logger

from ..core.generator import SubtitleGenerator
from ..stream.processor import StreamProcessor
from ..core.config import config

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """SubtitleGenius - 基于GenAI的实时MP4音频流字幕生成工具"""
    # 配置日志
    logger.add(
        config.log_file,
        level=config.log_level,
        rotation="10 MB",
        retention="7 days"
    )


@main.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', help='输出文件路径')
@click.option('--model', '-m', default='openai-whisper', 
              type=click.Choice(['openai-whisper', 'openai-gpt', 'claude']),
              help='使用的AI模型')
@click.option('--language', '-l', default='zh-CN', help='字幕语言')
@click.option('--format', '-f', default='srt', 
              type=click.Choice(['srt', 'vtt']),
              help='字幕格式')
def process(input_file, output, model, language, format):
    """处理单个音频/视频文件生成字幕"""
    
    async def _process():
        input_path = Path(input_file)
        
        if output:
            output_path = Path(output)
        else:
            output_path = input_path.with_suffix(f'.{format}')
        
        console.print(f"[green]处理文件:[/green] {input_path}")
        console.print(f"[green]使用模型:[/green] {model}")
        console.print(f"[green]输出格式:[/green] {format}")
        
        try:
            generator = SubtitleGenerator(
                model=model,
                language=language,
                output_format=format
            )
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("正在生成字幕...", total=None)
                
                if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    subtitles = await generator.process_video(input_path, output_path)
                else:
                    subtitles = await generator.generate_from_file(input_path)
                    generator.save_subtitles(subtitles, output_path)
                
                progress.update(task, description=f"完成! 生成了 {len(subtitles)} 条字幕")
            
            console.print(f"[green]✓[/green] 字幕已保存到: {output_path}")
            
        except Exception as e:
            console.print(f"[red]✗ 处理失败:[/red] {e}")
            raise click.ClickException(str(e))
    
    asyncio.run(_process())


@main.command()
@click.option('--input', '-i', help='输入流URL或设备')
@click.option('--output', '-o', help='输出文件路径')
@click.option('--model', '-m', default='openai-whisper',
              type=click.Choice(['openai-whisper', 'openai-gpt', 'claude']),
              help='使用的AI模型')
@click.option('--language', '-l', default='zh-CN', help='字幕语言')
@click.option('--format', '-f', default='srt',
              type=click.Choice(['srt', 'vtt']),
              help='字幕格式')
def stream(input, output, model, language, format):
    """实时处理音频流生成字幕"""
    
    async def _stream():
        console.print(f"[green]开始实时字幕生成[/green]")
        console.print(f"[green]使用模型:[/green] {model}")
        console.print(f"[green]输入源:[/green] {input or '麦克风'}")
        
        try:
            generator = SubtitleGenerator(
                model=model,
                language=language,
                output_format=format
            )
            
            stream_processor = StreamProcessor()
            
            # 选择音频流源
            if input:
                if input.startswith('rtmp://'):
                    audio_stream = stream_processor.process_rtmp_stream(input)
                else:
                    audio_stream = stream_processor.process_file_stream(input)
            else:
                audio_stream = stream_processor.start_microphone_stream()
            
            subtitles = []
            
            console.print("[yellow]开始监听音频流... (按 Ctrl+C 停止)[/yellow]")
            
            async for subtitle in generator.generate_realtime(audio_stream):
                console.print(f"[cyan][{subtitle.start:.1f}s][/cyan] {subtitle.text}")
                subtitles.append(subtitle)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]停止实时处理[/yellow]")
            
            if output and subtitles:
                generator.save_subtitles(subtitles, Path(output))
                console.print(f"[green]✓[/green] 字幕已保存到: {output}")
                
        except Exception as e:
            console.print(f"[red]✗ 流处理失败:[/red] {e}")
            raise click.ClickException(str(e))
    
    asyncio.run(_stream())


@main.command()
@click.option('--input-dir', '-i', required=True, type=click.Path(exists=True),
              help='输入目录')
@click.option('--output-dir', '-o', required=True, type=click.Path(),
              help='输出目录')
@click.option('--model', '-m', default='openai-whisper',
              type=click.Choice(['openai-whisper', 'openai-gpt', 'claude']),
              help='使用的AI模型')
@click.option('--language', '-l', default='zh-CN', help='字幕语言')
@click.option('--format', '-f', default='srt',
              type=click.Choice(['srt', 'vtt']),
              help='字幕格式')
def batch(input_dir, output_dir, model, language, format):
    """批量处理目录中的音频/视频文件"""
    
    async def _batch():
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 支持的文件格式
        supported_formats = {'.mp4', '.avi', '.mov', '.mkv', '.wav', '.mp3', '.m4a'}
        
        # 查找所有支持的文件
        files = []
        for ext in supported_formats:
            files.extend(input_path.glob(f'*{ext}'))
            files.extend(input_path.glob(f'*{ext.upper()}'))
        
        if not files:
            console.print(f"[yellow]在 {input_path} 中未找到支持的文件[/yellow]")
            return
        
        console.print(f"[green]找到 {len(files)} 个文件待处理[/green]")
        
        generator = SubtitleGenerator(
            model=model,
            language=language,
            output_format=format
        )
        
        with Progress(console=console) as progress:
            task = progress.add_task("批量处理中...", total=len(files))
            
            for file_path in files:
                try:
                    output_file = output_path / f"{file_path.stem}.{format}"
                    
                    progress.update(task, description=f"处理: {file_path.name}")
                    
                    if file_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                        await generator.process_video(file_path, output_file)
                    else:
                        subtitles = await generator.generate_from_file(file_path)
                        generator.save_subtitles(subtitles, output_file)
                    
                    console.print(f"[green]✓[/green] {file_path.name} -> {output_file.name}")
                    
                except Exception as e:
                    console.print(f"[red]✗[/red] {file_path.name}: {e}")
                
                progress.advance(task)
        
        console.print(f"[green]批量处理完成![/green]")
    
    asyncio.run(_batch())


if __name__ == '__main__':
    main()
