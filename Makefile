.PHONY: install dev test lint format clean sync

# 使用uv安装依赖
install:
	uv sync

# 开发模式安装
dev:
	uv sync --extra dev

# 运行测试
test:
	uv run pytest tests/ -v

# 代码检查
lint:
	uv run flake8 subtitle_genius/
	uv run mypy subtitle_genius/

# 代码格式化
format:
	uv run black subtitle_genius/ tests/

# 同步依赖
sync:
	uv lock
	uv sync

# 清理
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

# 构建
build:
	uv build

# 运行示例
example:
	uv run subtitle-genius --help

# 添加依赖
add:
	uv add $(PACKAGE)

# 添加开发依赖
add-dev:
	uv add --dev $(PACKAGE)
