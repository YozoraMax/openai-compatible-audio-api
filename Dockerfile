# OpenAI Compatible Audio API Docker Image
FROM ubuntu:22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    ffmpeg \
    libsndfile1 \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制requirements.txt并安装Python依赖
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 尝试安装可选的预编译包（如果失败会跳过）
RUN pip3 install editdistance --only-binary=all --prefer-binary || echo "editdistance skipped" && \
    pip3 install pyworld --only-binary=all --prefer-binary || echo "pyworld skipped"

# 复制项目文件
COPY . .

# 创建模型缓存目录
RUN mkdir -p /app/CosyVoice/pretrained_models && \
    mkdir -p /root/.cache/dolphin

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["python3", "openai_compatible_api.py", "--host", "0.0.0.0", "--port", "8000"]