# OpenAI Compatible Audio API

OpenAI兼容的音频API服务器，基于CosyVoice (TTS) 和 Dolphin (ASR) 实现。

## 功能特性

- **文本转语音 (TTS)**: 通过 `/v1/audio/speech` 端点使用 CosyVoice
- **语音转文本 (ASR)**: 通过 `/v1/audio/transcriptions` 端点使用 Dolphin  
- **OpenAI兼容**: 完全兼容 OpenAI Audio API 格式
- **多语言支持**: Dolphin 支持40种东方语言和22种中文方言

## 项目结构

```
openai-compatible-audio-api/
├── openai_compatible_api.py    # 主API服务器
├── requirements.txt            # Python依赖
├── download_models.py          # 模型下载脚本
├── Dockerfile                  # Docker镜像配置
├── .dockerignore              # Docker忽略文件
├── models/                    # 模型存储目录
│   ├── cosyvoice/            # CosyVoice模型
│   └── dolphin/              # Dolphin模型
├── CosyVoice/                 # CosyVoice TTS项目
├── Dolphin/                   # Dolphin ASR项目
└── README.md                  # 本文件
```

## 部署方式

### 方式1: Docker部署（推荐）

#### 1. 预下载模型到项目目录

```bash
# 安装模型下载依赖（使用--user避免权限问题）
python3 -m pip install --user modelscope funasr

# 下载模型到项目目录
python3 download_models.py
```

#### 2. 构建Docker镜像

```bash
docker build -t openai-audio-api .
```

#### 3. 运行容器

```bash
# 使用预下载的模型运行（推荐）
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models/cosyvoice:/app/CosyVoice/pretrained_models \
  -v $(pwd)/models/dolphin:/root/.cache/dolphin \
  --name audio-api \
  --restart unless-stopped \
  openai-audio-api
```

#### 4. 验证部署

```bash
# 检查容器状态
docker ps

# 查看日志
docker logs audio-api

# 健康检查
curl http://localhost:8000/health
```

#### Docker容器管理

```bash
# 停止容器
docker stop audio-api

# 重启容器
docker restart audio-api

# 删除容器
docker rm audio-api

# 删除镜像
docker rmi openai-audio-api
```

### 方式2: 本地直接运行

#### 1. 安装Python依赖

```bash
pip install -r requirements.txt
```

#### 2. 处理编译问题的包 (macOS)

某些包在macOS上可能编译失败，可以跳过：

```bash
# 可选：安装预编译版本
pip install editdistance --only-binary=all --prefer-binary || echo "editdistance skipped"
pip install pyworld --only-binary=all --prefer-binary || echo "pyworld skipped"
```

#### 3. 运行服务器

```bash
# 基本运行
python3 openai_compatible_api.py

# 自定义配置
python3 openai_compatible_api.py \
  --host 0.0.0.0 \
  --port 8000 \
  --cosyvoice-model "CosyVoice/pretrained_models/CosyVoice2-0.5B" \
  --dolphin-model "small"
```

默认在 `http://127.0.0.1:8000` 启动服务。

## API使用

### 文本转语音 (TTS)

```bash
curl -X POST "http://127.0.0.1:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello, this is a test.",
    "voice": "alloy",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

### 语音转文本 (ASR)

```bash
curl -X POST "http://127.0.0.1:8000/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav" \
  -F "model=whisper-1"
```

### 健康检查

```bash
curl http://127.0.0.1:8000/health
```

### 查看可用模型

```bash
curl http://127.0.0.1:8000/v1/models
```

## 支持的声音

| OpenAI Voice | CosyVoice Speaker |
|--------------|-------------------|
| alloy        | 中文女             |
| echo         | 中文男             |
| fable        | 英文女             |
| onyx         | 英文男             |
| nova         | 中文女             |
| shimmer      | 中文女             |

## 故障排除

### 常见问题

1. **端口被占用**
   ```bash
   lsof -ti:8000 | xargs kill -9
   ```

2. **依赖安装失败（权限问题）**
   ```bash
   # 使用--user选项避免权限问题
   python3 -m pip install --user modelscope funasr
   ```

3. **依赖编译失败（macOS）**
   - 某些包（pyworld, editdistance）在macOS上可能编译失败
   - 可以跳过这些包，API仍能正常工作（功能可能受限）

4. **模型下载失败**
   - 检查网络连接，确保能访问ModelScope
   - 检查磁盘空间是否充足
   - 可以使用方案2让容器自动下载模型

5. **内存不足**
   - CosyVoice和Dolphin模型较大，建议至少8GB内存
   - 可以只启用其中一个模型

6. **Docker相关问题**
   ```bash
   # 如果预下载模型失败，可以让容器自动下载
   docker run -d \
     -p 8000:8000 \
     -v $(pwd)/models:/app/models \
     --name audio-api \
     openai-audio-api

   # 查看容器内模型下载进度
   docker logs -f audio-api
   ```

## 开发

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- 至少8GB内存
- 网络连接（首次运行下载模型）
- Docker（推荐使用Docker部署）

### 代码结构

- `openai_compatible_api.py`: 主API服务器
- `initialize_cosyvoice()`: CosyVoice模型初始化
- `initialize_dolphin()`: Dolphin模型初始化
- `/v1/audio/speech`: TTS端点实现
- `/v1/audio/transcriptions`: ASR端点实现

## 许可证

本项目遵循相关开源项目的许可证：
- CosyVoice: Apache 2.0
- Dolphin: 请查看Dolphin项目许可证

## 致谢

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - 阿里巴巴达摩院文本转语音模型
- [Dolphin](https://github.com/DataoceanAI/Dolphin) - DataoceanAI语音识别模型
- [FastAPI](https://fastapi.tiangolo.com/) - 现代Web框架