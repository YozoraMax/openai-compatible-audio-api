# OpenAI Compatible Audio API

OpenAI兼容的音频API服务器，基于CosyVoice (TTS) 和 FunASR (ASR) 实现。

## 功能特性

- **文本转语音 (TTS)**: 通过 `/v1/audio/speech` 端点使用 CosyVoice
- **语音转文本 (ASR)**: 通过 `/v1/audio/transcriptions` 端点使用 FunASR
- **OpenAI兼容**: 完全兼容 OpenAI Audio API 格式
- **多语言支持**: FunASR 支持中文、英文等多种语言

## 项目结构

```
openai-compatible-audio-api/
├── openai_compatible_api.py    # 主API服务器
├── requirements.txt            # Python依赖文件
├── README.md                  # 项目说明文档
├── .gitignore                 # Git忽略文件配置
├── CosyVoice/                 # CosyVoice TTS项目（自动下载）
├── Dolphin/                   # 历史项目目录（已不使用）
└── models/                    # 模型存储目录（运行时自动创建）
    ├── cosyvoice/            # CosyVoice模型文件
    └── funasr/               # FunASR模型缓存
```

## 部署方式

### Conda环境部署（推荐）

#### 1. 创建Conda环境

```bash
# 创建Python 3.11环境（解决matcha-tts兼容性问题）
conda create -n myenv311 python=3.11

# 激活环境
conda activate myenv311

# 安装依赖
pip install -r requirements.txt
```

#### 2. 启动服务

```bash
# 基本运行（完整功能）
python3 openai_compatible_api.py

# 快速启动（推荐demo使用，仅TTS功能）
python3 openai_compatible_api.py --fast

# 仅TTS模式（最快启动）
python3 openai_compatible_api.py --tts-only

# 自定义配置
python3 openai_compatible_api.py \
  --host 0.0.0.0 \
  --port 8000 \
  --cosyvoice-model "CosyVoice/pretrained_models/CosyVoice2-0.5B" \
  --asr-model "paraformer-zh-streaming"
```

**启动选项说明：**
- `--fast`: 快速模式，自动启用TTS专用模式，跳过ASR模型加载
- `--tts-only`: TTS专用模式，仅启用文本转语音功能，跳过语音识别
- `--asr-model`: 指定ASR模型（仅在完整模式下有效）
  - `paraformer-zh`: 标准中文模型 (~1GB，高精度)
  - `paraformer-zh-streaming`: 流式模型 (~840MB，快速启动，低延迟)

**首次启动说明：**
- 服务会自动检测并下载必要的模型文件
- 加载过程会显示详细进度和耗时
- CosyVoice模型约2GB，FunASR大模型约1GB
- 使用 `--fast` 或 `--tts-only` 选项可显著减少启动时间
- 默认服务地址：`http://127.0.0.1:8000`

**性能对比：**

| 启动模式 | 启动时间 | 模型大小 | 功能 |
|---------|---------|---------|------|
| 完整模式 | ~5-10分钟 | ~3GB | TTS + ASR |
| 快速模式 (`--fast`) | ~2-3分钟 | ~2GB | 仅TTS |
| TTS专用 (`--tts-only`) | ~2-3分钟 | ~2GB | 仅TTS |

### 环境管理

```bash
# 激活环境
conda activate myenv311

# 退出环境
conda deactivate

# 删除环境（如需重新安装）
conda env remove -n myenv311
```

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

**注意：ASR功能仅在完整模式下可用，TTS专用模式不支持语音转文本**

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

2. **matcha-tts安装失败（Python 3.12兼容性问题）**
   ```bash
   # 解决方案：使用Python 3.11
   conda create -n myenv311 python=3.11
   conda activate myenv311
   pip install -r requirements.txt
   ```

3. **依赖编译失败**
   ```bash
   # 某些包可能编译失败，可以跳过
   pip install editdistance --only-binary=all --prefer-binary || echo "editdistance skipped"
   ```

4. **模型下载失败**
   - 检查网络连接，确保能访问ModelScope
   - 检查磁盘空间是否充足
   - 模型会在首次启动时自动下载

5. **内存不足**
   - CosyVoice和FunASR模型较大，建议至少8GB内存
   - 使用 `--fast` 选项可减少内存占用
   - 可以只启用其中一个模型

6. **模型加载时间长**
   ```bash
   # 使用TTS专用模式（最快）
   python3 openai_compatible_api.py --tts-only

   # 或使用快速模式
   python3 openai_compatible_api.py --fast

   # 或手动指定流式模型（仍较慢）
   python3 openai_compatible_api.py --asr-model paraformer-zh-streaming
   ```

7. **Conda环境问题**
   ```bash
   # 如果conda未安装，可以下载Miniconda
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh

   # 重新初始化shell
   source ~/.bashrc
   ```

## 开发

### 环境要求

- Python 3.11（推荐，解决matcha-tts兼容性问题）
- Conda或Miniconda
- PyTorch 2.0+
- 至少8GB内存
- 网络连接（首次运行下载模型）

### 代码结构

- `openai_compatible_api.py`: 主API服务器
- `initialize_cosyvoice()`: CosyVoice模型初始化
- `initialize_funasr()`: FunASR模型初始化
- `/v1/audio/speech`: TTS端点实现
- `/v1/audio/transcriptions`: ASR端点实现

## 许可证

本项目遵循相关开源项目的许可证：
- CosyVoice: Apache 2.0
- FunASR: 请查看FunASR项目许可证

## 致谢

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - 阿里巴巴达摩院文本转语音模型
- [FunASR](https://github.com/modelscope/FunASR) - 达摩院语音识别工具包
- [FastAPI](https://fastapi.tiangolo.com/) - 现代Web框架