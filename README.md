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
├── CosyVoice/                 # CosyVoice项目代码（需要手动克隆）
└── models/                    # 统一模型存储目录（自动创建）
    ├── cosyvoice/            # CosyVoice模型文件
    │   ├── iic/             # ModelScope下载的模型
    │   │   └── CosyVoice2-0.5B/  # 主要TTS模型
    │   └── asset/           # 零样本推理音频文件（可选）
    └── funasr/              # FunASR模型缓存
        └── [模型文件]        # ASR模型自动下载到此处
```

## 部署方式

### Conda环境部署（推荐）

#### 1. 准备环境

```bash
# 1. 安装系统依赖工具（必需）
# Ubuntu/Debian:
sudo apt update && sudo apt install build-essential ffmpeg

# 2. 创建Python 3.11环境（解决matcha-tts兼容性问题）
conda create -n myenv311 python=3.11

# 3. 激活环境
conda activate myenv311

# 4. 克隆CosyVoice项目（必需）
git clone https://github.com/FunAudioLLM/CosyVoice.git

# 5. 安装Python依赖
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
  --cosyvoice-model "models/cosyvoice/CosyVoice2-0.5B" \
  --asr-model "paraformer-zh-streaming"
```

**启动选项说明：**
- `--fast`: 快速模式，自动启用TTS专用模式，跳过ASR模型加载
- `--tts-only`: TTS专用模式，仅启用文本转语音功能，跳过语音识别
- `--asr-model`: 指定ASR模型（仅在完整模式下有效）
  - `paraformer-zh`: 标准中文模型 (~1GB，高精度)
  - `paraformer-zh-streaming`: 流式模型 (~840MB，快速启动，低延迟)

**首次启动说明：**
- 🏗️ 服务会自动创建 `models` 目录并下载必要的模型文件
- 📁 所有模型统一保存在 `models/` 目录下，便于管理
- 📊 CosyVoice模型约2GB（保存到 `models/cosyvoice/`）
- 📊 FunASR模型约1GB（保存到 `models/funasr/`）
- ⏱️ 加载过程会显示详细进度和耗时
- 🚀 使用 `--fast` 或 `--tts-only` 选项可显著减少启动时间
- 🌐 默认服务地址：`http://127.0.0.1:8000`

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

### 模型目录管理

**统一模型目录结构：**
```bash
models/
├── cosyvoice/                    # CosyVoice TTS 模型目录
│   ├── iic/CosyVoice2-0.5B/     # 从ModelScope自动下载的模型
│   └── asset/                   # 零样本推理音频文件
└── funasr/                      # FunASR ASR 模型目录
    └── [自动下载的ASR模型文件]
```

**模型目录优势：**
- 📁 统一管理：所有模型集中在 `models/` 目录
- 🧹 易于清理：删除 `models/` 目录即可清理所有模型
- 💾 节省空间：避免重复下载模型文件
- 🔧 简化路径：无需复杂的软链接，直接从models目录加载
- 🎵 音频文件：零样本推理音频文件优先使用CosyVoice原始asset文件

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

1. **编译依赖缺失（gcc/g++未找到）**
   ```bash
   # Ubuntu/Debian:
   sudo apt update && sudo apt install build-essential
   
   # CentOS/RHEL:
   sudo yum groupinstall "Development Tools"
   
   # macOS:
   xcode-select --install
   
   # 验证编译工具安装
   gcc --version
   g++ --version
   ```

2. **端口被占用**
   ```bash
   lsof -ti:8000 | xargs kill -9
   ```

3. **模型下载失败**
   - 检查网络连接，确保能访问ModelScope
   - 检查磁盘空间是否充足（需要至少4GB可用空间）
   - 模型会自动下载到 `models/` 目录
   - 如需重新下载，删除对应的模型子目录即可

4. **内存不足**
   - CosyVoice和FunASR模型较大，建议至少8GB内存
   - 使用 `--fast` 选项可减少内存占用
   - 可以只启用其中一个模型

5. **Conda环境问题**
   ```bash
   # 如果conda未安装，可以下载Miniconda
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh

   # 重新初始化shell
   source ~/.bashrc
   ```

6. **模型相关问题**
   ```bash
   # 清理所有模型文件（重新下载）
   rm -rf models/
   
   # 仅清理CosyVoice模型
   rm -rf models/cosyvoice/
   
   # 仅清理FunASR模型
   rm -rf models/funasr/
   
   # 查看模型文件大小
   du -sh models/
   ```

## 开发

### 环境要求

- **系统编译工具**：gcc/g++（必需，用于编译matcha-tts和pyworld）
- **Python 3.11**（推荐，解决matcha-tts兼容性问题）
- **Conda或Miniconda**
- **PyTorch 2.0+**
- **至少8GB内存**
- **网络连接**（首次运行下载模型）
- **磁盘空间**：至少4GB可用空间

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