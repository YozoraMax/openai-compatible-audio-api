#!/usr/bin/env python3
"""
OpenAI-compatible API server for CosyVoice (TTS) and FunASR (ASR)
Provides /v1/audio/speech and /v1/audio/transcriptions endpoints
"""

import sys
import os
import io
import base64
import tempfile
import traceback
from pathlib import Path
from typing import Optional, List

import torch
import torchaudio
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import uvicorn

# Add CosyVoice to path
sys.path.append(str(Path(__file__).parent / "CosyVoice" / "third_party" / "Matcha-TTS"))
sys.path.append(str(Path(__file__).parent / "CosyVoice"))

# Note: Dolphin directory is kept for historical reference but not used

# Import CosyVoice
CosyVoice = None
CosyVoice2 = None
load_wav = None

try:
    # 设置环境变量禁用一些可选功能
    os.environ['MATCHA_DISABLE_COMPILE'] = '1'

    # 尝试创建matcha模块存根（如果不存在）
    matcha_dir = Path(__file__).parent / "CosyVoice" / "third_party" / "Matcha-TTS"
    if not matcha_dir.exists():
        matcha_dir.mkdir(parents=True, exist_ok=True)
        (matcha_dir / "__init__.py").write_text("")

    # 添加到Python路径
    if str(matcha_dir) not in sys.path:
        sys.path.insert(0, str(matcha_dir))

    from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
    from cosyvoice.utils.file_utils import load_wav
    print("✓ CosyVoice导入成功")
except ImportError as e:
    print(f"⚠️ CosyVoice导入失败: {e}")
    print("💡 解决方案：")
    print("   1. 运行: python3 install_dependencies.py")
    print("   2. 或手动安装: pip install matcha-tts einops phonemizer")
except Exception as e:
    print(f"❌ CosyVoice初始化错误: {e}")
    print("💡 这可能是模型加载问题，服务器将继续启动但TTS功能不可用")

# FunASR for speech recognition
print("ℹ️ 使用FunASR进行语音识别")

app = FastAPI(title="OpenAI Compatible Audio API", version="1.0.0")

# Global models
cosyvoice_model = None
funasr_model = None

def fix_cosyvoice_model_path(expected_model_path: str = "models/cosyvoice/CosyVoice2-0.5B"):
    """修复CosyVoice模型路径问题 - 统一使用models目录"""
    base_dir = Path(__file__).parent

    # 确保models目录结构存在
    models_dir = base_dir / "models"
    cosyvoice_dir = models_dir / "cosyvoice"
    cosyvoice_dir.mkdir(parents=True, exist_ok=True)

    # API期望的路径
    expected_path = base_dir / expected_model_path

    # 如果期望路径已存在且有效，直接返回
    if expected_path.exists():
        config_files = list(expected_path.glob("*.yaml"))
        if config_files:
            print(f"✓ CosyVoice模型路径正确: {expected_model_path}")
            return True

    # 检查可能的旧路径（用于迁移）
    potential_old_paths = [
        base_dir / "CosyVoice" / "pretrained_models" / "iic" / "CosyVoice2-0.5B",
        base_dir / "CosyVoice" / "pretrained_models" / "CosyVoice2-0.5B", 
        base_dir / "CosyVoice" / "pretrained_models" / "CosyVoice-300M-SFT",
        base_dir / "CosyVoice" / "pretrained_models" / "CosyVoice-300M",
    ]

    # 检查models目录下可能的路径
    potential_paths = [
        base_dir / "models" / "cosyvoice" / "iic" / "CosyVoice2-0.5B",
        base_dir / "models" / "cosyvoice" / "CosyVoice2-0.5B",
        base_dir / "models" / "cosyvoice" / "CosyVoice-300M-SFT", 
        base_dir / "models" / "cosyvoice" / "CosyVoice-300M",
    ] + potential_old_paths

    for actual_path in potential_paths:
        if actual_path.exists() and actual_path.is_dir():
            # 检查是否有配置文件
            config_files = list(actual_path.glob("*.yaml"))
            if config_files:
                print(f"找到CosyVoice模型: {actual_path}")
                print(f"配置文件: {[f.name for f in config_files]}")

                # 如果模型在旧路径，迁移到新的models目录
                if not str(actual_path).startswith(str(models_dir)):
                    new_path = cosyvoice_dir / actual_path.name
                    print(f"🔄 迁移模型到统一目录: {actual_path} -> {new_path}")
                    try:
                        import shutil
                        if new_path.exists():
                            shutil.rmtree(new_path)
                        shutil.move(str(actual_path), str(new_path))
                        actual_path = new_path
                        print(f"✅ 模型迁移完成: {new_path}")
                    except Exception as e:
                        print(f"⚠️ 模型迁移失败，使用原路径: {e}")

                # 确保期望路径指向正确的模型
                if actual_path != expected_path:
                    expected_path.parent.mkdir(parents=True, exist_ok=True)
                    if not expected_path.exists():
                        try:
                            expected_path.symlink_to(actual_path, target_is_directory=True)
                            print(f"✓ 创建符号链接: {expected_path} -> {actual_path}")
                        except OSError:
                            print(f"⚠️ 符号链接创建失败，直接使用路径: {actual_path}")
                
                return True

    return False

def check_and_download_models(cosyvoice_model_path: str = "models/cosyvoice/CosyVoice2-0.5B"):
    """检查并下载必要的模型到统一的models目录"""
    print("📁 检查并创建models目录结构...")
    
    base_dir = Path(__file__).parent
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    print("✅ models目录已创建")
    print("🔍 检查模型文件...")

    # 检查并修复CosyVoice模型路径
    cosyvoice_ready = fix_cosyvoice_model_path(cosyvoice_model_path)

    if not cosyvoice_ready:
        print(f"✗ CosyVoice模型不存在，开始下载到: {cosyvoice_model_path}")
        try:
            download_cosyvoice_model(cosyvoice_model_path)
            # 下载后再次尝试修复路径
            cosyvoice_ready = fix_cosyvoice_model_path(cosyvoice_model_path)
        except Exception as e:
            print(f"CosyVoice模型下载失败: {e}")
            cosyvoice_ready = False

    # 检查FunASR模型
    funasr_ready = False
    try:
        from funasr import AutoModel
        # 简单检查是否可以初始化模型（会自动下载）
        print("🔍 检查FunASR模型...")
        model_dir = models_dir / "funasr"
        model_dir.mkdir(parents=True, exist_ok=True)

        # 设置缓存目录
        os.environ['FUNASR_CACHE_HOME'] = str(model_dir)

        # 尝试加载模型（如果不存在会自动下载）
        print("⬇️ 初始化FunASR模型（如需要会自动下载到models/funasr）...")
        test_model = AutoModel(model="paraformer-zh", cache_dir=str(model_dir))
        print("✅ FunASR模型准备就绪")
        funasr_ready = True

    except ImportError:
        print("✗ FunASR未安装，请运行: pip install funasr")
    except Exception as e:
        print(f"✗ FunASR模型初始化失败: {e}")

    return cosyvoice_ready, funasr_ready

def download_cosyvoice_model(model_path: str):
    """下载CosyVoice模型到models目录"""
    try:
        from modelscope import snapshot_download

        base_dir = Path(__file__).parent
        models_dir = base_dir / "models" / "cosyvoice"
        models_dir.mkdir(parents=True, exist_ok=True)

        print("⬇️ 从ModelScope下载CosyVoice模型到models/cosyvoice目录...")
        print("📊 模型大小约2GB，请耐心等待...")
        
        # 下载到models/cosyvoice目录
        downloaded_path = snapshot_download(
            'iic/CosyVoice2-0.5B',
            cache_dir=str(models_dir)
        )
        print(f"✅ CosyVoice模型下载完成: {downloaded_path}")
        
        # 检查下载的模型是否在预期位置
        expected_model_path = models_dir / "iic" / "CosyVoice2-0.5B"
        if expected_model_path.exists():
            print(f"✅ 模型已保存到: {expected_model_path}")
        else:
            print(f"⚠️ 模型可能在其他位置: {downloaded_path}")

        # 下载完成后，调用路径修复函数
        fix_cosyvoice_model_path(model_path)

        return downloaded_path

    except ImportError:
        print("✗ ModelScope未安装，请运行: pip install modelscope")
        raise RuntimeError("无法下载CosyVoice模型，ModelScope未安装")
    except Exception as e:
        print(f"✗ CosyVoice模型下载失败: {e}")
        raise RuntimeError(f"CosyVoice模型下载失败: {e}")

class TTSRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "alloy"
    response_format: str = "mp3"
    speed: float = 1.0

class TranscriptionResponse(BaseModel):
    text: str

def create_default_prompt_audio():
    """创建默认的零样本提示音频文件到models目录"""
    try:
        import numpy as np

        # 创建一个简单的正弦波作为默认提示音频（1秒，16kHz）
        sample_rate = 16000
        duration = 1.0  # 1秒
        frequency = 220  # 低音A，更接近人声

        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # 创建一个复合波形，模拟简单语音
        audio = (np.sin(2 * np.pi * frequency * t) * 0.5 +
                np.sin(2 * np.pi * frequency * 2 * t) * 0.3 +
                np.sin(2 * np.pi * frequency * 3 * t) * 0.2)

        # 添加包络，使其更像语音
        envelope = np.exp(-t * 1.5) * (1 - np.exp(-t * 10))
        audio = audio * envelope * 0.3

        # 转换为torch tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)

        # 保存音频文件到models目录
        asset_dir = Path(__file__).parent / "models" / "cosyvoice" / "asset"
        asset_dir.mkdir(parents=True, exist_ok=True)

        output_path = asset_dir / "zero_shot_prompt.wav"
        torchaudio.save(str(output_path), audio_tensor, sample_rate)

        print(f"✅ 默认提示音频文件已创建: {output_path}")
        return output_path

    except Exception as e:
        print(f"⚠️ 创建默认提示音频失败: {e}")
        return None

def initialize_cosyvoice(model_path: str = "models/cosyvoice/CosyVoice2-0.5B"):
    """Initialize CosyVoice model"""
    global cosyvoice_model
    import time

    if not CosyVoice or not CosyVoice2:
        raise RuntimeError("CosyVoice not available")

    print(f"🔄 开始加载 CosyVoice 模型...")
    start_time = time.time()

    full_path = Path(__file__).parent / model_path
    print(f"📂 模型路径: {model_path}")

    if not full_path.exists():
        # Try alternative paths including models directory and legacy paths
        alt_paths = [
            "models/cosyvoice/iic/CosyVoice2-0.5B",  # ModelScope下载到models目录的路径
            "models/cosyvoice/CosyVoice-300M-SFT",
            "models/cosyvoice/CosyVoice-300M", 
            "CosyVoice/pretrained_models/iic/CosyVoice2-0.5B",  # 旧路径兼容
            "CosyVoice/pretrained_models/CosyVoice2-0.5B",
            "CosyVoice/pretrained_models/CosyVoice-300M-SFT",
            "CosyVoice/pretrained_models/CosyVoice-300M",
        ]
        for alt_path in alt_paths:
            alt_full_path = Path(__file__).parent / alt_path
            if alt_full_path.exists():
                full_path = alt_full_path
                print(f"✅ 使用发现的模型路径: {alt_path}")
                break
        else:
            raise FileNotFoundError(f"CosyVoice model not found at {model_path} or alternative paths: {alt_paths}")

    # Check which config file exists and use appropriate class
    cosyvoice2_yaml = full_path / "cosyvoice2.yaml"
    cosyvoice_yaml = full_path / "cosyvoice.yaml"

    try:
        print("⏳ 正在初始化模型参数...")
        if cosyvoice2_yaml.exists():
            print("📄 使用 CosyVoice2 配置")
            cosyvoice_model = CosyVoice2(str(full_path), load_jit=False, load_trt=False, fp16=False)
            model_type = "CosyVoice2"
        elif cosyvoice_yaml.exists():
            print("📄 使用 CosyVoice 配置")
            cosyvoice_model = CosyVoice(str(full_path), load_jit=False, load_trt=False, fp16=False)
            model_type = "CosyVoice"
        else:
            raise FileNotFoundError(f"Neither cosyvoice2.yaml nor cosyvoice.yaml found in {full_path}")

        elapsed = time.time() - start_time
        print(f"✅ {model_type} 模型加载完成 (耗时: {elapsed:.1f}秒)")

        # 检查零样本推理音频文件
        # 优先查找CosyVoice原生asset文件
        cosyvoice_asset_paths = [
            Path(__file__).parent / "CosyVoice" / "asset" / "zero_shot_prompt.wav",
            Path(__file__).parent / "CosyVoice" / "asset" / "samples" / "zero_shot_prompt.wav",
            Path(__file__).parent / "CosyVoice" / "asset" / "samples" / "cross_lingual_prompt.wav",
            Path(__file__).parent / "CosyVoice" / "asset" / "instruct_prompt.wav",
        ]
        
        original_asset_found = False
        for asset_path in cosyvoice_asset_paths:
            if asset_path.exists():
                print(f"✅ 发现CosyVoice原生提示音频文件: {asset_path}")
                original_asset_found = True
                break
        
        if not original_asset_found:
            # 如果没有找到原生文件，创建备用文件到models目录
            asset_dir = Path(__file__).parent / "models" / "cosyvoice" / "asset"
            asset_dir.mkdir(parents=True, exist_ok=True)
            prompt_path = asset_dir / "zero_shot_prompt.wav"
            
            if not prompt_path.exists():
                print("🎵 未找到CosyVoice原生音频文件，正在创建备用文件...")
                create_default_prompt_audio()
            else:
                print(f"✅ 备用提示音频文件已存在: {prompt_path}")
        else:
            print("✅ 将优先使用CosyVoice原生提示音频文件进行零样本推理")

    except Exception as e:
        raise RuntimeError(f"Failed to load CosyVoice model: {e}")

def initialize_funasr(model_name: str = "paraformer-zh"):
    """Initialize FunASR model"""
    global funasr_model

    try:
        from funasr import AutoModel
        import time

        model_dir = Path(__file__).parent / "models" / "funasr"
        model_dir.mkdir(parents=True, exist_ok=True)

        # 设置缓存目录
        os.environ['FUNASR_CACHE_HOME'] = str(model_dir)

        print(f"🔄 开始加载 FunASR 模型: {model_name}")
        start_time = time.time()

        # 小模型映射 - 使用FunASR实际支持的模型名称
        model_sizes = {
            "paraformer-zh": "标准中文模型 (~1GB)",
            "iic/speech_paraformer-zh-small_asr_nat-zh-cn-16k-common-vocab8404-pytorch": "小模型 (~300MB)",
            "paraformer-zh-streaming": "流式模型 (~840MB)",
            "paraformer-en": "英文模型 (~800MB)"
        }

        size_info = model_sizes.get(model_name, "未知大小")
        print(f"📊 模型信息: {size_info}")
        print("⏳ 正在下载/加载模型，请稍候...")
        print("💡 提示: 模型加载可能需要几分钟，请耐心等待...")

        # 添加加载进度提示
        import threading
        import time

        def progress_indicator():
            dots = 0
            while not hasattr(progress_indicator, 'stop'):
                dots = (dots + 1) % 4
                print(f"\r⏳ 模型加载中{'.' * dots}{' ' * (3 - dots)}", end='', flush=True)
                time.sleep(1)

        progress_thread = threading.Thread(target=progress_indicator, daemon=True)
        progress_thread.start()

        try:
            funasr_model = AutoModel(model=model_name, cache_dir=str(model_dir))
        finally:
            progress_indicator.stop = True
            print("\r", end='')  # 清除进度指示器

        elapsed = time.time() - start_time
        print(f"✅ FunASR {model_name} 模型加载完成 (耗时: {elapsed:.1f}秒)")

    except ImportError:
        raise RuntimeError("FunASR not available, please install: pip install funasr")
    except Exception as e:
        raise RuntimeError(f"Failed to load FunASR model: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    # 获取配置的模型路径 - 默认使用models目录
    cosyvoice_model_path = globals().get('COSYVOICE_MODEL_PATH', 'models/cosyvoice/CosyVoice2-0.5B')
    asr_model_name = globals().get('ASR_MODEL_NAME', 'paraformer-zh')
    tts_only_mode = globals().get('TTS_ONLY_MODE', False)

    print("=" * 50)
    print("OpenAI Compatible Audio API 启动中...")
    if tts_only_mode:
        print("🚀 TTS专用模式 - 仅启用文本转语音功能")
    print("=" * 50)

    print("🔍 检查模型状态...")

    # 检查并下载模型
    try:
        cosyvoice_ready, funasr_ready = check_and_download_models(cosyvoice_model_path)
    except Exception as e:
        print(f"❌ 模型检查/下载失败: {e}")
        print("⚠️ 将尝试继续启动，某些功能可能不可用")

    print("\n🚀 初始化模型...")

    # 初始化CosyVoice
    try:
        initialize_cosyvoice(cosyvoice_model_path)
        print("✅ CosyVoice TTS服务已启动")
    except Exception as e:
        print(f"❌ CosyVoice初始化失败: {e}")
        print("⚠️ TTS功能将不可用")

    # 初始化FunASR (除非是TTS专用模式)
    if not tts_only_mode:
        try:
            initialize_funasr(asr_model_name)
            print("✅ FunASR转录服务已启动")
        except Exception as e:
            print(f"❌ FunASR初始化失败: {e}")
            print("⚠️ ASR功能将不可用")
    else:
        print("⏭️ TTS专用模式，跳过ASR模型加载")

    print("\n" + "=" * 50)
    print("🎉 API服务器启动完成！")
    print(f"🎙️  CosyVoice TTS: {'✅ 可用' if cosyvoice_model else '❌ 不可用'}")
    if tts_only_mode:
        print("🎧 FunASR 转录: ⏭️ TTS专用模式已禁用")
    else:
        print(f"🎧 FunASR 转录: {'✅ 可用' if funasr_model else '❌ 不可用'}")
    print("🌐 服务地址: http://127.0.0.1:8000")
    print("📖 API文档: http://127.0.0.1:8000/docs")
    print("=" * 50)

@app.post("/v1/audio/speech")
async def create_speech(request: TTSRequest):
    """
    Create speech from text using CosyVoice
    Compatible with OpenAI's /v1/audio/speech endpoint
    """
    if cosyvoice_model is None:
        raise HTTPException(status_code=503, detail="CosyVoice model not available")
    
    try:
        # Map OpenAI voices to CosyVoice speakers
        voice_mapping = {
            "alloy": "中文女",
            "echo": "中文男", 
            "fable": "英文女",
            "onyx": "英文男",
            "nova": "中文女",
            "shimmer": "中文女"
        }
        
        speaker = voice_mapping.get(request.voice, "中文女")
        
        # Check available speakers and methods
        print(f"🔍 检查CosyVoice模型可用方法...")
        available_methods = [method for method in dir(cosyvoice_model) if not method.startswith('_')]
        print(f"📋 可用方法: {available_methods}")

        available_spks = None
        try:
            available_spks = cosyvoice_model.list_available_spks()
            print(f"🎭 可用说话人: {available_spks}")
        except Exception as e:
            print(f"⚠️ 获取说话人列表失败: {e}")

        if available_spks and speaker not in available_spks:
            speaker = available_spks[0] if available_spks else "中文女"
            print(f"🔄 切换到可用说话人: {speaker}")
        else:
            print(f"🎭 使用说话人: {speaker}")
        
        # Generate speech
        speech_data = None

        # Helper function for zero-shot inference
        def try_zero_shot_inference():
            # 优先查找CosyVoice原生asset目录下的音频文件
            cosyvoice_asset_paths = [
                Path(__file__).parent / "CosyVoice" / "asset" / "zero_shot_prompt.wav",
                Path(__file__).parent / "CosyVoice" / "asset" / "samples" / "zero_shot_prompt.wav",
                Path(__file__).parent / "CosyVoice" / "asset" / "samples" / "cross_lingual_prompt.wav",
                Path(__file__).parent / "CosyVoice" / "asset" / "instruct_prompt.wav",
            ]

            # 次优选择：models目录和其他位置
            fallback_paths = [
                Path(__file__).parent / "models" / "cosyvoice" / "asset" / "zero_shot_prompt.wav",
                Path(__file__).parent / "CosyVoice" / "zero_shot_prompt.wav",
                Path(__file__).parent / "zero_shot_prompt.wav"
            ]

            # 组合所有路径，优先使用原生asset文件
            all_prompt_paths = cosyvoice_asset_paths + fallback_paths

            for path in all_prompt_paths:
                if path.exists():
                    try:
                        print(f"🎵 找到提示音频文件: {path}")
                        prompt_speech = load_wav(str(path), 16000)
                        return cosyvoice_model.inference_zero_shot(
                            request.input, "希望你以后能够做的比我还好呦。", prompt_speech,
                            stream=False, speed=request.speed
                        )
                    except Exception as e:
                        print(f"⚠️ 零样本模式路径 {path} 失败: {e}")
                        continue

            # 如果没有找到任何提示音频文件，尝试无提示推理
            print("📝 未找到提示音频文件，尝试使用空提示进行零样本推理")
            try:
                # Some CosyVoice2 models support inference without explicit prompt audio
                return cosyvoice_model.inference_zero_shot(
                    request.input, "希望你以后能够做的比我还好呦。", None,
                    stream=False, speed=request.speed
                )
            except Exception as e:
                print(f"⚠️ 无提示音频零样本推理失败: {e}")
                return None

        # Try different inference methods based on what's available
        # For CosyVoice2-0.5B, prioritize cross_lingual and instruct modes
        inference_methods = [
            ('inference_sft', lambda: cosyvoice_model.inference_sft(request.input, speaker, stream=False, speed=request.speed) if available_spks else None),
            ('inference_cross_lingual', lambda: cosyvoice_model.inference_cross_lingual(request.input, None, stream=False) if hasattr(cosyvoice_model, 'inference_cross_lingual') else None),
            ('inference_instruct2', lambda: cosyvoice_model.inference_instruct2(request.input, '用自然的语调说这句话', None, stream=False) if hasattr(cosyvoice_model, 'inference_instruct2') else None),
            ('inference_zero_shot', try_zero_shot_inference),
            ('inference', lambda: cosyvoice_model.inference(request.input, stream=False, speed=request.speed)),
            ('tts', lambda: cosyvoice_model.tts(request.input, speaker=speaker)),
            ('generate', lambda: cosyvoice_model.generate(request.input)),
        ]

        for method_name, method_func in inference_methods:
            if hasattr(cosyvoice_model, method_name) and speech_data is None:
                try:
                    print(f"🎤 尝试 {method_name} 模式生成语音...")

                    # Skip methods that require unavailable resources
                    if method_name == 'inference_sft' and not available_spks:
                        print(f"⏭️ 跳过 {method_name}：无可用说话人")
                        continue

                    if method_name in ['inference_cross_lingual', 'inference_instruct2'] and not hasattr(cosyvoice_model, method_name):
                        print(f"⏭️ 跳过 {method_name}：方法不可用")
                        continue

                    result = method_func()
                    if result is None:
                        continue

                    # Handle different result formats
                    if hasattr(result, '__iter__') and not isinstance(result, (str, bytes)):
                        for i, output in enumerate(result):
                            if isinstance(output, dict) and 'tts_speech' in output:
                                speech_data = output['tts_speech']
                                break
                            elif hasattr(output, 'audio') or hasattr(output, 'speech'):
                                speech_data = getattr(output, 'audio', getattr(output, 'speech', None))
                                break
                            elif isinstance(output, torch.Tensor):
                                speech_data = output
                                break
                    elif isinstance(result, torch.Tensor):
                        speech_data = result
                    elif isinstance(result, dict) and 'tts_speech' in result:
                        speech_data = result['tts_speech']

                    if speech_data is not None:
                        print(f"✅ {method_name} 模式生成成功")
                        break

                except Exception as e:
                    print(f"⚠️ {method_name} 模式失败: {e}")
                    continue
        
        if speech_data is None:
            raise HTTPException(status_code=500, detail="Failed to generate speech")
        
        # Convert to numpy and save as wav
        audio_numpy = speech_data.cpu().numpy().flatten()
        
        # Create temporary file for audio conversion
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            torchaudio.save(tmp_file.name, speech_data, cosyvoice_model.sample_rate)
            
            # Read the audio file
            with open(tmp_file.name, "rb") as f:
                audio_bytes = f.read()
            
            # Clean up temp file
            os.unlink(tmp_file.name)
        
        # Return appropriate format
        if request.response_format == "mp3":
            # For simplicity, return WAV with MP3 content-type
            # In production, you'd want to convert to actual MP3
            return Response(content=audio_bytes, media_type="audio/mpeg")
        elif request.response_format == "wav":
            return Response(content=audio_bytes, media_type="audio/wav")
        else:
            return Response(content=audio_bytes, media_type="audio/wav")
            
    except Exception as e:
        print(f"TTS Error: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Speech generation failed: {str(e)}")

@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0)
):
    """
    Transcribe audio to text using FunASR
    Compatible with OpenAI's /v1/audio/transcriptions endpoint
    """
    tts_only_mode = globals().get('TTS_ONLY_MODE', False)
    if tts_only_mode:
        raise HTTPException(status_code=503, detail="ASR functionality disabled in TTS-only mode")

    if funasr_model is None:
        raise HTTPException(status_code=503, detail="FunASR model not available")

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            # Use FunASR for transcription
            result = funasr_model(tmp_file_path)

            # Clean up temp file
            os.unlink(tmp_file_path)

            # Extract text from result
            if hasattr(result, 'text'):
                text = result.text
            elif isinstance(result, list) and len(result) > 0:
                # Handle list output
                text = ""
                for item in result:
                    if hasattr(item, 'text'):
                        text += item.text
                    elif isinstance(item, dict) and 'text' in item:
                        text += item['text']
                    elif isinstance(item, str):
                        text += item
            elif isinstance(result, dict) and 'text' in result:
                text = result['text']
            else:
                text = str(result)

            return TranscriptionResponse(text=text.strip())

        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            raise e

    except Exception as e:
        print(f"Transcription Error: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "cosyvoice_available": cosyvoice_model is not None,
        "funasr_available": funasr_model is not None
    }

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)"""
    models = []
    
    if cosyvoice_model is not None:
        models.extend([
            {"id": "tts-1", "object": "model", "created": 1677610602, "owned_by": "cosyvoice"},
            {"id": "tts-1-hd", "object": "model", "created": 1677610602, "owned_by": "cosyvoice"}
        ])
    
    if funasr_model is not None:
        models.extend([
            {"id": "whisper-1", "object": "model", "created": 1677610602, "owned_by": "funasr"}
        ])
    
    return {"object": "list", "data": models}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to") 
    parser.add_argument("--cosyvoice-model", type=str, 
                       default="models/cosyvoice/CosyVoice2-0.5B",
                       help="CosyVoice model path (now defaults to models directory)")
    parser.add_argument("--asr-model", type=str, default="paraformer-zh",
                       help="FunASR model name (paraformer-zh, paraformer-zh-streaming)")
    parser.add_argument("--fast", action="store_true",
                       help="Use faster/smaller models for demo (enables TTS-only mode)")
    parser.add_argument("--tts-only", action="store_true",
                       help="TTS-only mode: only enable text-to-speech, skip ASR for fastest startup")

    args = parser.parse_args()

    # Apply fast mode
    if args.fast:
        args.tts_only = True
        print("🚀 快速模式已启用，使用TTS专用模式")

    # Apply TTS-only mode
    if args.tts_only:
        globals()['TTS_ONLY_MODE'] = True
        print("🎙️ TTS专用模式已启用，将跳过ASR模型加载")
    else:
        globals()['TTS_ONLY_MODE'] = False
    
    # Override defaults
    globals()['COSYVOICE_MODEL_PATH'] = args.cosyvoice_model
    globals()['ASR_MODEL_NAME'] = args.asr_model

    print(f"Starting OpenAI-compatible API server on {args.host}:{args.port}")
    print(f"CosyVoice model: {args.cosyvoice_model}")
    if not args.tts_only:
        print(f"FunASR model: {args.asr_model}")
    else:
        print("ASR model: 已禁用 (TTS专用模式)")
    
    uvicorn.run(app, host=args.host, port=args.port)