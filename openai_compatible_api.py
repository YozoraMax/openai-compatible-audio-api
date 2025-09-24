#!/usr/bin/env python3
"""
OpenAI-compatible API server for CosyVoice (TTS) and Dolphin (ASR)
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

# Add Dolphin to path  
sys.path.append(str(Path(__file__).parent / "Dolphin"))

# Import CosyVoice
try:
    from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
    from cosyvoice.utils.file_utils import load_wav
except ImportError as e:
    print(f"Failed to import CosyVoice: {e}")
    CosyVoice = None
    CosyVoice2 = None

# Import Dolphin
try:
    sys.path.append(str(Path(__file__).parent / "Dolphin"))
    import dolphin
except ImportError as e:
    print(f"Failed to import Dolphin: {e}")
    dolphin = None

app = FastAPI(title="OpenAI Compatible Audio API", version="1.0.0")

# Global models
cosyvoice_model = None
dolphin_model = None

def fix_cosyvoice_model_path(expected_model_path: str = "CosyVoice/pretrained_models/CosyVoice2-0.5B"):
    """修复CosyVoice模型路径问题"""
    base_dir = Path(__file__).parent

    # API期望的路径
    expected_path = base_dir / expected_model_path

    # 如果期望路径已存在且有效，直接返回
    if expected_path.exists():
        config_files = list(expected_path.glob("*.yaml"))
        if config_files:
            print(f"✓ CosyVoice模型路径正确: {expected_model_path}")
            return True

    # 检查ModelScope下载的实际路径
    potential_paths = [
        base_dir / "CosyVoice" / "pretrained_models" / "iic" / "CosyVoice2-0.5B",
        base_dir / "models" / "cosyvoice" / "iic" / "CosyVoice2-0.5B",
        base_dir / "CosyVoice" / "pretrained_models" / "CosyVoice-300M-SFT",
        base_dir / "CosyVoice" / "pretrained_models" / "CosyVoice-300M",
    ]

    for actual_path in potential_paths:
        if actual_path.exists() and actual_path.is_dir():
            # 检查是否有配置文件
            config_files = list(actual_path.glob("*.yaml"))
            if config_files:
                print(f"找到CosyVoice模型: {actual_path}")
                print(f"配置文件: {[f.name for f in config_files]}")

                # 确保期望路径的父目录存在
                expected_path.parent.mkdir(parents=True, exist_ok=True)

                # 尝试创建符号链接
                if not expected_path.exists():
                    try:
                        expected_path.symlink_to(actual_path, target_is_directory=True)
                        print(f"✓ 创建符号链接: {expected_path} -> {actual_path}")
                        return True
                    except OSError:
                        # 如果符号链接失败，尝试移动
                        try:
                            import shutil
                            if expected_path.exists():
                                shutil.rmtree(expected_path)
                            shutil.move(str(actual_path), str(expected_path))
                            print(f"✓ 移动模型文件: {actual_path} -> {expected_path}")
                            return True
                        except Exception as e:
                            print(f"✗ 移动模型失败: {e}")
                            continue
                else:
                    return True

    return False

def check_and_download_models(cosyvoice_model_path: str = "CosyVoice/pretrained_models/CosyVoice2-0.5B"):
    """检查并下载必要的模型"""
    print("检查模型文件...")

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

    # 检查Dolphin模型（使用FunASR）
    dolphin_ready = False
    try:
        from funasr import AutoModel
        # 简单检查是否可以初始化模型（会自动下载）
        print("检查Dolphin/FunASR模型...")
        model_dir = Path(__file__).parent / "models" / "dolphin"
        model_dir.mkdir(parents=True, exist_ok=True)

        # 设置缓存目录
        os.environ['FUNASR_CACHE_HOME'] = str(model_dir)

        # 尝试加载模型（如果不存在会自动下载）
        print("初始化FunASR模型（如需要会自动下载）...")
        test_model = AutoModel(model="paraformer-zh", cache_dir=str(model_dir))
        print("✓ Dolphin/FunASR模型准备就绪")
        dolphin_ready = True

    except ImportError:
        print("✗ FunASR未安装，请运行: pip install funasr")
    except Exception as e:
        print(f"✗ Dolphin/FunASR模型初始化失败: {e}")

    return cosyvoice_ready, dolphin_ready

def download_cosyvoice_model(model_path: str):
    """下载CosyVoice模型"""
    try:
        from modelscope import snapshot_download

        # 创建模型目录
        full_path = Path(__file__).parent / model_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        print("从ModelScope下载CosyVoice模型...")
        downloaded_path = snapshot_download(
            'iic/CosyVoice2-0.5B',
            cache_dir=str(full_path.parent)
        )
        print(f"✓ CosyVoice模型下载完成: {downloaded_path}")

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

def initialize_cosyvoice(model_path: str = "CosyVoice/pretrained_models/CosyVoice2-0.5B"):
    """Initialize CosyVoice model"""
    global cosyvoice_model

    if not CosyVoice or not CosyVoice2:
        raise RuntimeError("CosyVoice not available")

    full_path = Path(__file__).parent / model_path
    if not full_path.exists():
        # Try alternative paths including the downloaded ModelScope path
        alt_paths = [
            "CosyVoice/pretrained_models/iic/CosyVoice2-0.5B",  # ModelScope下载路径
            "CosyVoice/pretrained_models/CosyVoice-300M-SFT",
            "CosyVoice/pretrained_models/CosyVoice-300M",
        ]
        for alt_path in alt_paths:
            alt_full_path = Path(__file__).parent / alt_path
            if alt_full_path.exists():
                full_path = alt_full_path
                print(f"使用替代路径: {alt_path}")
                break
        else:
            raise FileNotFoundError(f"CosyVoice model not found at {model_path} or alternative paths: {alt_paths}")

    # Check which config file exists and use appropriate class
    cosyvoice2_yaml = full_path / "cosyvoice2.yaml"
    cosyvoice_yaml = full_path / "cosyvoice.yaml"

    try:
        if cosyvoice2_yaml.exists():
            cosyvoice_model = CosyVoice2(str(full_path), load_jit=False, load_trt=False, fp16=False)
            print(f"Loaded CosyVoice2 from {full_path}")
        elif cosyvoice_yaml.exists():
            cosyvoice_model = CosyVoice(str(full_path), load_jit=False, load_trt=False, fp16=False)
            print(f"Loaded CosyVoice from {full_path}")
        else:
            raise FileNotFoundError(f"Neither cosyvoice2.yaml nor cosyvoice.yaml found in {full_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load CosyVoice model: {e}")

def initialize_dolphin(model_name: str = "paraformer-zh"):
    """Initialize Dolphin/FunASR model"""
    global dolphin_model

    try:
        from funasr import AutoModel

        model_dir = Path(__file__).parent / "models" / "dolphin"
        model_dir.mkdir(parents=True, exist_ok=True)

        # 设置缓存目录
        os.environ['FUNASR_CACHE_HOME'] = str(model_dir)

        dolphin_model = AutoModel(model=model_name, cache_dir=str(model_dir))
        print(f"Loaded FunASR {model_name} model")

    except ImportError:
        raise RuntimeError("FunASR not available, please install: pip install funasr")
    except Exception as e:
        raise RuntimeError(f"Failed to load FunASR model: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    # 获取配置的模型路径
    cosyvoice_model_path = globals().get('COSYVOICE_MODEL_PATH', 'CosyVoice/pretrained_models/CosyVoice2-0.5B')
    dolphin_model_name = globals().get('DOLPHIN_MODEL_NAME', 'paraformer-zh')

    print("=" * 50)
    print("OpenAI Compatible Audio API 启动中...")
    print("=" * 50)

    print("🔍 检查模型状态...")

    # 检查并下载模型
    try:
        cosyvoice_ready, dolphin_ready = check_and_download_models(cosyvoice_model_path)
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

    # 初始化Dolphin/FunASR
    try:
        initialize_dolphin(dolphin_model_name)
        print("✅ FunASR转录服务已启动")
    except Exception as e:
        print(f"❌ FunASR初始化失败: {e}")
        print("⚠️ ASR功能将不可用")

    print("\n" + "=" * 50)
    print("🎉 API服务器启动完成！")
    print(f"🎙️  CosyVoice TTS: {'✅ 可用' if cosyvoice_model else '❌ 不可用'}")
    print(f"🎧 FunASR 转录: {'✅ 可用' if dolphin_model else '❌ 不可用'}")
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
        
        # Check available speakers
        available_spks = cosyvoice_model.list_available_spks()
        if available_spks and speaker not in available_spks:
            speaker = available_spks[0] if available_spks else "中文女"
        
        # Generate speech
        speech_data = None
        if hasattr(cosyvoice_model, 'inference_sft') and available_spks:
            # Use SFT mode if available
            for i, output in enumerate(cosyvoice_model.inference_sft(
                request.input, speaker, stream=False, speed=request.speed
            )):
                speech_data = output['tts_speech']
                break
        elif hasattr(cosyvoice_model, 'inference_zero_shot'):
            # Use zero-shot mode with default prompt
            prompt_path = Path(__file__).parent / "CosyVoice" / "asset" / "zero_shot_prompt.wav"
            if prompt_path.exists():
                prompt_speech = load_wav(str(prompt_path), 16000)
                for i, output in enumerate(cosyvoice_model.inference_zero_shot(
                    request.input, "希望你以后能够做的比我还好呦。", prompt_speech, 
                    stream=False, speed=request.speed
                )):
                    speech_data = output['tts_speech']
                    break
            else:
                raise HTTPException(status_code=500, detail="No prompt audio found for zero-shot")
        
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
    if dolphin_model is None:
        raise HTTPException(status_code=503, detail="FunASR model not available")

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            # Use FunASR for transcription
            result = dolphin_model(tmp_file_path)

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
        "dolphin_available": dolphin_model is not None
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
    
    if dolphin_model is not None:
        models.extend([
            {"id": "whisper-1", "object": "model", "created": 1677610602, "owned_by": "dolphin"}
        ])
    
    return {"object": "list", "data": models}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to") 
    parser.add_argument("--cosyvoice-model", type=str, 
                       default="CosyVoice/pretrained_models/CosyVoice2-0.5B",
                       help="CosyVoice model path")
    parser.add_argument("--dolphin-model", type=str, default="paraformer-zh",
                       help="FunASR model name (paraformer-zh, paraformer-en, etc.)")
    
    args = parser.parse_args()
    
    # Override defaults
    globals()['COSYVOICE_MODEL_PATH'] = args.cosyvoice_model
    globals()['DOLPHIN_MODEL_NAME'] = args.dolphin_model
    
    print(f"Starting OpenAI-compatible API server on {args.host}:{args.port}")
    print(f"CosyVoice model: {args.cosyvoice_model}")
    print(f"Dolphin model: {args.dolphin_model}")
    
    uvicorn.run(app, host=args.host, port=args.port)