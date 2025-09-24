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
        # Try alternative paths
        alt_paths = [
            "CosyVoice/pretrained_models/CosyVoice-300M-SFT",
            "CosyVoice/pretrained_models/CosyVoice-300M",
        ]
        for alt_path in alt_paths:
            alt_full_path = Path(__file__).parent / alt_path
            if alt_full_path.exists():
                full_path = alt_full_path
                break
        else:
            raise FileNotFoundError(f"CosyVoice model not found at {model_path} or alternative paths")
    
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

def initialize_dolphin(model_name: str = "small"):
    """Initialize Dolphin model"""
    global dolphin_model
    
    if not dolphin:
        raise RuntimeError("Dolphin not available")
    
    model_dir = Path.home() / ".cache" / "dolphin"
    model_dir.mkdir(exist_ok=True)
    
    try:
        dolphin_model = dolphin.load_model(model_name, str(model_dir))
        print(f"Loaded Dolphin {model_name} model")
    except Exception as e:
        raise RuntimeError(f"Failed to load Dolphin model: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    try:
        initialize_cosyvoice()
    except Exception as e:
        print(f"Warning: CosyVoice initialization failed: {e}")
    
    try:
        initialize_dolphin()
    except Exception as e:
        print(f"Warning: Dolphin initialization failed: {e}")

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
    Transcribe audio to text using Dolphin
    Compatible with OpenAI's /v1/audio/transcriptions endpoint
    """
    if dolphin_model is None:
        raise HTTPException(status_code=503, detail="Dolphin model not available")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Map language codes (optional)
            lang_mapping = {
                "zh": "zh",
                "en": "en", 
                "ja": "ja",
                "ko": "ko"
            }
            
            lang_sym = None
            region_sym = None
            if language:
                lang_sym = lang_mapping.get(language[:2], language[:2])
            
            # Transcribe using Dolphin
            waveform = dolphin.load_audio(tmp_file_path)
            result = dolphin_model(waveform, lang_sym=lang_sym, region_sym=region_sym)
            
            # Clean up temp file
            os.unlink(tmp_file_path)
            
            return TranscriptionResponse(text=result.text)
            
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
    parser.add_argument("--dolphin-model", type=str, default="small",
                       help="Dolphin model name (base, small)")
    
    args = parser.parse_args()
    
    # Override defaults
    globals()['COSYVOICE_MODEL_PATH'] = args.cosyvoice_model
    globals()['DOLPHIN_MODEL_NAME'] = args.dolphin_model
    
    print(f"Starting OpenAI-compatible API server on {args.host}:{args.port}")
    print(f"CosyVoice model: {args.cosyvoice_model}")
    print(f"Dolphin model: {args.dolphin_model}")
    
    uvicorn.run(app, host=args.host, port=args.port)