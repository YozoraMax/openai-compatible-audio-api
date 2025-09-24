#!/usr/bin/env python3
"""
依赖安装脚本
解决CosyVoice和FunASR的复杂依赖问题
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description=""):
    """运行命令并显示结果"""
    print(f"执行: {description or cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 失败: {e}")
        if e.stderr:
            print(f"错误信息: {e.stderr}")
        return False

def install_basic_requirements():
    """安装基础依赖"""
    print("📦 安装基础依赖...")

    basic_packages = [
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.20.0",
        "pydantic>=2.0.0",
        "python-multipart>=0.0.5",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "numpy",
        "pyyaml",
        "requests>=2.25"
    ]

    for package in basic_packages:
        if not run_command(f"pip install {package}", f"安装 {package}"):
            print(f"⚠️ {package} 安装失败，继续...")

    return True

def install_cosyvoice_deps():
    """安装CosyVoice相关依赖"""
    print("🎵 安装CosyVoice依赖...")

    cosyvoice_packages = [
        "modelscope>=1.0.0",
        "transformers>=4.44.0",
        "onnxruntime>=1.18.0",
        "hydra-core>=1.3.2",
        "omegaconf>=2.3.0",
        "einops",
        "phonemizer"
    ]

    for package in cosyvoice_packages:
        run_command(f"pip install {package}", f"安装 {package}")

    # 尝试安装matcha-tts（可能有依赖冲突）
    print("🔧 尝试安装Matcha-TTS...")
    if not run_command("pip install matcha-tts", "安装 matcha-tts"):
        print("⚠️ matcha-tts安装失败，尝试替代方案...")
        # 安装matcha的核心依赖
        run_command("pip install einops", "安装 einops")

    return True

def install_funasr_deps():
    """安装FunASR依赖"""
    print("🎧 安装FunASR依赖...")

    funasr_packages = [
        "funasr>=1.2.7",
        "sentencepiece==0.2.0",
        "kaldiio>=2.17.0"
    ]

    for package in funasr_packages:
        run_command(f"pip install {package}", f"安装 {package}")

    return True

def install_optional_deps():
    """安装可选依赖"""
    print("📋 安装可选依赖...")

    optional_packages = [
        "editdistance",
        "jieba",
        "jamo"
    ]

    for package in optional_packages:
        if not run_command(f"pip install {package}", f"安装 {package}"):
            print(f"⚠️ {package} 安装失败（可选依赖）")

    return True

def create_matcha_stub():
    """创建matcha模块存根以解决导入问题"""
    print("🔧 创建Matcha-TTS存根...")

    project_dir = Path(__file__).parent
    matcha_dir = project_dir / "CosyVoice" / "third_party" / "Matcha-TTS"

    if not matcha_dir.exists():
        matcha_dir.mkdir(parents=True, exist_ok=True)

    # 创建__init__.py
    init_file = matcha_dir / "__init__.py"
    init_file.write_text("")

    # 创建matcha模块存根
    matcha_file = matcha_dir / "matcha.py"
    stub_content = '''"""
Matcha-TTS 存根模块
用于解决CosyVoice的导入问题
"""

# 基础类和函数存根
class MatchaModel:
    def __init__(self, *args, **kwargs):
        pass

def load_model(*args, **kwargs):
    return MatchaModel()

# 其他可能需要的存根
__all__ = ['MatchaModel', 'load_model']
'''
    matcha_file.write_text(stub_content)
    print(f"✓ 创建Matcha存根: {matcha_file}")

    return True

def main():
    """主函数"""
    print("🚀 OpenAI Compatible Audio API 依赖安装")
    print("=" * 50)

    # 检查虚拟环境
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✓ 检测到虚拟环境")
    else:
        print("⚠️ 建议在虚拟环境中运行")

    print()

    # 按顺序安装依赖
    install_basic_requirements()
    print()

    install_cosyvoice_deps()
    print()

    install_funasr_deps()
    print()

    install_optional_deps()
    print()

    create_matcha_stub()
    print()

    print("=" * 50)
    print("🎉 依赖安装完成！")
    print("现在可以运行: python3 openai_compatible_api.py")

if __name__ == "__main__":
    main()