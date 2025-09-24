#!/usr/bin/env python3
"""
模型预下载脚本
在构建Docker镜像前运行此脚本，将模型下载到项目目录中
"""

import os
import sys
from pathlib import Path

def download_cosyvoice_model():
    """下载CosyVoice模型"""
    print("正在下载CosyVoice模型...")

    try:
        from modelscope import snapshot_download

        # 模型下载到项目的models/cosyvoice目录
        model_dir = "./models/cosyvoice"
        os.makedirs(model_dir, exist_ok=True)

        # 下载CosyVoice-0.5B模型
        model_path = snapshot_download(
            'iic/CosyVoice2-0.5B',
            cache_dir=model_dir
        )

        print(f"CosyVoice模型下载完成: {model_path}")
        return True

    except ImportError:
        print("modelscope未安装，请先运行: pip install modelscope")
        return False
    except Exception as e:
        print(f"CosyVoice模型下载失败: {e}")
        return False

def download_dolphin_model():
    """下载Dolphin模型"""
    print("正在下载Dolphin模型...")

    try:
        from funasr import AutoModel

        # 模型下载到项目的models/dolphin目录
        model_dir = "./models/dolphin"
        os.makedirs(model_dir, exist_ok=True)

        # 设置缓存目录
        os.environ['FUNASR_CACHE_HOME'] = model_dir

        # 下载ASR模型 (使用FunASR支持的模型)
        model = AutoModel(
            model="paraformer-zh",
            cache_dir=model_dir
        )

        print(f"Dolphin ASR模型下载完成: {model_dir}")
        return True

    except ImportError:
        print("funasr未安装，请先运行: pip install funasr")
        return False
    except Exception as e:
        print(f"Dolphin模型下载失败: {e}")
        return False

def main():
    """主函数"""
    print("开始下载模型...")

    # 确保在正确的目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    success_count = 0

    # 下载CosyVoice模型
    if download_cosyvoice_model():
        success_count += 1

    # 下载Dolphin模型
    if download_dolphin_model():
        success_count += 1

    print(f"\n下载完成！成功下载 {success_count}/2 个模型")

    if success_count < 2:
        print("部分模型下载失败，请检查网络连接或手动下载")
        sys.exit(1)
    else:
        print("所有模型下载成功！现在可以构建Docker镜像了")

if __name__ == "__main__":
    main()