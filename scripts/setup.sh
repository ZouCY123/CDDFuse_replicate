#!/usr/bin/env bash
# setup.sh — 一键配置 conda 环境
# 自动检测服务器 CUDA 版本，安装匹配的 PyTorch
#
# 用法：bash scripts/setup.sh [env_name]
#   env_name 默认为 cddfuse

set -e

ENV_NAME="${1:-cddfuse-env}"
PYTHON_VER="3.11"

# ── 1. 检测 CUDA 版本 ─────────────────────────────────────
echo ">>> 检测 CUDA 版本..."
if command -v nvcc &> /dev/null; then
    CUDA_RAW=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    echo "    nvcc CUDA: $CUDA_RAW"
elif [ -f /usr/local/cuda/version.txt ]; then
    CUDA_RAW=$(cat /usr/local/cuda/version.txt | awk '{print $3}')
    echo "    文件读取 CUDA: $CUDA_RAW"
else
    CUDA_RAW="cpu"
    echo "    未检测到 CUDA，将使用 CPU 版 PyTorch"
fi

# 取主次版本号，如 11.3.1 -> 11.3
CUDA_VER=$(echo "$CUDA_RAW" | cut -d. -f1,2)

# ── 2. 选择 PyTorch 安装命令 ─────────────────────────────
case "$CUDA_VER" in
    11.1)
        TORCH_CMD="pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 \
            -f https://download.pytorch.org/whl/torch_stable.html"
        ;;
    11.3|11.4|11.5)
        TORCH_CMD="pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
            -f https://download.pytorch.org/whl/torch_stable.html"
        ;;
    11.6|11.7)
        TORCH_CMD="pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 \
            -f https://download.pytorch.org/whl/torch_stable.html"
        ;;
    11.8)
        TORCH_CMD="pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
            -f https://download.pytorch.org/whl/torch_stable.html"
        ;;
    12.1|12.2)
        TORCH_CMD="pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
            -f https://download.pytorch.org/whl/torch_stable.html"
        ;;
    cpu)
        TORCH_CMD="pip install torch==2.1.2 torchvision==0.16.2"
        ;;
    *)
        echo "⚠️  未匹配到预设 CUDA 版本 ($CUDA_VER)"
        echo "   请手动安装 PyTorch: https://pytorch.org/get-started/previous-versions/"
        exit 1
        ;;
esac

# ── 3. 创建 conda 环境 ────────────────────────────────────
echo ">>> 创建 conda 环境: $ENV_NAME (python=$PYTHON_VER)"
conda create -n "$ENV_NAME" python="$PYTHON_VER" -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ── 4. 安装 PyTorch ───────────────────────────────────────
echo ">>> 安装 PyTorch (CUDA $CUDA_VER)..."
echo "    $TORCH_CMD"
eval "$TORCH_CMD"

# ── 5. 安装其余依赖 ───────────────────────────────────────
echo ">>> 安装项目依赖..."
pip install -r requirements.txt

# ── 6. 验证 ──────────────────────────────────────────────
echo ">>> 验证安装..."
python - << 'PYEOF'
import torch
print(f"    PyTorch : {torch.__version__}")
print(f"    CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    GPU     : {torch.cuda.get_device_name(0)}")
PYEOF

echo ""
echo "✓ 完成！激活命令: conda activate $ENV_NAME"
