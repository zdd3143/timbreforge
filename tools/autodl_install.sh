#!/usr/bin/env bash
set -Eeuo pipefail

# AutoDL GPU setup for the local Qwen path. Run inside the repository root.
# It never downloads model weights automatically; pass --download-models only
# after reviewing the model licenses and the instance disk budget.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${TIMBREFORGE_AUTODL_ENV:-${ROOT_DIR}/.autodl-venv}"
MODEL_DIR="${TIMBREFORGE_AUTODL_MODELS:-${ROOT_DIR}/models}"
DOWNLOAD_MODELS=0
for arg in "$@"; do
  [[ "$arg" == "--download-models" ]] && DOWNLOAD_MODELS=1
done

command -v python3 >/dev/null || { echo "缺少 python3" >&2; exit 1; }
python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel

# Qwen3-TTS 0.1.1 is the tested API baseline. Install the project core/UI
# separately so model dependencies remain isolated from the CPU package.
python -m pip install -e "$ROOT_DIR"
python -m pip install 'qwen-tts==0.1.1' 'transformers==4.57.3' accelerate safetensors
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
    print('capability:', torch.cuda.get_device_capability(0))
PY

mkdir -p "$MODEL_DIR"
cat > "$ROOT_DIR/timbreforge.autodl.toml" <<EOF
[qwen]
python = "${ENV_DIR}/bin/python"
design_model = "${MODEL_DIR}/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
base_model = "${MODEL_DIR}/Qwen3-TTS-12Hz-0.6B-Base"

[planner]
python = "${ENV_DIR}/bin/python"
model = "${MODEL_DIR}/Qwen3-1.7B"

[paths]
outputs_dir = "${ROOT_DIR}/.local-data/outputs"
characters_dir = "${ROOT_DIR}/.local-data/characters"
EOF

if (( DOWNLOAD_MODELS )); then
  python -m pip install -U huggingface_hub
  python - <<PY
from huggingface_hub import snapshot_download
for repo, target in {
    'Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign': '${MODEL_DIR}/Qwen3-TTS-12Hz-1.7B-VoiceDesign',
    'Qwen/Qwen3-TTS-12Hz-0.6B-Base': '${MODEL_DIR}/Qwen3-TTS-12Hz-0.6B-Base',
    'Qwen/Qwen3-1.7B': '${MODEL_DIR}/Qwen3-1.7B',
}.items():
    print('Downloading', repo, 'to', target)
    snapshot_download(repo_id=repo, local_dir=target)
PY
fi

python -m timbreforge.launcher --config "$ROOT_DIR/timbreforge.autodl.toml" --check || true
echo
echo "配置已写入 $ROOT_DIR/timbreforge.autodl.toml"
echo "启动：python -m timbreforge.launcher --config timbreforge.autodl.toml --host 0.0.0.0 --port 6006"
echo "模型下载需显式执行：bash tools/autodl_install.sh --download-models"
