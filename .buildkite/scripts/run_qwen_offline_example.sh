#!/usr/bin/env bash

set -euo pipefail
sudo apt update
sudo apt install ffmpeg -y

# Move to repo root (script lives in .buildkite/scripts/)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

UV_INSTALL_DIR="${ROOT_DIR}/.local/bin"
case ":${PATH}:" in
  *":${UV_INSTALL_DIR}:"*) ;;
  *) export PATH="${UV_INSTALL_DIR}:${PATH}" ;;
esac

if ! command -v uv >/dev/null 2>&1; then
  mkdir -p "${UV_INSTALL_DIR}"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${UV_INSTALL_DIR}" sh
fi

UV_BIN=(uv)

"${UV_BIN[@]}" python install 3.12
"${UV_BIN[@]}" venv --python 3.12 .venv
source .venv/bin/activate

"${UV_BIN[@]}" pip install -e .
"${UV_BIN[@]}" pip install vllm==0.11.0 --torch-backend=auto

EXAMPLE_DIR="examples/offline_inference/qwen2_5_omni"
cd "${EXAMPLE_DIR}"

python end2end.py --output-wav output_audio \
                  --query-type use_audio_in_video \
                  --no-save-results
