#!/bin/bash
# Prepare the Python/runtime environment for unit tests.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

PLATFORM=""
PKG_MGR="uv"
ENV_NAME=""
ENV_PATH="/opt/venv"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform) PLATFORM="$2"; shift 2 ;;
        --pkg-mgr) PKG_MGR="$2"; shift 2 ;;
        --env-name) ENV_NAME="$2"; shift 2 ;;
        --env-path) ENV_PATH="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

[ -n "$PLATFORM" ] || { echo "platform is required" >&2; exit 1; }

cd "$PROJECT_ROOT"
source ./tools/install/utils/pyenv_utils.sh

activate_python_env() {
    case "$PKG_MGR" in
        conda)
            if [ -n "$ENV_NAME" ] && [ -n "$ENV_PATH" ]; then
                activate_conda "$ENV_NAME" "$ENV_PATH" || {
                    echo "Conda activation failed"
                    exit 1
                }
            fi
            ;;
        uv)
            if [ -n "$ENV_PATH" ] && [ -d "$ENV_PATH" ]; then
                activate_uv_env "$ENV_PATH" || {
                    echo "UV activation failed"
                    exit 1
                }
            fi
            ;;
        pip)
            echo "Using system Python with pip"
            ;;
        *)
            echo "Unsupported package manager: $PKG_MGR" >&2
            exit 1
            ;;
    esac
}

install_common_python_deps() {
    python -m pip install coverage pytest-mock  diffusers==0.36.0 transformers==4.57.6 --quiet --root-user-action=ignore 
}

setup_cuda_unit_env() {
    local install_dir=""
    if [ "$PKG_MGR" = "conda" ] && [ -n "$ENV_PATH" ]; then
        install_dir=$(dirname "$ENV_PATH")
    fi

    local install_args=(
        --platform cuda
        --task train
        --pkg-mgr "$PKG_MGR"
        --no-system --no-dev --no-base --no-task
        --src-deps megatron-lm
        --pip-deps typer
        --force-build
        --retry-count 3
    )
    [ -n "$ENV_NAME" ] && install_args+=(--env-name "$ENV_NAME")
    [ -n "$install_dir" ] && install_args+=(--install-dir "$install_dir")

    ./tools/install/install.sh "${install_args[@]}"

    # TODO: remove after CI images contain these dependencies.
    python -m pip install \
        qwen_vl_utils==0.0.14 \
        diffusers==0.36.0 \
        websocket-client==1.8.0 \
        websocket==0.2.1 \
        websockets==15.0.1 \
        msgpack==1.1.0 \
        datasets==4.5.0
}

setup_metax_unit_env() {
    
    git clone https://github.com/flagos-ai/Megatron-LM-FL.git
    cd Megatron-LM-FL
    git checkout d092f8df49f7c0b5b4cae42d036b7e4a26b8fc81
    pip install . --no-build-isolation 

    git clone --depth 1 https://github.com/flagos-ai/TransformerEngine-FL.git
    cd TransformerEngine-FL
    TE_FL_SKIP_CUDA=1  pip install . --no-build-isolation 
}

setup_ascend_unit_env() {
    pip install datasets==4.5.0 omegaconf==2.3.0 diffusers==0.36.0 hydra-core==1.3.2
    echo "Ascend CI image is expected to provide platform runtime dependencies"
}

echo "Preparing unit test environment"
echo "Platform: $PLATFORM"
echo "Package Manager: $PKG_MGR"
echo "Environment Name: $ENV_NAME"
echo "Environment Path: $ENV_PATH"

activate_python_env

echo "Python location: $(command -v python)"
echo "Python version: $(python --version)"

install_common_python_deps

case "$PLATFORM" in
    cuda) setup_cuda_unit_env ;;
    ascend) setup_ascend_unit_env ;;
    metax) setup_metax_unit_env ;;
    *) echo "No platform-specific unit setup for $PLATFORM" ;;
esac

mkdir -p /opt/data
cp -r /home/gitlab-runner/data/Megatron-LM/* /opt/data/ 2>/dev/null || true
cp -r /home/gitlab-runner/tokenizers/Megatron-LM/* /opt/data/ 2>/dev/null || true

echo "Unit test environment ready"
