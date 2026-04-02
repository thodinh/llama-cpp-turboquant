#!/usr/bin/env bash
# =============================================================================
#  install.sh — TurboQuant one-click installer
#  Repo: https://github.com/thodinh/llama-cpp-turboquant
#
#  Usage:
#    curl -fsSL https://raw.githubusercontent.com/thodinh/llama-cpp-turboquant/master/install.sh | bash
#    # Or, for system-wide install:
#    sudo bash install.sh --system
# =============================================================================

set -euo pipefail

REPO="thodinh/llama-cpp-turboquant"
INSTALL_BASE="${HOME}/.local/share/turboquant"
BIN_DIR="${HOME}/.local/bin"
SYSTEM_INSTALL=false

for arg in "$@"; do
  case $arg in
    --system) SYSTEM_INSTALL=true ;;
    --help|-h)
      echo "Usage: install.sh [--system] [--help]"
      echo "  --system   Install to /usr/local (requires sudo)"
      exit 0
      ;;
  esac
done

if $SYSTEM_INSTALL; then
  INSTALL_BASE="/usr/local/share/turboquant"
  BIN_DIR="/usr/local/bin"
  if [[ $EUID -ne 0 ]]; then
    echo "[error] --system requires root. Run with sudo." >&2
    exit 1
  fi
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()  { echo -e "${GREEN}[turboquant]${NC} $*"; }
info() { echo -e "${CYAN}[info]${NC}       $*"; }
warn() { echo -e "${YELLOW}[warn]${NC}       $*"; }
err()  { echo -e "${RED}[error]${NC}      $*" >&2; exit 1; }

echo -e "${BOLD}"
echo "  ████████╗██╗   ██╗██████╗ ██████╗  ██████╗  ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗"
echo "  ╚══██╔══╝██║   ██║██╔══██╗██╔══██╗██╔═══██╗██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝"
echo "     ██║   ██║   ██║██████╔╝██████╔╝██║   ██║██║   ██║██║   ██║███████║██╔██╗ ██║   ██║   "
echo "     ██║   ██║   ██║██╔══██╗██╔══██╗██║   ██║██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║   "
echo "     ██║   ╚██████╔╝██║  ██║██████╔╝╚██████╔╝╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║   "
echo "     ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═════╝  ╚═════╝  ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   "
echo -e "${NC}"
echo -e "  ${CYAN}llama.cpp TurboQuant Edition${NC} — https://github.com/${REPO}"
echo ""

command -v curl &>/dev/null || err "curl is required. Install it and retry."
command -v tar  &>/dev/null || err "tar is required. Install it and retry."

OS=$(uname -s)
ARCH=$(uname -m)

[[ "$OS" == "Linux" ]]  || err "Unsupported OS: ${OS}. Only Linux is supported."
[[ "$ARCH" == "x86_64" ]] || err "Unsupported architecture: ${ARCH}. Only x86_64 is supported."

detect_cuda() {
  if command -v nvidia-smi &>/dev/null; then
    if nvidia-smi --query-gpu=name --format=csv,noheader &>/dev/null 2>&1; then
      return 0
    fi
  fi
  if ls /dev/nvidia[0-9]* &>/dev/null 2>&1; then
    return 0
  fi
  if lsmod 2>/dev/null | grep -q "^nvidia "; then
    return 0
  fi
  return 1
}

VARIANT="linux-cpu-x64"
if detect_cuda; then
  info "NVIDIA GPU detected → selecting CUDA variant"
  VARIANT="linux-cuda-x64"
else
  info "No NVIDIA GPU detected → selecting CPU-only variant"
fi

log "Fetching latest release from GitHub..."

API_URL="https://api.github.com/repos/${REPO}/releases/latest"
LATEST_JSON=$(curl -fsSL \
  -H "Accept: application/vnd.github.v3+json" \
  "${API_URL}" 2>/dev/null) \
  || err "Failed to reach GitHub API. Check your internet connection."

LATEST_TAG=$(echo "${LATEST_JSON}" | grep '"tag_name"' \
  | sed 's/.*"tag_name": *"\(.*\)".*/\1/' | head -n1)

[[ -n "${LATEST_TAG}" ]] || err "Could not determine latest release tag."
info "Latest release: ${BOLD}${LATEST_TAG}${NC}"

FILENAME="turboquant-${LATEST_TAG}-${VARIANT}.tar.gz"
DOWNLOAD_URL="https://github.com/${REPO}/releases/download/${LATEST_TAG}/${FILENAME}"

echo "${LATEST_JSON}" | grep -q "\"${FILENAME}\"" \
  || err "Artifact '${FILENAME}' not found in release ${LATEST_TAG}.\nCheck: https://github.com/${REPO}/releases/tag/${LATEST_TAG}"

TMP_DIR=$(mktemp -d)
trap 'rm -rf "${TMP_DIR}"' EXIT

log "Downloading ${FILENAME}..."
info "URL: ${DOWNLOAD_URL}"
curl -fL --progress-bar "${DOWNLOAD_URL}" -o "${TMP_DIR}/${FILENAME}" \
  || err "Download failed."

VERSIONED_DIR="${INSTALL_BASE}/${LATEST_TAG}"
mkdir -p "${VERSIONED_DIR}" "${BIN_DIR}"

log "Extracting to ${VERSIONED_DIR}..."
tar -xzf "${TMP_DIR}/${FILENAME}" -C "${VERSIONED_DIR}" --strip-components=1

log "Linking binaries → ${BIN_DIR}/"
LINKED=0
for bin_path in "${VERSIONED_DIR}"/llama-*; do
  [[ -f "${bin_path}" && -x "${bin_path}" ]] || continue
  bin_name=$(basename "${bin_path}")
  ln -sf "${bin_path}" "${BIN_DIR}/${bin_name}"
  info "  ${BIN_DIR}/${bin_name}"
  (( LINKED++ )) || true
done
[[ "${LINKED}" -gt 0 ]] || warn "No executables found in the archive."

echo "${LATEST_TAG}" > "${VERSIONED_DIR}/.version"

if ! echo ":${PATH}:" | grep -q ":${BIN_DIR}:"; then
  echo ""
  warn "${BIN_DIR} is not in your PATH."
  warn "Add the following line to your shell config (~/.bashrc or ~/.zshrc):"
  echo ""
  echo -e "    ${CYAN}export PATH=\"\${HOME}/.local/bin:\${PATH}\"${NC}"
  echo ""
  warn "Then reload your shell:"
  echo -e "    ${CYAN}source ~/.bashrc${NC}"
fi

echo ""
echo -e "${GREEN}${BOLD}Installation complete!${NC}"
echo ""
echo -e "  Version   : ${BOLD}${LATEST_TAG}${NC}"
echo -e "  Variant   : ${BOLD}${VARIANT}${NC}"
echo -e "  Install   : ${BOLD}${VERSIONED_DIR}${NC}"
echo -e "  Binaries  : ${BOLD}${BIN_DIR}/llama-*${NC}"
echo ""
echo -e "  ${CYAN}Quick start:${NC}"
echo -e "    llama-cli -hf unsloth/Qwen3.5-35B-A3B-GGUF:Q2_K_XL -ngl 99 -c 4096 -fa on -rea off -cnv"
echo ""
echo -e "  ${CYAN}Run server (OpenAI-compatible API):${NC}"
echo -e "    llama-server -hf unsloth/Qwen3.5-35B-A3B-GGUF:Q2_K_XL -ngl 99 --port 8080"
echo ""
