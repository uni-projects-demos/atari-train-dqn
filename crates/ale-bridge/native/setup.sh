#!/usr/bin/env bash
set -euo pipefail

BRIDGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENDOR_DIR="${BRIDGE_DIR}/vendor"

ALE_REPO="${ALE_REPO:-https://github.com/Farama-Foundation/Arcade-Learning-Environment.git}"
ZLIB_REPO="${ZLIB_REPO:-https://github.com/madler/zlib.git}"

ALE_REF="${ALE_REF:-${1:-v0.12.1}}"
ZLIB_REF="${ZLIB_REF:-${2:-v1.3.2}}"

mkdir -p "${VENDOR_DIR}"

clone_or_update() {
  local name="$1"
  local repo="$2"
  local ref="$3"
  local dest="${VENDOR_DIR}/${name}"

  if [[ -d "${dest}/.git" ]]; then
    echo "Updating ${name}..."
    git -C "${dest}" fetch --tags --prune origin
  elif [[ -e "${dest}" ]]; then
    echo "error: ${dest} already exists but not Git checkout" >&2
    exit 1
  else
    echo "Cloning ${name}..."
    git clone --filter=blob:none "${repo}" "${dest}"
  fi

  if [[ -n "${ref}" ]]; then
    echo "Checking out ${name} ref: ${ref}"

    if ! git -C "${dest}" checkout --detach "${ref}" 2>/dev/null; then
      git -C "${dest}" fetch --depth 1 origin "${ref}"
      git -C "${dest}" checkout --detach FETCH_HEAD
    fi
  fi

  local commit
  commit="$(git -C "${dest}" rev-parse HEAD)"
  echo "${name}: ${commit}"
}

clone_or_update "ale" "${ALE_REPO}" "${ALE_REF}"
clone_or_update "zlib" "${ZLIB_REPO}" "${ZLIB_REF}"

echo
echo "Vendor dependencies are ready:"
echo "  ${VENDOR_DIR}/ale"
echo "  ${VENDOR_DIR}/zlib"
