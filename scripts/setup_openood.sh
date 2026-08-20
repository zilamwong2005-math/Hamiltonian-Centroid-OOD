#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENOOD_DIR="${PROJECT_ROOT}/OpenOOD"
OPENOOD_URL="https://github.com/Jingkang50/OpenOOD.git"
OPENOOD_COMMIT="3c35632ee91b54b09d1f085d04f94744cece7d0b"
OVERRIDE_ROOT="${PROJECT_ROOT}/openood_overrides"

if [[ ! -d "${OPENOOD_DIR}/.git" ]]; then
  git clone "${OPENOOD_URL}" "${OPENOOD_DIR}"
fi

git -C "${OPENOOD_DIR}" fetch --tags origin
git -C "${OPENOOD_DIR}" checkout --detach "${OPENOOD_COMMIT}"

cp -f \
  "${OVERRIDE_ROOT}/openood/postprocessors/dice_postprocessor.py" \
  "${OPENOOD_DIR}/openood/postprocessors/dice_postprocessor.py"
cp -f \
  "${OVERRIDE_ROOT}/openood/postprocessors/rmds_postprocessor.py" \
  "${OPENOOD_DIR}/openood/postprocessors/rmds_postprocessor.py"
cp -f \
  "${OVERRIDE_ROOT}/openood/postprocessors/she_postprocessor.py" \
  "${OPENOOD_DIR}/openood/postprocessors/she_postprocessor.py"

echo "OpenOOD prepared at ${OPENOOD_COMMIT}"
echo "Local postprocessor overrides installed from ${OVERRIDE_ROOT}"

