#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH="/usr/local/lib/cudnn8:${LD_LIBRARY_PATH:-}"

echo "[entrypoint] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# Fail tidlig hvis cuDNN8 ikke er synlig
if ! ldconfig -p | grep -q "libcudnn_ops_infer.so.8"; then
  echo "[entrypoint] ERROR: libcudnn_ops_infer.so.8 ikke funnet via ldconfig."
  echo "[entrypoint] Innhold i /usr/local/lib/cudnn8:"
  ls -la /usr/local/lib/cudnn8 || true
  exit 1
fi

exec "$@"
