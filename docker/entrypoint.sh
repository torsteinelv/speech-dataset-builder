#!/usr/bin/env bash
set -euo pipefail

# Sørg for at vår cuDNN8-mappe er i søkestien (i tillegg til system/default)
export LD_LIBRARY_PATH="/usr/local/lib/cudnn8:${LD_LIBRARY_PATH:-}"

echo "[entrypoint] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# Debug/hard fail hvis cuDNN8-liba mangler (gir mye bedre feilmelding i logs)
if ! ldconfig -p | grep -q "libcudnn_ops_infer.so.8"; then
  echo "[entrypoint] ERROR: libcudnn_ops_infer.so.8 ikke funnet i ldconfig cache."
  echo "[entrypoint] Innhold i /usr/local/lib/cudnn8:"
  ls -la /usr/local/lib/cudnn8 || true
  echo "[entrypoint] Treffer på libcudnn_ops_infer:"
  find /usr/local/lib/cudnn8 -maxdepth 1 -name 'libcudnn_ops_infer.so.8*' -print || true
  exit 1
fi

exec "$@"
