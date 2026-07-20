#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$root/validation_logs"
nvidia-smi dmon -s pucvmet -d 1 -c 30 > "$root/validation_logs/hybrid_nvidia_smi_dmon.log" &
monitor_pid=$!
"$root/scripts/run_hybrid_smoke.sh"
wait "$monitor_pid" || true
