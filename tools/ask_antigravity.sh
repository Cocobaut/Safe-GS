#!/usr/bin/env bash
# Gọi Antigravity CLI (agy) từ project Safe-GS, không tương tác.
#   bash tools/ask_antigravity.sh <model-id> <file-prompt.md>
#   model-id: xem `agy.exe models` (vd claude-opus-5-5-medium, gemini-3.8-flash-medium)
# Kết quả in ra stdout. Quyền công cụ giữ mặc định (không --dangerously-skip-permissions).
set -euo pipefail
AGY="C:/Users/Admin/.gemini/bin/agy.exe"
MODEL="${1:?model-id}"
PROMPT_FILE="${2:?file prompt}"
exec "$AGY" --model "$MODEL" --add-dir "E:/Safe-GS" --output-format text --print-timeout 0 --print="$(cat "$PROMPT_FILE")"
