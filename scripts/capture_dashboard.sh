#!/usr/bin/env bash
# Start the dashboard on the crisis replay, wait until the whole replay is processed, capture
# light and dark screenshots of every tab (docs/screenshots/), compress them, stop the server.
#
#   bash scripts/capture_dashboard.sh [path-to-python-with-playwright] [path-to-runtime-python]
set -euo pipefail
cd "$(dirname "$0")/.."
SHOT_PY="${1:-.venv-train/Scripts/python.exe}"
RUN_PY="${2:-.venv/Scripts/python.exe}"
PORT=8010

PYTHONIOENCODING=utf-8 "$RUN_PY" -W ignore main.py serve --no-browser --speed 7200 --port "$PORT" > data/output/capture_server.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do curl -s -o /dev/null "http://127.0.0.1:$PORT/api/status" && break; sleep 2; done
echo "server up; waiting for the replay to finish ..."
for _ in $(seq 1 300); do
  state=$(curl -s "http://127.0.0.1:$PORT/api/status" | "$RUN_PY" -c "import json,sys; d=json.load(sys.stdin); print(d['replay']['finished'] and not d['busy'])" || echo False)
  [ "$state" = "True" ] && break
  sleep 2
done
sleep 3
for theme in light dark; do
  PYTHONIOENCODING=utf-8 "$SHOT_PY" scripts/screenshots.py --url "http://127.0.0.1:$PORT" --theme "$theme" --wait 4
done
# Quantise to 256 colours: dashboards are flat colour, so this cuts file size ~3x with no visible change.
"$RUN_PY" - <<'EOF'
from pathlib import Path
from PIL import Image
for png in sorted(Path("docs/screenshots").glob("*.png")):
    before = png.stat().st_size
    Image.open(png).convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(png, optimize=True)
    print(f"{png.name}: {before / 1e6:.2f} MB -> {png.stat().st_size / 1e6:.2f} MB")
EOF
