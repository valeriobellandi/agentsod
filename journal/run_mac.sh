#!/usr/bin/env bash
# Stage 2 with LLM agents on a Mac (Apple Silicon, >= 32 GB; 64 GB recommended), all local.
#
#   ./run_mac.sh setup     # Python venv, tests, Ollama models (~33 GB download, once)
#   ./run_mac.sh smoke     # a few tasks per configuration, then a calibrated time estimate
#   ./run_mac.sh run       # the reduced plan: B0,B1,B2,L2,L3 x 3 seeds (resumable)
#   ./run_mac.sh status    # how many runs are done
#   ./run_mac.sh a5        # long-horizon pairing (A5) with LLM agents, local, L2 vs L3 (~5-7 h)
#   ./run_mac.sh cloud-smoke | cloud | a5-cloud
#                          # larger models on hosted providers (config/providers.large.json);
#                          # needs GROQ_API_KEY, TOGETHER_API_KEY, DEEPINFRA_API_KEY in the environment
#
# The three model families are the same as in config/providers.example.json:
# gpt-oss:20b (M1, also judge for B2), qwen2.5:7b (M2), mistral-small3.2:24b (M3).
# The Mac stays awake while running (caffeinate); keep it on power.
set -euo pipefail
cd "$(dirname "$0")"

MODELS=(gpt-oss:20b qwen2.5:7b mistral-small3.2:24b)
CONFIGS="${CONFIGS:-B0,B1,B2,L2,L3}"
SEEDS="${SEEDS:-1,2,3}"
WORKERS="${WORKERS:-2}"
OUT="${OUT:-results/stage2_llm_local}"
PY=.venv/bin/python
export PYTHONWARNINGS=ignore

ollama_up() {
  if ! command -v ollama >/dev/null; then
    echo "Ollama is not installed: download it from https://ollama.com/download (or: brew install ollama)"; exit 1
  fi
  if ! curl -s http://localhost:11434/api/tags >/dev/null; then
    echo "starting ollama serve..."
    # keep all three models in memory and let two requests run at once
    OLLAMA_MAX_LOADED_MODELS=3 OLLAMA_NUM_PARALLEL="$WORKERS" nohup ollama serve >ollama.log 2>&1 &
    for _ in $(seq 30); do curl -s http://localhost:11434/api/tags >/dev/null && break; sleep 1; done
  fi
}

case "${1:-}" in
  setup)
    python3 -c 'import sys; assert sys.version_info >= (3, 10), "Python >= 3.10 needed"'
    [ -d .venv ] || python3 -m venv .venv
    $PY -m pip install -q --upgrade pip
    $PY -m pip install -q pandapower pytest
    $PY -m pytest -q tests
    ollama_up
    for m in "${MODELS[@]}"; do ollama pull "$m"; done
    echo "setup done"
    ;;
  smoke)
    ollama_up
    caffeinate -i $PY bench/stage2_llm.py --providers config/providers.example.json --local \
      --configs "$CONFIGS" --workers "$WORKERS" --out "$OUT" --smoke
    $PY bench/estimate_cost.py --providers config/providers.example.json --configs "$CONFIGS" \
      --seeds "$(echo "$SEEDS" | tr ',' '\n' | wc -l | tr -d ' ')" --workers "$WORKERS" \
      --calibrate "$OUT/smoke/turns.jsonl" | grep -E "calibrated|Tasks|Sequential"
    echo "(ignore the USD figures above: local runs cost nothing)"
    ;;
  run)
    ollama_up
    caffeinate -i $PY bench/stage2_llm.py --providers config/providers.example.json --local \
      --configs "$CONFIGS" --seeds "$SEEDS" --workers "$WORKERS" --out "$OUT" 2>&1 | tee -a run.log
    ;;
  a5)
    ollama_up
    caffeinate -i $PY bench/a5_llm.py --providers config/providers.example.json --local \
      --seeds "${A5_SEEDS:-1,2,3,4}" --workers "$WORKERS" --out results/a5_llm_local 2>&1 | tee -a run.log
    ;;
  cloud-smoke)
    $PY bench/estimate_cost.py --providers config/providers.large.json | tail -4
    $PY bench/stage2_llm.py --providers config/providers.large.json --configs "$CONFIGS" \
      --workers 4 --out results/stage2_llm_large --smoke
    $PY bench/estimate_cost.py --providers config/providers.large.json --configs "$CONFIGS" \
      --calibrate results/stage2_llm_large/smoke/turns.jsonl | tail -4
    ;;
  cloud)
    caffeinate -i $PY bench/stage2_llm.py --providers config/providers.large.json \
      --configs "$CONFIGS" --seeds "$SEEDS" --workers 4 --out results/stage2_llm_large 2>&1 | tee -a run.log
    ;;
  a5-cloud)
    caffeinate -i $PY bench/a5_llm.py --providers config/providers.large.json \
      --seeds "${A5_SEEDS:-1,2,3,4}" --workers 4 --out results/a5_llm_large 2>&1 | tee -a run.log
    ;;
  status)
    if [ -f "$OUT/rows.jsonl" ]; then
      $PY -c "import json,collections;c=collections.Counter((r['config'],r['seed']) for r in map(json.loads,open('$OUT/rows.jsonl')));print(len(c),'runs done:',sorted(c))"
    else
      echo "no runs yet"
    fi
    ;;
  *)
    sed -n '2,16p' "$0"; exit 1 ;;
esac
