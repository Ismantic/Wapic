#!/usr/bin/env bash
# Wait for the 3M NER PID to finish, then launch full OpenNews NER.
set -e
cd "$(dirname "$0")/.."

PREV_PID=${1:-1474418}
LIMIT_OUT=${2:-10000000}

# Wait for previous NER process to exit
while kill -0 "$PREV_PID" 2>/dev/null; do
    sleep 60
done
echo "[chain] previous NER PID $PREV_PID exited at $(date)"

# Activate venv and launch full OpenNews NER
source ~/.venv310/bin/activate
exec python3 scripts/ner_opennews_full.py \
    --input /home/tfbao/Data/data/OpenNews.sentences.txt \
    --input-format txt \
    --output data/raw/opennews_full_nh.jsonl \
    --batch 32 --max-chars 100 \
    --filter-nh-only \
    --limit-out "$LIMIT_OUT" \
    --progress-every 200
