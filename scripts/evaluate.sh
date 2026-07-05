#!/usr/bin/env bash
# Evaluate one or more models on the published PDMP and 12M test sets.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR=${WAPIC_DATA_DIR:-data/dataset/test}
WAPIC_BIN=${WAPIC_BIN:-./build/wapic}
RESULT=$(mktemp)
trap 'rm -f "$RESULT"' EXIT

eval_one() {
    local M=$1
    local GOLD=$2
    local NOLBL=$3
    "$WAPIC_BIN" test -m "$M" "$NOLBL" "$RESULT" 2>/dev/null >/dev/null
    python3 - "$GOLD" "$RESULT" <<'PY'
import sys

def rd_gold(p):
    s=[];c=[];t=[]
    for line in open(p,encoding='utf-8'):
        line=line.rstrip()
        if not line:
            if c: s.append((c,t)); c,t=[],[]; continue
        else:
            x=line.split(); c.append(x[0]); t.append(x[-1])
    if c: s.append((c,t))
    return s
def rd_pred(p):
    s=[];c=[]
    for line in open(p,encoding='utf-8'):
        line=line.rstrip()
        if not line:
            if c: s.append(c); c=[]; continue
        else:
            if line.startswith('score='): continue
            c.append(line.split()[0])
    if c: s.append(c)
    return s

def spans(cc,tt):
    sp=[];p=0;cur=''
    for c,t in zip(cc,tt):
        if t in ('B','S'):
            if cur: sp.append((p-len(cur),p))
            cur=c
        else: cur+=c
        p+=1
    if cur: sp.append((p-len(cur),p))
    return set(sp)

gold=rd_gold(sys.argv[1]); pred=rd_pred(sys.argv[2])
n=min(len(gold),len(pred)); tp=fp=fn=0
for (cc,gt),pt in zip(gold[:n],pred[:n]):
    if len(cc)!=len(pt): continue
    g=spans(cc,gt); p=spans(cc,pt)
    tp+=len(g&p); fp+=len(p-g); fn+=len(g-p)
P=tp/(tp+fp); R=tp/(tp+fn); F=2*P*R/(P+R)
print(f'{F*100:.2f}')
PY
}

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 MODEL [MODEL ...]" >&2
    exit 2
fi

for M in "$@"; do
    [[ -f "$M" ]] || { echo "Model not found: $M" >&2; exit 1; }
    SIZE=$(ls -lh "$M" | awk '{print $5}')
    F_PDMP=$(eval_one "$M" "$DATA_DIR/pdmp_test.txt" "$DATA_DIR/pdmp_test_nolabel.txt")
    F_12M=$(eval_one "$M" "$DATA_DIR/12m_test.txt" "$DATA_DIR/12m_test_nolabel.txt")
    echo "RESULT $(basename "$M" .wac): size=$SIZE F1_pdmp=$F_PDMP F1_12m=$F_12M"
done
