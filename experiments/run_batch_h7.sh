#!/usr/bin/env bash
# Batch H7: Stage 3 with OLD PeopleDaily 3M cut (1940s-50s era) — minimal check.
# Expect: no improvement (vocab too far from PD-1998 era).
# Modern PD slice goes to H8.
set -e
cd "$(dirname "$0")/.."

INIT_H23=data/exp_h2_3_opennh500k_clean_anchor1m_pd2x_i50.wac
COMMON_BASE="-p data/pattern.txt -a l-bfgs -2 0.0001 -t 4 --histsz 5 -e 1e-9 --save-binary"

eval_one() {
    local M=$1; local GOLD=$2; local NOLBL=$3
    ./build/wapic test -m "$M" "$NOLBL" /tmp/r.txt 2>/dev/null >/dev/null
    python3 -c "
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
gold=rd_gold('$GOLD'); pred=rd_pred('/tmp/r.txt')
n=min(len(gold),len(pred)); tp=fp=fn=0
for (cc,gt),pt in zip(gold[:n],pred[:n]):
    if len(cc)!=len(pt): continue
    g=spans(cc,gt); p=spans(cc,pt)
    tp+=len(g&p); fp+=len(p-g); fn+=len(g-p)
P=tp/(tp+fp); R=tp/(tp+fn); F=2*P*R/(P+R)
print(f'{F*100:.2f}')
"
}

eval_full() {
    local M=$1
    local F_PDMP=$(eval_one "$M" data/pd_mp_test.txt data/pd_mp_test_nolabel.txt)
    local F_12M=$(eval_one "$M" data/all12m_test.txt data/all12m_test_nolabel.txt)
    local NAME=$(python3 scripts/test_name_cases.py --model "$M" 2>&1 | grep 'TOTAL' | grep -oE '[0-9]+/[0-9]+')
    local SIZE=$(ls -lh "$M" | awk '{print $5}')
    echo "RESULT $(basename "$M" .wac): size=$SIZE F1_pdmp=$F_PDMP F1_12m=$F_12M names=$NAME" | tee -a experiments/results.log
}

eval_pruned() {
    local M=$1; local THR=$2
    local OUT=/tmp/$(basename "$M" .wac)_p${THR}.wac
    ./build/wapic convert -m "$M" --save-binary --save-prune --prune-threshold "$THR" "$OUT" 2>/dev/null
    local F_PDMP=$(eval_one "$OUT" data/pd_mp_test.txt data/pd_mp_test_nolabel.txt)
    local F_12M=$(eval_one "$OUT" data/all12m_test.txt data/all12m_test_nolabel.txt)
    local NAME=$(python3 scripts/test_name_cases.py --model "$OUT" 2>&1 | grep 'TOTAL' | grep -oE '[0-9]+/[0-9]+')
    local SIZE=$(ls -lh "$OUT" | awk '{print $5}')
    echo "RESULT $(basename "$M" .wac)_p${THR}: size=$SIZE F1_pdmp=$F_PDMP F1_12m=$F_12M names=$NAME" | tee -a experiments/results.log
}

run_stage3() {
    local NAME=$1; local INIT=$2; local DATA=$3; local ITER=$4; local L1=$5
    ./build/wapic fit $COMMON_BASE --init-from $INIT -i $ITER -1 $L1 \
        $DATA data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
    eval_full data/exp_${NAME}.wac
    eval_pruned data/exp_${NAME}.wac 0.05
}

echo "=== Batch H7: Stage 3 with old PD3M cut (reference, expect no help) ===" | tee -a experiments/results.log
date | tee -a experiments/results.log

# Just 3 quick reference experiments to confirm hypothesis
run_stage3 h7_1_h23_pd3m100k_i5  $INIT_H23 data/pd3m_100k.txt 5  0.01
run_stage3 h7_2_h23_pd3m300k_i5  $INIT_H23 data/pd3m_300k.txt 5  0.01
run_stage3 h7_3_h23_pd3m500k_i5  $INIT_H23 data/pd3m_500k.txt 5  0.01

echo "===Batch H7 DONE===" | tee -a experiments/results.log
date | tee -a experiments/results.log
