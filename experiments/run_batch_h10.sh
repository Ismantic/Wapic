#!/usr/bin/env bash
# Batch H10: Stage 3 from best H9 base, using pd_mp_train (NOW UNSEEN since H9 didn't include it).
# Goal: finally push F1_pdmp ≥ 97.67 (model has fresh PD signal to learn from).
set -e
cd "$(dirname "$0")/.."

# Pick best H9 by F1_pdmp + 15/15
INIT=$(ls -t data/exp_h9_*.wac 2>/dev/null | grep -v "p0.05" | head -1)
if [[ -z "$INIT" ]]; then
    INIT=data/exp_h9_1_opennh_antiv2_modPD300k_anchor1m_i50.wac
fi
echo "[H10] using base: $INIT"
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
    local OUT="${M%.wac}_p${THR}.wac"
    ./build/wapic convert -m "$M" --save-binary --save-prune --prune-threshold "$THR" "$OUT" 2>/dev/null
    local F_PDMP=$(eval_one "$OUT" data/pd_mp_test.txt data/pd_mp_test_nolabel.txt)
    local F_12M=$(eval_one "$OUT" data/all12m_test.txt data/all12m_test_nolabel.txt)
    local NAME=$(python3 scripts/test_name_cases.py --model "$OUT" 2>&1 | grep 'TOTAL' | grep -oE '[0-9]+/[0-9]+')
    local SIZE=$(ls -lh "$OUT" | awk '{print $5}')
    echo "RESULT $(basename "$OUT" .wac): size=$SIZE F1_pdmp=$F_PDMP F1_12m=$F_12M names=$NAME" | tee -a experiments/results.log
}

run_stage3() {
    local NAME=$1; local DATA=$2; local ITER=$3; local L1=$4
    ./build/wapic fit $COMMON_BASE --init-from $INIT -i $ITER -1 $L1 \
        $DATA data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
    eval_full data/exp_${NAME}.wac
    eval_pruned data/exp_${NAME}.wac 0.05
}

echo "=== Batch H10: Stage 3 from H9 base, pd_mp_train (UNSEEN in Stage 2) ===" | tee -a experiments/results.log
date | tee -a experiments/results.log

# Build Stage 3 datasets
cat data/pd_mp_train.txt data/anchor_100k.txt > data/h10_pd_anc.txt
cat data/pd_mp_train.txt data/pd_mp_train.txt > data/h10_pd2x.txt
cat data/pd_mp_train.txt data/pd_mp_train.txt data/anchor_100k.txt > data/h10_pd2x_anc.txt
cat data/pd_mp_train.txt data/pd_mp_train.txt data/antimerge_v2_train.txt > data/h10_pd2x_anti.txt

# H10.1: pd_mp i=3 (lightest)
run_stage3 h10_1_pd_i3 data/pd_mp_train.txt 3 0.01

# H10.2: pd_mp i=5
run_stage3 h10_2_pd_i5 data/pd_mp_train.txt 5 0.01

# H10.3: pd_mp i=10
run_stage3 h10_3_pd_i10 data/pd_mp_train.txt 10 0.01

# H10.4: pd_mp + anchor 100k, i=5
run_stage3 h10_4_pd_anc100k_i5 data/h10_pd_anc.txt 5 0.01

# H10.5: pd_mp×2 + anchor 100k, i=5
run_stage3 h10_5_pd2x_anc100k_i5 data/h10_pd2x_anc.txt 5 0.01

# H10.6: pd_mp×2 + antimerge_v2, i=5
run_stage3 h10_6_pd2x_anti_i5 data/h10_pd2x_anti.txt 5 0.01

# H10.7: pd_mp i=5 L1=0.005 (lighter regularization)
run_stage3 h10_7_pd_i5_l005 data/pd_mp_train.txt 5 0.005

echo "===Batch H10 DONE===" | tee -a experiments/results.log
date | tee -a experiments/results.log
