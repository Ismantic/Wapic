#!/usr/bin/env bash
# Batch H11: pure Stage 2 exploration (A0 base, single training pass, no Stage 3).
# Strategy: H2.3 + various aug / iter / data variations to find 15/15 + F1_pdmp ≥ 97.67.
set -e
cd "$(dirname "$0")/.."

INIT=data/all12m_compact_v2.wac
COMMON="-p data/pattern.txt --init-from $INIT -a l-bfgs -1 0.01 -2 0.0001 -t 4 --histsz 5 -e 1e-9 --save-binary"

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

run_s2() {
    local NAME=$1; local DATA=$2; local ITER=$3; local L1=$4
    ./build/wapic fit $COMMON -i $ITER -1 $L1 \
        $DATA data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
    eval_full data/exp_${NAME}.wac
    eval_pruned data/exp_${NAME}.wac 0.05
}

echo "=== Batch H11: pure Stage 2 exploration ===" | tee -a experiments/results.log
date | tee -a experiments/results.log

# Build common data files
cat data/opennews_nh_500k_train.txt data/clean_aug_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/antimerge_train.txt data/pd_modern_300k.txt data/anchor_1m.txt > data/h11_a.txt
cat data/opennews_nh_500k_train.txt data/clean_aug_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/antimerge_train.txt data/pd_modern_500k.txt data/anchor_1m.txt > data/h11_b.txt
cat data/opennews_nh_500k_train.txt data/opennews_nh_500k_train.txt data/clean_aug_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/antimerge_train.txt data/anchor_1m.txt > data/h11_c.txt
cat data/opennews_nh_500k_train.txt data/clean_aug_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/antimerge_train.txt data/pd_modern_300k.txt data/anchor_1m.txt > data/h11_d.txt
cat data/opennews_nh_500k_train.txt data/clean_aug_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/antimerge_train.txt data/anchor_1m.txt > data/h11_e.txt  # exactly H2.3 recipe
# sample anchor 2M
python3 scripts/sample_sentences.py --input data/all12m_train.txt --output data/anchor_2m.txt --n 2000000 --seed 42 2>&1 | tail -1
cat data/opennews_nh_500k_train.txt data/clean_aug_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/antimerge_train.txt data/anchor_2m.txt > data/h11_f.txt

# H11.1: H2.3 + modPD 300k (additive), i=50
run_s2 h11_1_h23_PLUS_modPD300k_i50 data/h11_a.txt 50 0.01

# H11.2: H2.3 + modPD 500k, i=50
run_s2 h11_2_h23_PLUS_modPD500k_i50 data/h11_b.txt 50 0.01

# H11.3: H2.3 + opennh ×2 (more name signal), i=50
run_s2 h11_3_h23_opennhx2_i50 data/h11_c.txt 50 0.01

# H11.4: H2.3 + pd ×3 (heavier PD), i=50
run_s2 h11_4_h23_pd3x_modPD_i50 data/h11_d.txt 50 0.01

# H11.5: H2.3 recipe but i=60 (more iter, may unlock more)
run_s2 h11_5_h23_i60 data/h11_e.txt 60 0.01

# H11.6: H2.3 recipe but i=70
run_s2 h11_6_h23_i70 data/h11_e.txt 70 0.01

# H11.7: H2.3 recipe but i=40 (less iter — see case stability vs F1)
run_s2 h11_7_h23_i40 data/h11_e.txt 40 0.01

# H11.8: H2.3 + larger anchor (2M), i=50
run_s2 h11_8_h23_anchor2m_i50 data/h11_f.txt 50 0.01

# H11.9: H2.3 + modPD 300k, i=60 (combine more iter + more data)
run_s2 h11_9_h23_PLUS_modPD300k_i60 data/h11_a.txt 60 0.01

# H11.10: H2.3 + modPD 300k, L1=0.005, i=50 (lighter reg)
run_s2 h11_10_h23_PLUS_modPD300k_l005_i50 data/h11_a.txt 50 0.005

echo "===Batch H11 DONE===" | tee -a experiments/results.log
date | tee -a experiments/results.log
