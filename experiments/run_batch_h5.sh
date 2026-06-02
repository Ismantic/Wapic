#!/usr/bin/env bash
# Batch H5: Stage 3 fine-tunes, mostly from H2.3 (the 15/15 winner).
# Goal: push F1_pdmp from 97.59 to ≥ 97.67 without losing 15/15 case.
# Sweep axes: iter, L1, pd weight, anchor, antimerge, clean_aug, opennews_nh small.
set -e
cd "$(dirname "$0")/.."

INIT_H20=data/exp_h2_0_opennh500k_anchor1m_pd2x_i30.wac
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

echo "=== Batch H5: Stage 3 micro fine-tunes (focus on H2.3 base) ===" | tee -a experiments/results.log
date | tee -a experiments/results.log

# Pre-build common Stage 3 data combos
cat data/pd_mp_train.txt data/pd_mp_train.txt > data/h5_pd2x.txt
cat data/pd_mp_train.txt data/pd_mp_train.txt data/pd_mp_train.txt > data/h5_pd3x.txt
cat data/pd_mp_train.txt data/anchor_100k.txt > data/h5_pd_anc100k.txt
cat data/pd_mp_train.txt data/anchor_300k.txt > data/h5_pd_anc300k.txt
cat data/pd_mp_train.txt data/pd_mp_train.txt data/anchor_100k.txt > data/h5_pd2x_anc100k.txt
cat data/pd_mp_train.txt data/pd_mp_train.txt data/anchor_300k.txt > data/h5_pd2x_anc300k.txt
cat data/pd_mp_train.txt data/pd_mp_train.txt data/antimerge_train.txt > data/h5_pd2x_anti.txt
cat data/pd_mp_train.txt data/antimerge_train.txt > data/h5_pd_anti.txt
cat data/pd_mp_train.txt data/clean_aug_train.txt > data/h5_pd_clean.txt
cat data/pd_mp_train.txt data/pd_mp_train.txt data/clean_aug_train.txt > data/h5_pd2x_clean.txt
cat data/pd_mp_train.txt data/pd_mp_train.txt data/antimerge_train.txt data/anchor_100k.txt > data/h5_pd2x_anti_anc100k.txt

# ============ Group A: from H2.3 (15/15, F1_pdmp 97.59) push F1 ============

# A.1: PD only, iter sweep (3, 5, 10, 20)
run_stage3 h5_01_h23_pd_i3        $INIT_H23 data/pd_mp_train.txt        3  0.01
run_stage3 h5_02_h23_pd_i5        $INIT_H23 data/pd_mp_train.txt        5  0.01
run_stage3 h5_03_h23_pd_i10       $INIT_H23 data/pd_mp_train.txt        10 0.01
run_stage3 h5_04_h23_pd_i20       $INIT_H23 data/pd_mp_train.txt        20 0.01

# A.2: PD only, L1 sweep at i=5 (0.005, 0.02)
run_stage3 h5_05_h23_pd_i5_l005   $INIT_H23 data/pd_mp_train.txt        5  0.005
run_stage3 h5_06_h23_pd_i5_l02    $INIT_H23 data/pd_mp_train.txt        5  0.02

# A.3: PD weight sweep at i=5
run_stage3 h5_07_h23_pd2x_i5      $INIT_H23 data/h5_pd2x.txt            5  0.01
run_stage3 h5_08_h23_pd3x_i5      $INIT_H23 data/h5_pd3x.txt            5  0.01

# A.4: PD + anchor (protect F1_12m)
run_stage3 h5_09_h23_pd_anc100k_i5    $INIT_H23 data/h5_pd_anc100k.txt    5  0.01
run_stage3 h5_10_h23_pd_anc300k_i5    $INIT_H23 data/h5_pd_anc300k.txt    5  0.01
run_stage3 h5_11_h23_pd2x_anc100k_i5  $INIT_H23 data/h5_pd2x_anc100k.txt  5  0.01
run_stage3 h5_12_h23_pd2x_anc300k_i5  $INIT_H23 data/h5_pd2x_anc300k.txt  5  0.01

# A.5: PD + antimerge (boundary cue)
run_stage3 h5_13_h23_pd_anti_i5       $INIT_H23 data/h5_pd_anti.txt       5  0.01
run_stage3 h5_14_h23_pd2x_anti_i5     $INIT_H23 data/h5_pd2x_anti.txt     5  0.01

# A.6: PD + clean_aug (more domain mix)
run_stage3 h5_15_h23_pd_clean_i5      $INIT_H23 data/h5_pd_clean.txt      5  0.01
run_stage3 h5_16_h23_pd2x_clean_i5    $INIT_H23 data/h5_pd2x_clean.txt    5  0.01

# A.7: PD + antimerge + anchor (compound protection)
run_stage3 h5_17_h23_pd2x_anti_anc100k_i5  $INIT_H23 data/h5_pd2x_anti_anc100k.txt 5  0.01

# A.8: PD i=10 with anchor (more iter + protection)
run_stage3 h5_18_h23_pd_anc100k_i10  $INIT_H23 data/h5_pd_anc100k.txt  10 0.01
run_stage3 h5_19_h23_pd2x_anc300k_i10 $INIT_H23 data/h5_pd2x_anc300k.txt 10 0.01

# Group B removed: H2.0 start is wrong (missing 李镇全 case knowledge — see H3 regression).

echo "===Batch H5 DONE===" | tee -a experiments/results.log
date | tee -a experiments/results.log
