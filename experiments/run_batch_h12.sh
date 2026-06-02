#!/usr/bin/env bash
# Batch H12: H2.3 recipe data scaling extreme (no pattern change, no Stage 3).
# H2.3 = opennh500k + clean_aug + pd_mp×2 + antimerge + anchor1m, i=50
# Vary: opennh size (1M, 2M), anchor size (2M, 3M, 5M), antimerge weight, histsz.
set -e
cd "$(dirname "$0")/.."

INIT=data/all12m_compact_v2.wac
COMMON="-p data/pattern.txt --init-from $INIT -a l-bfgs -2 0.0001 -t 4 -e 1e-9 --save-binary"

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
    local NAME=$1; local DATA=$2; local ITER=$3; local L1=$4; local HISTSZ=${5:-5}
    ./build/wapic fit $COMMON --histsz $HISTSZ -1 $L1 -i $ITER \
        $DATA data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
    eval_full data/exp_${NAME}.wac
    eval_pruned data/exp_${NAME}.wac 0.05
}

echo "=== Batch H12: H2.3 data scaling extreme ===" | tee -a experiments/results.log
date | tee -a experiments/results.log

# Build data combos (all keep clean_aug + pd_mp×2 + antimerge as H2.3 core)
PD2X="data/pd_mp_train.txt data/pd_mp_train.txt"
CLEAN="data/clean_aug_train.txt"
ANTI="data/antimerge_train.txt"

# H12.1: opennh 1M (vs 500k baseline)
cat data/opennews_nh_1m_train.txt $CLEAN $PD2X $ANTI data/anchor_1m.txt > data/h12_1.txt
run_s2 h12_1_oh1m_h23_i50 data/h12_1.txt 50 0.01

# H12.2: opennh 2M
cat data/opennews_nh_2m_train.txt $CLEAN $PD2X $ANTI data/anchor_1m.txt > data/h12_2.txt
run_s2 h12_2_oh2m_h23_i50 data/h12_2.txt 50 0.01

# H12.3: anchor 3M (vs 1M)
cat data/opennews_nh_500k_train.txt $CLEAN $PD2X $ANTI data/anchor_3m.txt > data/h12_3.txt
run_s2 h12_3_anc3m_h23_i50 data/h12_3.txt 50 0.01

# H12.4: anchor 5M
cat data/opennews_nh_500k_train.txt $CLEAN $PD2X $ANTI data/anchor_5m.txt > data/h12_4.txt
run_s2 h12_4_anc5m_h23_i50 data/h12_4.txt 50 0.01

# H12.5: opennh 1M + anchor 3M (both scaled up)
cat data/opennews_nh_1m_train.txt $CLEAN $PD2X $ANTI data/anchor_3m.txt > data/h12_5.txt
run_s2 h12_5_oh1m_anc3m_h23_i50 data/h12_5.txt 50 0.01

# H12.6: H2.3 + antimerge × 5 (heavier antimerge signal, H6.1 showed antimerge critical)
cat data/opennews_nh_500k_train.txt $CLEAN $PD2X $ANTI $ANTI $ANTI $ANTI $ANTI data/anchor_1m.txt > data/h12_6.txt
run_s2 h12_6_anti5x_h23_i50 data/h12_6.txt 50 0.01

# H12.7: H2.3 + histsz=10 (L-BFGS history longer, may find better optimum)
run_s2 h12_7_h23_hist10_i50 data/h11_e.txt 50 0.01 10

# H12.8: H2.3 + histsz=3 (less L-BFGS smoothing)
run_s2 h12_8_h23_hist3_i50 data/h11_e.txt 50 0.01 3

# H12.9: opennh 1M + anchor 5M (max data both ways)
cat data/opennews_nh_1m_train.txt $CLEAN $PD2X $ANTI data/anchor_5m.txt > data/h12_9.txt
run_s2 h12_9_oh1m_anc5m_h23_i50 data/h12_9.txt 50 0.01

# H12.10: H2.3 recipe i=80 (more iter than H2.3's 50)
run_s2 h12_10_h23_i80 data/h11_e.txt 80 0.01

echo "===Batch H12 DONE===" | tee -a experiments/results.log
date | tee -a experiments/results.log
