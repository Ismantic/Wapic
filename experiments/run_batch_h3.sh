#!/usr/bin/env bash
# Batch H3: attack last case "李镇全是著名的学者" (sentence-start 3-char name).
# Uses opennews_start3 (sentence-start 3-char Nh real news, 300k) + prune 0.05.
# Start from A0. Anchor 1M.
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
    local OUT=/tmp/$(basename "$M" .wac)_p${THR}.wac
    ./build/wapic convert -m "$M" --save-binary --save-prune --prune-threshold "$THR" "$OUT" 2>/dev/null
    local F_PDMP=$(eval_one "$OUT" data/pd_mp_test.txt data/pd_mp_test_nolabel.txt)
    local F_12M=$(eval_one "$OUT" data/all12m_test.txt data/all12m_test_nolabel.txt)
    local NAME=$(python3 scripts/test_name_cases.py --model "$OUT" 2>&1 | grep 'TOTAL' | grep -oE '[0-9]+/[0-9]+')
    local SIZE=$(ls -lh "$OUT" | awk '{print $5}')
    echo "RESULT $(basename "$M" .wac)_p${THR}: size=$SIZE F1_pdmp=$F_PDMP F1_12m=$F_12M names=$NAME" | tee -a experiments/results.log
}

echo "=== Batch H3: attack last case with start-3 data, prune 0.05 ===" | tee -a experiments/results.log
date | tee -a experiments/results.log

# H3.1: H2.0 recipe + start3 300k
NAME=h3_1_opennh500k_start3_anchor1m_pd2x_i30
cat data/opennews_nh_500k_train.txt data/opennews_start3_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/antimerge_train.txt data/anchor_1m.txt > data/h3_1.txt
./build/wapic fit $COMMON -i 30 data/h3_1.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac
eval_pruned data/exp_${NAME}.wac 0.05

# H3.2: H2.1 recipe + start3
NAME=h3_2_opennh500k_clean_start3_anchor1m_pd2x_i30
cat data/opennews_nh_500k_train.txt data/clean_aug_train.txt data/opennews_start3_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/antimerge_train.txt data/anchor_1m.txt > data/h3_2.txt
./build/wapic fit $COMMON -i 30 data/h3_2.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac
eval_pruned data/exp_${NAME}.wac 0.05

# H3.3: H2.0 + start3 doubled (heavier signal for sentence-start)
NAME=h3_3_opennh500k_start3x2_anchor1m_pd2x_i30
cat data/opennews_nh_500k_train.txt data/opennews_start3_train.txt data/opennews_start3_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/antimerge_train.txt data/anchor_1m.txt > data/h3_3.txt
./build/wapic fit $COMMON -i 30 data/h3_3.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac
eval_pruned data/exp_${NAME}.wac 0.05

# H3.4: H3.1 with i=50 more iter
NAME=h3_4_opennh500k_start3_anchor1m_pd2x_i50
./build/wapic fit $COMMON -i 50 data/h3_1.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac
eval_pruned data/exp_${NAME}.wac 0.05

echo "===Batch H3 DONE===" | tee -a experiments/results.log
date | tee -a experiments/results.log
