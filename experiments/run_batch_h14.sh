#!/usr/bin/env bash
# Batch H14: heavier foreign-dot attack. H11.2 recipe + bigger foreign-dot data.
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
    local BCV2=$(python3 scripts/eval_badcase_set.py --model "$M" --input data/badcase_eval_v2.jsonl --show-fail 0 2>&1 | grep OVERALL | grep -oE '[0-9]+/[0-9]+')
    local SIZE=$(ls -lh "$M" | awk '{print $5}')
    echo "RESULT $(basename "$M" .wac): size=$SIZE F1_pdmp=$F_PDMP F1_12m=$F_12M names=$NAME bcv2=$BCV2" | tee -a experiments/results.log
}

eval_pruned() {
    local M=$1; local THR=$2
    local OUT="${M%.wac}_p${THR}.wac"
    ./build/wapic convert -m "$M" --save-binary --save-prune --prune-threshold "$THR" "$OUT" 2>/dev/null
    local F_PDMP=$(eval_one "$OUT" data/pd_mp_test.txt data/pd_mp_test_nolabel.txt)
    local F_12M=$(eval_one "$OUT" data/all12m_test.txt data/all12m_test_nolabel.txt)
    local NAME=$(python3 scripts/test_name_cases.py --model "$OUT" 2>&1 | grep 'TOTAL' | grep -oE '[0-9]+/[0-9]+')
    local BCV2=$(python3 scripts/eval_badcase_set.py --model "$OUT" --input data/badcase_eval_v2.jsonl --show-fail 0 2>&1 | grep OVERALL | grep -oE '[0-9]+/[0-9]+')
    local SIZE=$(ls -lh "$OUT" | awk '{print $5}')
    echo "RESULT $(basename "$OUT" .wac): size=$SIZE F1_pdmp=$F_PDMP F1_12m=$F_12M names=$NAME bcv2=$BCV2" | tee -a experiments/results.log
}

echo "=== Batch H14: heavier foreign-dot attack ===" | tee -a experiments/results.log
date | tee -a experiments/results.log

# H11.2 core
H112_CORE="data/opennews_nh_500k_train.txt data/clean_aug_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/antimerge_train.txt data/pd_modern_500k.txt data/anchor_1m.txt"

# H14.1: H11.2 + foreign_dot_ALL 306k
cat $H112_CORE data/foreign_dot_ALL_train.txt > data/h14_1.txt
NAME=h14_1_h112_PLUS_fdotALL_i50
./build/wapic fit $COMMON -i 50 data/h14_1.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac
eval_pruned data/exp_${NAME}.wac 0.15

# H14.2: H11.2 + foreign_dot_ALL + foreign_dot_1plus ×5 (heavily weight 1-char prefix)
cat $H112_CORE data/foreign_dot_ALL_train.txt data/foreign_dot_1plus_train.txt data/foreign_dot_1plus_train.txt data/foreign_dot_1plus_train.txt data/foreign_dot_1plus_train.txt data/foreign_dot_1plus_train.txt > data/h14_2.txt
NAME=h14_2_h112_PLUS_fdotALL_1plus5x_i50
./build/wapic fit $COMMON -i 50 data/h14_2.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac
eval_pruned data/exp_${NAME}.wac 0.15

# H14.3: same as H14.2 but i=60 (more iter)
NAME=h14_3_h112_PLUS_fdotALL_1plus5x_i60
./build/wapic fit $COMMON -i 60 data/h14_2.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac
eval_pruned data/exp_${NAME}.wac 0.15

echo "===Batch H14 DONE===" | tee -a experiments/results.log
date | tee -a experiments/results.log
