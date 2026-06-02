#!/usr/bin/env bash
# Batch H17: more natural OpenNews data instead of synthetic title-attach.
# H17.1: H15.2 + opennh 3M (replace 2M) — wider name distribution
# H17.2: H15.2 + threechar_zh 1M (replace 500k) — denser 3-char Chinese names
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
    local BCV3=$(python3 scripts/eval_badcase_set.py --model "$M" --input data/badcase_eval_v3.jsonl --show-fail 0 2>&1 | grep OVERALL | grep -oE '[0-9]+/[0-9]+')
    local SIZE=$(ls -lh "$M" | awk '{print $5}')
    echo "RESULT $(basename "$M" .wac): size=$SIZE F1_pdmp=$F_PDMP F1_12m=$F_12M names=$NAME bcv2=$BCV2 bcv3=$BCV3" | tee -a experiments/results.log
}

eval_pruned() {
    local M=$1; local THR=$2
    local OUT="${M%.wac}_p${THR}.wac"
    ./build/wapic convert -m "$M" --save-binary --save-prune --prune-threshold "$THR" "$OUT" 2>/dev/null
    eval_full "$OUT"
}

echo "=== Batch H17: natural OpenNews 3M / 3char 1M scaling ===" | tee -a experiments/results.log
date | tee -a experiments/results.log

# H17.1: H15.2 + opennh 3M (replaces 2M)
NAME=h17_1_h152_oh3m_i50
cat data/opennews_nh_3m_train.txt data/clean_aug_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/antimerge_train.txt data/anchor_1m.txt data/threechar_zh_train.txt > data/h17_1.txt
./build/wapic fit $COMMON -i 50 data/h17_1.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac
eval_pruned data/exp_${NAME}.wac 0.15

# H17.2: H15.2 + threechar_zh 1M (replaces 500k)
NAME=h17_2_h152_3charzh1m_i50
cat data/opennews_nh_2m_train.txt data/clean_aug_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/antimerge_train.txt data/anchor_1m.txt data/threechar_zh_1m_train.txt > data/h17_2.txt
./build/wapic fit $COMMON -i 50 data/h17_2.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac
eval_pruned data/exp_${NAME}.wac 0.15

echo "===Batch H17 DONE===" | tee -a experiments/results.log
date | tee -a experiments/results.log
