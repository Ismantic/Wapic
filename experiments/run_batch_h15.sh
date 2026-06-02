#!/usr/bin/env bash
# Batch H15: attack 3-char Chinese name ambiguity using mined 500k pure Chinese 3-char Nh data
# (excluded names that appear in v3 test set to avoid leak)
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
    local F_PDMP=$(eval_one "$OUT" data/pd_mp_test.txt data/pd_mp_test_nolabel.txt)
    local F_12M=$(eval_one "$OUT" data/all12m_test.txt data/all12m_test_nolabel.txt)
    local NAME=$(python3 scripts/test_name_cases.py --model "$OUT" 2>&1 | grep 'TOTAL' | grep -oE '[0-9]+/[0-9]+')
    local BCV2=$(python3 scripts/eval_badcase_set.py --model "$OUT" --input data/badcase_eval_v2.jsonl --show-fail 0 2>&1 | grep OVERALL | grep -oE '[0-9]+/[0-9]+')
    local BCV3=$(python3 scripts/eval_badcase_set.py --model "$OUT" --input data/badcase_eval_v3.jsonl --show-fail 0 2>&1 | grep OVERALL | grep -oE '[0-9]+/[0-9]+')
    local SIZE=$(ls -lh "$OUT" | awk '{print $5}')
    echo "RESULT $(basename "$OUT" .wac): size=$SIZE F1_pdmp=$F_PDMP F1_12m=$F_12M names=$NAME bcv2=$BCV2 bcv3=$BCV3" | tee -a experiments/results.log
}

echo "=== Batch H15: 3-char Chinese name ambiguity attack ===" | tee -a experiments/results.log
date | tee -a experiments/results.log

# H15.1: H12.1 (current release) recipe + 3char_zh 500k
NAME=h15_1_h121_PLUS_3charzh500k_i50
cat data/opennews_nh_1m_train.txt data/clean_aug_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/antimerge_train.txt data/anchor_1m.txt data/threechar_zh_train.txt > data/h15_1.txt
./build/wapic fit $COMMON -i 50 data/h15_1.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac
eval_pruned data/exp_${NAME}.wac 0.15

# H15.2: H12.2 (best F1 candidate) recipe + 3char_zh 500k
NAME=h15_2_h122_PLUS_3charzh500k_i50
cat data/opennews_nh_2m_train.txt data/clean_aug_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/antimerge_train.txt data/anchor_1m.txt data/threechar_zh_train.txt > data/h15_2.txt
./build/wapic fit $COMMON -i 50 data/h15_2.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac
eval_pruned data/exp_${NAME}.wac 0.15

# H15.3: H12.1 recipe + 3char_zh 500k + anchor 2M (try to also boost F1_12m)
NAME=h15_3_h121_3charzh_anc2m_i50
cat data/opennews_nh_1m_train.txt data/clean_aug_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/antimerge_train.txt data/anchor_2m.txt data/threechar_zh_train.txt > data/h15_3.txt
./build/wapic fit $COMMON -i 50 data/h15_3.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac
eval_pruned data/exp_${NAME}.wac 0.15

echo "===Batch H15 DONE===" | tee -a experiments/results.log
date | tee -a experiments/results.log
