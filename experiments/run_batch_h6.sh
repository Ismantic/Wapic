#!/usr/bin/env bash
# Batch H6: H2.3 full ablation — leave one out of each data component.
# H2.3 recipe = opennews_nh_500k + clean_aug + pd_mp×2 + antimerge + anchor_1m, i=50
# Compare case + F1 to know which piece(s) actually unlock the 15/15.
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

echo "=== Batch H6: H2.3 leave-one-out ablation (i=50) ===" | tee -a experiments/results.log
date | tee -a experiments/results.log

# H6.1: minus antimerge
cat data/opennews_nh_500k_train.txt data/clean_aug_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/anchor_1m.txt > data/h6_1_no_anti.txt
NAME=h6_1_h23_NO_anti_i50
./build/wapic fit $COMMON -i 50 data/h6_1_no_anti.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac
eval_pruned data/exp_${NAME}.wac 0.05

# H6.2: minus clean_aug
cat data/opennews_nh_500k_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/antimerge_train.txt data/anchor_1m.txt > data/h6_2_no_clean.txt
NAME=h6_2_h23_NO_clean_i50
./build/wapic fit $COMMON -i 50 data/h6_2_no_clean.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac
eval_pruned data/exp_${NAME}.wac 0.05

# H6.3: minus opennews_nh (THE big one — 500k real Nh sentences)
cat data/clean_aug_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/antimerge_train.txt data/anchor_1m.txt > data/h6_3_no_opennh.txt
NAME=h6_3_h23_NO_opennh_i50
./build/wapic fit $COMMON -i 50 data/h6_3_no_opennh.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac
eval_pruned data/exp_${NAME}.wac 0.05

# H6.4: minus pd_mp (no PD modernization data at all)
cat data/opennews_nh_500k_train.txt data/clean_aug_train.txt data/antimerge_train.txt data/anchor_1m.txt > data/h6_4_no_pd.txt
NAME=h6_4_h23_NO_pd_i50
./build/wapic fit $COMMON -i 50 data/h6_4_no_pd.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac
eval_pruned data/exp_${NAME}.wac 0.05

# H6.5: minus anchor 1M
cat data/opennews_nh_500k_train.txt data/clean_aug_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/antimerge_train.txt > data/h6_5_no_anchor.txt
NAME=h6_5_h23_NO_anchor_i50
./build/wapic fit $COMMON -i 50 data/h6_5_no_anchor.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac
eval_pruned data/exp_${NAME}.wac 0.05

echo "===Batch H6 DONE===" | tee -a experiments/results.log
date | tee -a experiments/results.log
