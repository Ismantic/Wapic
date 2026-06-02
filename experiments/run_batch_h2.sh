#!/usr/bin/env bash
# Batch H2: add OpenNews NER 500k as legit name aug (replaces/augments clean_aug).
# Start from A0. Anchor 1M kept from H1.3 winner. No hardcase.
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

echo "=== Batch H2: add OpenNews NER 500k from A0 base, anchor 1M ===" | tee -a experiments/results.log
date | tee -a experiments/results.log

# H2.0: opennews_nh REPLACES clean_aug, anchor 1M, pd×2, antimerge
NAME=h2_0_opennh500k_anchor1m_pd2x_i30
cat data/opennews_nh_500k_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/antimerge_train.txt data/anchor_1m.txt > data/h2_0.txt
./build/wapic fit $COMMON -i 30 data/h2_0.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac

# H2.1: opennews_nh + clean_aug + anchor 1M + pd×2 + antimerge
NAME=h2_1_opennh500k_clean_anchor1m_pd2x_i30
cat data/opennews_nh_500k_train.txt data/clean_aug_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/antimerge_train.txt data/anchor_1m.txt > data/h2_1.txt
./build/wapic fit $COMMON -i 30 data/h2_1.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac

# H2.2: opennews_nh ×2 (heavier weight) + anchor 1M + pd×2 + antimerge
NAME=h2_2_opennh500kx2_anchor1m_pd2x_i30
cat data/opennews_nh_500k_train.txt data/opennews_nh_500k_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/antimerge_train.txt data/anchor_1m.txt > data/h2_2.txt
./build/wapic fit $COMMON -i 30 data/h2_2.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac

# H2.3: H2.1 with i=50 (more iter — anchor should still hold F1_12m)
NAME=h2_3_opennh500k_clean_anchor1m_pd2x_i50
./build/wapic fit $COMMON -i 50 data/h2_1.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac

# H2.4: H2.1 with pd×3 (heavier PD modernization)
NAME=h2_4_opennh500k_clean_anchor1m_pd3x_i30
cat data/opennews_nh_500k_train.txt data/clean_aug_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/antimerge_train.txt data/anchor_1m.txt > data/h2_4.txt
./build/wapic fit $COMMON -i 30 data/h2_4.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac

# H2.5: opennews_nh×2 + clean + anchor1m + pd2x (combo)
NAME=h2_5_opennh500kx2_clean_anchor1m_pd2x_i30
cat data/opennews_nh_500k_train.txt data/opennews_nh_500k_train.txt data/clean_aug_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/antimerge_train.txt data/anchor_1m.txt > data/h2_5.txt
./build/wapic fit $COMMON -i 30 data/h2_5.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac

echo "===Batch H2 DONE===" | tee -a experiments/results.log
date | tee -a experiments/results.log
