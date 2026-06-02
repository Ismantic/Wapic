#!/usr/bin/env bash
# Batch H9: NEW Stage 2 architecture, validating H6 ablation insights.
# Recipe: opennews_nh_500k (essential, H6.3) + antimerge_v2 (essential, H6.1) +
#         modern_PD (replaces pd_mp_train, user insight) + anchor_1m (essential, H6.5)
# Drop clean_aug (H6.2 showed redundant).
# Drop pd_mp_train from Stage 2 (saves it for Stage 3 H10).
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

echo "=== Batch H9: NEW Stage 2 architecture (no clean_aug, no pd_mp_train) ===" | tee -a experiments/results.log
date | tee -a experiments/results.log

# H9.1 main: opennh + antimerge_v2 + modPD300k + anchor1m, i=50
cat data/opennews_nh_500k_train.txt data/antimerge_v2_train.txt data/pd_modern_300k.txt data/anchor_1m.txt > data/h9_1.txt
NAME=h9_1_opennh_antiv2_modPD300k_anchor1m_i50
./build/wapic fit $COMMON -i 50 data/h9_1.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac
eval_pruned data/exp_${NAME}.wac 0.05

# H9.2 same but modPD 500k (more modern data)
cat data/opennews_nh_500k_train.txt data/antimerge_v2_train.txt data/pd_modern_500k.txt data/anchor_1m.txt > data/h9_2.txt
NAME=h9_2_opennh_antiv2_modPD500k_anchor1m_i50
./build/wapic fit $COMMON -i 50 data/h9_2.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac
eval_pruned data/exp_${NAME}.wac 0.05

# H9.3 same with old antimerge (control vs antimerge_v2 equivalence)
cat data/opennews_nh_500k_train.txt data/antimerge_train.txt data/pd_modern_300k.txt data/anchor_1m.txt > data/h9_3.txt
NAME=h9_3_opennh_antiOLD_modPD300k_anchor1m_i50
./build/wapic fit $COMMON -i 50 data/h9_3.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac
eval_pruned data/exp_${NAME}.wac 0.05

# H9.4 modPD weighted ×2 (more PD-style signal)
cat data/opennews_nh_500k_train.txt data/antimerge_v2_train.txt data/pd_modern_300k.txt data/pd_modern_300k.txt data/anchor_1m.txt > data/h9_4.txt
NAME=h9_4_opennh_antiv2_modPD300kx2_anchor1m_i50
./build/wapic fit $COMMON -i 50 data/h9_4.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac
eval_pruned data/exp_${NAME}.wac 0.05

echo "===Batch H9 DONE===" | tee -a experiments/results.log
date | tee -a experiments/results.log
