#!/usr/bin/env bash
# Batch H8: Stage 3 with MODERN PD (last 5M of PeopleDaily, NER-processed) on H2.3 base.
# This is the main breakthrough attempt — fresh data the model has never seen.
set -e
cd "$(dirname "$0")/.."

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

echo "=== Batch H8: Stage 3 with MODERN PD (NER-processed) ===" | tee -a experiments/results.log
date | tee -a experiments/results.log

# Filter out any pd_mp_test exact matches (mostly short standard phrases like 致读者)
python3 scripts/filter_test_leak.py \
    --input data/raw/pd_modern_ner.jsonl \
    --output data/raw/pd_modern_ner_clean.jsonl \
    --test-bmes data/pd_mp_test.txt

# Build modern PD BMES from cleaned NER jsonl
python3 scripts/jsonl_to_bmes.py \
    --input data/raw/pd_modern_ner_clean.jsonl \
    --out-bmes data/pd_modern_full.txt \
    --no-type --min-chars 3 --max-chars 200 2>&1 | tail -3

# Sample subsets
python3 scripts/sample_sentences.py --input data/pd_modern_full.txt --output data/pd_modern_100k.txt --n 100000  --seed 42 2>/dev/null
python3 scripts/sample_sentences.py --input data/pd_modern_full.txt --output data/pd_modern_300k.txt --n 300000  --seed 42 2>/dev/null
python3 scripts/sample_sentences.py --input data/pd_modern_full.txt --output data/pd_modern_500k.txt --n 500000  --seed 42 2>/dev/null
python3 scripts/sample_sentences.py --input data/pd_modern_full.txt --output data/pd_modern_1m.txt   --n 1000000 --seed 42 2>/dev/null

ls -lh data/pd_modern_*.txt

# H8.1: modern PD 100k, i=5
run_stage3 h8_1_h23_pdmod100k_i5  $INIT_H23 data/pd_modern_100k.txt 5  0.01

# H8.2: modern PD 300k, i=5
run_stage3 h8_2_h23_pdmod300k_i5  $INIT_H23 data/pd_modern_300k.txt 5  0.01

# H8.3: modern PD 500k, i=5
run_stage3 h8_3_h23_pdmod500k_i5  $INIT_H23 data/pd_modern_500k.txt 5  0.01

# H8.4: modern PD 1M, i=5
run_stage3 h8_4_h23_pdmod1m_i5    $INIT_H23 data/pd_modern_1m.txt   5  0.01

# H8.5: modern PD 300k, i=10
run_stage3 h8_5_h23_pdmod300k_i10 $INIT_H23 data/pd_modern_300k.txt 10 0.01

# H8.6: modern PD 300k + anchor 300k, i=5 (protect F1_12m)
cat data/pd_modern_300k.txt data/anchor_300k.txt > data/h8_6.txt
run_stage3 h8_6_h23_pdmod300k_anchor300k_i5 $INIT_H23 data/h8_6.txt 5 0.01

# H8.7: modern PD 500k + pd_mp + anchor 100k, i=5  (mix old+new PD)
cat data/pd_modern_500k.txt data/pd_mp_train.txt data/anchor_100k.txt > data/h8_7.txt
run_stage3 h8_7_h23_pdmod500k_pdmp_anchor100k_i5 $INIT_H23 data/h8_7.txt 5 0.01

# H8.8: modern PD 1M + anchor 300k, i=5
cat data/pd_modern_1m.txt data/anchor_300k.txt > data/h8_8.txt
run_stage3 h8_8_h23_pdmod1m_anchor300k_i5 $INIT_H23 data/h8_8.txt 5 0.01

echo "===Batch H8 DONE===" | tee -a experiments/results.log
date | tee -a experiments/results.log
