#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

eval_full() {
    local M=$1
    ./build/wapic test -m $M data/pd_mp_test_nolabel.txt /tmp/r.txt 2>&1 >/dev/null
    local F1=$(python3 -c "
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
gold=rd_gold('data/pd_mp_test.txt'); pred=rd_pred('/tmp/r.txt')
n=min(len(gold),len(pred)); tp=fp=fn=0
for (cc,gt),pt in zip(gold[:n],pred[:n]):
    if len(cc)!=len(pt): continue
    g=spans(cc,gt); p=spans(cc,pt)
    tp+=len(g&p); fp+=len(p-g); fn+=len(g-p)
P=tp/(tp+fp); R=tp/(tp+fn); F=2*P*R/(P+R)
print(f'{F*100:.2f}')
")
    local NAME=$(python3 scripts/test_name_cases.py --model $M 2>&1 | grep 'TOTAL' | grep -oE '[0-9]+/[0-9]+')
    local SIZE=$(ls -lh $M | awk '{print $5}')
    echo "RESULT $(basename $M .wac): size=$SIZE F1=$F1 names=$NAME" | tee -a experiments/results.log
}

echo "=== Batch 7: add anti-merge templates ===" | tee -a experiments/results.log

# G1: E5 配方 + antimerge
cat data/clean_aug_train.txt data/hardcase_train.txt data/hardcase_train.txt data/hardcase_train.txt data/antimerge_train.txt data/pd_mp_train.txt data/pd_mp_train.txt > data/combined_e5_anti.txt
NAME=g1_clean_hard3x_anti_pd2x_i30
./build/wapic fit -p data/pattern.txt --init-from data/wapic-20260529.wac \
    -a l-bfgs -i 30 -1 0.01 -2 0.0001 -t 4 --histsz 5 -e 1e-9 \
    --save-binary \
    data/combined_e5_anti.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac

# G2: anti 2x
cat data/clean_aug_train.txt data/hardcase_train.txt data/hardcase_train.txt data/hardcase_train.txt data/antimerge_train.txt data/antimerge_train.txt data/pd_mp_train.txt data/pd_mp_train.txt > data/combined_e5_anti2x.txt
NAME=g2_clean_hard3x_anti2x_pd2x_i30
./build/wapic fit -p data/pattern.txt --init-from data/wapic-20260529.wac \
    -a l-bfgs -i 30 -1 0.01 -2 0.0001 -t 4 --histsz 5 -e 1e-9 \
    --save-binary \
    data/combined_e5_anti2x.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac

# G3: D3 base + antimerge
cat data/clean_aug_train.txt data/hardcase_train.txt data/antimerge_train.txt data/pd_mp_train.txt data/pd_mp_train.txt > data/combined_d3_anti.txt
NAME=g3_clean_hard_anti_pd2x_i30
./build/wapic fit -p data/pattern.txt --init-from data/wapic-20260529.wac \
    -a l-bfgs -i 30 -1 0.01 -2 0.0001 -t 4 --histsz 5 -e 1e-9 \
    --save-binary \
    data/combined_d3_anti.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac

# G4: anti 3x
cat data/clean_aug_train.txt data/hardcase_train.txt data/hardcase_train.txt data/hardcase_train.txt data/antimerge_train.txt data/antimerge_train.txt data/antimerge_train.txt data/pd_mp_train.txt data/pd_mp_train.txt > data/combined_e5_anti3x.txt
NAME=g4_clean_hard3x_anti3x_pd2x_i30
./build/wapic fit -p data/pattern.txt --init-from data/wapic-20260529.wac \
    -a l-bfgs -i 30 -1 0.01 -2 0.0001 -t 4 --histsz 5 -e 1e-9 \
    --save-binary \
    data/combined_e5_anti3x.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac

# G5: full mix with pd3x
cat data/clean_aug_train.txt data/hardcase_train.txt data/hardcase_train.txt data/antimerge_train.txt data/antimerge_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/pd_mp_train.txt > data/combined_full_mix.txt
NAME=g5_clean_hard2x_anti2x_pd3x_i30
./build/wapic fit -p data/pattern.txt --init-from data/wapic-20260529.wac \
    -a l-bfgs -i 30 -1 0.01 -2 0.0001 -t 4 --histsz 5 -e 1e-9 \
    --save-binary \
    data/combined_full_mix.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac

echo "===Batch 7 DONE===" | tee -a experiments/results.log
