#!/usr/bin/env bash
# Clean resume: skip PD/antimerge duplicates, only run new param/data combos
set -e
cd /home/tfbao/Shiyu/Wapic

INIT=data/all12m_compact_v2.wac

eval_one() {
    ./build/wapic test -m "$1" "$3" /tmp/r.txt 2>/dev/null >/dev/null
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
gold=rd_gold('$2'); pred=rd_pred('/tmp/r.txt')
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
    [[ ! -f "$M" || $(stat -c%s "$M") -lt 100000 ]] && { echo "SKIP $(basename $M): empty/missing" | tee -a experiments/results.log; return; }
    local F_PDMP=$(eval_one "$M" data/pd_mp_test.txt data/pd_mp_test_nolabel.txt)
    local F_12M=$(eval_one "$M" data/all12m_test.txt data/all12m_test_nolabel.txt)
    local NAME=$(python3 scripts/test_name_cases.py --model "$M" 2>&1 | grep 'TOTAL' | grep -oE '[0-9]+/[0-9]+')
    local BCV2=$(python3 scripts/eval_badcase_set.py --model "$M" --input data/badcase_eval_v2.jsonl --show-fail 0 2>&1 | grep OVERALL | grep -oE '[0-9]+/[0-9]+')
    local SIZE=$(ls -lh "$M" | awk '{print $5}')
    echo "RESULT $(basename "$M" .wac): size=$SIZE F1_pdmp=$F_PDMP F1_12m=$F_12M names=$NAME bcv2=$BCV2" | tee -a experiments/results.log
}

echo "=== Clean Resume (skipping PD/antimerge duplicates) ===" | tee -a experiments/results.log
date | tee -a experiments/results.log

# H11.5/6/7: iter sweep on H2.3 recipe (uses h11_e bin)
for ITER in 60 70 40; do
    if [[ "$ITER" == "60" ]]; then NAME=h11_5_h23_i60; fi
    if [[ "$ITER" == "70" ]]; then NAME=h11_6_h23_i70; fi
    if [[ "$ITER" == "40" ]]; then NAME=h11_7_h23_i40; fi
    ./build/wapic fit -p data/pattern.txt --init-from $INIT \
        -a l-bfgs -i $ITER -1 0.01 -2 0.0001 -t 4 --histsz 5 -e 1e-9 --save-binary \
        data/h11_e.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
    eval_full data/exp_${NAME}.wac
done

# H11.8: H2.3 + anchor 2M (new data scaling)
NAME=h11_8_h23_anchor2m_i50
./build/wapic fit -p data/pattern.txt --init-from $INIT \
    -a l-bfgs -i 50 -1 0.01 -2 0.0001 -t 4 --histsz 5 -e 1e-9 --save-binary \
    data/h11_f.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac

# H11.9/10: H2.3 + modPD 300k variants (uses h11_a bin)
NAME=h11_9_h23_PLUS_modPD300k_i60
./build/wapic fit -p data/pattern.txt --init-from $INIT \
    -a l-bfgs -i 60 -1 0.01 -2 0.0001 -t 4 --histsz 5 -e 1e-9 --save-binary \
    data/h11_a.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac

NAME=h11_10_h23_PLUS_modPD300k_l005_i50
./build/wapic fit -p data/pattern.txt --init-from $INIT \
    -a l-bfgs -i 50 -1 0.005 -2 0.0001 -t 4 --histsz 5 -e 1e-9 --save-binary \
    data/h11_a.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac

echo "===Resume H11 DONE===" | tee -a experiments/results.log

# H12: skip H12.6 (antimerge ×5 duplicate)
COMMON_H12="-p data/pattern.txt --init-from $INIT -a l-bfgs -2 0.0001 -t 4 -e 1e-9 --save-binary"

NAME=h12_1_oh1m_h23_i50
[[ ! -f data/h12_1.txt ]] && {
    cat data/opennews_nh_1m_train.txt data/clean_aug_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/antimerge_train.txt data/anchor_1m.txt > data/h12_1.txt
}
./build/wapic fit $COMMON_H12 --histsz 5 -1 0.01 -i 50 data/h12_1.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac

NAME=h12_2_oh2m_h23_i50
[[ ! -f data/h12_2.txt ]] && {
    cat data/opennews_nh_2m_train.txt data/clean_aug_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/antimerge_train.txt data/anchor_1m.txt > data/h12_2.txt
}
./build/wapic fit $COMMON_H12 --histsz 5 -1 0.01 -i 50 data/h12_2.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac

NAME=h12_3_anc3m_h23_i50
[[ ! -f data/h12_3.txt ]] && {
    cat data/opennews_nh_500k_train.txt data/clean_aug_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/antimerge_train.txt data/anchor_3m.txt > data/h12_3.txt
}
./build/wapic fit $COMMON_H12 --histsz 5 -1 0.01 -i 50 data/h12_3.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac

NAME=h12_4_anc5m_h23_i50
[[ ! -f data/h12_4.txt ]] && {
    cat data/opennews_nh_500k_train.txt data/clean_aug_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/antimerge_train.txt data/anchor_5m.txt > data/h12_4.txt
}
./build/wapic fit $COMMON_H12 --histsz 5 -1 0.01 -i 50 data/h12_4.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac

NAME=h12_5_oh1m_anc3m_h23_i50
[[ ! -f data/h12_5.txt ]] && {
    cat data/opennews_nh_1m_train.txt data/clean_aug_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/antimerge_train.txt data/anchor_3m.txt > data/h12_5.txt
}
./build/wapic fit $COMMON_H12 --histsz 5 -1 0.01 -i 50 data/h12_5.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac

# Skip H12.6 (antimerge ×5 duplicate)

NAME=h12_7_h23_hist10_i50
./build/wapic fit $COMMON_H12 --histsz 10 -1 0.01 -i 50 data/h11_e.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac

NAME=h12_8_h23_hist3_i50
./build/wapic fit $COMMON_H12 --histsz 3 -1 0.01 -i 50 data/h11_e.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac

NAME=h12_9_oh1m_anc5m_h23_i50
[[ ! -f data/h12_9.txt ]] && {
    cat data/opennews_nh_1m_train.txt data/clean_aug_train.txt data/pd_mp_train.txt data/pd_mp_train.txt data/antimerge_train.txt data/anchor_5m.txt > data/h12_9.txt
}
./build/wapic fit $COMMON_H12 --histsz 5 -1 0.01 -i 50 data/h12_9.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac

NAME=h12_10_h23_i80
./build/wapic fit $COMMON_H12 --histsz 5 -1 0.01 -i 80 data/h11_e.txt data/exp_${NAME}.wac > experiments/log_${NAME}.txt 2>&1
eval_full data/exp_${NAME}.wac

echo "===Clean Resume DONE===" | tee -a experiments/results.log
date | tee -a experiments/results.log
