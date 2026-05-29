#!/usr/bin/env bash
# Stage 2 parameter sweep: warm-start from 12M base, train on 500k name-rich.
# Evaluates F1 on pd_mp_test + name case 15-句 pass rate.
# Writes a markdown row per experiment to logs_sweep_stage2.md.

set -e
cd "$(dirname "$0")/.."

BASE=data/all12m_compact_v2.wac
TRAIN=data/name_aug_v2_train.txt
PATTERN=data/pattern.txt
OUT_MD=logs_sweep_stage2.md
TEST_GOLD=data/pd_mp_test.txt
TEST_NOLAB=data/pd_mp_test_nolabel.txt

echo "| name | iter | L1 | threshold | size(MB) | F1 | name 15句 |" > $OUT_MD
echo "|---|---|---|---|---|---|---|" >> $OUT_MD

run_one() {
    local NAME=$1 ITER=$2 L1=$3 TH=$4
    local OUT=data/sweep_s2_${NAME}.wac
    local LOG=logs_sweep_s2_${NAME}.txt
    echo "=== $NAME : -i $ITER -1 $L1 --prune-threshold $TH ==="
    ./build/wapic fit -p $PATTERN --init-from $BASE \
        -a l-bfgs -i $ITER -1 $L1 -2 0.0001 -t 4 --histsz 5 -e 1e-9 \
        --save-binary --save-prune --prune-threshold $TH \
        $TRAIN $OUT > $LOG 2>&1
    local SIZE=$(stat -c %s $OUT 2>/dev/null)
    local SIZE_MB=$(echo "scale=1; $SIZE/1048576" | bc)
    # F1
    ./build/wapic test -m $OUT $TEST_NOLAB /tmp/sweep_pred.txt 2>&1 >/dev/null
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
def spans(chars,tags):
    sp=[];p=0;cur=''
    for ch,tg in zip(chars,tags):
        if tg in ('B','S'):
            if cur: sp.append((p-len(cur),p))
            cur=ch
        else: cur+=ch
        p+=1
    if cur: sp.append((p-len(cur),p))
    return set(sp)
gold=rd_gold('$TEST_GOLD'); pred=rd_pred('/tmp/sweep_pred.txt')
n=min(len(gold),len(pred)); tp=fp=fn=0
for (cc,gt),pt in zip(gold[:n],pred[:n]):
    if len(cc)!=len(pt): continue
    g=spans(cc,gt); p=spans(cc,pt)
    tp+=len(g&p); fp+=len(p-g); fn+=len(g-p)
P=tp/(tp+fp); R=tp/(tp+fn); F=2*P*R/(P+R)
print(f'{F*100:.2f}')
")
    # name case
    local NAME_OK=$(python3 scripts/test_name_cases.py --model $OUT 2>&1 | grep TOTAL | grep -oE '[0-9]+/[0-9]+' | head -1)
    echo "| $NAME | $ITER | $L1 | $TH | $SIZE_MB | $F1 | $NAME_OK |" >> $OUT_MD
    echo "RESULT $NAME: size=${SIZE_MB}MB F1=$F1 names=$NAME_OK"
}

run_one i20_l03  20  0.3  0.05
run_one i30_l03  30  0.3  0.05
run_one i50_l03  50  0.3  0.05
run_one i50_l01  50  0.1  0.05
run_one i100_l01 100 0.1  0.05
run_one i100_l005 100 0.05 0.05

echo "===DONE==="
cat $OUT_MD
