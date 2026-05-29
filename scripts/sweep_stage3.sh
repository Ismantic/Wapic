#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

BASE=data/sweep_s2_i100_l01.wac
TRAIN=data/pd_mp_train.txt
PATTERN=data/pattern.txt
OUT_MD=logs_sweep_stage3.md
TEST_GOLD=data/pd_mp_test.txt
TEST_NOLAB=data/pd_mp_test_nolabel.txt

echo "| name | iter | L1 | size(MB) | F1 | name 15句 |" > $OUT_MD
echo "|---|---|---|---|---|---|" >> $OUT_MD

run_one() {
    local NAME=$1 ITER=$2 L1=$3 TH=$4
    local OUT=data/sweep_s3_${NAME}.wac
    local LOG=logs_sweep_s3_${NAME}.txt
    echo "=== $NAME : -i $ITER -1 $L1 ==="
    ./build/wapic fit -p $PATTERN --init-from $BASE \
        -a l-bfgs -i $ITER -1 $L1 -2 0.0001 -t 4 --histsz 5 -e 1e-9 \
        --save-binary --save-prune --prune-threshold $TH \
        $TRAIN $OUT > $LOG 2>&1
    local SIZE_MB=$(stat -c %s $OUT | awk '{printf "%.1f", $1/1048576}')
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
    local NAME_OK=$(python3 scripts/test_name_cases.py --model $OUT 2>&1 | grep TOTAL | grep -oE '[0-9]+/[0-9]+' | head -1)
    echo "| $NAME | $ITER | $L1 | $SIZE_MB | $F1 | $NAME_OK |" >> $OUT_MD
    echo "RESULT $NAME: size=${SIZE_MB}MB F1=$F1 names=$NAME_OK"
}

run_one i10_l03  10 0.3  0.05
run_one i20_l03  20 0.3  0.05
run_one i30_l03  30 0.3  0.05
run_one i10_l01  10 0.1  0.05
run_one i20_l01  20 0.1  0.05
run_one i30_l01  30 0.1  0.05

echo "===DONE==="
cat $OUT_MD
