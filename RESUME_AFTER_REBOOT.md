# 重启后恢复指南

## 当前 release 候选已就绪
```
data/wapic-20260601-h11_2.wac  (52M)
F1_pdmp=97.67  F1_12m=97.07  case=15/15  badcase_eval=99.7%
```

## 重启时被中断的任务

1. **H13.1 BC001 攻击实验**（iter 1/50，需重训）
2. **chain_resume**: H11.4-H11.10 + H12（已 chain，未跑）
3. **Mining v2**（H2.3 badcase 挖矿，0 confirmed）

## 重启后启动顺序

```bash
cd /home/tfbao/Shiyu/Wapic

# 1) 立即启动 H13 (~80 min) — 攻 BC001
nohup bash experiments/run_batch_h13.sh > experiments/batch_h13_console.log 2>&1 &

# 2) Chain H11 resume + H12 after H13
nohup bash /tmp/chain_resume.sh > experiments/chain_resume.log 2>&1 &
#   或者重写一个，把它放到 /home/tfbao/Shiyu/Wapic/experiments/chain_resume_v2.sh

# 3) Mining 后台（CPU 空闲时挖）
nohup python3 scripts/mine_badcases_v2.py \
    --model data/exp_h2_3_opennh500k_clean_anchor1m_pd2x_i50_p0.05.wac \
    --output data/badcases_h23_v2.jsonl \
    --limit-in 500000 --limit-out 200 --batch 64 \
    > logs/mine_h23_v2.log 2>&1 &
```

## 跑完后选 release

按优先级看：
1. `data/wapic-20260601-h11_2.wac` (52M, 97.67/97.07/15-15) — 保底
2. H13.x 如果解了 BC001 且 F1 不掉 → 升级为新 release
3. H11.5-H11.10 / H12 如果出现更好 (15/15 + F1_pdmp > 97.68 + size ~50M) → 升级

## 当前 best 三大指标对比

| | size | F1_pdmp | F1_12m | case | badcase |
|---|---|---|---|---|---|
| 旧 release wapic-20260529 | 47M | 97.70 | 96.83 | 12/15 | 97.0% |
| **新 wapic-20260601-h11_2** | 52M | 97.67 | 97.07 | 15/15 | 99.7% |

唯一未解：BC001 (波·索提拉克 / 拉·甘地，foreign_dot 模式)
