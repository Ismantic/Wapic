# Wapic 三阶段训练参数

## TL;DR

```bash
# Stage 1: 12M LTP base (一次性，用现有 all12m_compact_v2.wac)
# (训练命令略 — 见 RESULTS.md, 11月-训出来的)

# Stage 2: 500k name-rich, fine-tune from Stage 1
./build/wapic fit -p data/pattern.txt \
    --init-from data/all12m_compact_v2.wac \
    -a l-bfgs -i 100 -1 0.1 -2 0.0001 -t 4 --histsz 5 -e 1e-9 \
    --save-binary --save-prune --prune-threshold 0.05 \
    data/name_aug_v2_train.txt data/stage2.wac

# Stage 3: 102k PD modern+punct, fine-tune from Stage 2
./build/wapic fit -p data/pattern.txt \
    --init-from data/stage2.wac \
    -a l-bfgs -i 30 -1 0.1 -2 0.0001 -t 4 --histsz 5 -e 1e-9 \
    --save-binary --save-prune --prune-threshold 0.05 \
    data/pd_mp_train.txt data/release.wac
```

## 数据 ratio

| Stage | 数据 | 句数 |
|---|---|---|
| 1 | 12M LTP/base1 标注（4 源混合） | 9.5M |
| 2 | OpenNews + 1.14M 人名词表 grep + LTP cws | **500k** |
| 3 | PD-1998 (01-05) → modernize + split punct | 100k |

比例 100 : 5 : 1。Stage 2 用 grep 比 LTP NER 召回多 10×，速度快几十倍。

## Stage 2 参数 sweep（从 12M base warm-start）

| iter | L1 | F1 (pd_mp) | name 15句 | 备注 |
|---|---|---|---|---|
| 20 | 0.3 | 97.73 | 11/15 | 迭代不够 |
| 30 | 0.3 | **97.91** | 12/15 | F1 高但名字没提升 |
| 50 | 0.3 | 97.46 | 11/15 | L1=0.3 + i50 过压 |
| 50 | 0.1 | 97.55 | 12/15 | |
| **100** | **0.1** | 97.34 | **14/15** | ★ name 最佳 |
| 100 | 0.05 | 97.30 | 14/15 | L1 再弱无收益 |

**结论**：i100 + L1=0.1 学名字最透，但 F1 暂掉，靠 Stage 3 拉回。

## Stage 3 参数 sweep（从 Stage 2 i100_l01 warm-start）

| iter | L1 | size | F1 | name |
|---|---|---|---|---|
| 10 | 0.3 | 11.8 MB | 97.40 | 14/15 |
| 20 | 0.3 | 11.9 MB | 97.40 | 14/15 |
| 30 | 0.3 | 8.3 MB | 97.45 | 13/15 ← 多压会丢名字 |
| 10 | 0.1 | 11.8 MB | 97.41 | 14/15 |
| 20 | 0.1 | 12.2 MB | 97.48 | 14/15 |
| **30** | **0.1** | **11.4 MB** | **97.58** | **14/15** ★ |

**结论**：i30 + L1=0.1。再多 iter 名字会塌。

## 关键观察

1. **L1 越大 → 越稀疏**。0.3 在 i30 后开始过压，i50 已劣化
2. **i20 名字学不到**（11-12/15），i100 才到 14/15
3. **Stage 3 不能跑太多** —— PD 标准数据少（100k），fine-tune 过头会"忘记"Stage 2 学的名字模式
4. **L1=0.1 是 Stage 3 sweet spot** —— 既能继续学 PD 标准又不过压
5. **bigram pattern 极限是 14/15 + F1 97.58**。要破"缉毒警林强峰"那道难题需要 trigram

## 三阶段路径总结

```
all12m_compact_v2.wac (61 MB / F1 96.48)
    ↓ Stage 2: i100 L1=0.1 on 500k name-rich
sweep_s2_i100_l01.wac (15 MB / F1 97.34 / 14/15) ★
    ↓ Stage 3: i30 L1=0.1 on 100k PD-mp
sweep_s3_i30_l01.wac (11.4 MB / F1 97.58 / 14/15) ★ FINAL
```

## 跟老 release 对比

| 模型 | 大小 | F1 | name 15 |
|---|---|---|---|
| 老 wapic-20260529 (3-stage, 140k aug) | 47 MB | **98.07** | 12/15 |
| 新 sweep_s3_i30_l01 (3-stage, 500k aug) | **11.4 MB** | 97.58 | **14/15** |

trade-off：用 35 MB 体积 + 0.49 F1 换 2 个 case + 名字结构泛化。

## ⚠️ 注意

- **fine-tune 不能用 `--from-bin`**：bin 的 trie ID 跟 init model 不匹配会 segfault
- **必须用 text 输入**：fit 时动态查 trie，LockFeatures 把新 feature drop 掉
- 想修这个，得给 `wapic build` 加 `--init-from` 选项，让 build 用 init model 的 trie 编号

## TODO

- 14 → 15：解决 "缉毒警林强峰"
  - 选项 A：trigram pattern + 500k 数据（数据小内存够）
  - 选项 B：更多 stage 2 数据（合并 140k 老 aug + 500k 新 = 640k）
  - 选项 C：调整 case 评估，重视"邻接 警/林"这种 adversarial 上下文
