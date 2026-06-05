# Released Models Summary

每个 `data/wapic-*.wac` 的当前指标。详细训练配方见 [RELEASE.md](RELEASE.md) 和 [experiments/results.log](experiments/results.log)。

## Active releases (in git)

| 文件 | 来源 | size | F1_pdmp | F1_12m | names | bcv2 | bcv3 | 备注 |
|---|---|---|---|---|---|---|---|---|
| `wapic-20260529.wac` | 早期 sweep | 47M | 97.70 | 96.83 | 12/15 | 171/200 | 273/500 | 历史 baseline |
| `wapic-20260601.wac` | H15.2 p0.15 | 56M | 97.69 | 96.97 | 15/15 | 182/200 | 374/500 | 第一个 15/15 |
| `wapic-20260601-h11_2.wac` | H11.2 p0.05 | 52M | 97.68 | 97.12 | 15/15 | - | - | 历史中间产物 |
| `wapic-20260601-h12_1.wac` | H12.1 raw | 51M | 97.71 | 97.23 | 15/15 | 179/200 | - | 历史中间产物 |
| `wapic-20260601-h13_1.wac` | H13.1 p0.15 | 52M | 97.69 | 97.09 | 15/15 | - | - | 历史中间产物 |
| **`wapic-20260602.wac`** | **H17.2** | **78M** | **97.79** | **97.10** | **15/15** | **183/200** | **377/500** | **当前 release** |
| `wapic-20260602-h19_1.wac` | H19.1 p0.15 | **54M** | 97.77 | 97.07 | 15/15 | 182/200 | **398/500** | size 最小 + bcv3 高（部分 overfit） |
| `wapic-20260602-h19_1-full.wac` | H19.1 unpruned | 77M | 97.79 | 97.10 | 15/15 | 183/200 | **404/500** | bcv3 历史最高 |
| **`wapic-20260604-h25_5.wac`** | **H25.5 p=0.03** | **50M** | **97.83** | **97.69** | **15/15** | **190/200** | **399/500** | **size 减半 vs 20260602**；F1/F1_12m/bcv2 全面超 release；bcv3 差 36 |
| **`wapic-20260605-h25_20.wac`** | **H25.20 p=0.01** | **42M** | **97.65** | **97.37** | **15/15** | **190/200** | **410/500** | **bcv3 历史最高**；v5_p0.01 base + i=150 (50+50+50 chained)；F1_pdmp 比 release 微低 0.14 |

## 已知 case 状态（截至当前 release H17.2）

| case | release | H19.1 | 备注 |
|---|---|---|---|
| 林强峰（unseen 三字名） | ✅ | ✅ | |
| 波·索提拉克（外国名） | ✅ | ✅ | |
| 包江苏（人名/地名歧义） | ❌ | ❌ | 仍未解 |
| 俞黎新（姓+双字名） | ❌ | ✅ | H19.1 修了 |
| 1.25亿英镑（数字小数点） | ❌ | ❌ | H19.2 在攻 |

## 配方差异

- **H15.2** (release_20260601): + threechar_zh 500k
- **H17.2** (release_20260602): + threechar_zh **1M** (replaces 500k)
- **H19.1**: H17.2 + bcv3_fail_mining 105 句 + L1=0.015
- **H19.2** (running): H19.1 + decimal_focus_v2 6.9k 句

## 推理性能（基于 H17.2，但所有 release 速度/内存接近）

- 速度：1131 sent/s（单线程 CPU，pd_mp_test 5k 句）
- 内存：215 MB RSS（DAT 加速 + obs string 释放）
- 详见 [commit 9be358b](https://github.com/Ismantic/Wapic/commit/9be358b)
