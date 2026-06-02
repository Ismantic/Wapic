# 自主探索计划

时间预算：60 小时
约束：F1 ≥ 98 + 15/15 names + size ≥ 40 MB

## 探索方向（按 ROI 排序）

### A. Stage 3 优化（最快试错，每个 ~2 min）
- A1: i50 L1=0.001 no prune (more stage 3 iter)
- A2: i100 L1=0.001 no prune
- A3: i50 L1=0.005

### B. Stage 2 数据扩展（每个 ~15 min）
- B1: 140k (老 aug) + 500k (新 namelist) = 640k 合并
- B2: 加合成 "X[警/兵/官]Y[3字名]Z" 的 hard-case templates
- B3: 把 OpenNews 3M 全扫一遍（不止 300k limit）→ 可能 100w name-rich

### C. Stage 1 重训（每个 ~1 小时）
- C1: 12M + 加 type 列（pattern_type.txt）
- C2: 12M 但去掉 LCCC（保留新闻为主）= 9M News+PD+Zhihu，再 LTP cws 一遍
- C3: 12M + bigram only，但用更紧的 -e 1e-12

### D. 代码改动（每个 ~3 小时）
- D1: anchor reg：L1 改成 |w - w_init| 形式，防权重漂离 12M base
- D2: 给 build 加 --init-from，让 bin 兼容 init model trie → 解锁 --from-bin warm-start
- D3: 加 weight clamping：训练中限制 |w| ≤ X，避免极值

### E. Pattern 改进（高 ROI，高风险）
- E1: trigram 在 500k name-rich 上 from scratch（数据小内存够）→ 看是否真破 14 ceiling
- E2: 加跳跃 bigram：U:%x[-3,0]/%x[0,0]
- E3: cross feature：char × 位置

## 执行顺序

阶段 1 (快): A1 → A2 → A3，找最佳 Stage 3 配方
阶段 2 (中): B1 (合并数据)
阶段 3 (中): B2 (hard-case 合成)
阶段 4 (慢): C1 或 C2 (重训 stage 1)
阶段 5 (慢): D1 (anchor) 或 E1 (trigram)

## 已知失败

- 12M + trigram 训练内存挂（≥ 27 GB） → station 跑不动
- Stage 3 i30+ 名字 case 反而掉 → 不能跑太多 PD 数据
