# Wapic 训练交接文档

最后更新：2026-05-29
执行环境：另一台机（u8700，31 GB RAM，28 GB available，OpenMP 4.5，g++ 15）

## 任务

训练一个 Wapic（线性链 CRF + L-BFGS）中文分词模型。目标按**优先级**排：

1. **李镇全 case 5/5 一致**（人名整体识别稳定，不分姓+名）— 用户核心诉求
2. F1 ≥ 98 在 PD-1998 测试集上即可，不必追 98.5+
3. 模型体积 < 100 MB

## 用户偏好

- 数据要"累积"不"丢弃"——大数据+小数据 fine-tune 是核心策略
- 人名要靠 CRF **学到**而非靠词典后处理
- 现代中文标准（人名整体、标点分开、数字英文不切碎）
- 不喜欢"PD-only 训练 PD 测试"这种自评，要在 12M 数据上有真泛化

## 当前状态

### 数据已在 `data/`

| 文件 | 大小 | 内容 | 用途 |
|---|---|---|---|
| `all12m_tri_bin.{meta,obs,trie}.*` | 11.5 GB | 12M 句 LTP/base1 标注 + trigram 已 build | Stage 1 训练 |
| `pd_aug_train.txt` | 65 MB | name-rich aug（267k 句） | Stage 2 训练 |
| `pd_mp_train.txt` | 57 MB | PD-1998 modern+punct（102k 句） | Stage 3 训练 |
| `pd_mp_test.txt` | 12 MB | 测试集（modern+punct 标准） | 评测 |
| `pd_test.txt` | 12 MB | 测试集（原 PD 标准） | 备用评测 |
| `pattern_tri.txt` | 200 B | trigram feature 模板 | 训练用 |

### 代码状态
- 仓库 git clone 完成，commit `2018bb7` 起步
- 已编译 `build/wapic`
- 关键新功能：`build`/`convert`/`fit` 子命令；`--from-bin`、`--save-binary`、`--save-prune`、`--prune-threshold`、`--init-from`（warm-start）

## 三阶段训练计划

### Stage 1：12M LTP base + trigram（首要）

```bash
nohup ./build/wapic fit -p data/pattern_tri.txt --from-bin \
    -a l-bfgs -i 100 -1 0.3 -2 0.0001 \
    -t 8 --histsz 5 -e 1e-9 \
    --save-binary --save-prune --prune-threshold 0.05 \
    data/all12m_tri_bin data/wapic_12m_tri.wac \
    > logs_12m_tri.txt 2>&1 &
```

- 预期 1-2 小时，~80-100 MB 模型
- F1 在 12M test 上应 ~96.5（用 LTP 标准）
- 用 `tail -f logs_12m_tri.txt` 看进度
- 关注：RSS 别超 26 GB（28 GB 可用）；earlyoom 在 32 GB 机上没装也无所谓

### Stage 2：name-rich warm-start fine-tune

先 build name-rich 数据：
```bash
./build/wapic build -p data/pattern_tri.txt -t 8 \
    data/pd_aug_train.txt data/pd_aug_tri_bin
```

然后 fine-tune（**关键：--init-from + LockFeatures，不会扩 feature 空间**）：
```bash
nohup ./build/wapic fit -p data/pattern_tri.txt --from-bin \
    --init-from data/wapic_12m_tri.wac \
    -a l-bfgs -i 50 -1 0.3 -2 0.0001 \
    -t 8 --histsz 5 -e 1e-9 \
    --save-binary --save-prune --prune-threshold 0.05 \
    data/pd_aug_tri_bin data/wapic_12m_tri_s2.wac \
    > logs_s2.txt 2>&1 &
```

- 用户明确：**多塞数据 + 多跑几轮**（i30 → i50）让 name pattern 充分学到
- 预期 20-40 min，体积差不多
- 评测此阶段在 pd_mp_test 上 F1 应 ≥ 97.5

### Stage 3：PD modern+punct fine-tune（可选）

用户明确：**少跑、轻 fine-tune**，避免把 stage 2 学的 name pattern 覆盖：

```bash
./build/wapic build -p data/pattern_tri.txt -t 8 \
    data/pd_mp_train.txt data/pd_mp_tri_bin

nohup ./build/wapic fit -p data/pattern_tri.txt --from-bin \
    --init-from data/wapic_12m_tri_s2.wac \
    -a l-bfgs -i 30 -1 0.5 -2 0.0001 \
    -t 8 --histsz 5 -e 1e-9 \
    --save-binary --save-prune --prune-threshold 0.05 \
    data/pd_mp_tri_bin data/wapic_12m_tri_s3.wac \
    > logs_s3.txt 2>&1 &
```

- 只 i30，L1 加大到 0.5 防过拟
- 预期 5-10 min
- **用户提示：如果 stage 2 出来 F1 ≈ 98 + 李镇全稳，stage 3 可跳过**

## 数据生成 pipeline（重新生成时参考）

### Stage 2 数据 `pd_aug_train.txt` （name-rich）

输入需要：`news_cut.jsonl`（cut_corpus.py 用 LTP/base1 切好的 News 数据，每行 `{"source", "cut"}`）

```bash
python scripts/extract_name_rich.py \
    --input data/news_cut.jsonl \
    --output data/pd_aug.jsonl \
    --limit-in 300000 --limit-out 80000

python scripts/jsonl_to_bmes.py \
    --input data/pd_aug.jsonl \
    --out-bmes data/pd_aug_train.txt \
    --out-nolabel data/pd_aug_nolabel.txt \
    --max-chars 200
```

`extract_name_rich.py` 用 LTP NER 筛 `Nh`-tagged 句子 + 内置 split_punct。

**替代路径**（如要更多人名覆盖）：
- `extract_with_namelist.py` —— 1.14M `Chinese-Names-Corpus` 词表 grep
- `name_synth.py` —— 纯合成（不依赖语料）
- `name_amplify.py` —— 抽常见人名 + 25 模板放大

### Stage 3 数据 `pd_mp_train.txt` （PD modern+punct）

输入需要：PD-1998 原始 `199801.txt` ~ `199805.txt`（199806 留测试）

```bash
# 1. 解析 PD 原始（去 POS / NER 括号 / 句子 ID）
for m in 199801 199802 199803 199804 199805; do
  python scripts/parse_pd1998.py \
    --src data/pd_raw/${m}.txt \
    --out data/pd_parsed_${m}.jsonl
done
cat data/pd_parsed_*.jsonl > data/pd_parsed.jsonl

# 2. 现代化（仅合并不拆分）：LTP NER 把连续的 Nh/Ni/Ns token 合并
python scripts/modernize_pd.py \
    --input data/pd_parsed.jsonl \
    --output data/pd_modern.jsonl

# 3. 标点拆 token：13:10 → 13 : 10
python scripts/split_punct.py \
    --input data/pd_modern.jsonl \
    --output data/pd_mp.jsonl

# 4. 转 BMES
python scripts/jsonl_to_bmes.py \
    --input data/pd_mp.jsonl \
    --out-bmes data/pd_mp_train.txt \
    --out-nolabel data/pd_mp_nolabel.txt \
    --max-chars 200
```

测试集 `pd_mp_test.txt` 同 pipeline，输入用 `199806.txt`。

⚠️ 重要：parse_pd1998.py **必须用 UTF-8** 编码读（不是 gb18030）。早期 bug 导致 F1 虚高。

## 评测脚本

### 跑测试集 F1

```bash
# infer
./build/wapic test -m data/wapic_12m_tri_s2.wac \
    data/pd_mp_test_nolabel.txt data/pd_mp_pred.txt

# 计算 F1（compare_1998.py 严格版）
python3 scripts/compare_1998.py \
    --gold data/pd_mp_test.txt \
    --pred data/pd_mp_pred.txt
```

### 李镇全 case 测试

5 个上下文，看 5 次是否一致输出 `李镇全`（不分姓）：

```python
# 在 scripts/ 下新建测试 py 或直接 echo 喂 wapic test
cases = [
    "李镇全 是 著名 的 学者 。",
    "据 李镇全 介绍 ， 项目 进展 顺利 。",
    "李镇全 担任 该 公司 的 总经理 。",
    "记者 李镇全 报道",
    "中央 领导 李镇全 同志 发表 讲话 。",
]
```

对每条 source（去空格）跑 wapic infer，看输出是否把 `李镇全` 整体保留。

## 当前最佳基线（旧机器训练，供参考）

| 模型 | 大小 | F1 (pd_test) | 备注 |
|---|---|---|---|
| `wapic_pd_best.wac` | 4.8 MB | 97.32 | PD-only 训练，超小 |
| `wapic_stage2_v2.wac` | 47 MB | ~98+ | 三阶段无 trigram，目前 F1 最高 |
| `wapic_tri_v2.wac` | 16 MB | 97.64 | trigram + 340k，**李镇全 4/5 一致** |

**目标新模型 `wapic_12m_tri_s2.wac`** 应优于上述所有，且李镇全 5/5。

## 内存/性能注意

- 12M obs.bin 9 GB 走 mmap，**不算 RSS**（但 free 显示偏低）
- 训练峰值实测 17 GB RSS（trigram feature trie + L-BFGS state + gradient）
- `-t 8` 在 32 GB 机上稳；如 RSS 超 26 GB 改 `-t 4`
- `--histsz 5` 比 3 收敛快但多 ~10% 内存

## 完成后回传

把这三个模型 rsync 回起源机器：
```bash
# 在 u8700 上
rsync -avP data/wapic_12m_tri.wac \
            data/wapic_12m_tri_s2.wac \
            data/wapic_12m_tri_s3.wac \
            data/logs_s*.txt logs_12m_tri.txt \
    tfbao@<起源机IP>:/home/tfbao/Shiyu/Wapic/data/
```

## 故障排查

| 现象 | 解决 |
|---|---|
| `Permission denied` 跑 wapic | `chmod +x build/wapic` |
| F1 远低于预期 | 检查 pattern_tri.txt 是否正确（有 U10 trigram 行） |
| 训练立刻 OOM | `-t 8 → -t 4` 或 `--histsz 5 → 3` |
| Stage 2/3 feature 空间扩张 | 确认用了 `--init-from`，LockFeatures 不会加新 feature |
| 李镇全分姓+名 | 增大 Stage 2 数据量或 iter，或 Stage 3 跑得更轻（i20） |
