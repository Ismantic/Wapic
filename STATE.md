# Autonomous Run State (last update: 2026-06-01 13:00)

## Concurrent background processes

| Job | PID | What | Output | ETA |
|---|---|---|---|---|
| 3M NER | 1474422 | LTP/base1 NER on OpenNews.3M.cut.jsonl | data/raw/opennews_ner.jsonl | ~40 min |
| H1 batch | 1476347 | Stage 2 fine-tune sweep (4 experiments) | experiments/results.log | ~3-4 h |
| Chain | 1476727 | Will trigger full OpenNews NER after 3M finishes | logs/ner_chain.log | activates ~T+40min |

## Objectives (signed off by user)

- F1_pdmp ≥ 97.67 (LTP/base1 - 0.005)
- F1_12m ≥ 96.83 hard / 97.92 soft (allowed to drop a bit)
- case 15/15
- size ≥ 40 MB
- Stage 1 ≥ 10M sentences (using existing 9.5M base, no retrain)
- Stage 2 ≥ 500k sentences
- Don't target specific case vocabulary

## Architecture

Stage 1 base (all12m_compact_v2.wac, 61M, **fixed, no retraining**) → Stage 2 fine-tune (with anchor mix) → optional Stage 3 light pass.

Start point fixed: **A0 = all12m_compact_v2.wac**. A1 (wapic-20260529.wac) is comparison only, not a training start.

## Clean data inventory

| File | Sentences | Clean | Purpose |
|---|---|---|---|
| data/all12m_train.txt | 9.5M | ✓ Stage 1 source | anchor source |
| data/all_train.txt | 1.14M | ✓ | (smaller subset, less used) |
| data/clean_aug_train.txt | 55k | ✓ LTP NER OpenNews | name aug |
| data/pd_mp_train.txt | 102k | ✓ PD-1998 modernized | PD distribution |
| data/antimerge_train.txt | 6.3k | ✓ no case names | anti-merge prior |
| data/name_aug_v2_train.txt | 500k | ✓ namelist filtered | extra name aug |
| data/anchor_100k.txt | 100k | ✓ | anchor sample (seed=42) |
| data/anchor_300k.txt | 300k | ✓ | anchor sample (seed=42) |
| data/anchor_1m.txt | 1M | ✓ | anchor sample (seed=42) |
| ~~data/hardcase_train.txt~~ | ~~15k~~ | ❌ **CHEATS** (553×"缉毒警") | DELETED from new exps |

## Coming data

- data/raw/opennews_ner.jsonl — 3M OpenNews + LTP NER (in progress)
- data/raw/opennews_full_nh.jsonl — up to 10M Nh-only from full 226M OpenNews (chained)

## Baselines

| Model | size | F1_pdmp | F1_12m | case |
|---|---|---|---|---|
| Stage 1 base (A0) | 61M | 96.55 | 97.92 | 11/15 |
| Old release (A1) | 47M | 97.70 | 96.83 | 12/15 |
| G1 (deprecated, used hardcase) | 52M | 97.53 | 96.23 | 15/15 |
| D1 (deprecated, used hardcase) | 56M | 97.71 | 96.67 | 14/15 |
| E5 (deprecated, used hardcase) | 52M | 97.50 | 96.26 | 15/15 |

## Active queue (next 48h)

- [x] T+0: Launch 3M OpenNews NER
- [x] T+0: Launch H1 (anchor sweep, no Stage 1 retrain)
- [ ] T+40min: 3M NER done → chain auto-fires full OpenNews NER (filter-nh-only, limit 10M)
- [ ] T+3h: H1 done → choose best, design H2 with OpenNews data
- [ ] T+5-8h: H2 batch (best H1 recipe + OpenNews Nh data)
- [ ] T+8-16h: H3 batch (Stage 3 light pass on best H2)
- [ ] T+16-40h: refinement with growing OpenNews corpus
- [ ] T+40-48h: final candidate selection, document, commit

## Key scripts

- experiments/run_batch_h1.sh — current running batch
- scripts/sample_sentences.py — sentence-level subsample of BMES files
- scripts/eval_both.sh — F1_pdmp + F1_12m + case in one shot
- scripts/test_name_cases.py — 15-case test
- scripts/ner_opennews_full.py — OpenNews NER (supports txt/jsonl input, filter-Nh-only, limit-out)
- scripts/chain_ner_full.sh — waits for PID, then starts full OpenNews NER

## Pending after H5

- [ ] H6.1: H2.3 recipe MINUS antimerge, i=50 — measure antimerge contribution.
       If contribution < 0.05 F1 + same case → drop antimerge from release (cleaner repro).
       If contribution significant → recreate generation script and pin in REPRODUCE.md.
