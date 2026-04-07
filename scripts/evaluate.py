#!/usr/bin/env python3
"""Evaluate CRF segmentation: compare predicted BMES tags with gold tags,
and show sample segmentation results."""

import os
import sys


def read_sentences(filepath, has_score=False):
    """Read CRF column file into list of sentences (list of (char, tag) or just tag)."""
    sentences = []
    current = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            if line.startswith('score='):
                continue
            parts = line.split()
            current.append(parts)
    if current:
        sentences.append(current)
    return sentences


def tags_to_words(chars, tags):
    """Convert chars + BMES tags to segmented words."""
    words = []
    current = ''
    for c, t in zip(chars, tags):
        if t == 'B':
            if current:
                words.append(current)
            current = c
        elif t == 'M':
            current += c
        elif t == 'E':
            current += c
            words.append(current)
            current = ''
        elif t == 'S':
            if current:
                words.append(current)
            words.append(c)
            current = ''
        else:
            current += c
    if current:
        words.append(current)
    return words


def evaluate(gold_file, pred_file, n_samples=10):
    gold_sens = read_sentences(gold_file)
    pred_sens = read_sentences(pred_file)

    # Pred file has only tags (+ scores), gold has char + tag
    total_tokens = 0
    correct_tokens = 0
    total_sens = 0
    correct_sens = 0

    # For word-level P/R/F1
    total_gold_words = 0
    total_pred_words = 0
    total_correct_words = 0

    samples = []

    for i, (gs, ps) in enumerate(zip(gold_sens, pred_sens)):
        gold_chars = [t[0] for t in gs]
        gold_tags = [t[1] for t in gs]
        pred_tags = [t[0] for t in ps]  # pred file: tag score

        # Token-level accuracy
        sen_correct = True
        for gt, pt in zip(gold_tags, pred_tags):
            total_tokens += 1
            if gt == pt:
                correct_tokens += 1
            else:
                sen_correct = False
        total_sens += 1
        if sen_correct:
            correct_sens += 1

        # Word-level F1
        gold_words = tags_to_words(gold_chars, gold_tags)
        pred_words = tags_to_words(gold_chars, pred_tags)

        # Use position-based matching
        gold_spans = set()
        pos = 0
        for w in gold_words:
            gold_spans.add((pos, pos + len(w)))
            pos += len(w)

        pred_spans = set()
        pos = 0
        for w in pred_words:
            pred_spans.add((pos, pos + len(w)))
            pos += len(w)

        total_gold_words += len(gold_spans)
        total_pred_words += len(pred_spans)
        total_correct_words += len(gold_spans & pred_spans)

        if i < n_samples:
            samples.append((
                ''.join(gold_chars),
                ' '.join(gold_words),
                ' '.join(pred_words)
            ))

    # Results
    token_acc = correct_tokens / total_tokens * 100
    sen_acc = correct_sens / total_sens * 100

    precision = total_correct_words / total_pred_words * 100 if total_pred_words else 0
    recall = total_correct_words / total_gold_words * 100 if total_gold_words else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print(f"=== Evaluation on {total_sens} sentences ===\n")
    print(f"Token accuracy:  {token_acc:.2f}% ({correct_tokens}/{total_tokens})")
    print(f"Sentence accuracy: {sen_acc:.2f}% ({correct_sens}/{total_sens})")
    print(f"\nWord-level metrics:")
    print(f"  Precision: {precision:.2f}%")
    print(f"  Recall:    {recall:.2f}%")
    print(f"  F1:        {f1:.2f}%")

    print(f"\n=== Sample Results (first {len(samples)}) ===\n")
    for raw, gold, pred in samples:
        print(f"原文: {raw}")
        print(f"标注: {gold}")
        print(f"预测: {pred}")
        match = "✓" if gold == pred else "✗"
        print(f"  [{match}]")
        print()


if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    evaluate(os.path.join(data_dir, 'test.txt'),
             os.path.join(data_dir, 'test_result.txt'))
