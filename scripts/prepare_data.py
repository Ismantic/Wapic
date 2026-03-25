#!/usr/bin/env python3
"""Convert JSON segmentation data to CRF BMES format.

Input:  JSON lines with {"source": "...", "cut": "词1 词2 ..."}
Output: CRF columnar format, one char per line: char label
        Blank lines separate sentences.
"""

import json
import sys
import os
import random


def word_to_bmes(word):
    """Convert a word to BMES tags."""
    chars = list(word)
    if len(chars) == 1:
        return [(chars[0], 'S')]
    tags = []
    for i, c in enumerate(chars):
        if i == 0:
            tags.append((c, 'B'))
        elif i == len(chars) - 1:
            tags.append((c, 'E'))
        else:
            tags.append((c, 'M'))
    return tags


def convert_line(line):
    """Convert one JSON line to list of (char, tag) pairs."""
    obj = json.loads(line)
    cut = obj['cut']
    words = cut.split()
    result = []
    for word in words:
        # Skip empty tokens
        if not word:
            continue
        # Remove leading [ (entity marker in the data)
        word = word.lstrip('[')
        if not word:
            continue
        result.extend(word_to_bmes(word))
    return result


def process_files(input_files, output_file):
    """Process multiple input files into one CRF output file."""
    count = 0
    with open(output_file, 'w', encoding='utf-8') as out:
        for fname in input_files:
            with open(fname, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        pairs = convert_line(line)
                    except (json.JSONDecodeError, KeyError):
                        continue
                    if not pairs:
                        continue
                    for char, tag in pairs:
                        out.write(f'{char} {tag}\n')
                    out.write('\n')
                    count += 1
    return count


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    # Use refined version for 01, original for others
    train_files = [
        os.path.join(data_dir, '1998-01-refined.txt'),
        os.path.join(data_dir, '1998-02.txt'),
        os.path.join(data_dir, '1998-03.txt'),
        os.path.join(data_dir, '1998-04.txt'),
        os.path.join(data_dir, '1998-05.txt'),
    ]
    test_files = [
        os.path.join(data_dir, '1998-06.txt'),
    ]

    os.makedirs(data_dir, exist_ok=True)

    print("Converting training data...")
    n_train = process_files(train_files, os.path.join(data_dir, 'train.txt'))
    print(f"  {n_train} sentences written to data/train.txt")

    print("Converting test data...")
    n_test = process_files(test_files, os.path.join(data_dir, 'test.txt'))
    print(f"  {n_test} sentences written to data/test.txt")

    # Also create a small test input without labels (for inference)
    with open(os.path.join(data_dir, 'test.txt'), 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(os.path.join(data_dir, 'test_nolabel.txt'), 'w', encoding='utf-8') as out:
        for line in lines:
            line = line.strip()
            if not line:
                out.write('\n')
            else:
                parts = line.split()
                out.write(f'{parts[0]}\n')

    print("Done.")


if __name__ == '__main__':
    main()
