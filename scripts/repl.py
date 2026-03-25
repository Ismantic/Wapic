#!/usr/bin/env python3
"""Interactive Chinese word segmentation REPL."""

import subprocess
import tempfile
import os
import sys

BINARY = os.path.join(os.path.dirname(__file__), '..', 'build', 'src', 'isma_wapiti_test')
MODEL = os.path.join(os.path.dirname(__file__), '..', 'data', 'model.crf')


def segment(text, binary, model):
    """Segment Chinese text using the CRF model."""
    chars = list(text.strip())
    if not chars:
        return ''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as fin:
        for c in chars:
            fin.write(f'{c}\n')
        fin.write('\n')
        in_path = fin.name

    out_path = in_path + '.out'

    try:
        subprocess.run(
            [binary, 'label', '-m', model, in_path, out_path],
            capture_output=True, text=True
        )

        with open(out_path, 'r', encoding='utf-8') as f:
            tags = []
            for line in f:
                line = line.strip()
                if not line or line.startswith('score='):
                    continue
                tags.append(line.split()[0])

        # Reconstruct words from BMES tags
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

        return ' '.join(words)
    finally:
        os.unlink(in_path)
        if os.path.exists(out_path):
            os.unlink(out_path)


def main():
    binary = os.path.abspath(BINARY)
    model = os.path.abspath(MODEL)

    if not os.path.exists(binary):
        print(f'Error: binary not found: {binary}')
        sys.exit(1)
    if not os.path.exists(model):
        print(f'Error: model not found: {model}')
        sys.exit(1)

    # Preload model by doing a dummy run (the model loads each time anyway)
    print('IsmaWapiti 中文分词 REPL')
    print('输入中文文本，回车分词。输入 q 退出。')
    print()

    while True:
        try:
            text = input('>>> ')
        except (EOFError, KeyboardInterrupt):
            print()
            break

        text = text.strip()
        if not text:
            continue
        if text in ('q', 'quit', 'exit'):
            break

        result = segment(text, binary, model)
        print(result)
        print()


if __name__ == '__main__':
    main()
