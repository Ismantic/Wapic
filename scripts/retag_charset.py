#!/usr/bin/env python3
"""
Rewrite BMES files: split tokens by character-type boundary.

Rules within each existing token:
  - Digit run (0-9 + full-width 0-9)       → one new token
  - Latin letter run (ascii alpha + full-width letters) → one new token
  - CJK run (U+4E00-U+9FFF + Ext A)        → one new token
  - Any other char (punct/symbol)          → standalone S, each

Existing CJK token boundaries are preserved (CJK runs only split when type changes).
Multi-char Latin words (iPhone, OPPO, Wi-Fi → Wi/Fi) keep contiguous letters.

Examples:
  1998年        → 1998 年
  20.3          → 20 . 3
  iPhone手机     → iPhone 手机
  OPPO         → OPPO (unchanged)
  中科院-中国     → 中科院 - 中国
  5个PIN码       → 5 个 PIN 码
  Wi-Fi         → Wi - Fi
  波·索提拉克     → 波 · 索提拉克 (unchanged, was already correct)

Each input line: "<char> <tag>" (or "<char> <type> <tag>" for 3-col data).
Blank line = sentence separator.
"""
import sys

def char_type(c):
    if c.isdigit():
        return 'D'  # digits (incl. full-width)
    if c.isalpha():
        if c.isascii():
            return 'L'
        co = ord(c)
        if 0x4E00 <= co <= 0x9FFF: return 'C'
        if 0x3400 <= co <= 0x4DBF: return 'C'
        # full-width latin and other letters: treat as Latin
        return 'L'
    return 'P'

def split_token(chars):
    """Split a list of chars (one token) into list of new sub-tokens.

    All punct chars become standalone. Same-type runs (D/L/C) stay grouped.
    """
    out, i, n = [], 0, len(chars)
    while i < n:
        t = char_type(chars[i])
        if t == 'P':
            out.append([chars[i]])
            i += 1
        else:
            j = i
            while j < n and char_type(chars[j]) == t:
                j += 1
            out.append(chars[i:j])
            i = j
    return out

def chars_to_bmes(chars):
    if len(chars) == 1: return ['S']
    return ['B'] + ['M']*(len(chars)-2) + ['E']

def process(in_path, out_path, has_type=False):
    n_sent = n_token_in = n_token_out = 0
    with open(in_path, encoding='utf-8') as fi, open(out_path, 'w', encoding='utf-8') as fo:
        sent_lines = []
        for line in fi:
            line = line.rstrip('\n')
            if not line:
                if sent_lines:
                    n_sent += 1
                    tokens = []
                    cur_chars, cur_types = [], []
                    for ch, tp, tg in sent_lines:
                        if tg == 'B':
                            if cur_chars: tokens.append((cur_chars, cur_types))
                            cur_chars = [ch]; cur_types = [tp]
                        elif tg == 'M' or tg == 'E':
                            cur_chars.append(ch); cur_types.append(tp)
                        else:
                            if cur_chars: tokens.append((cur_chars, cur_types))
                            tokens.append(([ch], [tp]))
                            cur_chars = []; cur_types = []
                    if cur_chars: tokens.append((cur_chars, cur_types))
                    n_token_in += len(tokens)
                    new_tokens = []
                    for chars, types in tokens:
                        sub = split_token(chars)
                        ci = 0
                        for sub_chars in sub:
                            sub_types = types[ci:ci+len(sub_chars)]
                            ci += len(sub_chars)
                            new_tokens.append((sub_chars, sub_types))
                    n_token_out += len(new_tokens)
                    for chars, types in new_tokens:
                        tags = chars_to_bmes(chars)
                        for ch, tp, tg in zip(chars, types, tags):
                            if has_type:
                                fo.write(f"{ch} {tp} {tg}\n")
                            else:
                                fo.write(f"{ch} {tg}\n")
                    fo.write("\n")
                    sent_lines = []
            else:
                parts = line.split()
                if has_type and len(parts) == 3:
                    sent_lines.append((parts[0], parts[1], parts[2]))
                elif len(parts) == 2:
                    sent_lines.append((parts[0], None, parts[1]))
                else:
                    fo.write(line + "\n")
    return n_sent, n_token_in, n_token_out

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: retag_charset.py <input.txt> <output.txt> [--type]")
        sys.exit(1)
    inp, outp = sys.argv[1], sys.argv[2]
    has_type = "--type" in sys.argv
    s, ti, to = process(inp, outp, has_type)
    print(f"{inp}: sentences={s}  tokens {ti} -> {to} (+{to-ti})")
