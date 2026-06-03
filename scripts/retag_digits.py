#!/usr/bin/env python3
"""
Rewrite BMES files to make digit runs independent tokens.

Rules within each existing token:
  - Pure digit run → one new token (BMES per length)
  - '.' / '+' / '-' / '/' between digits → standalone S
  - Non-digit, non-punct (Chinese/Latin) → keep as one new token

Examples:
  1998年   (BMMME)   → 1998 (BMME) + 年 (S)
  20.3     (BMME)    → 20 (BE) + . (S) + 3 (S)
  1.25亿元 (BMMMME)  → 1 (S) + . (S) + 25 (BE) + 亿元 (BE)

Each input line: "<char> <tag>" (or "<char> <type> <tag>" for 3-col data).
Blank line = sentence separator.
"""
import sys

NUMERIC_PUNCT = set(".")  # split-out: '.' only for now (decimal point)

def split_token(chars):
    """Split a list of chars (one token) into list of new token char-lists."""
    out, i, n = [], 0, len(chars)
    while i < n:
        c = chars[i]
        if c.isdigit():
            j = i
            while j < n and chars[j].isdigit(): j += 1
            out.append(chars[i:j])
            i = j
        elif c in NUMERIC_PUNCT and i > 0 and i < n-1 \
             and chars[i-1].isdigit() and chars[i+1].isdigit():
            # punct between digits: standalone
            out.append([c])
            i += 1
        else:
            # non-digit: keep until next digit (or punct-between-digits)
            j = i
            while j < n:
                if chars[j].isdigit(): break
                if (chars[j] in NUMERIC_PUNCT and j > 0 and j < n-1
                    and chars[j-1].isdigit() and chars[j+1].isdigit()):
                    break
                j += 1
            out.append(chars[i:j])
            i = j
    return out

def chars_to_bmes(chars):
    """Build BMES tags for a token of given length."""
    if len(chars) == 1: return ['S']
    return ['B'] + ['M']*(len(chars)-2) + ['E']

def process(in_path, out_path, has_type=False):
    n_sent = n_token_in = n_token_out = 0
    with open(in_path, encoding='utf-8') as fi, open(out_path, 'w', encoding='utf-8') as fo:
        sent_lines = []  # (char, type_or_None, tag)
        for line in fi:
            line = line.rstrip('\n')
            if not line:
                if sent_lines:
                    n_sent += 1
                    # Group into tokens by tag
                    tokens = []  # list of (chars, type or None)
                    cur_chars = []
                    cur_types = []
                    for ch, tp, tg in sent_lines:
                        if tg == 'B':
                            if cur_chars: tokens.append((cur_chars, cur_types))
                            cur_chars = [ch]; cur_types = [tp]
                        elif tg == 'M' or tg == 'E':
                            cur_chars.append(ch); cur_types.append(tp)
                        else: # S
                            if cur_chars: tokens.append((cur_chars, cur_types))
                            tokens.append(([ch], [tp]))
                            cur_chars = []; cur_types = []
                    if cur_chars: tokens.append((cur_chars, cur_types))
                    n_token_in += len(tokens)
                    # Transform each token
                    new_tokens = []
                    for chars, types in tokens:
                        sub = split_token(chars)
                        # rebuild type per sub-token
                        ci = 0
                        for sub_chars in sub:
                            sub_types = types[ci:ci+len(sub_chars)]
                            ci += len(sub_chars)
                            new_tokens.append((sub_chars, sub_types))
                    n_token_out += len(new_tokens)
                    # Write
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
                    # nolabel format: just char
                    fo.write(line + "\n")
        if sent_lines:
            # tail without final blank line, same as above
            pass
    return n_sent, n_token_in, n_token_out

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: retag_digits.py <input.txt> <output.txt> [--type]")
        sys.exit(1)
    inp, outp = sys.argv[1], sys.argv[2]
    has_type = "--type" in sys.argv
    s, ti, to = process(inp, outp, has_type)
    print(f"{inp}: sentences={s}  tokens {ti} -> {to} (+{to-ti})")
