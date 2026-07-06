"""Shared retag2 / PreSegment helpers for the data scripts.

`classify` mirrors ClassifyCodePoint in src/preprocess.cc — keep the two in sync
if the character-type rules ever change. Imported by check_retag2.py,
normalize_retag2.py, convert.py and prepare.py so the rules live in one place.
"""


def classify(cp):
    """Unicode code point -> run category: S space, D digit, L latin, H han, P punct."""
    if cp in (0x20, 0x09, 0x0A, 0x0D, 0x0C, 0x0B, 0x00A0, 0x3000):
        return "S"
    if 0x30 <= cp <= 0x39 or 0xFF10 <= cp <= 0xFF19:
        return "D"
    if (0x41 <= cp <= 0x5A or 0x61 <= cp <= 0x7A or 0x00C0 <= cp <= 0x024F
            or 0xFF21 <= cp <= 0xFF3A or 0xFF41 <= cp <= 0xFF5A):
        return "L"
    if (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF
            or 0xF900 <= cp <= 0xFAFF or cp == 0x3007
            or 0x20000 <= cp <= 0x2A6DF or 0x2A700 <= cp <= 0x2EBEF):
        return "H"
    return "P"


def resegment(source, words):
    """Retokenize `source` to the PreSegment convention.

    Han word boundaries from `words` are preserved; non-Han is merged/split by
    character type (latin/digit run -> one token, latin|digit split, each
    punctuation mark its own token) and whitespace is dropped.

    Returns (new_words, aligned). aligned is False — and new_words == words —
    when `words` don't line up with `source` (a data anomaly; caller may skip).
    """
    starts = [False] * len(source)
    si = 0
    for w in words:
        while si < len(source) and classify(ord(source[si])) == "S":
            si += 1
        if source[si:si + len(w)] != w:
            return words, False
        starts[si] = True
        si += len(w)

    out, cur, prev_cat, prev_space = [], "", None, True
    for i, ch in enumerate(source):
        cat = classify(ord(ch))
        if cat == "S":
            prev_space = True
            continue
        if cur == "":
            start_new = True
        elif prev_space or cat != prev_cat or cat == "P":
            start_new = True
        elif cat == "H":
            start_new = starts[i]
        else:
            start_new = False
        if start_new and cur:
            out.append(cur)
            cur = ""
        cur += ch
        prev_cat = cat
        prev_space = False
    if cur:
        out.append(cur)
    return out, True


def words_to_bmes(words):
    """A list of words -> BMES column lines ('<char> <B|M|E|S>')."""
    lines = []
    for w in words:
        if len(w) == 1:
            lines.append(w + " S")
        else:
            lines.append(w[0] + " B")
            for c in w[1:-1]:
                lines.append(c + " M")
            lines.append(w[-1] + " E")
    return lines
