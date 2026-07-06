#include "preprocess.h"

namespace wati {

int Utf8CharLen(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

int Utf8Decode(const std::string& s, size_t i, uint32_t* cp) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    const int len = Utf8CharLen(c);
    if (i + static_cast<size_t>(len) > s.size()) {  // truncated
        *cp = c;
        return 1;
    }
    uint32_t v;
    switch (len) {
        case 2: v = c & 0x1F; break;
        case 3: v = c & 0x0F; break;
        case 4: v = c & 0x07; break;
        default: *cp = c; return 1;
    }
    for (int k = 1; k < len; ++k) {
        const unsigned char cc = static_cast<unsigned char>(s[i + k]);
        if ((cc & 0xC0) != 0x80) {  // malformed continuation
            *cp = c;
            return 1;
        }
        v = (v << 6) | (cc & 0x3F);
    }
    *cp = v;
    return len;
}

namespace {
inline bool InRange(uint32_t c, uint32_t lo, uint32_t hi) {
    return c >= lo && c <= hi;
}
}  // namespace

RunType ClassifyCodePoint(uint32_t c) {
    // Whitespace (ASCII + ideographic space + no-break space).
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
        c == '\v' || c == 0x00A0 || c == 0x3000)
        return RunType::Space;

    // Digits: ASCII + fullwidth.
    if (InRange(c, '0', '9') || InRange(c, 0xFF10, 0xFF19))
        return RunType::Digit;

    // Latin letters: ASCII + Latin-1/extended + fullwidth forms.
    if (InRange(c, 'A', 'Z') || InRange(c, 'a', 'z') ||
        InRange(c, 0x00C0, 0x024F) ||
        InRange(c, 0xFF21, 0xFF3A) || InRange(c, 0xFF41, 0xFF5A))
        return RunType::Latin;

    // Han (CJK ideographs): BMP blocks + extensions + compatibility + 〇.
    if (InRange(c, 0x4E00, 0x9FFF) ||   // CJK Unified Ideographs
        InRange(c, 0x3400, 0x4DBF) ||   // Extension A
        InRange(c, 0xF900, 0xFAFF) ||   // Compatibility Ideographs
        c == 0x3007 ||                  // 〇 ideographic number zero
        InRange(c, 0x20000, 0x2A6DF) || // Extension B
        InRange(c, 0x2A700, 0x2EBEF))   // Extensions C-F
        return RunType::Han;

    // Everything else (punctuation, symbols, unknown) is a punctuation token.
    return RunType::Punct;
}

std::vector<Run> PreSegment(const std::string& utf8) {
    std::vector<Run> runs;
    const size_t n = utf8.size();
    size_t i = 0;
    while (i < n) {
        uint32_t cp;
        const int len = Utf8Decode(utf8, i, &cp);
        const RunType t = ClassifyCodePoint(cp);
        // Punctuation is always split into single-mark tokens; other categories
        // merge consecutive same-category code points into one run.
        if (!runs.empty() && runs.back().type == t && t != RunType::Punct) {
            runs.back().text.append(utf8, i, len);
        } else {
            runs.push_back(Run{utf8.substr(i, len), t});
        }
        i += static_cast<size_t>(len);
    }
    return runs;
}

}  // namespace wati
