#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Inference-only text pre-segmentation. Splits raw UTF-8 text into maximal
// same-category runs so that only Han (CJK) runs are handed to the CRF, while
// latin / digit / punctuation runs become tokens directly. This mirrors, at
// runtime, the offline tokenization the model was trained on (retag2: split
// punctuation, split at CN/EN/digit boundaries).
//
// This module is pure text logic — it does NOT depend on the model and is NOT
// used anywhere in the training path (fit / LoadData / RawToSentence).
namespace wati {

enum class RunType { Han, Latin, Digit, Punct, Space };

struct Run {
    std::string text;
    RunType type;
};

// Bytes in the UTF-8 char whose lead byte is c (1-4; 1 on malformed lead).
int Utf8CharLen(unsigned char c);

// Decode one UTF-8 code point at s[i]; returns bytes consumed (1-4, or 1 on a
// malformed/truncated sequence) and writes the code point to *cp.
int Utf8Decode(const std::string& s, size_t i, uint32_t* cp);

// Classify a Unicode code point into a run category.
RunType ClassifyCodePoint(uint32_t cp);

// Split raw UTF-8 into runs, left to right. Han / Latin / Digit / Space runs are
// maximal (consecutive same-category code points merge); each Punct code point
// is its own run (punctuation is split, per retag2). Space runs are emitted so
// callers can see boundaries; they are normally dropped.
std::vector<Run> PreSegment(const std::string& utf8);

}  // namespace wati
