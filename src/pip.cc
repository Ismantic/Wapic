// Python binding for Wapic CRF Chinese segmenter.
// Build as Python extension:  wapic._core
//
// Usage:
//   import wapic
//   seg = wapic.Segmenter("model.wac")
//   words = seg.cut("中华人民共和国是一个伟大的国家")
//   tags  = seg.tag("中华人民共和国是一个伟大的国家")  # list[(char, "B/M/E/S")]
//   batch = seg.cut_batch(["第一句", "第二句"])
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <mutex>

#include "data.h"
#include "model.h"
#include "preprocess.h"
#include "score.h"
#include "sentence.h"

namespace py = pybind11;

// Decode a single UTF-8 character, return bytes consumed (1-4), or 0 on error
static int utf8_char_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

// Split a UTF-8 string into individual characters
static std::vector<std::string> utf8_chars(const std::string& s) {
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < s.size()) {
        int len = utf8_char_len(static_cast<unsigned char>(s[i]));
        chars.push_back(s.substr(i, len));
        i += len;
    }
    return chars;
}

// Char-level chunking: process at most max_chars_per_seg per CRF call,
// so very long inputs don't blow scorer state. The CRF sees each chunk as a
// separate sentence (small loss of context at chunk boundaries; default 1024
// chars per chunk keeps boundary loss negligible for normal text).
class Segmenter {
public:
    Segmenter(const std::string& model_path) {
        model_ = std::make_unique<wati::Model>(std::make_unique<wati::DataProcessor>());
        model_->Load(model_path);
        scorer_ = std::make_unique<wati::Scorer>(model_.get());
        processor_ = model_->GetDataProcessor();
        // Cache label -> string for fast Viterbi -> tag conversion.
        int64_t n = model_->LabelCount();
        label_strs_.resize(n);
        for (int64_t i = 0; i < n; i++) {
            label_strs_[i] = processor_->GetLabelStr(i);
        }
        // Inference-only: obs strings won't be needed (DAT handles lookups).
        model_->FreeObservationStrings();
    }

    // Get all chars + BMES tags for one text.
    // Returns (chars, tags). On empty input, returns ({}, {}).
    std::pair<std::vector<std::string>, std::vector<std::string>>
    tag(const std::string& text) {
        auto chars = utf8_chars(text);
        std::vector<std::string> tags;
        if (chars.empty()) return {chars, tags};
        tags.reserve(chars.size());

        // Single-threaded scorer access: use one mutex per Segmenter.
        // CRF Viterbi mutates scorer internal buffers, so concurrent calls
        // on the same instance must serialize.
        std::lock_guard<std::mutex> lock(mtx_);

        const size_t kChunk = 1024;
        for (size_t start = 0; start < chars.size(); start += kChunk) {
            size_t end = std::min(start + kChunk, chars.size());
            // Build RawStrs directly (skip string buffer + istringstream).
            wati::RawStrs raw;
            raw.strs.reserve(end - start);
            for (size_t i = start; i < end; i++) raw.strs.push_back(chars[i]);
            wati::Sentence* sen = processor_->RawToSentence(&raw, false);
            if (!sen) {
                for (size_t i = start; i < end; i++) tags.push_back("S");
                continue;
            }
            std::vector<int64_t> labels;
            scorer_->Viterbi(*sen, labels);
            for (size_t t = 0; t < labels.size() && (start + t) < end; t++) {
                int64_t li = labels[t];
                if (li >= 0 && (size_t)li < label_strs_.size())
                    tags.push_back(label_strs_[li]);
                else
                    tags.push_back("S");
            }
            // Pad with S if Viterbi returned fewer than expected.
            while (tags.size() < end) tags.push_back("S");
            delete sen;
        }
        return {chars, tags};
    }

    // Cut with pre-segmentation (default): split raw text into runs, send only
    // Han runs to the CRF; latin/digit/punctuation become tokens directly. This
    // matches the retag2 tokenization the model was trained on.
    std::vector<std::string> cut(const std::string& text) {
        std::vector<std::string> out;
        for (const auto& run : wati::PreSegment(text)) {
            if (run.type == wati::RunType::Space) continue;  // word boundary
            if (run.type == wati::RunType::Han) {
                for (auto& w : cut_raw(run.text)) out.push_back(std::move(w));
            } else {
                out.push_back(run.text);  // latin / digit / punct token
            }
        }
        return out;
    }

    // Raw char-level cut: hand the whole string to the CRF one char at a time,
    // no pre-segmentation. Kept for callers needing the training-columnar path.
    std::vector<std::string> cut_raw(const std::string& text) {
        auto [chars, tags] = tag(text);
        std::vector<std::string> words;
        std::string cur;
        size_t n = std::min(chars.size(), tags.size());
        for (size_t i = 0; i < n; i++) {
            cur += chars[i];
            const std::string& t = tags[i];
            if (t == "E" || t == "S") {
                words.push_back(cur);
                cur.clear();
            }
        }
        if (!cur.empty()) words.push_back(cur);
        return words;
    }

    std::vector<std::vector<std::string>>
    cut_batch(const std::vector<std::string>& texts) {
        std::vector<std::vector<std::string>> out;
        out.reserve(texts.size());
        for (const auto& t : texts) out.push_back(cut(t));
        return out;
    }

    // Word boundaries (char-level start indices of each word, + final length).
    // Suited for WWM mask: zip with chars to know "is this char a word start".
    std::vector<int> word_starts(const std::string& text) {
        auto [chars, tags] = tag(text);
        std::vector<int> starts;
        size_t n = std::min(chars.size(), tags.size());
        for (size_t i = 0; i < n; i++) {
            const std::string& t = tags[i];
            if (t == "B" || t == "S") starts.push_back((int)i);
        }
        starts.push_back((int)chars.size());
        return starts;
    }

    // cut_smart: kept for backward compatibility. The pre-segmentation logic it
    // used to carry now lives in cut()/PreSegment (proper Unicode classification,
    // punctuation and CN/EN/digit boundaries), so this just delegates.
    std::vector<std::string> cut_smart(const std::string& text) {
        return cut(text);
    }

    std::vector<std::vector<std::string>>
    cut_smart_batch(const std::vector<std::string>& texts) {
        std::vector<std::vector<std::string>> out;
        out.reserve(texts.size());
        for (const auto& t : texts) out.push_back(cut_smart(t));
        return out;
    }

    int64_t label_count() const { return model_->LabelCount(); }
    int64_t feature_count() const { return model_->FeatureCount(); }

private:
    std::unique_ptr<wati::Model> model_;
    std::unique_ptr<wati::Scorer> scorer_;
    const wati::DataProcessor* processor_ = nullptr;
    std::vector<std::string> label_strs_;
    std::mutex mtx_;
};

PYBIND11_MODULE(_core, m) {
    m.doc() = "Wapic: C++ CRF Chinese segmenter Python binding";

    py::class_<Segmenter>(m, "Segmenter")
        .def(py::init<const std::string&>(), py::arg("model_path"),
             "Load a Wapic CRF model from disk.")
        .def("cut", &Segmenter::cut, py::arg("text"),
             "Segment a string into list[str]. Pre-segments first (whitespace, "
             "punctuation, CN/EN/digit boundaries); only Han runs go to the CRF.")
        .def("cut_batch", &Segmenter::cut_batch, py::arg("texts"),
             "Segment a list of strings, returns list[list[str]].")
        .def("cut_raw", &Segmenter::cut_raw, py::arg("text"),
             "Raw char-level CRF cut with NO pre-segmentation (whole string fed "
             "one char at a time). Matches the training-columnar path.")
        .def("tag", &Segmenter::tag, py::arg("text"),
             "Return (chars: list[str], tags: list[str]). BMES tags.")
        .def("word_starts", &Segmenter::word_starts, py::arg("text"),
             "Char indices of word starts, plus final sentinel = len(chars). "
             "Useful for WWM mask building.")
        .def("cut_smart", &Segmenter::cut_smart, py::arg("text"),
             "Smart cut for mixed CN/EN: split by whitespace first; "
             "全英文 segment 整体 1 word(不让 wapic 切散),含中文 segment 用 CRF 切。"
             "适用 BERT WWM 数据 prep 等 LLM pretrain scenario。")
        .def("cut_smart_batch", &Segmenter::cut_smart_batch, py::arg("texts"),
             "Batch version of cut_smart.")
        .def_property_readonly("label_count", &Segmenter::label_count)
        .def_property_readonly("feature_count", &Segmenter::feature_count);
}
