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
            std::string buf;
            for (size_t i = start; i < end; i++) { buf += chars[i]; buf += '\n'; }
            buf += '\n';
            std::istringstream iss(buf);
            wati::RawStrs* raw = processor_->ReadRawStrs(iss);
            if (!raw) { for (size_t i = start; i < end; i++) tags.push_back("S"); continue; }
            wati::Sentence* sen = processor_->RawToSentence(raw, false);
            if (!sen) {
                delete raw;
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
            delete raw;
            delete sen;
        }
        return {chars, tags};
    }

    // Cut: char + tag → joined words.
    std::vector<std::string> cut(const std::string& text) {
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
             "Segment a single string into list[str] of words.")
        .def("cut_batch", &Segmenter::cut_batch, py::arg("texts"),
             "Segment a list of strings, returns list[list[str]].")
        .def("tag", &Segmenter::tag, py::arg("text"),
             "Return (chars: list[str], tags: list[str]). BMES tags.")
        .def("word_starts", &Segmenter::word_starts, py::arg("text"),
             "Char indices of word starts, plus final sentinel = len(chars). "
             "Useful for WWM mask building.")
        .def_property_readonly("label_count", &Segmenter::label_count)
        .def_property_readonly("feature_count", &Segmenter::feature_count);
}
