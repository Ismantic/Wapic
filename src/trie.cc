#include "trie.h"
#include "misc.h"

#include <cstdio>
#include <algorithm>
#ifdef __GLIBC__
#include <malloc.h>
#endif

namespace wati {

// ----- DartArray (XOR Double-Array Trie) implementation ------------------

namespace {

struct DartBuilder {
    // Children kept sorted by byte in a flat vector. std::map would allocate a
    // red-black node per child and a tree per node; for a trie over ~2M sorted
    // strings that dominates build-time peak memory. Since Build() feeds strings
    // in sorted order, children almost always arrive in increasing byte order,
    // so find_or_add() hits the cheap append path (binary-search insert is the
    // rare fallback, kept only for correctness on unsorted input).
    struct TrieNode {
        std::vector<std::pair<uint8_t, std::unique_ptr<TrieNode>>> down;
        bool eow = false;
        int32_t value = 0;

        TrieNode* find_or_add(uint8_t t) {
            if (!down.empty() && down.back().first < t) {
                down.emplace_back(t, std::make_unique<TrieNode>());
                return down.back().second.get();
            }
            auto it = std::lower_bound(
                down.begin(), down.end(), t,
                [](const auto& p, uint8_t k) { return p.first < k; });
            if (it != down.end() && it->first == t) return it->second.get();
            it = down.emplace(it, t, std::make_unique<TrieNode>());
            return it->second.get();
        }
    };

    std::vector<DartArray::ArrayUnit> units;
    std::vector<bool> uses;
    std::unique_ptr<TrieNode> root;
    uint32_t prev_pos = 0;

    void EnsureSize(std::size_t n) {
        if (n > units.size()) {
            std::size_t ns = std::max(n, units.size() * 2);
            units.resize(ns);
            uses.resize(ns, false);
        }
    }

    void Insert(const std::string& s, int32_t v) {
        TrieNode* cur = root.get();
        for (char c : s) {
            cur = cur->find_or_add(static_cast<uint8_t>(c));
        }
        cur->eow = true;
        cur->value = v;
    }

    uint32_t GetFreeIndex(const std::vector<uint8_t>& es) {
        if (es.empty()) return 0;
        uint32_t start = (prev_pos > 256) ? prev_pos - 256 : 1;
        for (uint32_t i = start; ; ++i) {
            bool ok = true;
            for (uint8_t e : es) {
                std::size_t p = i ^ e;
                EnsureSize(p + 1);
                if (uses[p]) { ok = false; break; }
            }
            if (ok) { prev_pos = i; return i; }
        }
    }

    uint32_t SetupDown(const std::vector<uint8_t>& es,
                       std::size_t pos, TrieNode* node) {
        if (es.empty()) return 0;
        uint32_t index = GetFreeIndex(es);
        units[pos].u.index = pos ^ index;
        for (std::size_t i = 0; i < es.size(); ++i) {
            uint8_t e = es[i];
            std::size_t p = index ^ e;
            EnsureSize(p + 1);
            uses[p] = true;
            if (e == '\0') {
                units[p].label = 0;
                units[p].u.value = node->value;
                units[p].eow = true;
                units[p].parent = pos;
                units[pos].eow = true;
            } else {
                units[p].label = e;
                units[p].parent = pos;
            }
        }
        return index;
    }

    void ConvertNode(TrieNode* node, std::size_t pos) {
        std::vector<uint8_t> es;
        std::vector<TrieNode*> downs;
        es.reserve(node->down.size() + (node->eow ? 1 : 0));
        downs.reserve(es.capacity());
        for (auto& kv : node->down) {
            es.push_back(kv.first);
            downs.push_back(kv.second.get());
        }
        if (node->eow) { es.push_back('\0'); downs.push_back(nullptr); }
        if (es.empty()) return;
        uint32_t index = SetupDown(es, pos, node);
        for (std::size_t i = 0; i < es.size(); ++i) {
            uint8_t e = es[i];
            if (e != '\0') {
                std::size_t dp = index ^ e;
                ConvertNode(downs[i], dp);
            }
        }
    }

    void Build(const std::vector<const std::string*>& strs,
               const std::vector<int32_t>& vals) {
        root = std::make_unique<TrieNode>();
        for (std::size_t i = 0; i < strs.size(); ++i) Insert(*strs[i], vals[i]);
        units.assign(1024, {});
        uses.assign(1024, false);
        uses[0] = true;
        units[0].label = 0;
        if (!root->down.empty()) ConvertNode(root.get(), 0);
        while (!units.empty() && !uses.back()) {
            units.pop_back();
            uses.pop_back();
        }
    }
};

}  // namespace

void DartArray::Build(const std::vector<const std::string*>& strs,
                      const std::vector<int32_t>& values) {
    if (strs.size() != values.size() || strs.empty()) {
        size_ = 0;
        array_.reset();
        return;
    }
    DartBuilder b;
    b.Build(strs, values);
    size_ = b.units.size();
    array_ = std::make_unique<ArrayUnit[]>(size_);
    for (std::size_t i = 0; i < size_; ++i) array_[i] = b.units[i];
}

DartArray::SearchResult DartArray::Lookup(const std::string& str) const {
    if (size_ == 0) return SearchResult();
    std::size_t pos = 0;
    const char* s = str.c_str();
    std::size_t n = str.length();
    for (std::size_t i = 0; i < n; ++i) {
        if (pos >= size_) return SearchResult();
        const ArrayUnit& u = array_[pos];
        uint32_t idx = u.u.index;
        uint8_t ch = static_cast<uint8_t>(s[i]);
        std::size_t prev = pos;
        std::size_t next = pos ^ idx ^ ch;
        if (next >= size_) return SearchResult();
        pos = next;
        const ArrayUnit& nu = array_[pos];
        if (nu.label != ch || nu.parent != prev) return SearchResult();
    }
    const ArrayUnit& cur = array_[pos];
    if (!cur.eow) return SearchResult();
    std::size_t vp = pos ^ cur.u.index;
    if (vp >= size_) return SearchResult();
    const ArrayUnit& vn = array_[vp];
    if (!vn.HasValue()) return SearchResult();
    return SearchResult(vn.u.value);
}

// ----- Trie methods -------------------------------------------------------

Trie::~Trie() {
    for (auto* v : data_) delete v;
}

int64_t Trie::Insert(const std::string& value) {
    // Locked & DAT built: lookup via DAT (cache-friendly, read-only).
    if (is_lock_ && !dat_.Empty()) {
        auto r = dat_.Lookup(value);
        return r.found ? static_cast<int64_t>(r.value) : -1;
    }

    // Mutable phase (or DAT-overflow fallback): hash index over data_.
    auto it = index_.find(std::string_view(value));
    if (it != index_.end()) return it->second;
    if (is_lock_) return -1;

    auto* v = new Value(value, static_cast<int64_t>(data_.size()));
    data_.push_back(v);
    index_.emplace(std::string_view(v->value), v->i);
    return v->i;
}

const std::string& Trie::GetValue(int64_t i) const {
    return data_[i]->value;
}

void Trie::Save(std::ostream& file) const {
    file << "#Trie#" << data_.size() << "\n";
    for (const auto& v : data_) {
        WriteStr(file, v->value);
    }
}

void Trie::SaveBin(std::ostream& file) const {
    file << "#TrieBin#" << data_.size() << "\n";
    for (const auto& v : data_) {
        WriteStrBin(file, v->value);
    }
}

void Trie::Load(std::istream& file) {
    std::string line;
    std::getline(file, line);

    size_t start = line.find("#Trie#")+6;

    int64_t count = std::stoll(line.substr(start));

    for (int64_t i = 0; i < count; ++i) {
        std::string line;
        line = ReadStr(file);
        Insert(line);
    }
}

void Trie::LoadAuto(std::istream& file) {
    std::string line;
    if (!std::getline(file, line)) {
        throw std::runtime_error("Invalid model: missing trie header");
    }

    // On the load path every entry is unique and its id is exactly its position
    // in the file (that is how SaveBin/Save emit them, and the weight indices in
    // the model depend on this order), so values are appended directly. The
    // hash index is not built either: loaded tries are locked right after
    // (Sync -> BuildDAT) and lookups go through the DAT.
    const bool bin = line.rfind("#TrieBin#", 0) == 0;
    const bool text = line.rfind("#Trie#", 0) == 0;
    if (!bin && !text) {
        throw std::runtime_error("Invalid model: malformed trie header");
    }
    const size_t start = bin ? 9 : 6;
    const int64_t count = std::stoll(line.substr(start));
    if (count < 0 || count > INT32_MAX) {
        throw std::runtime_error("Invalid model: trie size out of range");
    }

    data_.reserve(data_.size() + count);
    for (int64_t i = 0; i < count; ++i) {
        std::string s = bin ? ReadStrBin(file) : ReadStr(file);
        data_.push_back(new Value(std::move(s), static_cast<int64_t>(data_.size())));
    }
}


void Trie::FreeValueStrings() {
    // Inference-only: drop Value objects entirely (strings + structs + ptr vec).
    // After this call, GetValue(i) becomes invalid; only DAT lookup works.
    std::unordered_map<std::string_view, int64_t>().swap(index_);  // views dangle
    for (auto* v : data_) delete v;
    std::vector<Value*>().swap(data_);
#ifdef __GLIBC__
    // Return arenas to the OS so RSS actually drops.
    malloc_trim(0);
#endif
}

void Trie::BuildDAT() {
    if (!dat_.Empty()) return;  // already built; idempotent

    // DAT needs the entries sorted by value. Sort an index permutation and hand
    // the builder pointers into data_'s strings rather than copying every string
    // into a temporary (which, on a large model, doubles peak memory for the
    // duration of the build).
    const size_t N = data_.size();
    for (auto* v : data_) {
        if (v->i > INT32_MAX) {
            // Overflow: skip building the DAT; the hash index (kept alive in
            // this case) keeps serving lookups.
            return;
        }
    }
    std::vector<std::pair<const std::string*, int32_t>> order;
    order.reserve(N);
    for (auto* v : data_)
        order.emplace_back(&v->value, static_cast<int32_t>(v->i));
    std::sort(order.begin(), order.end(),
              [](const auto& a, const auto& b) { return *a.first < *b.first; });

    std::vector<const std::string*> strs(N);
    std::vector<int32_t> ids(N);
    for (size_t k = 0; k < N; ++k) {
        strs[k] = order[k].first;
        ids[k]  = order[k].second;
    }
    dat_.Build(strs, ids);

    // The DAT supersedes the hash index; free it (data_ Value* are kept for
    // GetValue(i) inverse lookup and for saving).
    std::unordered_map<std::string_view, int64_t>().swap(index_);
}

} // namespace wati
