#include "trie.h"
#include "misc.h"

#include <cstdio>
#include <algorithm>
#include <map>
#ifdef __GLIBC__
#include <malloc.h>
#endif

namespace wati {

// ----- DartArray (XOR Double-Array Trie) implementation ------------------

namespace {

struct DartBuilder {
    struct TrieNode {
        std::map<uint8_t, std::unique_ptr<TrieNode>> down;
        bool eow = false;
        int32_t value = 0;
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
            uint8_t t = static_cast<uint8_t>(c);
            auto it = cur->down.find(t);
            if (it == cur->down.end()) {
                auto p = std::make_unique<TrieNode>();
                TrieNode* raw = p.get();
                cur->down[t] = std::move(p);
                cur = raw;
            } else {
                cur = it->second.get();
            }
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

    void Build(const std::vector<std::string>& strs,
               const std::vector<int32_t>& vals) {
        root = std::make_unique<TrieNode>();
        for (std::size_t i = 0; i < strs.size(); ++i) Insert(strs[i], vals[i]);
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

void DartArray::Build(const std::vector<std::string>& strs,
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
    const uint32_t max = 1024;
    if (root_) {
        Node* s[max];
        uint32_t cnt = 0;
        s[cnt++] = root_;
        while (cnt != 0) {
            Node* n = s[--cnt];
            if (IsValue(n)) {
                continue;
            }
            s[cnt++] = n->data[0];
            s[cnt++] = n->data[1];
            delete n;
        }
    }
    // Value* objects are owned via data_ regardless of whether the Node tree
    // is still alive (BuildDAT may have freed the inner Node* tree).
    for (auto* v : data_) delete v;
}

int64_t Trie::Insert(const std::string& value) {
    // Fast path: when locked & DAT built, lookup via DAT (cache-friendly).
    if (is_lock_ && !dat_.Empty()) {
        auto r = dat_.Lookup(value);
        return r.found ? static_cast<int64_t>(r.value) : -1;
    }

    // Handle empty tree case
    if (data_.empty()) {
        if (is_lock_) return -1;

        auto v = new Value(value, 0);
        data_.push_back(v);
        root_ = ValueToNode(data_.back());
    }

    // Search down the tree
    Node* node = root_;
    while (!IsValue(node)) {
        const uint8_t c = node->pos < value.length() ? value[node->pos] : 0;
        const int s = ((c | node->byte) + 1) >> 8;
        node = node->data[s];
    }

    // Compare with found value
    Value* e = NodeToValue(node);
    const std::string& t = e->value;

    size_t pos = 0;
    const size_t shared = std::min(value.length(), t.length());
    for (; pos < shared; pos++) {
        if (value[pos] != t[pos]) break;
    }

    uint8_t byte;
    if (pos < value.length() && pos < t.length()) {
        byte = value[pos] ^ t[pos];
    } else if (pos < t.length()) { // v is Prefix of t
        byte = t[pos];
    } else if (pos < value.length()) { // t is Prefix of v
        byte = value[pos];
    } else { // v == t
        return e->i;
    }

    if (is_lock_) return -1;

    // Find critical bit
    while (byte & (byte-1)) {
        byte &= byte -1;
    }
    byte ^= 255;

    // Create new Value and internal Node
    const uint8_t c = t[pos];
    const int s = ((c | byte) + 1) >> 8;

    auto new_node = new Node();
    auto new_value = new Value(value, data_.size());

    new_node->pos = pos;
    new_node->byte = byte;
    new_node->data[1-s] = ValueToNode(new_value);

    // Insert new node in tree
    Node** tx = &root_;
    while (true) {
        Node* n = *tx;
        if (IsValue(n) || n->pos > pos) break;
        if (n->pos == pos && n->byte > byte) break;

        const uint8_t c = n->pos < value.length() ? value[n->pos] : 0;
        const int s = ((c | n->byte) + 1) >> 8;
        tx = &n->data[s];
    }

    new_node->data[s] = *tx;
    *tx = new_node;

    data_.push_back(new_value);
    return new_value->i;
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

void Trie::Save(const std::string& filename) const {
    std::ofstream file(filename);
    Save(file);
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
    std::getline(file, line);
    if (line.find("#TrieBin#") == 0) {
        int64_t count = std::stoll(line.substr(9));
        for (int64_t i = 0; i < count; ++i) {
            Insert(ReadStrBin(file));
        }
    } else {
        // legacy text
        size_t start = line.find("#Trie#") + 6;
        int64_t count = std::stoll(line.substr(start));
        for (int64_t i = 0; i < count; ++i) {
            Insert(ReadStr(file));
        }
    }
}

void Trie::LoadAuto(std::istream& file, std::vector<int64_t>& id_map) {
    std::string line;
    std::getline(file, line);
    if (line.find("#TrieBin#") == 0) {
        int64_t count = std::stoll(line.substr(9));
        id_map.resize(count);
        for (int64_t i = 0; i < count; ++i) {
            id_map[i] = Insert(ReadStrBin(file));
        }
    } else {
        size_t start = line.find("#Trie#") + 6;
        int64_t count = std::stoll(line.substr(start));
        id_map.resize(count);
        for (int64_t i = 0; i < count; ++i) {
            id_map[i] = Insert(ReadStr(file));
        }
    }
}


void Trie::Load(const std::string &filename) {
    std::ifstream file(filename);
    Load(file);
}

void Trie::FreeValueStrings() {
    // Inference-only: drop Value objects entirely (strings + structs + ptr vec).
    // After this call, GetValue(i) becomes invalid; only DAT lookup works.
    for (auto* v : data_) delete v;
    std::vector<Value*>().swap(data_);
#ifdef __GLIBC__
    // Return arenas to the OS so RSS actually drops.
    malloc_trim(0);
#endif
}

void Trie::BuildDAT() {
    if (!dat_.Empty()) return;  // already built; idempotent

    // Collect (value, id) pairs and sort by value (DAT requires sorted input).
    std::vector<std::pair<std::string, int32_t>> pairs;
    pairs.reserve(data_.size());
    for (auto* v : data_) {
        if (v->i > INT32_MAX) {
            // overflow: skip building DAT (fall back to binary trie)
            return;
        }
        pairs.emplace_back(v->value, static_cast<int32_t>(v->i));
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<std::string> strs;
    std::vector<int32_t> ids;
    strs.reserve(pairs.size());
    ids.reserve(pairs.size());
    for (auto& p : pairs) {
        strs.push_back(std::move(p.first));
        ids.push_back(p.second);
    }
    dat_.Build(strs, ids);

    // NOTE: previously we freed root_ here (binary tree was redundant with the
    // DAT for inference). That broke a "lock → unlock → insert" cycle used by
    // training when warm-starting from an existing model and then loading a
    // larger binary cache (LoadBinary inserts new obs strings via Insert,
    // which needs root_). Keep the binary tree alive so SetLock(false) +
    // Insert continues to work; memory cost is small relative to theta_.
}

} // namespace wati
