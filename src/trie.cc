#include "trie.h"
#include "misc.h"

namespace wati {


Trie::~Trie() {
    const uint32_t max = 1024;
    if (data_.size() != 0) {
        Node* s[max];
        uint32_t cnt = 0;
        s[cnt++] = root_;
        while (cnt != 0) {
            Node* n = s[--cnt];
            if (IsValue(n)) {
                delete NodeToValue(n);
                continue;
            }
            s[cnt++] = n->data[0];
            s[cnt++] = n->data[1];
            delete n;
        }
    }
}

int64_t Trie::Insert(const std::string& value) {
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
    for (; pos < value.length(); pos++) {
        if (value[pos] != t[pos]) break;
    }

    uint8_t byte;
    if (pos != value.length()) { //  Prefix of v 
        byte = value[pos] ^ t[pos];
    } else if (pos < t.length()) { // v is Prefix of t
        byte = t[pos];
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


void Trie::Load(const std::string &filename) {
    std::ifstream file(filename);
    Load(file);
}

} // namespace wati