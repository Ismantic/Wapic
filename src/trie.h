#pragma once 

#include <stdint.h>
#include <stdlib.h>

#include <vector>
#include <string>
#include <memory>
#include <iostream>

namespace wati {

class Trie {
private:
    struct Node {
        Node* data[2];
        uint32_t pos;
        uint8_t byte;
    };

    struct Value {
        int64_t i;
        std::string value;

        Value(const std::string& v, int64_t i) : i(i), value(v) {}
    };

    static Node* ValueToNode(Value* v) {
        return reinterpret_cast<Node*>(reinterpret_cast<uintptr_t>(v)|1);
    }
    static Value* NodeToValue(Node* n) {
        return reinterpret_cast<Value*>(reinterpret_cast<uintptr_t>(n) & ~1);
    }
    static bool IsValue(Node* n) {
        return reinterpret_cast<uintptr_t>(n) & 1;    
    }

    Node* root_;
    std::vector<Value*> data_;
    bool is_lock_;

public:
    Trie() : is_lock_(false) {}
    ~Trie();

    Trie(const Trie&) = delete;
    Trie& operator=(const Trie&) = delete;

    Trie(Trie&&) = default;
    Trie& operator=(Trie&&) = default;

    int64_t Insert(const std::string& value);
    const std::string& GetValue(int64_t i) const;

    void Save(const std::string& filename) const;
    void Load(const std::string& filename);

    void Save(std::ostream& file) const;
    void Load(std::istream& file);

    size_t Size() const { return data_.size(); }
    bool SetLock(bool n) {
        bool o = is_lock_;
        is_lock_ = n;
        return o;
    }
};

} // namespace wati