#pragma once

#include <stdint.h>
#include <stdlib.h>

#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <stdexcept>

namespace wati {

// XOR Double-Array Trie (Darts-style), built once after Load() for fast
// string -> id lookup. Lookup is O(strlen) with cache-friendly contiguous array
// access vs binary trie's O(strlen * 8) pointer chasing.
//
// Self-contained (no separate file). Only string -> int32 lookup is exposed;
// the inverse (id -> string) keeps using Trie::data_.
class DartArray {
public:
    struct ArrayUnit {
        uint8_t label = 0;
        bool eow = false;
        union {
            uint32_t index;
            int32_t value;
        } u = {0};
        uint32_t parent = 0;

        bool HasValue() const { return label == '\0' && eow; }
        bool IsEmpty() const {
            return u.index == 0 && label == 0 && !eow && parent == 0;
        }
    };

    struct SearchResult {
        int32_t value;
        bool found;

        SearchResult() : value(0), found(false) {}
        SearchResult(int32_t v) : value(v), found(true) {}
    };

private:
    std::size_t size_ = 0;
    std::unique_ptr<ArrayUnit[]> array_;

public:
    DartArray() = default;

    DartArray(DartArray&& other) noexcept
        : size_(other.size_), array_(std::move(other.array_)) {
        other.size_ = 0;
    }
    DartArray& operator=(DartArray&& other) noexcept {
        if (this != &other) {
            size_ = other.size_;
            array_ = std::move(other.array_);
            other.size_ = 0;
        }
        return *this;
    }
    DartArray(const DartArray&) = delete;
    DartArray& operator=(const DartArray&) = delete;

    // Build from sorted (str, id) pairs. Both vectors must be the same length
    // and `strs` must be lexicographically sorted.
    void Build(const std::vector<std::string>& strs,
               const std::vector<int32_t>& values);

    SearchResult Lookup(const std::string& str) const;

    std::size_t Size() const { return size_; }
    bool Empty() const { return size_ == 0; }
};

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
        Value(std::string&& v, int64_t i) : i(i), value(std::move(v)) {}
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

    // Inference accelerator (only built when locked & SyncDAT() called).
    DartArray dat_;

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

    // Compact binary save: "#TrieBin#N\n" then N entries of (u16 len, bytes).
    void SaveBin(std::ostream& file) const;
    // Auto-detect header: #TrieBin# vs #Trie#
    void LoadAuto(std::istream& file);

    size_t Size() const { return data_.size(); }
    bool SetLock(bool n) {
        bool o = is_lock_;
        is_lock_ = n;
        if (n) BuildDAT();  // lock -> build inference accelerator
        return o;
    }

    // Build the DAT accelerator from current data_. Idempotent.
    void BuildDAT();

    // Inference-only: drop the per-entry value strings from data_ (saves
    // memory). GetValue(i) becomes invalid afterwards; lookups via DAT still
    // work because the DAT array embeds its own labels. Safe to call only when
    // the model will not be saved/exported again.
    void FreeValueStrings();
};

} // namespace wati
