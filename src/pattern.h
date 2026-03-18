#pragma once 

#include <string>
#include <vector>
#include <memory>

#include <stdint.h>

#include "sentence.h"

namespace wati {

class Pattern {
public:
    Pattern(const std::string& p);
    ~Pattern() = default;

    std::string Execute(const TokenStrs& tokens, uint32_t at);

    uint32_t TokenNum() const { return token_num; }
    std::string GetSource() const { return src; }

private:
    struct Item {
        char type;
        bool caps;
        std::string value;
        bool absolute;
        int32_t offset;
        uint32_t column;

        Item() : type('s'), caps(false), absolute(false), offset(0), column(0) {}
    };

    std::string src;
    uint32_t token_num;
    uint32_t item_num;
    std::vector<Item> items;
};

} // namespace wati