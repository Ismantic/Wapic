#include "pattern.h"

#include <algorithm>
#include <stdexcept>

namespace wati {

bool MatchCharacter(const std::string& pattern, size_t pos, char c) {
    if (c == '\0') return false;
    if (pattern[pos] == '.') return true; 
    if (pattern[pos] == '\\' && pos + 1 < pattern.length()) {
        switch (pattern[pos + 1]) {
            case 'a': return std::isalpha(c);
            case 'd': return std::isdigit(c);
            case 'l': return std::islower(c);
            case 'p': return std::ispunct(c);
            case 's': return std::isspace(c);
            case 'u': return std::isupper(c);
            case 'w': return std::isalnum(c);
            case 'A': return !std::isalpha(c);
            case 'D': return !std::isdigit(c);
            case 'L': return !std::islower(c);
            case 'P': return !std::ispunct(c);
            case 'S': return !std::isspace(c);
            case 'U': return !std::isupper(c);
            case 'W': return !std::isalnum(c);
            default: return pattern[pos + 1] == c;
        }
    }
    return pattern[pos] == c;
}

bool MatchPattern(const std::string& re, const std::string& str, uint32_t& n) {
    if (re.empty()) return true;
    if (re[0] == '$' && re.length() == 1) return str.empty();

    size_t cn = (re[0] == '\\') ? 2 : 1;
    std::string next = re.substr(cn);

    if (!next.empty() && next[0] == '*') {
        next = next.substr(1);
        size_t pos = 0;
        do {
            uint32_t save = n;
            if (MatchPattern(next, str.substr(pos), n)) return true;
            n = save + 1;
            pos++;
        } while (pos <= str.length() && MatchCharacter(re, 0, str[pos-1]));
        return false;
    }

    if (!next.empty() && next[0] == '?') {
        next = next.substr(1);
        if (!str.empty() && MatchCharacter(re, 0, str[0])) {
            ++n;
            if (MatchPattern(next, str.substr(1), n)) return true;
            --n;
        }
        return MatchPattern(next, str, n);
    }

    ++n;
    return !str.empty() && MatchCharacter(re, 0, str[0]) &&
           MatchPattern(next, str.substr(1), n);
}

int32_t MatchRegex(const std::string re, const std::string& str, uint32_t& n) {
    if (re[0] == '^') {
        n = 0;
        if (MatchPattern(re.substr(1), str, n)) return 0;
        return -1;
    }    

    for (size_t pos = 0; pos <= str.length(); ++pos) {
        n = 0;
        if (MatchPattern(re, str.substr(pos), n)) return pos;
    }
    return -1;
}

Pattern::Pattern(const std::string& p)
    : src(p), token_num(0), item_num(0) {
    
    size_t pos = 0;
    while (pos < src.length()) {
        Item item;

        if (src[pos] == '%') {
            char type = std::tolower(src[pos+1]);
            item.type = type;
            item.caps = (src[pos+1] != type);
            pos += 2;

            // Parse [offset, column] format
            item.absolute = (src[pos+1] == '@');
            size_t start = pos + (item.absolute ? 2 : 1);

            // Parse numbers
            size_t num_end;
            item.offset = std::stoi(src.substr(start), &num_end);
            start += num_end;
            
            start++;
            item.column = std::stoul(src.substr(start), &num_end);
            start += num_end;
            
            token_num = std::max(token_num, item.column+1);

            // Parse regexp for 't' and 'm' commands
            if (type == 't' || type == 'm') {
                start += 2;
                
                std::string value;
                while (start < src.length() && src[start] != '"') {
                    if (src[start] == '\\' && start + 1 < src.length()) {
                        value += src[start + 1];
                        start += 2;
                    } else {
                        value += src[start];
                        start++;
                    }
                }
                
                item.value = value;
                start++;
            }
            pos = start + 1;
        } else {
            item.type = 's';
            item.caps = false;
            size_t end = src.find('%', pos);
            if (end == std::string::npos) end = src.length();
            item.value = src.substr(pos, end - pos);
            pos = end;
        }

        items.push_back(item);
        item_num++;
    }
}

std::string Pattern::Execute(const TokenStrs& tokens, uint32_t at) {
    static const std::vector<std::string> bval = {"_x-1", "_x-2", "_x-3", "_x-4", "_x-#"};
    static const std::vector<std::string> eval = {"_x+1", "_x+2", "_x+3", "_x+4", "_x+#"};

    std::string result;
    result.reserve(16);

    size_t m = tokens.Size();
    for (const auto& item: items) {
        std::string value;
        uint32_t n = 0;

        if (item.type != 's') {
            int pos = item.offset;
            if (item.absolute) {
                pos = (pos < 0) ? pos + tokens.tokens.size() : pos - 1;
            } else {
                pos += at;
            }
            if (pos < 0) {
                value = bval[std::min(-pos - 1, 4)];
            } else if (pos >= static_cast<int32_t>(m)) {
                value = eval[std::min(pos - static_cast<int32_t>(m), 4)];
            } else {
                value = tokens.tokens[pos][item.column];
            }
        }

        if (item.type == 's') {
            value = item.value;
        } else if (item.type == 't') {
            value = (MatchRegex(item.value, value, n) == -1) ? "false" : "true";
        } else if (item.type == 'm') {
            int32_t pos = MatchRegex(item.value, value, n);
            if (pos == -1) {
                continue;  // No match found
            }
            value = value.substr(pos, n);
        }        

        if (item.caps) {
            std::string lowered;
            lowered.reserve(value.length());
            for (char c : value) {
                lowered += std::tolower(c);
            }
            result += lowered;
        } else {
            result += value;
        }
    }

    return result;
}

} // namespace wati