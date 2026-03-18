#include "misc.h"

#include <stdint.h>

namespace wati {

std::string ReadStr(std::istream& file) {
    uint32_t n;
    char c;
    file >> n >> c;

    std::string str(n, '\0');
    file.read(&str[0], n);

    char comma;
    file.get(comma);
    
    file.get();

    return str;
}

void WriteStr(std::ostream& file, const std::string& str) {
    file << str.length() << ":" << str << ",\n";
}

std::vector<std::string> SplitLine(const std::string& line) {
    std::vector<std::string> rs;
    size_t start = 0, end = 0;
    while ((end=line.find(' ', start)) != std::string::npos) {
        if (end > start) {
            rs.push_back(line.substr(start, end - start));
        }
        start = end + 1;
    }
    if (start < line.length()) {
        rs.push_back(line.substr(start));
    }
    return rs;
}

std::string TrimLine(const std::string& line) {
    auto start = line.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    auto end = line.find_last_not_of(" \t\r\n");
    return line.substr(start, end - start + 1);
}

std::string GetLine(std::istream& file) {
    std::string line;
    if (!std::getline(file, line)) {
        return "";
    }
    return TrimLine(line);
}

} // namespace wati
