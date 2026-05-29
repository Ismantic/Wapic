#include "misc.h"

#include <stdint.h>

namespace wati {

std::string ReadStrBin(std::istream& file) {
    uint16_t len;
    file.read(reinterpret_cast<char*>(&len), sizeof(len));
    std::string s(len, '\0');
    file.read(&s[0], len);
    return s;
}

void WriteStrBin(std::ostream& file, const std::string& str) {
    uint16_t len = static_cast<uint16_t>(str.size());
    file.write(reinterpret_cast<const char*>(&len), sizeof(len));
    file.write(str.data(), len);
}

void WriteVarUInt(std::ostream& file, uint64_t v) {
    while (v >= 0x80) {
        unsigned char b = (v & 0x7f) | 0x80;
        file.write(reinterpret_cast<const char*>(&b), 1);
        v >>= 7;
    }
    unsigned char b = static_cast<unsigned char>(v);
    file.write(reinterpret_cast<const char*>(&b), 1);
}

uint64_t ReadVarUInt(std::istream& file) {
    uint64_t v = 0;
    int shift = 0;
    while (true) {
        unsigned char b;
        file.read(reinterpret_cast<char*>(&b), 1);
        v |= static_cast<uint64_t>(b & 0x7f) << shift;
        if ((b & 0x80) == 0) break;
        shift += 7;
    }
    return v;
}

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
