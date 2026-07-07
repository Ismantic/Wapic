#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>

namespace wati {

std::string ReadStr(std::istream& file);
void WriteStr(std::ostream& file, const std::string& str);

// Binary (compact) string read/write: uint16 length + raw bytes (no newline/colon).
std::string ReadStrBin(std::istream& file);
void WriteStrBin(std::ostream& file, const std::string& str);

// Variable-length integer (unsigned), Protobuf-style.
void WriteVarUInt(std::ostream& file, uint64_t v);
uint64_t ReadVarUInt(std::istream& file);

std::vector<std::string> SplitLine(const std::string& line);
std::string TrimLine(const std::string& line);

} // namespace wati