#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

namespace wati {

std::string ReadStr(std::istream& file);
void WriteStr(std::ostream& file, const std::string& str);    

std::vector<std::string> SplitLine(const std::string& line);
std::string TrimLine(const std::string& line);
std::string GetLine(std::istream& file);

} // namespace wati