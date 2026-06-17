#pragma once
#include <cstdint>
#include <vector>
#include <fstream>
#include <string>
#include <iomanip>
#include <iostream>
#include <optional>

using MATRIX = std::vector<std::vector<double>>;

struct MnistData {
    std::vector<MATRIX> images;
    std::vector<uint8_t> labels;
};

class Parser {
private:

public:
    // this particular parseImage function does not flatten into 1D
    std::vector<MATRIX> parseImage(const std::string& filePath);
    std::vector<uint8_t> parseLabel(const std::string& filePath);
    std::optional<MnistData> parse(const std::string& label, const std::string& image);
};
