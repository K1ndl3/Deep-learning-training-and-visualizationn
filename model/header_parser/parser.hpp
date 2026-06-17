#pragma once
#include <cstdint>
#include <vector>
#include <fstream>
#include <string>
#include <iomanip>
#include <iostream>

struct MnistData {
    std::vector<std::vector<double>> images;
    std::vector<uint8_t> labels;
};

class Parser {
private:

public:
    std::vector<std::vector<double>> parseImage(const std::string& filePath);
    std::vector<uint8_t> parseLabel(const std::string& filePath);
    MnistData parse();
};
