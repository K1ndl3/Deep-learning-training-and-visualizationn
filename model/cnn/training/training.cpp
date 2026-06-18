#include "../../header_parser/parser.hpp"
#include <cstdint>
#include <optional>


int main() {
    Parser parser;

    std::string labelPath = "../../../dataset/train-labels.idx1-ubyte";
    std::string imagePath = "../../../dataset/train-images.idx3-ubyte";
    std::optional<MnistData> dataStruct= parser.parse(labelPath, imagePath);
    if (dataStruct == std::nullopt) {
        std::cerr << "[ERROR]: cannot read files\n";
        return 1;
    }

    std::vector<MATRIX> imageData = dataStruct->images;
    std::vector<uint8_t> labelData = dataStruct->labels;
    
    MATRIX firstIm = imageData[0];
    std::vector<MATRIX> tensorIm;
    tensorIm.push_back(firstIm);
    return 0;
}