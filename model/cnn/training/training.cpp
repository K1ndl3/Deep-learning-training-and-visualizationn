#include "../../header_parser/parser.hpp"
#include "../../structure_utility/utility.hpp"
#include <cstdint>
#include <optional>
#include <vector>


int main() {
    Parser parser;
    Utility util;

    std::string labelPath = "../../../dataset/train-labels.idx1-ubyte";
    std::string imagePath = "../../../dataset/train-images.idx3-ubyte";
    std::optional<MnistData> dataStruct= parser.parse(labelPath, imagePath);
    if (dataStruct == std::nullopt) {
        std::cerr << "[ERROR]: cannot read files\n";
        return 1;
    }

    std::vector<MATRIX> imageData = dataStruct->images;
    std::vector<uint8_t> labelData = dataStruct->labels;
    
    // tensor for the first image
    MATRIX firstIm = imageData[0];
    std::vector<MATRIX> tensor3D;
    tensor3D.push_back(firstIm);

    // weights and biases for the convolution layer
    std::vector<MATRIX> convWeight (8, MATRIX(3, (std::vector<double>(3,0.0))));
    util.initTensor(convWeight, 9.0);
    std::vector<double> convBias(8,0.0);

    
    return 0;
}