#include "../../header_parser/parser.hpp"
#include "../../structure_utility/utility.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#define NUM_FILTER 8
#define NUM_CHANNEL 1
#define CONV_WEIGHT_NUM_COL 3
#define CONV_WEIGHT_NUM_ROW 3

double Relu(double input);

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
    std::vector<MATRIX> convWeight (NUM_FILTER, MATRIX(CONV_WEIGHT_NUM_ROW, (std::vector<double>(CONV_WEIGHT_NUM_COL,0.0))));
    util.initTensor(convWeight, 9.0);
    std::vector<double> convBias(NUM_FILTER,0.0);

    // the output of the convolution layer, preactivation
    std::vector<MATRIX> convZ(NUM_FILTER, MATRIX(26, std::vector<double>(26, 0.0)));
    // the output of the convolution layer, activated
    std::vector<MATRIX> convA(NUM_FILTER, MATRIX(26, std::vector<double>(26, 0.0)));
    // forward pass for conv
    for (std::size_t f = 0; f < 8; ++f) {
        for (std::size_t row = 0; row < 26; ++row) {
            for (std::size_t col = 0; col < 26; ++col) {
                double sum = 0.0;
                for (std::size_t i = 0; i < 3; ++i) {
                    for (std::size_t j = 0; j < 3; ++j) {
                        sum += firstIm[row + i][col + j] * convWeight[f][i][j];
                    }
                }
                convZ[f][row][col] = sum + convBias[f];
            }
        }
    }

    for (std::size_t f = 0; f < NUM_FILTER; ++f) {
        for (std::size_t row = 0; row < 26; ++row) {
            for (std::size_t col = 0; col < 26; ++col) {
                convA[f][row][col] = Relu(convZ[f][row][col]);
            }
        }
    }
    return 0;
}


double Relu(double input) {
    return std::max(0.0, input);
}