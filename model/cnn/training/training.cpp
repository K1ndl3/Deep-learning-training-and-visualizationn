#include "../../header_parser/parser.hpp"
#include "../../structure_utility/utility.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <sys/types.h>
#include <vector>
#include <algorithm>
#include <cmath>

#define NUM_FILTER 8
#define NUM_CHANNEL 1
#define CONV_WEIGHT_NUM_COL 3
#define CONV_WEIGHT_NUM_ROW 3
#define CONV_LAYER_NUM_ROW_COL 26
#define MP_LAYER_STRIDE 2

#define MLP_INPUT_SIZE 1352
#define MLP_H1_SIZE 128
#define MLP_OUTPUT_SIZE 10
constexpr int MP_OUTPUT_SIZE = CONV_LAYER_NUM_ROW_COL /  MP_LAYER_STRIDE;
double Relu(double input);
void MLP(std::vector<double>& inputPool,
                        MATRIX& w1,
                        std::vector<double>& b1,
                        MATRIX& w2,
                        std::vector<double>& b2,
                        std::vector<double>& a1,
                        std::vector<double>& a2);

std::vector<double> softmax(const std::vector<double>& logits);
double crossEntropy(const std::vector<double>& probs, u_int8_t label);

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
    // label for the first image
    uint8_t label = labelData[0];
    // tensor3D.push_back(firstIm);

    // weights and biases for the convolution layer
    std::vector<MATRIX> convWeight (NUM_FILTER, MATRIX(CONV_WEIGHT_NUM_ROW, (std::vector<double>(CONV_WEIGHT_NUM_COL,0.0))));
    util.initTensor(convWeight, 9.0);
    std::vector<double> convBias(NUM_FILTER,0.0);

    // the output of the convolution layer, preactivation
    std::vector<MATRIX> convZ(NUM_FILTER, MATRIX(CONV_LAYER_NUM_ROW_COL, std::vector<double>(CONV_LAYER_NUM_ROW_COL, 0.0)));
    // the output of the convolution layer, activated
    std::vector<MATRIX> convA(NUM_FILTER, MATRIX(CONV_LAYER_NUM_ROW_COL, std::vector<double>(CONV_LAYER_NUM_ROW_COL, 0.0)));
    // forward pass for conv
    for (std::size_t f = 0; f < NUM_FILTER; ++f) {
        for (std::size_t row = 0; row < CONV_LAYER_NUM_ROW_COL; ++row) {
            for (std::size_t col = 0; col < CONV_LAYER_NUM_ROW_COL; ++col) {
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
        for (std::size_t row = 0; row < CONV_LAYER_NUM_ROW_COL; ++row) {
            for (std::size_t col = 0; col < CONV_LAYER_NUM_ROW_COL; ++col) {
                convA[f][row][col] = Relu(convZ[f][row][col]);
            }
        }
    }

    // MP layer
    std::vector<MATRIX> poolA(NUM_FILTER, MATRIX(MP_OUTPUT_SIZE, std::vector<double>(MP_OUTPUT_SIZE, 0.0)));
    for (std::size_t f = 0; f < NUM_FILTER; ++f) {
        for (std::size_t row = 0; row < CONV_LAYER_NUM_ROW_COL; row += MP_LAYER_STRIDE) {
            for (std::size_t col = 0; col < CONV_LAYER_NUM_ROW_COL; col += MP_LAYER_STRIDE) {
                double max = 0.0;
                for (std::size_t i = 0; i < MP_LAYER_STRIDE; ++i) {
                    for (std::size_t j = 0; j < MP_LAYER_STRIDE; ++j) {
                        if (max < convA[f][row + i][col + j]) {
                            max = convA[f][row + i][col + j];
                        }
                    }
                }
                poolA[f][row/MP_LAYER_STRIDE][col/MP_LAYER_STRIDE] = max;
            }
        }
    }
    // maybe we could have created a 1d and build it from flat but who knows
    // flatten the poolA 
    std::vector<double> poolAFlat(NUM_FILTER * MP_OUTPUT_SIZE * MP_OUTPUT_SIZE, 0.0);
    for (std::size_t f = 0; f < NUM_FILTER; ++f) {
        for (std::size_t row = 0; row < MP_OUTPUT_SIZE; ++row) {
            for (std::size_t col = 0; col < MP_OUTPUT_SIZE; ++col) {
                poolAFlat[f * MP_OUTPUT_SIZE * MP_OUTPUT_SIZE + row * MP_OUTPUT_SIZE + col] = poolA[f][row][col];
            }
        }
    }

    // weights and baises for the layers
    MATRIX w1(MLP_H1_SIZE, std::vector<double>(MLP_INPUT_SIZE, 0.0));
    std::vector<double> b1(MLP_H1_SIZE,0.0);

    MATRIX w2(MLP_OUTPUT_SIZE, std::vector<double>(MLP_H1_SIZE, 0.0));
    std::vector<double> b2(MLP_OUTPUT_SIZE,0.0);

    util.initMatrix(w1, MLP_INPUT_SIZE);
    util.initMatrix(w2, MLP_H1_SIZE);

    std::vector<double> hiddenMLP(MLP_H1_SIZE, 0.0);
    std::vector<double> outputMLP(MLP_OUTPUT_SIZE, 0.0);
    return 0;
}


double Relu(double input) {
    return std::max(0.0, input);
}

void MLP(std::vector<double>& inputPool,
    MATRIX& w1,
    std::vector<double>& b1,
    MATRIX& w2,
    std::vector<double>& b2,
    std::vector<double>& a1,
    std::vector<double>& a2) {
        // pass through the first layer
        for (std::size_t row = 0; row < MLP_H1_SIZE; ++row) {
            double sum = b1[row];
            for (std::size_t col = 0; col < MLP_INPUT_SIZE; ++col) {
                sum += inputPool[col] * w1[row][col];
            }
            a1[row] = Relu(sum);
        }

        // pass through the hidden layer
        for (std::size_t row = 0; row < MLP_OUTPUT_SIZE; ++row) {
            double sum = b2[row];
            for (std::size_t col = 0; col < MLP_H1_SIZE; ++col) {
                sum += a1[col] * w2[row][col];
            }
            a2[row] = sum;
        }
        
    }

    std::vector<double> softmax(const std::vector<double>& logits) {
        double sum = 0.0;
        for (auto logit : logits) {
            sum += std::exp(logit);
        }
        std::size_t size = logits.size();
        std::vector<double> probDist(size, 0.0);
        for (std::size_t i = 0; i < size; ++i) {
            probDist[i] = std::exp(logits[i]) / sum;
        }
        return probDist;
    }