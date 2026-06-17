#include "parser.hpp"
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>

#define REVERSE_BYTES(x) __builtin_bswap32(x)

std::vector<uint8_t> Parser::parseLabel(const std::string& filePath) {
    std::ifstream label_data(filePath, std::ios::binary);

    if (!label_data) {
        std::cerr << "[ERROR]: Cannot open label file\n";
        return {};
    }

    uint32_t magic;
    uint32_t count;
    label_data.read(reinterpret_cast<char*>(&magic), 4);
    label_data.read(reinterpret_cast<char*>(&count), 4);

    if (!label_data) {
        std::cerr << "[ERROR]: Failed to read label header\n";
        return {};
    }

    magic = REVERSE_BYTES(magic);
    count = REVERSE_BYTES(count);

    std::vector<uint8_t> label_vector(count);
    for (uint32_t i = 0; i < count; i++) {
        uint8_t label;
        label_data.read(reinterpret_cast<char*>(&label), 1);
        if (!label_data) {
            std::cerr << "[ERROR]: Failed to read label data\n";
            return {};
        }
        label_vector[i] = label;
    }

    return label_vector;
}

std::vector<MATRIX> Parser::parseImage(const std::string& filePath) {
    std::ifstream image_file(filePath, std::ios::binary);
    if (!image_file) {
        std::cerr << "[ERROR]: Failed to open image file\n";
        return {};
    }

    uint32_t magicNum;
    uint32_t numImages;
    uint32_t numRow;
    uint32_t numCol;

    image_file.read(reinterpret_cast<char*>(&magicNum), 4);
    magicNum = REVERSE_BYTES(magicNum);

    image_file.read(reinterpret_cast<char*>(&numImages), 4);
    numImages = REVERSE_BYTES(numImages);

    image_file.read(reinterpret_cast<char*>(&numRow), 4);
    numRow = REVERSE_BYTES(numRow);

    image_file.read(reinterpret_cast<char*>(&numCol), 4);
    numCol= REVERSE_BYTES(numCol);

    if (!image_file) {
        std::cerr << "[ERROR]: Failed to read image header\n";
        return {};
    }

    // parse the file
    std::vector<std::vector<std::vector<double>>> imagesData(numImages, 
                                    std::vector<std::vector<double>>(numRow,
                                    std::vector<double>(numCol, 0)));
    std::vector<std::vector<double>> rawPixelVector(numRow, std::vector<double>(numCol, 0));

    for (std::size_t i = 0; i < numImages; ++i) {
        for (std::size_t row = 0; row < numRow; ++row) {
            for (std::size_t col = 0; col < numCol; ++col) {
                uint8_t pixelValue;
                image_file.read(reinterpret_cast<char*>(&pixelValue), 1);
                if (!image_file) {
                    std::cerr << "[ERROR]: Failed to read image data\n";
                    return {};
                }
                rawPixelVector[row][col] = double(pixelValue) / 255.0;
            }
        }
        imagesData[i] = rawPixelVector;
    }

   return imagesData;
}

std::optional<MnistData> Parser::parse(const std::string& label, const std::string& image) {
    auto labels = parseLabel(label);
    if (labels.empty()) {
        return std::nullopt;
    }

    auto images = parseImage(image);
    if (images.empty()) {
        return std::nullopt;
    }

    if (labels.size() != images.size()) {
        std::cerr << "[ERROR]: Label count does not match image count\n";
        return std::nullopt;
    }

    return MnistData{std::move(images), std::move(labels)};
}


