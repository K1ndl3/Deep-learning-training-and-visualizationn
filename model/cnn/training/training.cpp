#include "../../header_parser/parser.hpp"
#include <optional>

int main() {
    Parser parser;

    std::string labelPath = "../../dataset/t10k-labels.idx1-ubyte";
    std::string imagePath = "../../dataset/t10k-images.idx3-ubyte";
    std::optional<MnistData> dataStruct= parser.parse(labelPath, imagePath);
    if (dataStruct == std::nullopt) {
        std::cerr << "[ERROR]: cannot read files\n";
    }

     

    return 0;
}