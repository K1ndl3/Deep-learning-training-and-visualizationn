#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define REVERSE_BYTES(x) __builtin_bswap32(x)

bool validateFileOpen(const std::ifstream& input) {
    return input.is_open();
}

bool loadMatrix(std::ifstream& in, std::vector<std::vector<double>>& matrix) {
    int rows = 0;
    int cols = 0;
    if (!(in >> rows >> cols)) {
        return false;
    }

    matrix.assign(rows, std::vector<double>(cols));
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (!(in >> matrix[r][c])) {
                return false;
            }
        }
    }
    return true;
}

bool loadVector(std::ifstream& in, std::vector<double>& vec) {
    int n = 0;
    if (!(in >> n)) {
        return false;
    }

    vec.resize(n);
    for (int i = 0; i < n; i++) {
        if (!(in >> vec[i])) {
            return false;
        }
    }
    return true;
}

double sigmoid(double z) {
    return 1 / (1 + std::exp(-z));
}

std::vector<double> forwardPass(std::vector<double> layer,
    std::vector<std::vector<double>> weight, std::vector<double> bias, bool final = false) {

        std::vector<double> h1(weight.size(),0.0);
        for(int row = 0; row < weight.size(); row++) {
            double w_sum = 0.0;
            for(int col = 0; col < weight[0].size(); col++) {
                w_sum += weight[row][col] * layer[col];
            }
            double z = w_sum + bias[row];
            if (!final) {
                h1[row] = sigmoid(z);
            } else {
                h1[row] = z;
            }
        }
        return h1;
    }

int argmax(const std::vector<double>& output) {
    int best = 0;
    for (size_t i = 1; i < output.size(); i++) {
        if (output[i] > output[best]) {
            best = static_cast<int>(i);
        }
    }
    return best;
}

int main() {
    const std::string test_label = "../dataset/t10k-labels.idx1-ubyte";
    const std::string test_images = "../dataset/t10k-images.idx3-ubyte";
    const std::string params_file = "./trained_params.txt";

    std::ifstream label_data(test_label, std::ios::binary);
    std::ifstream image_data(test_images, std::ios::binary);
    std::ifstream params_data(params_file);

    if (!validateFileOpen(image_data) || !validateFileOpen(label_data)) {
        std::cerr << "Cannot open test image/label files\n";
        return 1;
    }
    if (!validateFileOpen(params_data)) {
        std::cerr << "Cannot open " << params_file << '\n';
        return 1;
    }

    // parse label header
    uint32_t magic;
    uint32_t count;
    label_data.read(reinterpret_cast<char*>(&magic), 4);
    label_data.read(reinterpret_cast<char*>(&count), 4);
    magic = REVERSE_BYTES(magic);
    count = REVERSE_BYTES(count);

    std::vector<uint8_t> label_vector(count, 0);
    for (uint32_t i = 0; i < count; i++) {
        uint8_t label;
        label_data.read(reinterpret_cast<char*>(&label), 1);
        label_vector[i] = label;
    }

    // parse image header
    uint32_t num_images;
    uint32_t num_rows;
    uint32_t num_cols;

    image_data.read(reinterpret_cast<char*>(&magic), 4);
    image_data.read(reinterpret_cast<char*>(&num_images), 4);
    image_data.read(reinterpret_cast<char*>(&num_rows), 4);
    image_data.read(reinterpret_cast<char*>(&num_cols), 4);

    num_images = REVERSE_BYTES(num_images);
    num_rows = REVERSE_BYTES(num_rows);
    num_cols = REVERSE_BYTES(num_cols);

    const uint32_t pixels_per_image = num_rows * num_cols;

    std::vector<std::vector<uint8_t>> image_vector(
        num_images, std::vector<uint8_t>(pixels_per_image, 0));

    uint8_t pixel_value;
    for (uint32_t image = 0; image < num_images; image++) {
        for (uint32_t row = 0; row < num_rows; row++) {
            for (uint32_t col = 0; col < num_cols; col++) {
                image_data.read(reinterpret_cast<char*>(&pixel_value), 1);
                image_vector[image][row * num_cols + col] = pixel_value;
            }
        }
    }

    // x[image][pixel] — normalized inputs for the neural network
    std::vector<std::vector<double>> x(
        num_images, std::vector<double>(pixels_per_image, 0.0));

    for (uint32_t im = 0; im < num_images; im++) {
        for (uint32_t i = 0; i < pixels_per_image; i++) {
            x[im][i] = static_cast<double>(image_vector[im][i]) / 255.0;
        }
    }

    std::vector<std::vector<double>> w1, w2, w3;
    std::vector<double> b1, b2, b3;

    if (!loadMatrix(params_data, w1) || !loadVector(params_data, b1) ||
        !loadMatrix(params_data, w2) || !loadVector(params_data, b2) ||
        !loadMatrix(params_data, w3) || !loadVector(params_data, b3)) {
        std::cerr << "Failed to load trained parameters\n";
        return 1;
    }

    int correct = 0;
    for (uint32_t im = 0; im < num_images; im++) {
        std::vector<double> a1 = forwardPass(x[im], w1, b1);
        std::vector<double> a2 = forwardPass(a1, w2, b2);
        std::vector<double> a3 = forwardPass(a2, w3, b3, true);

        int prediction = argmax(a3);
        int actual = static_cast<int>(label_vector[im]);
        if (prediction == actual) {
            correct++;
        }
        std::cout << im << '\n';
    }

    const double accuracy = 100.0 * correct / static_cast<double>(num_images);
    std::cout << "Test images: " << num_images << '\n';
    std::cout << "Correct: " << correct << '\n';
    std::cout << "Accuracy: " << accuracy << "%\n";

    return 0;
}
