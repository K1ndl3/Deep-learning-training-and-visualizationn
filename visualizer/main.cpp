#include "SubmitButton/SubmitButton.hpp"
#include "DrawingGrid/DrawingGrid.hpp"
#include "model/ModelNN.hpp"
#include "raylib.h"
#include <iostream>

#define GRID_X_OFFSET 10
#define GRID_Y_OFFSET 10


int main(void)
{
    DrawingGrid grid;
    SubmitButton submit_btn;
    ModelNN model;

    if (!model.loadModel("../model/mlp/trained_params.txt")) {
        std::cerr << "Failed to load model from ../model/trained_params.txt\n";
        return 1;
    }
    std::cout << "Model loaded successfully\n";

    InitWindow(900, 600, "MNIST Handwritting Predictor");
    SetTargetFPS(60);

    while (!WindowShouldClose())
    {
        BeginDrawing();
        ClearBackground(RAYWHITE);
        grid.clear();
        grid.input(GRID_X_OFFSET, GRID_Y_OFFSET);
        grid.DrawingGrid::draw(GRID_X_OFFSET, GRID_Y_OFFSET);
        submit_btn.draw(grid.getWidth() + GRID_X_OFFSET, GRID_Y_OFFSET, 10);

        if (submit_btn.submit()) {
            model.setInput(grid.getNormalizedInput());
            int prediction = model.inference();
            std::cout << "prediction: " << prediction << std::endl;
        }
        EndDrawing();
    }

    CloseWindow();
    return 0;
}