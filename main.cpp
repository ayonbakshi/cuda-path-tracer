#include <iostream>

#include "include/kernels.cuh"
#include "include/writebmp.h"

int main() {
    int width = 1080, height = 720;
    int tx = 8, ty = 8;
    float *pixels = new float[width * height * 3];

    std::cerr << "Rendering a " << width << "x" << height << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    clock_t start, stop;
    start = clock();
    
    run(width, height, tx, ty, pixels);

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    
    drawbmp("out.bmp", width, height, pixels);

    delete [] pixels;
}