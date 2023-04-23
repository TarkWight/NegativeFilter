#include <algorithm>
#include <intrin.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <ctime>
#include <cstring>
#include <omp.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include "lodepng.cpp"
#include "lodepng.h"


using namespace std;


//@@@@@@@@@@@@@@@@@@@@      Consistent Filter     @@@@@@@@@@@@@@@@@@@@
void negateImage(const char* inputFilename, const char* outputFilename) {
    unsigned error;
    unsigned char* image;
    unsigned width, height;

    error = lodepng_decode24_file(&image, &width, &height, inputFilename);
    if (error) {
        std::cerr << "Failed to load image: " << lodepng_error_text(error) << endl;
        return;
    }

    for (unsigned y = 0; y < height; y++) {
        for (unsigned x = 0; x < width; x++) {
            unsigned index = (y * width + x) * 3;
            image[index + 0] = 255 - image[index + 0];
            image[index + 1] = 255 - image[index + 1];
            image[index + 2] = 255 - image[index + 2];
        }
    }

    error = lodepng_encode24_file(outputFilename, image, width, height);
    if (error) {
        std::cerr << "Failed to save image: " << lodepng_error_text(error) << endl;
        return;
    }

    free(image);
}


//@@@@@@@@@@@@@@@@@@@@      OpenMP Filter     @@@@@@@@@@@@@@@@@@@@
void negateImageOMP(const char* inputFilename, const char* outputFilename) {
    unsigned error;
    unsigned char* image;
    unsigned width, height;

    error = lodepng_decode24_file(&image, &width, &height, inputFilename);
    if (error) {
        std::cerr << "Failed to load image: " << lodepng_error_text(error) << endl;
        return;
    }

#pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            unsigned index = (y * width + x) * 3;
            image[index + 0] = 255 - image[index + 0];
            image[index + 1] = 255 - image[index + 1];
            image[index + 2] = 255 - image[index + 2];
        }
    }

    error = lodepng_encode24_file(outputFilename, image, width, height);
    if (error) {
        std::cerr << "Failed to save image: " << lodepng_error_text(error) << endl;
        return;
    }

    free(image);
}


//@@@@@@@@@@@@@@@@@@@@      Vectorisation Filter     @@@@@@@@@@@@@@@@@@@@
void vectorisationNegateFilter(const char* inputFilename, const char* outputFilename) {
    unsigned error;
    unsigned char* image;
    unsigned width, height;

    error = lodepng_decode24_file(&image, &width, &height, inputFilename);
    if (error) {
        std::cerr << "Failed to load image: " << lodepng_error_text(error) << std::endl;
        return;
    }

    const __m128i ones = _mm_set1_epi8(255);
    for (unsigned y = 0; y < height; ++y) {
        unsigned index = y * width * 3;
        unsigned length = width * 3;
        unsigned char* row = image + index;
        unsigned char* end = row + length;

        for (; row + 15 < end; row += 16) {
            __m128i data = _mm_load_si128((__m128i*)row);
            data = _mm_sub_epi8(ones, data);
            _mm_store_si128((__m128i*)row, data);
        }

        for (; row < end; ++row) {
            *row = 255 - *row;
        }
    }

    error = lodepng_encode24_file(outputFilename, image, width, height);
    if (error) {
        std::cerr << "Failed to save image: " << lodepng_error_text(error) << std::endl;
        return;
    }

    free(image);
}


//@@@@@@@@@@@@@@@@@@@@      OpenMP Vectorisation Filter     @@@@@@@@@@@@@@@@@@@@
void OpenMPVectorisationNnegateFilter(const char* inputFilename, const char* outputFilename) {
    unsigned error;
    unsigned char* image;
    unsigned width, height;

    error = lodepng_decode24_file(&image, &width, &height, inputFilename);
    if (error) {
        std::cerr << "Failed to load image: " << lodepng_error_text(error) << std::endl;
        return;
    }

    const __m128i ones = _mm_set1_epi8(255);

#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
    for (int y = 0; y < height; ++y) {
        unsigned index = y * width * 3;
        unsigned length = width * 3;
        unsigned char* row = image + index;
        unsigned char* end = row + length;

        for (; row + 15 < end; row += 16) {
            __m128i data = _mm_load_si128((__m128i*)row);
            data = _mm_sub_epi8(ones, data);
            _mm_store_si128((__m128i*)row, data);
        }

        for (; row < end; ++row) {
            *row = 255 - *row;
        }
    }

    error = lodepng_encode24_file(outputFilename, image, width, height);
    if (error) {
        std::cerr << "Failed to save image: " << lodepng_error_text(error) << std::endl;
        return;
    }

    free(image);
}


//# # # # # # # # # # #      Consistent Call     # # # # # # # # # # #
void userConsistentNegativeFilter() {
    clock_t start = clock();
    const char* inputFilename = "input.png";
    const char* outputFilename = "outputNegative.png";
    negateImage(inputFilename, outputFilename);
    clock_t end = clock();
    double elapsed_time = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Время выполнения функции негативного фильтра: " << elapsed_time << " секунд." << std::endl;
}


//# # # # # # # # # # #      OpenMP Call     # # # # # # # # # # #
void userOMPNegativeFilter() {
    clock_t start = clock();
    const char* inputFilename = "input.png";
    const char* outputFilename = "outputNegativeOMP.png";
    negateImageOMP(inputFilename, outputFilename);
    clock_t end = clock();
    double elapsed_time = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Время выполнения функции негативного фильтра: " << elapsed_time << " секунд." << std::endl;
}


//# # # # # # # # # # #      Vectorization Call     # # # # # # # # # # #
void userVectorizationNegativeFilter() {
    clock_t start = clock();
    const char* inputFilename = "input.png";
    const char* outputFilename = "outputNegativeVect.png";
    vectorisationNegateFilter(inputFilename, outputFilename);
    clock_t end = clock();
    double elapsed_time = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Время выполнения функции негативного фильтра: " << elapsed_time << " секунд." << std::endl;
}


//# # # # # # # # # # #      OpenMP Vectorization Call     # # # # # # # # # # #
void userOMPVectorizationNegativeFilter() {
    clock_t start = clock();
    const char* inputFilename = "input.png";
    const char* outputFilename = "outputNegativeOpenMPVect.png";
    OpenMPVectorisationNnegateFilter(inputFilename, outputFilename);
    clock_t end = clock();
    double elapsed_time = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Время выполнения функции негативного фильтра: " << elapsed_time << " секунд." << std::endl;
}


int main() {
    setlocale(LC_ALL, "Russian");
    std::cout << "Последовательный метод:\n";
    userConsistentNegativeFilter();
    std::cout << "\n\n";

    std::cout << "OpenMP:\n";
    userOMPNegativeFilter();
    std::cout << "\n\n";

    std::cout << "Vectorization:\n";
    userVectorizationNegativeFilter();
    std::cout << "\n\n";

    std::cout << "OpenMP Vectorization:\n";
    userOMPVectorizationFilters();
    std::cout << "\n";
    return 0;
}
