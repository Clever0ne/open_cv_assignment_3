#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "fast_furier_transformer.h"
#include "filter.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    /***** Задание 1. Прямое и обратное преобразования Фурье *****/

    Mat1f image_1 = imread("src/images/lena.bmp", IMREAD_GRAYSCALE);
    image_1 /= static_cast<float>(0xFF);
 
    // Реализованный класс FastFurierTransformer
    auto fft_1 = FastFurierTransformer();
    fft_1.setImage(image_1.clone());

    fft_1.directFastFurierTransform();
    fft_1.inverseFastFurierTransform();

    fft_1.showSpectrum();
    fft_1.showImage();

    // Встроенная функция
    auto cols = getOptimalDFTSize(image_1.cols);
    auto rows = getOptimalDFTSize(image_1.rows);

    auto openCVSpectrum_1 = Mat2f(Size2i(cols, rows), Vec2f(0, 0));
    auto openCVImage_1 = Mat1f(Size2i(cols, rows), 0);
    auto openCVImageROI_1 = image_1.clone();
    openCVImageROI_1.copyTo(openCVImage_1(Rect(0, 0, image_1.cols, image_1.rows)));

    dft(openCVImage_1, openCVSpectrum_1, DFT_COMPLEX_OUTPUT);
    idft(openCVSpectrum_1, openCVImage_1, DFT_REAL_OUTPUT);

    Mat1f openCVSpectrumComplex_1[2];
    split(openCVSpectrum_1, openCVSpectrumComplex_1);
    auto openCVSpectrumMagnitude_1 = Mat1f();

    magnitude(openCVSpectrumComplex_1[RE], openCVSpectrumComplex_1[IM], openCVSpectrumMagnitude_1);
    shiftSpectrum(openCVSpectrumMagnitude_1, cols / 2, rows / 2);

    openCVSpectrumMagnitude_1 += Scalar::all(1);
    log(openCVSpectrumMagnitude_1, openCVSpectrumMagnitude_1);
    normalize(openCVSpectrumMagnitude_1, openCVSpectrumMagnitude_1, 0, 1, NORM_MINMAX);

    imshow("Spectrum [OpenCV]", openCVSpectrumMagnitude_1);
    imshow("Image [OpenCV]", openCVImageROI_1);

    while (waitKey() != 27);

    /********************** Конец задания 1 **********************/

    /********* Задание 2. Фильтры верхних и нижних частот ********/

    Mat1f image_2 = imread("src/images/lena.bmp", IMREAD_GRAYSCALE);
    image_2 /= static_cast<float>(0xFF);

    // 1) Фильтр верхних частот
    auto fft_2_hpf = FastFurierTransformer();
    fft_2_hpf.setImage(image_2.clone());

    auto hpf = Filter();
    hpf.configure(FilterType::HIGH_PASS, FilterName::BUTTERWORTH, 50, 5);

    fft_2_hpf.setFilter(hpf);
    fft_2_hpf.filtrateImage();

    fft_2_hpf.showSpectrum("Spectrum with High Pass Butterworth Filter");
    fft_2_hpf.showImage("Image with High Pass Butterworth Filter");

    while (waitKey() != 27);

    // 2) Фильтр нижних частот
    auto fft_2_lpf = FastFurierTransformer();
    fft_2_lpf.setImage(image_2.clone());

    auto lpf = Filter();
    lpf.configure(FilterType::LOW_PASS, FilterName::BUTTERWORTH, 50, 5);

    fft_2_lpf.setFilter(lpf);
    fft_2_lpf.filtrateImage();

    fft_2_lpf.showSpectrum("Spectrum with Low Pass Butterworth Filter");
    fft_2_lpf.showImage("Image with Low Pass Butterworth Filter");

    while (waitKey() != 27);

    /********************** Конец задания 2 **********************/

    /** Задание 3. Фурье-образы свёртки изображения с фильтрами **/

    Mat1f image_3 = imread("src/images/lena.bmp", IMREAD_GRAYSCALE);
    image_3 /= static_cast<float>(0xFF);

    cols = getOptimalDFTSize(image_3.cols + 3 - 1);
    rows = getOptimalDFTSize(image_3.rows + 3 - 1);

    // 1) Фильтр Собеля
    float sobelH[3][3] =
    {
        { 1.0,  2.0,  1.0},
        { 0.0,  0.0,  0.0},
        {-1.0, -2.0, -1.0}
    };
    float sobelV[3][3] =
    {
        { 1.0,  0.0, -1.0},
        { 2.0,  0.0, -2.0},
        { 1.0,  0.0, -1.0}
    };

    auto sobelFilterH = Mat1f(3, 3, *sobelH);
    auto sobelFilterV = Mat1f(3, 3, *sobelV);

    auto fft_sobel = FastFurierTransformer();
    fft_sobel.setImage(image_3);
    fft_sobel.setSpectrumSize(Size2i(cols, rows));
    fft_sobel.directFastFurierTransform();

    auto fft_sobelH = FastFurierTransformer();
    fft_sobelH.setImage(sobelFilterH);
    fft_sobelH.setSpectrumSize(Size2i(cols, rows));
    fft_sobelH.directFastFurierTransform();

    auto fft_sobelV = FastFurierTransformer();
    fft_sobelV.setImage(sobelFilterV);
    fft_sobelV.setSpectrumSize(Size2i(cols, rows));
    fft_sobelV.directFastFurierTransform();

    fft_sobel.showSpectrum("Image Spectrum");
    fft_sobelH.showSpectrum("Horizontal Sobel Operator Spectrum");
    fft_sobelV.showSpectrum("Vertical Sobel Operator Spectrum");

    auto sobelSpectrum = fft_sobel.getSpectrum();
    auto sobelOperatorSpectrumH = fft_sobelH.getSpectrum();
    auto sobelOperatorSpectrumV = fft_sobelV.getSpectrum();
    multiplySpectrums(sobelSpectrum, sobelOperatorSpectrumH, sobelSpectrum);
    multiplySpectrums(sobelSpectrum, sobelOperatorSpectrumV, sobelSpectrum);

    fft_sobel.setSpectrum(sobelSpectrum);
    fft_sobel.showSpectrum("Sobel Filter Spectrum Result");

    while (waitKey() != 27);

    // 2) Усредняющий фильтр (Box Filter)
    float box[3][3] =
    {
        {1.0, 1.0, 1.0},
        {1.0, 1.0, 1.0},
        {1.0, 1.0, 1.0}
    };

    auto boxFilter = Mat1f(3, 3, *box);

    auto fft_box = FastFurierTransformer();
    fft_box.setImage(image_3);
    fft_box.setSpectrumSize(Size2i(cols, rows));
    fft_box.directFastFurierTransform();

    auto fft_boxFilter = FastFurierTransformer();
    fft_boxFilter.setImage(boxFilter);
    fft_boxFilter.setSpectrumSize(Size2i(cols, rows));
    fft_boxFilter.directFastFurierTransform();

    fft_box.showSpectrum("Image Spectrum");
    fft_boxFilter.showSpectrum("Box Filter Spectrum");

    auto boxSpectrum = fft_box.getSpectrum();
    auto boxFilterSpectrum = fft_boxFilter.getSpectrum();
    multiplySpectrums(boxSpectrum, boxFilterSpectrum, boxSpectrum);

    fft_box.setSpectrum(boxSpectrum);
    fft_box.showSpectrum("Box Filter Spectrum Result");

    while (waitKey() != 27);

    // 3) Фильтр Лапласа
    float laplace[3][3] =
    {
        {0.0,  1.0,  0.0},
        {1.0, -4.0,  1.0},
        {0.0,  1.0,  0.0}
    };

    auto laplaceFilter = Mat1f(3, 3, *laplace);

    auto fft_laplace = FastFurierTransformer();
    fft_laplace.setImage(image_3);
    fft_laplace.setSpectrumSize(Size2i(cols, rows));
    fft_laplace.directFastFurierTransform();

    auto fft_laplaceFilter = FastFurierTransformer();
    fft_laplaceFilter.setImage(laplaceFilter);
    fft_laplaceFilter.setSpectrumSize(Size2i(cols, rows));
    fft_laplaceFilter.directFastFurierTransform();

    fft_laplace.showSpectrum("Image Spectrum");
    fft_laplaceFilter.showSpectrum("Laplace Filter Spectrum");

    auto laplaceSpectrum = fft_laplace.getSpectrum();
    auto laplaceFilterSpectrum = fft_laplaceFilter.getSpectrum();
    multiplySpectrums(laplaceSpectrum, laplaceFilterSpectrum, laplaceSpectrum);

    fft_laplace.setSpectrum(laplaceSpectrum);
    fft_laplace.showSpectrum("Laplace Filter Spectrum Result");

    while (waitKey() != 27);

    /**** Задание 4. Результаты свёртки иображения с фильтрами ***/

    // 1) Фильтр Собеля
    fft_sobel.inverseFastFurierTransform();
    fft_sobel.showSpectrum("Sobel Filter Spectrum Result");
    fft_sobel.showImage("Sobel Filter Image Result");

    while (waitKey() != 27);

    // 2) Усредняющий фильтр (Box Filter)
    fft_box.inverseFastFurierTransform();
    fft_box.showSpectrum("Box Filter Spectrum Result");
    fft_box.showImage("Box Filter Image Result");

    while (waitKey() != 27);

    // 3) Фильтр Лапласа
    fft_laplace.inverseFastFurierTransform();
    fft_laplace.showSpectrum("Laplace Filter Spectrum Result");
    fft_laplace.showImage("Laplace Filter Image Result");

    while (waitKey() != 27);

    /********************** Конец задания 4 **********************/

    /************* Задание 4. Корелляция изображений *************/

    Mat1f image_5 = imread("src/images/car_number.bmp", IMREAD_GRAYSCALE);
    image_5 /= static_cast<float>(0xFF);

    // Корелляция с символом '6'
    Mat1f symbol_1 = imread("src/images/number_six.bmp", IMREAD_GRAYSCALE);
    symbol_1 /= static_cast<float>(0xFF);

    cols = getOptimalDFTSize(image_5.cols + symbol_1.cols - 1);
    rows = getOptimalDFTSize(image_5.rows + symbol_1.rows - 1);

    auto fft_carNumber_1 = FastFurierTransformer();
    fft_carNumber_1.setImage(image_5);
    fft_carNumber_1.setSpectrumSize(Size2i(cols, rows));
    fft_carNumber_1.directFastFurierTransform();

    auto fft_symbol_1 = FastFurierTransformer();
    fft_symbol_1.setImage(symbol_1);
    fft_symbol_1.setSpectrumSize(Size2i(cols, rows));
    fft_symbol_1.directFastFurierTransform();

    fft_carNumber_1.showImage("Car Number");
    fft_symbol_1.showImage("Symbol '6'");

    while (waitKey() != 27);

    auto carNumberSpectrum_1 = fft_carNumber_1.getSpectrum();
    auto symbolSpectrum_1 = fft_symbol_1.getSpectrum();
    multiplySpectrums(carNumberSpectrum_1, symbolSpectrum_1, carNumberSpectrum_1, true);

    fft_carNumber_1.setSpectrum(carNumberSpectrum_1);
    fft_carNumber_1.inverseFastFurierTransform();

    auto carNumberImage_1 = fft_carNumber_1.getImage();
    threshold(carNumberImage_1, carNumberImage_1, 0.95, 1.0, THRESH_BINARY);
    fft_carNumber_1.setImage(carNumberImage_1);

    fft_carNumber_1.showSpectrum("Correlation with Symbol '6' Spectrum");
    fft_carNumber_1.showImage("Correlation with Symbol '6' Image");

    while (waitKey() != 27);

    // Корелляция с символом '9'
    Mat1f symbol_2 = imread("src/images/number_nine.bmp", IMREAD_GRAYSCALE);
    symbol_2 /= static_cast<float>(0xFF);

    cols = getOptimalDFTSize(image_5.cols + symbol_2.cols - 1);
    rows = getOptimalDFTSize(image_5.rows + symbol_2.rows - 1);

    auto fft_carNumber_2 = FastFurierTransformer();
    fft_carNumber_2.setImage(image_5);
    fft_carNumber_2.setSpectrumSize(Size2i(cols, rows));
    fft_carNumber_2.directFastFurierTransform();

    auto fft_symbol_2 = FastFurierTransformer();
    fft_symbol_2.setImage(symbol_2);
    fft_symbol_2.setSpectrumSize(Size2i(cols, rows));
    fft_symbol_2.directFastFurierTransform();

    fft_carNumber_2.showImage("Car Number");
    fft_symbol_2.showImage("Symbol '9'");

    while (waitKey() != 27);

    auto carNumberSpectrum_2 = fft_carNumber_2.getSpectrum();
    auto symbolSpectrum_2 = fft_symbol_2.getSpectrum();
    multiplySpectrums(carNumberSpectrum_2, symbolSpectrum_2, carNumberSpectrum_2, true);

    fft_carNumber_2.setSpectrum(carNumberSpectrum_2);
    fft_carNumber_2.inverseFastFurierTransform();

    auto carNumberImage_2 = fft_carNumber_2.getImage();
    threshold(carNumberImage_2, carNumberImage_2, 0.95, 1.0, THRESH_BINARY);
    fft_carNumber_2.setImage(carNumberImage_2);

    fft_carNumber_2.showSpectrum("Correlation with Symbol '9' Spectrum");
    fft_carNumber_2.showImage("Correlation with Symbol '9' Image");

    while (waitKey() != 27);

    // Корелляция с символом 'O'
    Mat1f symbol_3 = imread("src/images/letter_o.bmp", IMREAD_GRAYSCALE);
    symbol_3 /= static_cast<float>(0xFF);

    cols = getOptimalDFTSize(image_5.cols + symbol_3.cols - 1);
    rows = getOptimalDFTSize(image_5.rows + symbol_3.rows - 1);

    auto fft_carNumber_3 = FastFurierTransformer();
    fft_carNumber_3.setImage(image_5);
    fft_carNumber_3.setSpectrumSize(Size2i(cols, rows));
    fft_carNumber_3.directFastFurierTransform();

    auto fft_symbol_3 = FastFurierTransformer();
    fft_symbol_3.setImage(symbol_3);
    fft_symbol_3.setSpectrumSize(Size2i(cols, rows));
    fft_symbol_3.directFastFurierTransform();

    fft_carNumber_3.showImage("Car Number");
    fft_symbol_3.showImage("Symbol 'O'");

    while (waitKey() != 27);

    auto carNumberSpectrum_3 = fft_carNumber_3.getSpectrum();
    auto symbolSpectrum_3 = fft_symbol_3.getSpectrum();
    multiplySpectrums(carNumberSpectrum_3, symbolSpectrum_3, carNumberSpectrum_3, true);

    fft_carNumber_3.setSpectrum(carNumberSpectrum_3);
    fft_carNumber_3.inverseFastFurierTransform();

    auto carNumberImage_3 = fft_carNumber_3.getImage();
    threshold(carNumberImage_3, carNumberImage_3, 0.95, 1.0, THRESH_BINARY);
    fft_carNumber_3.setImage(carNumberImage_3);

    fft_carNumber_3.showSpectrum("Correlation with Symbol 'O' Spectrum");
    fft_carNumber_3.showImage("Correlation with Symbol 'O' Image");

    while (waitKey() != 27);

    /********************** Конец задания 5 **********************/

    return 0;
}
