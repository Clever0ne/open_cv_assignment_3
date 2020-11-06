#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "fast_furier_transformer.h"

#include <stdint.h>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    auto image = imread("src/images/chess_medium.bmp", IMREAD_GRAYSCALE);
    auto fft = FastFurierTransformer();
    fft.setImage(image);
    fft.showImage();
    fft.directFastFurierTransform();
    fft.showSpectrum();
    fft.inverseFastFurierTransform();
    fft.showImage();
    while (waitKey() != 27)
    { }
    return 0;
}
