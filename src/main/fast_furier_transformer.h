#pragma once

#define _USE_MATH_DEFINES

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

#include <stdint.h>
#include <math.h>

enum ComplexPart
{
    RE = 0,
    IM = 1
};

enum PrimeFactor
{
    TWO = 0,
    THREE = 1,
    FIVE = 2,
    NUM_OF_PRIME_FACTORS = 3
};

class FastFurierTransformer
{
public:
    FastFurierTransformer() = default;
    ~FastFurierTransformer() = default;

    void setImage(const cv::Mat1f& image);
    void setImage(const cv::Mat& image);
    cv::Mat1f getImage() const;
    void showImage();

    void setSpectrum(const cv::Mat2f image);
    void setSpectrum(const cv::Mat image);
    cv::Mat2f getSpectrum() const;
    void showSpectrum();

    void directFastFurierTransform();
    void inverseFastFurierTransform();

private:
    cv::Mat1f spectrumMagnitude(cv::Mat2f spectrum);
    cv::Mat1f normalizeSpectrum(cv::Mat1f spectrumMagnitude, const float min, const float max);
    cv::Mat2f shiftSpectrum(cv::Mat2f spectrum, const int32_t shiftX, const int32_t shiftY);

    void directFFT(const cv::Mat1f& image, cv::Mat2f& spectrum);
    void inverseFFT(const cv::Mat2f& spectrum, cv::Mat1f& image);
    int32_t getFFTSize(const int32_t size);

    cv::Mat1f m_image;
    cv::Mat2f m_spectrum;
};
