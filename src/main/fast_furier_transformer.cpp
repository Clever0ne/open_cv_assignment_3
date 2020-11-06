#include "fast_furier_transformer.h"
#include <iostream>

using namespace std;
using namespace cv;

void FastFurierTransformer::setImage(const Mat1f& image)
{
    m_image = image.clone();
}

void FastFurierTransformer::setImage(const Mat& image)
{
    m_image = image.clone();
    m_image /= static_cast<float>(0xFF);
}

Mat1f FastFurierTransformer::getImage() const
{
    return m_image.clone();
}

void FastFurierTransformer::showImage()
{
    imshow("Image [FFT]", m_image);
}

void FastFurierTransformer::setSpectrum(const Mat2f spectrum)
{
    m_spectrum = spectrum.clone();
}

void FastFurierTransformer::setSpectrum(const Mat spectrum)
{
    auto newSpectrum = Mat2f(spectrum);
    m_spectrum = newSpectrum.clone();
}

Mat2f FastFurierTransformer::getSpectrum() const
{
    return m_spectrum.clone();
}

void FastFurierTransformer::showSpectrum()
{
    imshow("Spectrum [FFT]", spectrumMagnitude(m_spectrum));
}

Mat1f FastFurierTransformer::spectrumMagnitude(cv::Mat2f spectrum)
{
    spectrum = shiftSpectrum(spectrum, spectrum.cols / 2, spectrum.rows / 2);
    auto spectrumMagnitude = Mat1f(spectrum.size());
    auto max = FLT_MIN;
    auto min = FLT_MAX;
    for (auto row = 0; row < spectrum.rows; row++)
    {
        auto magnPtr = spectrumMagnitude.ptr<float>(row);
        auto spectrPtr = spectrum.ptr<Vec2f>(row);
        for (auto col = 0; col < spectrum.cols; col++)
        {
            auto magn = hypotf(spectrPtr[col][0], spectrPtr[col][1]);
            auto logMagn = logf(magn + 1);
            if (logMagn > max)
            {
                max = logMagn;
            }
            if (logMagn < min)
            {
                min = logMagn;
            }
            magnPtr[col] = logMagn;
        }
    }  
    spectrum = shiftSpectrum(spectrum, spectrum.cols / 2, spectrum.rows / 2);
    return normalizeSpectrum(spectrumMagnitude, min, max);
}

Mat2f FastFurierTransformer::shiftSpectrum(Mat2f spectrum, const int32_t shiftX, const int32_t shiftY)
{
    auto width  = (spectrum.cols - shiftX) % spectrum.cols;
    auto height = (spectrum.rows - shiftY) % spectrum.rows;

    auto buffer = Mat2f(spectrum.size());
    auto left   = Mat2f(spectrum, Rect(0    , 0, width                , spectrum.rows));
    auto right  = Mat2f(spectrum, Rect(width, 0, spectrum.cols - width, spectrum.rows));
    auto top    = Mat2f(buffer, Rect(0, 0     , buffer.cols, height              ));
    auto bottom = Mat2f(buffer, Rect(0, height, buffer.cols, buffer.rows - height));
    
    left.copyTo( buffer(Rect(buffer.cols - width, 0, width              , buffer.rows)));
    right.copyTo(buffer(Rect(0                  , 0, buffer.cols - width, buffer.rows)));
    
    top.copyTo(   spectrum(Rect(0, spectrum.rows - height, spectrum.cols, height                )));
    bottom.copyTo(spectrum(Rect(0, 0                     , spectrum.cols, spectrum.rows - height)));

    return spectrum;
}

Mat1f FastFurierTransformer::normalizeSpectrum(Mat1f spectrumMagnitude, const float min, const float max)
{
    for (auto row = 0; row < spectrumMagnitude.rows; row++)
    {
        auto magnPtr = spectrumMagnitude.ptr<float>(row);
        for (auto col = 0; col < spectrumMagnitude.cols; col++)
        {
            magnPtr[col] -= min;
            magnPtr[col] /= (max - min);
        }
    }
    return spectrumMagnitude;
}

void FastFurierTransformer::directFastFurierTransform()
{
    directFFT(m_image, m_spectrum);
    /*dft(m_image, m_spectrum, DFT_COMPLEX_OUTPUT);*/
}

void FastFurierTransformer::inverseFastFurierTransform()
{
    inverseFFT(m_spectrum, m_image);
    /*idft(m_spectrum, m_image, DFT_REAL_OUTPUT);*/
}

void FastFurierTransformer::directFFT(const Mat1f& image, Mat2f& spectrum)
{
    if (image.empty() == true)
    {
        return;
    }

    const auto cols = getFFTSize(image.cols);
    const auto rows = getFFTSize(image.rows);

    auto buffer = Mat2f(Size(cols, rows), Vec2f(0, 0));
    for (auto row = 0; row < image.rows; row++)
    {
        auto imagePtr = image.ptr<float>(row);
        auto spectrumPtr = buffer.ptr<Vec2f>(row);
        for (auto col = 0; col < image.cols; col++)
        {
            spectrumPtr[col][RE] = imagePtr[col];
        }
    }

    auto fft = [&buffer](int32_t rows, int32_t cols)
    {
        for (auto row = 0; row < rows; row++)
        {
            auto spcPtr = buffer.ptr<Vec2f>(row);

            for (auto i = 1, j = 0; i < cols; i++)
            {
                int bit = cols >> 1;
                for (; j >= bit; bit >>= 1)
                {
                    j -= bit;
                }
                j += bit;
                if (i < j)
                {
                    swap(spcPtr[i], spcPtr[j]);
                }
            }

            for (auto step = 2; step <= cols; step <<= 1)
            {
                float ang = -2 * M_PI / step;

                for (auto i = 0; i < cols; i += step)
                {
                    for (auto j = 0; j < step / 2; j++)
                    {
                        auto spcRe0 = spcPtr[i + j][RE];
                        auto spcIm0 = spcPtr[i + j][IM];
                        auto spcRe1 = spcPtr[i + j + step / 2][RE];
                        auto spcIm1 = spcPtr[i + j + step / 2][IM];
                        spcPtr[i + j][RE] = spcRe0 + spcRe1 * cosf(ang * j) - spcIm1 * sinf(ang * j);
                        spcPtr[i + j][IM] = spcIm0 + spcRe1 * sinf(ang * j) + spcIm1 * cosf(ang * j);
                        spcPtr[i + j + step / 2][RE] = spcRe0 - spcRe1 * cosf(ang * j) + spcIm1 * sinf(ang * j);
                        spcPtr[i + j + step / 2][IM] = spcIm0 - spcRe1 * sinf(ang * j) - spcIm1 * cosf(ang * j);
                    }
                }
            }
        }

        transpose(buffer, buffer);
        return buffer;
    };
    
    buffer = fft(cols, rows);
    buffer = fft(rows, cols);

    spectrum = buffer;
}

void FastFurierTransformer::inverseFFT(const Mat2f& spectrum, Mat1f& image)
{
    if (spectrum.empty() == true)
    {
        return;
    }

    const auto cols = getFFTSize(image.cols);
    const auto rows = getFFTSize(image.rows);

    auto buffer = spectrum.clone();

    auto ifft = [&buffer](int32_t rows, int32_t cols)
    {
        for (auto row = 0; row < rows; row++)
        {
            auto imPtr = buffer.ptr<Vec2f>(row);

            for (auto i = 1, j = 0; i < cols; i++)
            {
                int bit = cols >> 1;
                for (; j >= bit; bit >>= 1)
                {
                    j -= bit;
                }
                j += bit;
                if (i < j)
                {
                    swap(imPtr[i], imPtr[j]);
                }
            }

            for (auto step = 2; step <= cols; step <<= 1)
            {
                float ang = 2 * M_PI / step;

                for (auto i = 0; i < cols; i += step)
                {
                    for (auto j = 0; j < step / 2; j++)
                    {
                        auto imRe0 = imPtr[i + j][RE];
                        auto imIm0 = imPtr[i + j][IM];
                        auto imRe1 = imPtr[i + j + step / 2][RE];
                        auto imIm1 = imPtr[i + j + step / 2][IM];
                        imPtr[i + j][RE] = imRe0 + imRe1 * cosf(ang * j) - imIm1 * sinf(ang * j);
                        imPtr[i + j][IM] = imIm0 + imRe1 * sinf(ang * j) + imIm1 * cosf(ang * j);
                        imPtr[i + j + step / 2][RE] = imRe0 - imRe1 * cosf(ang * j) + imIm1 * sinf(ang * j);
                        imPtr[i + j + step / 2][IM] = imIm0 - imRe1 * sinf(ang * j) - imIm1 * cosf(ang * j);
                    }
                }
            }
        }

        transpose(buffer, buffer);
        return buffer;
    };

    buffer = ifft(cols, rows);
    buffer = ifft(rows, cols);

    for (auto row = 0; row < image.rows; row++)
    {
        auto imagePtr = image.ptr<float>(row);
        auto spectrumPtr = buffer.ptr<Vec2f>(row);
        for (auto col = 0; col < image.cols; col++)
        {
            imagePtr[col] = spectrumPtr[col][RE] / (cols * rows);
        }
    }
}

int32_t FastFurierTransformer::getFFTSize(const int32_t size)
{
    if ((size != 0) && ((size & (size - 1)) == 0))
    {
        return size;
    }
    
    auto newSize = 1;
    while (newSize < size)
    {
        newSize <<= 1;
    }
    return newSize;
}
