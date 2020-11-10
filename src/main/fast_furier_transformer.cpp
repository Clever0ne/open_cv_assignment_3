#include "fast_furier_transformer.h"
#include <iostream>

#define COMPLEX_MUL(complex1, complex2) \
        (COMPLEX(complex1, RE) * COMPLEX(complex2, RE) - COMPLEX(complex1, IM) * COMPLEX(complex2, IM)) : \
        (COMPLEX(complex1, RE) * COMPLEX(complex2, IM) + COMPLEX(complex1, IM) * COMPLEX(complex2, RE))

#define COMPLEX_EXP(angle) cosf(angle) : sinf(angle)

#define COMPLEX_VEC(ptr, index) ptr[index][RE] : ptr[index][IM]

#define COMPLEX(complex, complexPart) (complexPart == RE ? complex)

static const float k2_Re[2][2] =
{
    {1.0,  1.0},
    {1.0, -1.0}
};
static const float k2_Im[2][2] =
{
    {0.0, 0.0},
    {0.0, 0.0}
};
static const float k3_Re[3][3] =
{
    {1.0,  1.0,  1.0},
    {1.0, -0.5, -0.5},
    {1.0, -0.5, -0.5}
};
static const float k3_Im[3][3] =
{
    {0.0,  0.0         ,  0.0         },
    {0.0,  sqrtf(3) / 2, -sqrtf(3) / 2},
    {0.0, -sqrtf(3) / 2,  sqrtf(3) / 2}
};
static const float k5_Re[5][5] =
{
    {1.0, 1.0               , 1.0               , 1.0               , 1.0               },
    {1.0, cosf(2 * M_PI / 5), cosf(4 * M_PI / 5), cosf(6 * M_PI / 5), cosf(8 * M_PI / 5)},
    {1.0, cosf(4 * M_PI / 5), cosf(8 * M_PI / 5), cosf(2 * M_PI / 5), cosf(6 * M_PI / 5)},
    {1.0, cosf(6 * M_PI / 5), cosf(2 * M_PI / 5), cosf(8 * M_PI / 5), cosf(4 * M_PI / 5)},
    {1.0, cosf(8 * M_PI / 5), cosf(6 * M_PI / 5), cosf(4 * M_PI / 5), cosf(2 * M_PI / 5)}
};
static const float k5_Im[5][5] =
{
    {0.0, 0.0               , 0.0               , 0.0               , 0.0               },
    {0.0, sinf(2 * M_PI / 5), sinf(4 * M_PI / 5), sinf(6 * M_PI / 5), sinf(8 * M_PI / 5)},
    {0.0, sinf(4 * M_PI / 5), sinf(8 * M_PI / 5), sinf(2 * M_PI / 5), sinf(6 * M_PI / 5)},
    {0.0, sinf(6 * M_PI / 5), sinf(2 * M_PI / 5), sinf(8 * M_PI / 5), sinf(4 * M_PI / 5)},
    {0.0, sinf(8 * M_PI / 5), sinf(6 * M_PI / 5), sinf(4 * M_PI / 5), sinf(2 * M_PI / 5)}
};

#define K2(i, j, ang) k2_Re[i][j] : (ang > 0 ? 1 : -1) * k2_Im[i][j]
#define K3(i, j, ang) k3_Re[i][j] : (ang > 0 ? 1 : -1) * k3_Im[i][j]
#define K5(i, j, ang) k5_Re[i][j] : (ang > 0 ? 1 : -1) * k5_Im[i][j]

#define DFT2_RE(ptr, ptr0, ptr1, ang, i, j, step) ptr[i + j * step][RE] = \
        COMPLEX(COMPLEX_MUL(K2(0, j, ang), COMPLEX_MUL(COMPLEX_VEC(ptr0, i), COMPLEX_EXP(ang * i * 0))), RE) + \
        COMPLEX(COMPLEX_MUL(K2(1, j, ang), COMPLEX_MUL(COMPLEX_VEC(ptr1, i), COMPLEX_EXP(ang * i * 1))), RE)

#define DFT2_IM(ptr, ptr0, ptr1, ang, i, j, step) ptr[i + j * step][IM] = \
        COMPLEX(COMPLEX_MUL(K2(0, j, ang), COMPLEX_MUL(COMPLEX_VEC(ptr0, i), COMPLEX_EXP(ang * i * 0))), IM) + \
        COMPLEX(COMPLEX_MUL(K2(1, j, ang), COMPLEX_MUL(COMPLEX_VEC(ptr1, i), COMPLEX_EXP(ang * i * 1))), IM)

#define DFT2(ptr, ptr0, ptr1, ang, i, j, step) \
        DFT2_RE(ptr, ptr0, ptr1, ang, i, j, step); \
        DFT2_IM(ptr, ptr0, ptr1, ang, i, j, step)

#define DFT3_RE(ptr, ptr0, ptr1, ptr2, ang, i, j, step) ptr[i + j * step][RE] = \
        COMPLEX(COMPLEX_MUL(K3(0, j, ang), COMPLEX_MUL(COMPLEX_VEC(ptr0, i), COMPLEX_EXP(ang * i * 0))), RE) + \
        COMPLEX(COMPLEX_MUL(K3(1, j, ang), COMPLEX_MUL(COMPLEX_VEC(ptr1, i), COMPLEX_EXP(ang * i * 1))), RE) + \
        COMPLEX(COMPLEX_MUL(K3(2, j, ang), COMPLEX_MUL(COMPLEX_VEC(ptr2, i), COMPLEX_EXP(ang * i * 2))), RE)

#define DFT3_IM(ptr, ptr0, ptr1, ptr2, ang, i, j, step) ptr[i + j * step][IM] = \
        COMPLEX(COMPLEX_MUL(K3(0, j, ang), COMPLEX_MUL(COMPLEX_VEC(ptr0, i), COMPLEX_EXP(ang * i * 0))), IM) + \
        COMPLEX(COMPLEX_MUL(K3(1, j, ang), COMPLEX_MUL(COMPLEX_VEC(ptr1, i), COMPLEX_EXP(ang * i * 1))), IM) + \
        COMPLEX(COMPLEX_MUL(K3(2, j, ang), COMPLEX_MUL(COMPLEX_VEC(ptr2, i), COMPLEX_EXP(ang * i * 2))), IM)

#define DFT3(ptr, ptr0, ptr1, ptr2, ang, i, j, step) \
        DFT3_RE(ptr, ptr0, ptr1, ptr2, ang, i, j, step); \
        DFT3_IM(ptr, ptr0, ptr1, ptr2, ang, i, j, step)

#define DFT5_RE(ptr, ptr0, ptr1, ptr2, ptr3, ptr4, ang, i, j, step) ptr[i + j * step][RE] = \
        COMPLEX(COMPLEX_MUL(K5(0, j, ang), COMPLEX_MUL(COMPLEX_VEC(ptr0, i), COMPLEX_EXP(ang * i * 0))), RE) + \
        COMPLEX(COMPLEX_MUL(K5(1, j, ang), COMPLEX_MUL(COMPLEX_VEC(ptr1, i), COMPLEX_EXP(ang * i * 1))), RE) + \
        COMPLEX(COMPLEX_MUL(K5(2, j, ang), COMPLEX_MUL(COMPLEX_VEC(ptr2, i), COMPLEX_EXP(ang * i * 2))), RE) + \
        COMPLEX(COMPLEX_MUL(K5(3, j, ang), COMPLEX_MUL(COMPLEX_VEC(ptr3, i), COMPLEX_EXP(ang * i * 3))), RE) + \
        COMPLEX(COMPLEX_MUL(K5(4, j, ang), COMPLEX_MUL(COMPLEX_VEC(ptr4, i), COMPLEX_EXP(ang * i * 4))), RE)

#define DFT5_IM(ptr, ptr0, ptr1, ptr2, ptr3, ptr4, ang, i, j, step) ptr[i + j * step][IM] = \
        COMPLEX(COMPLEX_MUL(K5(0, j, ang), COMPLEX_MUL(COMPLEX_VEC(ptr0, i), COMPLEX_EXP(ang * i * 0))), IM) + \
        COMPLEX(COMPLEX_MUL(K5(1, j, ang), COMPLEX_MUL(COMPLEX_VEC(ptr1, i), COMPLEX_EXP(ang * i * 1))), IM) + \
        COMPLEX(COMPLEX_MUL(K5(2, j, ang), COMPLEX_MUL(COMPLEX_VEC(ptr2, i), COMPLEX_EXP(ang * i * 2))), IM) + \
        COMPLEX(COMPLEX_MUL(K5(3, j, ang), COMPLEX_MUL(COMPLEX_VEC(ptr3, i), COMPLEX_EXP(ang * i * 3))), IM) + \
        COMPLEX(COMPLEX_MUL(K5(4, j, ang), COMPLEX_MUL(COMPLEX_VEC(ptr4, i), COMPLEX_EXP(ang * i * 4))), IM)

#define DFT5(ptr, ptr0, ptr1, ptr2, ptr3, ptr4, ang, i, j, step) \
        DFT5_RE(ptr, ptr0, ptr1, ptr2, ptr3, ptr4, ang, i, j, step); \
        DFT5_IM(ptr, ptr0, ptr1, ptr2, ptr3, ptr4, ang, i, j, step)

using namespace std;
using namespace cv;

void FastFurierTransformer::setImage(const Mat1f& image)
{
    if (image.empty() == true)
    {
        return;
    }
    m_image = image.clone();
}

void FastFurierTransformer::setImage(const Mat& image)
{
    if (image.empty() == true)
    {
        return;
    }
    m_image = image.clone();
    m_image /= static_cast<float>(0xFF);
}

Mat1f FastFurierTransformer::getImage() const
{
    return m_image.clone();
}

void FastFurierTransformer::showImage(cv::String imageName)
{
    if (m_spectrum.empty() == true)
    {
        return;
    }
    imageName.append(" [FFT]");
    imshow(imageName, m_image);
}

void FastFurierTransformer::setSpectrum(const Mat2f spectrum)
{
    if (spectrum.empty() == true)
    {
        return;
    }
    m_spectrum = spectrum.clone();
}

void FastFurierTransformer::setSpectrum(const Mat spectrum)
{
    if (spectrum.empty() == true)
    {
        return;
    }
    auto newSpectrum = Mat2f(spectrum);
    m_spectrum = newSpectrum.clone();
}

Mat2f FastFurierTransformer::getSpectrum() const
{
    return m_spectrum.clone();
}

void FastFurierTransformer::showSpectrum(cv::String spectrumName)
{
    if (m_spectrum.empty() == true)
    {
        return;
    }
    spectrumName.append(" [FFT]");
    imshow(spectrumName, getSpectrumMagnitude());
}

void FastFurierTransformer::setSpectrumSize(Size2i spectrumSize)
{
    m_spectrumSize = spectrumSize;
}

Size2i FastFurierTransformer::getSpectrumSize() const
{
    return m_spectrumSize;
}

void FastFurierTransformer::setFilter(Filter& filter)
{
    m_filter = filter;
}

Filter& FastFurierTransformer::getFilter()
{
    return m_filter;
}

Mat1f FastFurierTransformer::getSpectrumMagnitude()
{
    shiftSpectrum(m_spectrum, m_spectrum.cols / 2, m_spectrum.rows / 2);
    auto buffer = Mat1f(m_spectrum.size());
    auto max = FLT_MIN;
    auto min = FLT_MAX;
    for (auto row = 0; row < m_spectrum.rows; row++)
    {
        auto magnPtr = buffer.ptr<float>(row);
        auto spcPtr = m_spectrum.ptr<Vec2f>(row);
        for (auto col = 0; col < m_spectrum.cols; col++)
        {
            auto magn = hypotf(spcPtr[col][0], spcPtr[col][1]);
            auto logMagn = log1pf(magn);
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
    shiftSpectrum(m_spectrum, m_spectrum.cols / 2, m_spectrum.rows / 2);
    return normalizeSpectrum(buffer, min, max);
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

void FastFurierTransformer::shiftSpectrum(Mat2f& spectrum, const int32_t shiftX, const int32_t shiftY)
{
    auto width = (spectrum.cols - shiftX) % spectrum.cols;
    auto height = (spectrum.rows - shiftY) % spectrum.rows;

    auto buffer = Mat2f(spectrum.size());
    auto left = Mat2f(spectrum, Rect(0, 0, width, spectrum.rows));
    auto right = Mat2f(spectrum, Rect(width, 0, spectrum.cols - width, spectrum.rows));
    auto top = Mat2f(buffer, Rect(0, 0, buffer.cols, height));
    auto bottom = Mat2f(buffer, Rect(0, height, buffer.cols, buffer.rows - height));

    left.copyTo(buffer(Rect(buffer.cols - width, 0, width, buffer.rows)));
    right.copyTo(buffer(Rect(0, 0, buffer.cols - width, buffer.rows)));

    top.copyTo(spectrum(Rect(0, spectrum.rows - height, spectrum.cols, height)));
    bottom.copyTo(spectrum(Rect(0, 0, spectrum.cols, spectrum.rows - height)));
}

void FastFurierTransformer::directFastFurierTransform()
{
    directFFT(m_image, m_spectrum);
}

void FastFurierTransformer::inverseFastFurierTransform()
{
    inverseFFT(m_spectrum, m_image);
}

void FastFurierTransformer::filtrateImage()
{
    if (m_image.empty() == true)
    {
        return;
    }

    directFFT(m_image, m_spectrum);
    shiftSpectrum(m_spectrum, m_spectrum.cols / 2, m_spectrum.rows / 2);

    m_filter.setSize(m_spectrum.size());
    for (auto row = 0; row < m_spectrum.rows; row++)
    {
        auto spcPtr = m_spectrum.ptr<Vec2f>(row);
        auto filter = m_filter.getFilter();
        auto fltPtr = filter.ptr<float>(row);
        for (auto col = 0; col < m_spectrum.cols; col++)
        {
            spcPtr[col] *= fltPtr[col];
        }
    }

    shiftSpectrum(m_spectrum, m_spectrum.cols / 2, m_spectrum.rows / 2);
    inverseFFT(m_spectrum, m_image);
}

void FastFurierTransformer::directFFT(const Mat1f& image, Mat2f& spectrum)
{
    if (image.empty() == true)
    {
        return;
    }

    auto cols = getOptimalDFTSize(image.cols);
    auto rows = getOptimalDFTSize(image.rows);

    if (cols < m_spectrumSize.width && rows < m_spectrumSize.height)
    {
        cols = m_spectrumSize.width;
        rows = m_spectrumSize.height;
    }
    else
    {
        m_spectrumSize = Size2i(cols, rows);
    }

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

    auto dfft = [this, &buffer](int32_t rows, int32_t cols)
    {
        for (auto row = 0; row < rows; row++)
        {
            auto spcPtr = buffer.ptr<Vec2f>(row);

            fft(spcPtr, cols, false);
        }

        transpose(buffer, buffer);
        return buffer;
    };
    
    buffer = dfft(rows, cols);
    buffer = dfft(cols, rows);

    spectrum = buffer;
}

void FastFurierTransformer::inverseFFT(const Mat2f& spectrum, Mat1f& image)
{
    if (spectrum.empty() == true)
    {
        return;
    }

    auto cols = getOptimalDFTSize(spectrum.cols);
    auto rows = getOptimalDFTSize(spectrum.rows);

    auto buffer = spectrum.clone();

    auto ifft = [this, &buffer](int32_t rows, int32_t cols)
    {
        for (auto row = 0; row < rows; row++)
        {
            auto imPtr = buffer.ptr<Vec2f>(row);

            fft(imPtr, cols, true);
        }

        transpose(buffer, buffer);
        return buffer;
    };

    buffer = ifft(rows, cols);
    buffer = ifft(cols, rows);

    float max = FLT_MIN;
    for (auto row = 0; row < image.rows; row++)
    {
        auto imagePtr = image.ptr<float>(row);
        auto spectrumPtr = buffer.ptr<Vec2f>(row);
        for (auto col = 0; col < image.cols; col++)
        { 
            float value = spectrumPtr[col][RE] / (cols * rows);
            imagePtr[col] = value;
            if (value > max)
            {
                max = value;
            }
        }
    }

    m_image /= max > 1 ? max : 1;
}

void FastFurierTransformer::fft(Vec2f *ptr, const int32_t size, const bool isInverse)
{
    if (size == 1) return;

    if (size % 2 == 0)
    {
        auto step = size / 2;
        auto vec = Mat2f(1, size, Vec2f(0, 0));
        auto ptr0 = vec.ptr<Vec2f>(0);
        auto ptr1 = ptr0 + step;

        for (int i = 0; i < step; i++)
        {
            ptr0[i] = ptr[2 * i];
            ptr1[i] = ptr[2 * i + 1];
        }

        fft(ptr0, step, isInverse);
        fft(ptr1, step, isInverse);

        float ang = 2 * M_PI / size * (isInverse ? 1 : -1);
        for (auto i = 0; i < step; i++)
        {
            DFT2(ptr, ptr0, ptr1, ang, i, 0, step);
            DFT2(ptr, ptr0, ptr1, ang, i, 1, step);
        }

        return;
    }

    if (size % 3 == 0)
    {
        auto step = size / 3;
        auto vec = Mat2f(1, size, Vec2f(0, 0));
        auto ptr0 = vec.ptr<Vec2f>(0);
        auto ptr1 = ptr0 + step;
        auto ptr2 = ptr1 + step;

        for (int i = 0; i < step; i++)
        {
            ptr0[i] = ptr[3 * i];
            ptr1[i] = ptr[3 * i + 1];
            ptr2[i] = ptr[3 * i + 2];
        }

        fft(ptr0, step, isInverse);
        fft(ptr1, step, isInverse);
        fft(ptr2, step, isInverse);

        float ang = 2 * M_PI / size * (isInverse ? 1 : -1);
        for (auto i = 0; i < step; i++)
        {
            DFT3(ptr, ptr0, ptr1, ptr2, ang, i, 0, step);
            DFT3(ptr, ptr0, ptr1, ptr2, ang, i, 1, step);
            DFT3(ptr, ptr0, ptr1, ptr2, ang, i, 2, step);
        }

        return;
    }

    if (size % 5 == 0)
    {
        auto step = size / 5;
        auto vec = Mat2f(1, size, Vec2f(0, 0));
        auto ptr0 = vec.ptr<Vec2f>(0);
        auto ptr1 = ptr0 + step;
        auto ptr2 = ptr1 + step;
        auto ptr3 = ptr2 + step;
        auto ptr4 = ptr3 + step;

        for (int i = 0; i < step; i++)
        {
            ptr0[i] = ptr[5 * i];
            ptr1[i] = ptr[5 * i + 1];
            ptr2[i] = ptr[5 * i + 2];
            ptr3[i] = ptr[5 * i + 3];
            ptr4[i] = ptr[5 * i + 4];
        }

        fft(ptr0, step, isInverse);
        fft(ptr1, step, isInverse);
        fft(ptr2, step, isInverse);
        fft(ptr3, step, isInverse);
        fft(ptr4, step, isInverse);

        float ang = 2 * M_PI / size * (isInverse ? 1 : -1);
        for (auto i = 0; i < step; i++)
        {
            DFT5(ptr, ptr0, ptr1, ptr2, ptr3, ptr4, ang, i, 0, step);
            DFT5(ptr, ptr0, ptr1, ptr2, ptr3, ptr4, ang, i, 1, step);
            DFT5(ptr, ptr0, ptr1, ptr2, ptr3, ptr4, ang, i, 2, step);
            DFT5(ptr, ptr0, ptr1, ptr2, ptr3, ptr4, ang, i, 3, step);
            DFT5(ptr, ptr0, ptr1, ptr2, ptr3, ptr4, ang, i, 4, step);
        }

        return;
    }
}

void shiftSpectrum(Mat1f& spectrum, const int32_t shiftX, const int32_t shiftY)
{
    auto width = (spectrum.cols - shiftX) % spectrum.cols;
    auto height = (spectrum.rows - shiftY) % spectrum.rows;

    auto buffer = Mat1f(spectrum.size());
    auto left   = Mat1f(spectrum, Rect(0    , 0, width                , spectrum.rows));
    auto right  = Mat1f(spectrum, Rect(width, 0, spectrum.cols - width, spectrum.rows));
    auto top    = Mat1f(buffer, Rect(0, 0     , buffer.cols, height              ));
    auto bottom = Mat1f(buffer, Rect(0, height, buffer.cols, buffer.rows - height));

    left.copyTo( buffer(Rect(buffer.cols - width, 0, width              , buffer.rows)));
    right.copyTo(buffer(Rect(0                  , 0, buffer.cols - width, buffer.rows)));

    top.copyTo(   spectrum(Rect(0, spectrum.rows - height, spectrum.cols, height                )));
    bottom.copyTo(spectrum(Rect(0, 0                     , spectrum.cols, spectrum.rows - height)));
}

void multiplySpectrums(Mat2f& first, Mat2f& second, Mat2f& result, bool isCorr)
{
    auto cols = max(first.cols, second.cols);
    auto rows = max(first.rows, second.rows);

    if (first.size != second.size)
    {
        return;
    }

    auto buffer = Mat2f(Size2i(cols, rows), Vec2f(0, 0));
    auto corr = isCorr ? -1 : 1;

    for (auto row = 0; row < rows; row++)
    {
        auto ptr = buffer.ptr<Vec2f>(row);
        auto ptr1 = first.ptr<Vec2f>(row);
        auto ptr2 = second.ptr<Vec2f>(row);

        for (auto col = 0; col < cols; col++)
        {
            ptr[col][RE] = COMPLEX(COMPLEX_MUL(COMPLEX_VEC(ptr1, col), COMPLEX_VEC(ptr2, col) * corr), RE);
            ptr[col][IM] = COMPLEX(COMPLEX_MUL(COMPLEX_VEC(ptr1, col), COMPLEX_VEC(ptr2, col) * corr), IM);
        }
    }

    result = buffer;
}
