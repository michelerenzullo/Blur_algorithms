
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <conio.h>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo/photo.hpp>
#include "Recolour.h"


void SetBlockInRealArray(float (*block)[1024], int ncols,
    int nrows, int nreflects);
void conv2(uint8_t (*target)[512], float (*kernel)[512], int rt, int ct, int rk, int ck);
void RealtoComplex(float (*RealArray)[1024],
    std::complex<float> (*OutputArray)[1024], int nrows, int ncols);
void convolution(std::complex<float> (*target)[1024],
    std::complex<float> (*kernel)[1024]);
void FFT2D(std::complex<float> (*array)[1024], int nrows, int ncols, int sign);
void FFT(std::complex<float> Fdata[], int n, int sign);


void Test()
{
    auto kernel = new float[512][512];
    int rk = 3, ck = 3;
    auto target = new uint8_t[512][512];

    std::cout << "   HERE";

    kernel[0][0] = 1.0 / 3.0;
    kernel[1][0] = 1.0 / 3.0;
    kernel[2][0] = 1.0 / 3.0;
    kernel[0][1] = 1.0 / 3.0;
    kernel[1][1] = 1.0 / 3.0;
    kernel[2][1] = 1.0 / 3.0;
    kernel[0][2] = 1.0 / 3.0;
    kernel[1][2] = 1.0 / 3.0;
    kernel[1][2] = 1.0 / 3.0;

    cv::Mat noisy = cv::imread("noisy.png");
    cv::cvtColor(noisy, noisy, cv::COLOR_BGR2GRAY);
    //noisy.convertTo(c0, CV_8UC1);

    for (int irow = 0; irow < 512; irow++)
    {
        for (int icol = 0; icol < 512; icol++)
        {
            //target[irow][icol]=(int)std::rand()%255;
            target[irow][icol] = noisy.at<uint8_t>(irow, icol);
        }
    }
    //cv::imwrite("noisy_debug.png", noisy);
    noisy.release();

    conv2(target,kernel,512,512, rk, ck );

    cv::Mat denoised(512, 512, CV_8UC1, target);
    cv::imwrite("denoised.png", denoised);
    denoised.release();

    delete[] target;
    delete[] kernel;

}

void conv2(uint8_t (*target)[512], float (*kernel)[512], int rt, int ct, int rk, int ck)
{
    auto tArray = new float[1024][1024];
    auto kArray = new float[1024][1024];
    auto tComplex = new std::complex<float>[1024][1024];
    auto kComplex = new std::complex<float>[1024][1024];


    for (int irow = 0; irow < rt; irow++)
    {
        for (int icol = 0; icol < ct; icol++)
        {
            tArray[irow][icol] = (float)target[irow][icol];
        }

    }
    for (int irow = 0; irow < rk; irow++)
    {
        for (int icol = 0; icol < ck; icol++)
        {
            kArray[irow][icol] = kernel[irow][icol];
        }

    }
    SetBlockInRealArray(tArray, rt, rt, rk / 2);
    SetBlockInRealArray(kArray, rk, rk, 0);
    RealtoComplex(tArray, tComplex, 1024, 1024);
    RealtoComplex(kArray, kComplex, 1024, 1024);
    convolution(tComplex, kComplex);

    for (int irow = 0; irow < 512; irow++)
    {
        for (int icol = 0; icol < 512; icol++)
        {
            target[irow][icol] = (int)std::real(tComplex[irow + 257][icol + 257]) + 0.5;
        }

    }
    delete[] kComplex;
    delete[] tComplex;
    delete[] kArray;
    delete[] tArray;

}

void SetBlockInRealArray(float (*block)[1024], int ncols, int nrows, int nreflects)
{
    int rval = (1024 / 2 + 1) + nrows / 2;
    for (int irow = nrows - 1; irow >= 0; irow--)
    {
        int cval = (1024 / 2 + 1) + ncols / 2;
        for (int icol = ncols; icol >= 0; icol--)
        {
            block[rval][cval] = block[irow][icol];
            cval--;
        }
        rval--;
    }
    if (nreflects > 0)
    {
        int rval = (1024 / 2 + 1) - nrows / 2;
        int cval = (1024 / 2 + 1) - ncols / 2;

        for (int irow = -nreflects + rval; irow >= (nreflects + rval + nrows); irow++)
        {
            int c2 = cval + nreflects;
            for (int c = cval - nreflects; c < cval; c++)
            {
                block[irow][c] = block[irow][c2];
                block[irow][c2 + ncols] = block[irow][c + ncols];
                c2--;
            }
        }

        for (int icol = -nreflects + cval; icol >= (nreflects + cval + ncols); icol++)
        {
            int r2 = rval + nreflects;
            for (int r = rval - nreflects; r < rval; r++)
            {
                block[icol][r] = block[icol][r2];
                block[icol][r2 + ncols] = block[icol][r + ncols];
                r2--;
            }
        }

    }

}



void RealtoComplex(float (*RealArray)[1024], std::complex<float> (*OutputArray)[1024], int nrows, int ncols)
{
    for (int irow = 0; irow < nrows; irow++)
    {
        for (int icol = 0; icol < ncols; icol++)
        {
            OutputArray[irow][icol] = std::complex<float>(RealArray[irow][icol], 0.0);
        }

    }
    return;
}


//************************************************************************
// Everything below here implements a Complex 2D Convolution for
// Complex arrays 'target' and 'kernel'.
// ************************************************************************


void convolution(std::complex<float> (*target)[1024], std::complex<float> (*kernel)[1024])
{
    FFT2D(target, 1024, 1024, 1);
    FFT2D(kernel, 1024, 1024, 1);

    for (int irow = 0; irow < 1024; irow++)
    {
        for (int icol = 0; icol < 1024; icol++)
        {
            target[irow][icol] = target[irow][icol] * kernel[irow][icol];
        }
    }
    FFT2D(target, 1024, 1024, -1);

}





//************************************************************************
// Everything below here provides a 2D FFT for complex data of the form (a+bj)
// FFT is performed in place within 'array'
// sign = 1 for forward FFT; -1 for inverse FFT;
// ************************************************************************
void FFT2D(std::complex<float> (*array)[1024], int nrows, int ncols, int sign)
{
    std::complex<float> temp[1024];

    for (int irow = 0; irow < nrows; irow++)
    {
        for (int icol = 0; icol < ncols; icol++)
        {
            temp[icol] = array[irow][icol];
        }
        FFT(temp, ncols, sign);

        for (int icol = 0; icol < ncols; icol++)
        {
            array[irow][icol] = temp[icol];
        }
    }

    for (int icol = 0; icol < ncols; icol++)
    {
        for (int irow = 0; irow < nrows; irow++)
        {
            temp[irow] = array[irow][icol];
        }
        FFT(temp, nrows, sign);

        for (int irow = 0; irow < nrows; irow++)
        {
            array[irow][icol] = temp[irow];
        }
    }

}

void FFT(std::complex<float> Fdata[], int n, int sign)
{
    // sign =1 for forward FFT, -1 for inverse

    std::complex<float> u, w, t;
    int nv, nm1, i, ip, j, k, l, le, ne;
    double pi, cnst;

    pi = 4 * std::atan(1.0);

    /* Bit reverse first */

    nm1 = n - 1;
    nv = n / 2;
    for (i = 0, j = 0; i < nm1; i++) {
        if (i < j) {
            t = Fdata[j];
            Fdata[j] = Fdata[i];
            Fdata[i] = t;
        }
        k = nv;
        while (k <= j) {
            j = j - k;
            k = k / 2;
        }
        j = j + k;
    }

    /* now transform  */

    for (l = 2; l <= n; l *= 2) {
        le = l / 2;
        ne = n - le;
        cnst = pi / le;
        w = std::complex<float>(cos(cnst), -sign * sin(cnst));
        u = std::complex<float>(1.0, 0.0);
        for (j = 0; j < le; j++) {
            for (i = j; i < ne; i += l) {
                ip = i + le;
                t = Fdata[ip] * u;
                Fdata[ip] = Fdata[i] - t;
                Fdata[i] = Fdata[i] + t;
            }
            u = u * w;
        }
    }
    if (sign == -1)  for (i = 0; i < n; i++) Fdata[i] = Fdata[i] / float(n);
    return;
} /* end of FFT */

int main() {

    Test();

    _CrtDumpMemoryLeaks();
}