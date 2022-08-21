
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <conio.h>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo/photo.hpp>
#include "Recolour.h"


void AddPadding(float (*block)[1024], int rows, int ncols, int nreflects);
void conv2(uchar (*target)[512], float (*kernel)[512], int rt, int ct, int rk, int ck);
void convolution(std::complex<float> (*target)[1024],
    std::complex<float> (*kernel)[1024]);
void FFT2D(std::complex<float> (*array)[1024], int nrows, int ncols, int sign);
void FFT(std::complex<float> Fdata[], int n, int sign);


void Test(cv::Mat monoimage)
{
    auto kernel = new float[512][512];

    int nsmooth = 5;

    int rk = nsmooth;
    int ck = nsmooth;

    for(int irows=0;irows<rk;irows++)
    {
        for(int icols=0;icols<ck;icols++)
        {
            kernel[irows][icols]=1.0/(float)(rk*ck);
        }
    }


    auto target = new uchar[512][512];

    for (int irow = 0; irow < 512; irow++)
    {
        for (int icol = 0; icol < 512; icol++)
        {
            target[irow][icol]=monoimage.at<uchar>(irow,icol);
        }
    }
    SaveImage(monoimage, "Mono in", 1,0,0);

    // The inputs have been set up. Now do
    // the convolution and display the output.
    conv2(target,kernel,512,512, rk, ck );

    for (int irow = 0; irow < 512; irow++)
    {
        for (int icol = 0; icol < 512; icol++)
        {
            monoimage.at<uchar>(irow,icol)=target[irow][icol];
        }
    }
    SaveImage(monoimage, "Mono out", 1,1,0);
    delete[] target;
    delete[] kernel;

}

void conv2(uchar (*target)[512], float (*kernel)[512], int rt, int ct, int rk, int ck)
{
    auto tArray = new float[1024][1024];
    auto kArray = new float[1024][1024];
    auto tComplex = new std::complex<float>[1024][1024];
    auto kComplex = new std::complex<float>[1024][1024];


    int d=1024;

    // Set arrays to zero
    for (int irow = 0; irow < d; irow++)
    {
        for (int icol = 0; icol < d; icol++)
        {
            tArray[irow][icol] = 0.0;
            kArray[irow][icol] = 0.0;
        }
    }

    // Write the input data into centre of an array.
    int rOffset=(d/2+1)-int(rt/2);
    int cOffset=(d/2+1)-int(ct/2);

    for (int irow = 0; irow < rt; irow++)
    {
        for (int icol = 0; icol < ct; icol++)
        {
            tArray[irow+rOffset][icol+cOffset]
                                = (float)target[irow][icol];
        }

    }
    // Bulk out by reflection at edges.
    //
    AddPadding(tArray, rt, ct, cv::max(rk/2,ck/2));

    // Write the kernel into an array centered on (0,0)
    for (int irow = 0; irow < rk; irow++)
    {
        int rval=(d+irow-int(rk/2))%d;
        for (int icol = 0; icol < ck; icol++)
        {
            int cval=(d+icol-int(ck/2))%d;
            kArray[rval][cval] = kernel[irow][icol];
        }

    }


    // Represent arrays as complex variables
    for (int irow = 0; irow <d; irow++)
    {
        for (int icol = 0; icol < d; icol++)
            {
                tComplex[irow][icol] = std::complex<float>(tArray[irow][icol], 0.0);
                kComplex[irow][icol] = std::complex<float>(kArray[irow][icol], 0.0);
            }
    }

    // Convolve arrays
    convolution(tComplex, kComplex);

    // Return convolved data as output.
    for (int irow = 0; irow < 512; irow++)
    {
        int ri=irow+rOffset;
        for (int icol = 0; icol < 512; icol++)
        {
            int ci=icol+cOffset;
            target[irow][icol] = (uchar) std::real(tComplex[ri][ci]) + 0.01;
        }

    }
    delete[] kComplex;
    delete[] tComplex;
    delete[] kArray;
    delete[] tArray;

}

void AddPadding(float (*block)[1024], int nrows, int ncols, int nreflects)
{
    // Pad out input image by reflection.
    if (nreflects > 0)
    {
        int rval = (1024 / 2 + 1) - nrows / 2;
        int cval = (1024 / 2 + 1) - ncols / 2;


        for (int irow = rval-nreflects; irow < (rval + nrows + nreflects); irow++)
        {
            int c2 = cval + nreflects;
            for (int c = cval - nreflects; c < cval; c++)
            {
                block[irow][c] = block[irow][c2];
                block[irow][c2 + ncols-1] = block[irow][c + ncols-1];
                c2--;
            }
        }

        for (int icol = cval-nreflects; icol <(cval + ncols+nreflects); icol++)
        {
            int r2 = rval + nreflects;
            for (int r = rval - nreflects; r < rval; r++)
            {
                block[r][icol] = block[r2][icol];
                block[r2 + nrows-1][icol] = block[r + nrows-1][icol];
                r2--;
            }
        }

    }

}


//************************************************************************
// Everything below here implements a Complex 2D Convolution for
// Complex arrays 'target' and 'kernel'.
// ************************************************************************


void convolution(std::complex<float> (*target)[1024], std::complex<float> (*kernel)[1024])
{
    // FFT the two arrays, multiply together and find inverse FFT of product.
    // Multiplication in the frequency domain is a convolution in the image domain.

    FFT2D(target, 1024, 1024, 1);
    FFT2D(kernel, 1024, 1024, 1);

    int testmode=1;
    int includeifft=1;

    for (int irow = 0; irow < 1024; irow++)
    {
        for (int icol = 0; icol < 1024; icol++)
        {

            if(testmode==1)
            {
                //Convolution
                target[irow][icol] = target[irow][icol]*kernel[irow][icol];
            }
            if(testmode==2)
            {
                // Write across only
                target[irow][icol] = target[irow][icol];
            }
            if(testmode==3)
            {
                // Scaled kernel;  (2295=255*9)
                target[irow][icol] = std::complex<float>(2295,0)*kernel[irow][icol];
            }
        }
    }
    if(includeifft==1)
    {
        FFT2D(target, 1024, 1024, -1);
    }


}





//************************************************************************
// Everything below here provides a 2D FFT for complex data of the form (a+bj)
// FFT is performed in place within 'array'
// sign = 1 for forward FFT; -1 for inverse FFT;
// ************************************************************************
void FFT2D(std::complex<float> (*array)[1024], int nrows, int ncols, int sign)
{
    std::complex<float> temp[1024];

    // FFT row by row
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

    //FFT the resultant column by column
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

void FFT(std::complex<float> Fdata[1], int n, int sign)
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
    if (sign == -1)  for (i = 0; i < n; i++) Fdata[i] = Fdata[i]/ float(n);
    return;
} /* end of FFT */



