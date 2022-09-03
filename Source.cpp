#include <complex>
#include <cmath>
#include <vector>
#include <stdio.h>
#include <conio.h>
#include <ppl.h>
#include <execution>
#include <iostream> 
#include <pffft.hpp>
#include "pocketfft_hdronly.h"
#define _CRTDBG_MAP_ALLOC
#include <cstdlib>
#include <crtdbg.h>

#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
// Replace _NORMAL_BLOCK with _CLIENT_BLOCK if you want the
// allocations to be of _CLIENT_BLOCK type

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo/photo.hpp>
#include "Recolour.h"

#define SIZE 512

std::chrono::time_point<std::chrono::steady_clock> start_0;

void AddPadding(float(*block)[SIZE * 2], int rows, int ncols, int nreflects);
void conv2(uint8_t(*target)[SIZE], float* kernel, int rt, int ct, int rk, int ck);
void convolution(std::complex<float>(*target)[SIZE * 2], std::complex<float>(*kernel)[SIZE * 2]);
void FFT2D(std::complex<float>(*array)[SIZE * 2], int nrows, int ncols, int sign);
inline void FFT(std::complex<float>* Fdata, int n, int sign);

void generate_table(uint32_t* crc_table)
{
	uint32_t r;
	for (int i = 0; i < 256; i++)
	{
		r = i;
		for (int j = 0; j < 8; j++)
		{
			if (r & 1)
			{
				r >>= 1;
				r ^= 0xEDB88320;
			}
			else
			{
				r >>= 1;
			}
		}
		crc_table[i] = r;
	}
}

uint32_t crc32c(uint8_t* data, size_t bytes)
{
	uint32_t* crc_table = new uint32_t[256];
	generate_table(crc_table);
	uint32_t crc = 0xFFFFFFFF;
	while (bytes--)
	{
		int i = (crc ^ *data++) & 0xFF;
		crc = (crc_table[i] ^ (crc >> 8));
	}
	delete[] crc_table;
	return crc ^ 0xFFFFFFFF;
}

void make_kernel(float* kernel, int kLen, const size_t iFTsize[])
{
	const double scale = 1. / ((float)kLen * (float)kLen * (float)kLen * (float)kLen);

#pragma omp parallel for
	for (int irow = -kLen + 1; irow <= (kLen - 1); irow++)
	{
		for (int icol = -kLen + 1; icol <= (kLen - 1); icol++)
		{
			double kval = ((kLen - abs(irow)) * (kLen - abs(icol)));
			int rval = (irow + iFTsize[0]) % iFTsize[0];
			int cval = (icol + iFTsize[1]) % iFTsize[1];
			kernel[rval * iFTsize[1] + cval] = kval * scale;
		}
	}
	printf("crc32: %02X\n", crc32c((uint8_t*)kernel, iFTsize[0] * iFTsize[1] * sizeof(float)));
}


void pocketfft_(cv::Mat image, int nsmooth)
{
	image.convertTo(image, CV_32FC3);
	size_t pad = nsmooth * nsmooth - 1;
	//pad = 0;
	cv::copyMakeBorder(image, image, pad, pad, pad, pad, CV_HAL_BORDER_REFLECT);
	const size_t sizes[2] = { image.size[0], image.size[1] };
	cv::Mat temp[3];
	cv::split(image, temp);

	int ndata = sizes[0] * sizes[1];

	std::vector<float> kernel(sizes[0] * sizes[1]);
	make_kernel(kernel.data(), (nsmooth * nsmooth), sizes);

	start_0 = std::chrono::steady_clock::now();
	pocketfft::shape_t shape{ sizes[0] , sizes[1] };

	pocketfft::stride_t strided(shape.size());
	size_t tmpf = sizeof(float);
	for (int i = shape.size() - 1; i >= 0; --i)
	{
		strided[i] = tmpf;
		tmpf *= shape[i];
	}

	size_t test_tmpf_out = sizeof(std::complex<float>);
	std::vector test{ sizes[0], (sizes[1] / 2 + 1) };
	pocketfft::stride_t test_strided_out(test.size());
	for (int i = test.size() - 1; i >= 0; --i)
	{
		test_strided_out[i] = test_tmpf_out;
		test_tmpf_out *= test[i];
	}

	std::vector<std::complex<float>> kerf(sizes[0] * (sizes[1] / 2 + 1));
	pocketfft::shape_t axes{ 0, 1 };

	pocketfft::r2c(shape, strided, test_strided_out, axes, pocketfft::FORWARD, kernel.data(), kerf.data(), 1.f, 0);
	//start_0 = std::chrono::steady_clock::now();

#pragma omp parallel for
	for (int i = 0; i < 3; ++i) {
		std::vector<std::complex<float>> resf(sizes[0] * (sizes[1] / 2 + 1));
		pocketfft::r2c(shape, strided, test_strided_out, axes, pocketfft::FORWARD, (float*)temp[i].data, resf.data(), 1.f, 0);

		transform(begin(kerf), end(kerf), begin(resf), begin(resf), std::multiplies<std::complex<float>>());

		/*
		for (int i = 0; i < sizes[0]; ++i)
			for (int j = 0; j < (sizes[1] / 2 + 1); ++j) {
				resf[i * (sizes[1] / 2 + 1) + j] *= kerf[i * (sizes[1] / 2 + 1) + j];
			}*/

		pocketfft::c2r(shape, test_strided_out, strided, axes, pocketfft::BACKWARD, resf.data(), (float*)temp[i].data, 1.f / ndata, 0);

		//transform(begin(resf), end(resf), &((float*)temp[i].data)[0], [](std::complex<float> i) { return std::real(i); });
		/*
		for (int row = 0; row < SIZE; ++row)
			for (int col = 0; col < (SIZE / 2 + 1); ++col)
				((float*)temp[i].data)[row * SIZE + col] = std::real(resf[row * SIZE + col]);*/
	}

	printf("PocketFFT: %f\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_0).count());
	cv::merge(temp, 3, image);


	image.convertTo(image, CV_8UC3);
	cv::Rect myroi(pad, pad, sizes[1] - pad * 2, sizes[0] - pad * 2);
	cv::imwrite("C:/Users/Michele/Downloads/c_.png", image(myroi));

}

void pffft_(cv::Mat image, int nsmooth, bool fast_ = true)
{

	image.convertTo(image, CV_32FC3);
	size_t pad[2] = { nsmooth * nsmooth - 1 , nsmooth * nsmooth - 1 }; //minimal required pad to fix the circular convolution
	size_t sizes[2] = { image.size[0] + pad[0] * 2, image.size[1] + pad[1] * 2 };
	size_t ndata = sizes[0] * sizes[1];

	//if the length of the data is not decomposable in small prime numbers 2 - 3 - 5, is necessary to update the size adding more pad
	if (!pffft::Fft<float>::isValidSize(ndata)) {
		printf("%d %d %d %d %d %d\n", sizes[0], sizes[1], pad[0], pad[1], ndata, pffft::Fft<float>::isValidSize(ndata));
		pad[0] += (pffft::Fft<float>::nearestTransformSize(sizes[0]) - sizes[0]) / 2;
		pad[1] += (pffft::Fft<float>::nearestTransformSize(sizes[1]) - sizes[1]) / 2;
		sizes[0] = pffft::Fft<float>::nearestTransformSize(sizes[0]);
		sizes[1] = pffft::Fft<float>::nearestTransformSize(sizes[1]);
		ndata = sizes[0] * sizes[1];
		printf("%d %d %d %d %d %d\n", sizes[0], sizes[1], pad[0], pad[1], ndata, pffft::Fft<float>::isValidSize(ndata));

	}
	cv::copyMakeBorder(image, image, pad[0], pad[0], pad[1], pad[1], CV_HAL_BORDER_REFLECT);

	pffft::Fft<float> fft_kernel(ndata);
	pffft::AlignedVector<float> kernel_aligned = pffft::AlignedVector<float>(fft_kernel.getLength());

	make_kernel(kernel_aligned.data(), nsmooth * nsmooth, sizes);

	cv::Mat temp[3];
	cv::split(image, temp);

	start_0 = std::chrono::steady_clock::now();

	bool fast_convolve = fast_;

	pffft::AlignedVector<float> kerf;
	pffft::AlignedVector<std::complex<float>> kerf_complex;
	if (fast_convolve) {
		kerf = fft_kernel.internalLayoutVector();
		fft_kernel.forwardToInternalLayout(kernel_aligned, kerf);
	}
	else {
		kerf_complex = fft_kernel.spectrumVector();
		fft_kernel.forward(kernel_aligned, kerf_complex);
	}
	kernel_aligned.clear();

	std::vector<pffft::AlignedVector<float>> resf(3, pffft::AlignedVector<float>(ndata) /* same as fft_kernel.valueVector() */);

#pragma omp parallel for
	for (int i = 0; i < 3; ++i) {
		pffft::Fft<float> fft(ndata);
		//std::cout << fft.getLength() << " " << fft.getSpectrumSize() << " " << fft.getInternalLayoutSize() << "\n";
		std::copy(&((float*)temp[i].data)[0], &((float*)temp[i].data)[ndata], resf[i].begin());
		temp[i].release();
		if (fast_convolve) {
			pffft::AlignedVector<float> work = fft.internalLayoutVector();
			fft.forwardToInternalLayout(resf[i], work);
			fft.convolve(work, kerf, work, 1.f / ndata);
			fft.inverseFromInternalLayout(work, resf[i]);
			work.clear();
		}
		else {
			pffft::AlignedVector<std::complex<float>> work = fft.spectrumVector();
			fft.forward(resf[i], work);
			transform(begin(kerf_complex), end(kerf_complex), begin(work), begin(work), [&ndata](auto& n, auto& m) { return n * m * (1.f / ndata); });
			fft.inverse(work, resf[i]);
			//pffft::AlignedVector<std::complex<float>> work = fft.spectrumVector();
			//fftshift1D(work_.data(), work.data(), sizes[0] * sizes[1]);
			//transform(begin(work), end(work), resf[i].begin(), [](std::complex<float> i) { return std::real(i); });
			work.clear();
		}
		temp[i] = cv::Mat(sizes[0], sizes[1], CV_32F, resf[i].data());
	}
	fast_convolve ? kerf.clear() : kerf_complex.clear();


	printf("pffft: %f\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_0).count());
	cv::merge(temp, 3, image);

	//cv::cvtColor(image, image, cv::COLOR_YCrCb2BGR);
	image.convertTo(image, CV_8UC3);
	cv::Rect myroi(pad[1], pad[0], sizes[1] - pad[1] * 2, sizes[0] - pad[0] * 2);
	cv::imwrite("C:/Users/Michele/Downloads/c_.png", image(myroi));
}

void NewBlur(cv::Mat monoimage, int nsmooth)
{
	auto kernel = new float[SIZE * SIZE * 4];

	int rk = 2 * nsmooth * nsmooth - 1;
	int ck = 2 * nsmooth * nsmooth - 1;

	const size_t sizes[2] = { SIZE * 2, SIZE * 2 };
	make_kernel(kernel, nsmooth * nsmooth, sizes);

	/*
	for (int irows = 0; irows < rk; irows++)
	{
		for (int icols = 0; icols < ck; icols++)
		{
			kernel[irows][icols] = 1.f / (rk * ck);
		}
	}*/


	auto target = new uint8_t[SIZE][SIZE];

	for (int irow = 0; irow < SIZE; irow++)
	{
		for (int icol = 0; icol < SIZE; icol++)
		{
			target[irow][icol] = monoimage.at<uint8_t>(irow, icol);
		}
	}

	//SaveImage(monoimage, "Mono in", 1, 0, 0);

	// The inputs have been set up. Now do
	// the convolution and display the output.
	conv2(target, kernel, SIZE, SIZE, rk, ck);

	for (int irow = 0; irow < SIZE; irow++)
	{
		for (int icol = 0; icol < SIZE; icol++)
		{
			monoimage.at<uint8_t>(irow, icol) = target[irow][icol];
		}
	}
	//SaveImage(monoimage, "Mono out", 0, 1, 0);
	delete[] target;
	delete[] kernel;

}

void Test(cv::Mat image, int flag = 0, int nsmooth = 5, bool fast_ = 1)
{

	start_0 = std::chrono::steady_clock::now();

	if (flag == 3) pffft_(image, nsmooth, fast_);
	else if (flag == 2) pocketfft_(image, nsmooth);
	else if (flag == 1)
	{
		cv::Mat temp[3];
		cv::split(image, temp);
		start_0 = std::chrono::steady_clock::now();
		NewBlur(temp[0], nsmooth);
		NewBlur(temp[1], nsmooth);
		NewBlur(temp[2], nsmooth);
		printf("NewBlur: %f\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_0).count());
		cv::merge(temp, 3, image);
		cv::imwrite("C:/Users/Michele/Downloads/c_.png", image);
	}
	else
	{
		cv::Mat temp[3];
		//cv::cvtColor(image, image, cv::COLOR_BGR2YCrCb);
		cv::split(image, temp);
		nsmooth *= nsmooth;
		Concurrency::parallel_for(0, 3, [&temp, &nsmooth](int i) {
			cv::blur(temp[i], temp[i], cv::Size(nsmooth, nsmooth));
			cv::blur(temp[i], temp[i], cv::Size(nsmooth, nsmooth));
			});
		printf("OpenCV blur: %f\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_0).count());

		cv::merge(temp, 3, image);
		//cv::cvtColor(image, image, cv::COLOR_YCrCb2BGR);
		cv::imwrite("C:/Users/Michele/Downloads/c_.png", image);
	}


}

void conv2(uint8_t(*target)[SIZE], float* kArray, int rt, int ct, int rk, int ck)
{
	const int d = SIZE * 2;
	auto tArray = new float[d][d]();
	auto tComplex = new std::complex<float>[d][d];
	auto kComplex = new std::complex<float>[d][d];

	// Write the input data into centre of an array.
	int rOffset = (d / 2 + 1) - int(rt / 2);
	int cOffset = (d / 2 + 1) - int(ct / 2);

	for (int irow = 0; irow < rt; irow++)
		for (int icol = 0; icol < ct; icol++)
			tArray[irow + rOffset][icol + cOffset] = (float)target[irow][icol];

	// Bulk out by reflection at edges.
	//
	AddPadding(tArray, rt, ct, cv::max(rk, ck));

	// Represent arrays as complex variables
	for (int irow = 0; irow < d; irow++)
	{
		for (int icol = 0; icol < d; icol++)
		{
			tComplex[irow][icol] = std::complex<float>(tArray[irow][icol], 0.0);
			kComplex[irow][icol] = std::complex<float>(kArray[irow * SIZE * 2 + icol], 0.0);
		}
	}

	// Convolve arrays
	convolution(tComplex, kComplex);

	// Return convolved data as output.
	for (int irow = 0; irow < SIZE; irow++)
	{
		for (int icol = 0, ri = irow + rOffset; icol < SIZE; icol++)
		{
			int ci = icol + cOffset;
			target[irow][icol] = (uint8_t)std::real(tComplex[ri][ci]) + 0.01;
		}

	}
	delete[] kComplex;
	delete[] tComplex;
	delete[] tArray;

}

void AddPadding(float(*block)[SIZE * 2], int nrows, int ncols, int nreflects)
{
	// Pad out input image by reflection.
	if (nreflects > 0)
	{
		int rval = (SIZE * 2 / 2 + 1) - nrows / 2;
		int cval = (SIZE * 2 / 2 + 1) - ncols / 2;


		for (int irow = rval - nreflects; irow < (rval + nrows + nreflects); irow++)
		{
			int c2 = cval + nreflects;
			for (int c = cval - nreflects; c < cval; c++)
			{
				block[irow][c] = block[irow][c2];
				block[irow][c2 + ncols - 1] = block[irow][c + ncols - 1];
				c2--;
			}
		}

		for (int icol = cval - nreflects; icol < (cval + ncols + nreflects); icol++)
		{
			int r2 = rval + nreflects;
			for (int r = rval - nreflects; r < rval; r++)
			{
				block[r][icol] = block[r2][icol];
				block[r2 + nrows - 1][icol] = block[r + nrows - 1][icol];
				r2--;
			}
		}
	}
}

void FFT2D(std::complex<float>(*array)[SIZE * 2], int nrows, int ncols, int sign)
{
	FFT((std::complex<float> *)array, SIZE * SIZE * 4, sign);
	printf("FFT2D_par %f\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_0).count());

}

//************************************************************************
// Everything below here implements a Complex 2D Convolution for
// Complex arrays 'target' and 'kernel'.
// ************************************************************************


void convolution(std::complex<float>(*target)[SIZE * 2], std::complex<float>(*kernel)[SIZE * 2])
{
	// FFT the two arrays, multiply together and find inverse FFT of product.
	// Multiplication in the frequency domain is a convolution in the image domain.
	std::chrono::time_point<std::chrono::steady_clock> start_0 = std::chrono::steady_clock::now();

	int par = 1;
	if (par) {
		Concurrency::parallel_invoke(
			[&target] {FFT2D(target, SIZE * 2, SIZE * 2, 1); },
			[&kernel] {FFT2D(kernel, SIZE * 2, SIZE * 2, 1); }
		);
		printf("parallel_invoke: %f\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_0).count());
	}
	else {
		FFT2D(target, SIZE * 2, SIZE * 2, 1);
		FFT2D(kernel, SIZE * 2, SIZE * 2, 1);
		printf("seq_invoke: %f\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_0).count());
	}

	int testmode = 1;
	int includeifft = 1;

	for (int irow = 0; irow < SIZE * 2; irow++)
	{
		for (int icol = 0; icol < SIZE * 2; icol++)
		{

			if (testmode == 1)
			{
				//Convolution
				target[irow][icol] = target[irow][icol] * kernel[irow][icol];
			}
			if (testmode == 2)
			{
				// Write across only
				target[irow][icol] = target[irow][icol];
			}
			if (testmode == 3)
			{
				// Scaled kernel;  (2295=255*9)
				target[irow][icol] = std::complex<float>(2295, 0) * kernel[irow][icol];
			}
		}
	}
	if (includeifft == 1)
	{
		FFT2D(target, SIZE * 2, SIZE * 2, -1);
	}
}


inline void FFT(std::complex<float>* Fdata, int n, int sign)
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

int main(int argc, char* argv[]) {
	cv::Mat noisy = cv::imread("C:/Users/Michele/Downloads/c.png");
	int flag = atoi(argv[1]); //3 pffft - 2 pocketfft - 1 TJ FFT - 0 OpenCV
	int nsmooth = atoi(argv[2]);
	bool fast = atoi(argv[3]); //fast convolve flaf for pffft ( skip reordering of z domain)
	Test(noisy, flag, nsmooth, fast);
	noisy.release();

	_CrtDumpMemoryLeaks();
}