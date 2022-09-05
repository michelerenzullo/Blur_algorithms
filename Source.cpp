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

void make_kernel_1D(float* kernel, const int kLen, const size_t iFTsize)
{
	const double scale = 1. / pow(kLen, 4); 
	for (int irow = -kLen + 1; irow <= (kLen - 1); irow++)
	{
		for (int icol = -kLen + 1; icol <= (kLen + 1); icol++) {
			double kval = (kLen - abs(irow)) * (kLen - abs(icol));
			kernel[(icol + iFTsize) % iFTsize] += std::clamp(kval * scale, 0., 1.);
		}
	}
	double rank = 0.;
	for (int i = 0; i < iFTsize; ++i) rank += kernel[i];
	printf("rank %.3f - crc32: %02X\n", rank, crc32c((uint8_t*)kernel, iFTsize * sizeof(float)));
}

void make_kernel(float* kernel, int kLen, const size_t iFTsize[])
{
	const double scale = 1. / pow(kLen, 4); 
	for (int irow = -kLen + 1; irow <= (kLen - 1); irow++)
	{
		for (int icol = -kLen + 1; icol <= (kLen - 1); icol++)
		{
			double kval = ((kLen - abs(irow)) * (kLen - abs(icol)));
			int rval = (irow + iFTsize[0]) % iFTsize[0];
			int cval = (icol + iFTsize[1]) % iFTsize[1];
			kernel[rval * iFTsize[1] + cval] += kval * scale;
		}
	}
	printf("crc32: %02X\n", crc32c((uint8_t*)kernel, iFTsize[0] * iFTsize[1] * sizeof(float)));
}


void pocketfft_(cv::Mat image, int nsmooth)
{
	//pocketfft can handle non-small prime numbers decomposition of ndata but becomes slower, pffft cannot handle them and force you to add more pad

	image.convertTo(image, CV_32FC3);
	size_t pad[2] = { nsmooth * nsmooth - 1, nsmooth * nsmooth - 1 };
	//pad = 0;

	size_t sizes[2] = { image.size[0] + pad[0] * 2, image.size[1] + pad[1] * 2 };

	//if the length of the data is not decomposable in small prime numbers 2 - 3 - 5, is necessary to update the size adding more pad
	if (!pffft::Fft<float>::isValidSize(sizes[0] * sizes[1])) {
		//printf("%d %d %d %d %d %d\n", sizes[0], sizes[1], pad[0], pad[1], sizes[0] * sizes[1], pffft::Fft<float>::isValidSize(sizes[0] * sizes[1]));
		pad[0] += (pffft::Fft<float>::nearestTransformSize(sizes[0]) - sizes[0]) / 2;
		pad[1] += (pffft::Fft<float>::nearestTransformSize(sizes[1]) - sizes[1]) / 2;
		sizes[0] = pffft::Fft<float>::nearestTransformSize(sizes[0]);
		sizes[1] = pffft::Fft<float>::nearestTransformSize(sizes[1]);

		//printf("%d %d %d %d %d %d\n", sizes[0], sizes[1], pad[0], pad[1], sizes[0] * sizes[1], pffft::Fft<float>::isValidSize(sizes[0] * sizes[1]));

	}
	cv::copyMakeBorder(image, image, pad[0], pad[0], pad[1], pad[1], CV_HAL_BORDER_REFLECT);

	cv::Mat temp[3];
	cv::split(image, temp);

	int ndata = sizes[0] * sizes[1];

	start_0 = std::chrono::steady_clock::now();

	/* arguments settings */
	pocketfft::shape_t shape{ sizes[0] , sizes[1] };
	pocketfft::stride_t strided_in{ (ptrdiff_t)(sizeof(float) * sizes[1]), sizeof(float) };
	pocketfft::stride_t strided_out{ (ptrdiff_t)(sizeof(std::complex<float>) * (sizes[1] / 2 + 1)), sizeof(std::complex<float>) };
	pocketfft::shape_t axes{ 0, 1 }; //argument setting to perform a 2D FFT so we can use 2 threads, instead of 1 . Output is the same, but faster

	//kernel 2 x 1D
	pocketfft::stride_t strided_1D{ sizeof(float) };
	pocketfft::stride_t strided_out_1D{ sizeof(std::complex<float>) };
	pocketfft::shape_t axes_1D{ 0 };
	pocketfft::shape_t shape_row{ sizes[0] };
	pocketfft::shape_t shape_col{ sizes[1] };

	std::complex<float>* kerf_1D_row = new std::complex<float>[sizes[0] / 2 + 1];
	std::complex<float>* kerf_1D_col;
	float* kernel_1D_row = new float[sizes[0]]();
	make_kernel_1D(kernel_1D_row, nsmooth * nsmooth, shape_row[0]);
	pocketfft::r2c(shape_row, strided_1D, strided_out_1D, axes_1D, pocketfft::FORWARD, kernel_1D_row, kerf_1D_row, 1.f, 0);
	delete[] kernel_1D_row;

	if (sizes[0] != sizes[1]) {
		kerf_1D_col = new std::complex<float>[sizes[1] / 2 + 1];
		float* kernel_1D_col = new float[sizes[1]]();
		make_kernel_1D(kernel_1D_col, nsmooth * nsmooth, shape_col[0]);
		pocketfft::r2c(shape_col, strided_1D, strided_out_1D, axes_1D, pocketfft::FORWARD, kernel_1D_col, kerf_1D_col, 1.f, 0);
		delete[] kernel_1D_col;
	}
	else
		kerf_1D_col = kerf_1D_row;

#pragma omp parallel for
	for (int i = 0; i < 3; ++i) {
		std::complex<float>* resf = new std::complex<float>[sizes[0] * (sizes[1] / 2 + 1)];
		pocketfft::r2c(shape, strided_in, strided_out, axes, pocketfft::FORWARD, (float*)temp[i].data, resf, 1.f, 0);

		// mul image_FFT with kernel_1D_row and kernel_1D_col
		for (int i = 0; i < sizes[0]; ++i) {
			for (int j = 0; j < (sizes[1] / 2 + 1); ++j) {
				resf[i * (sizes[1] / 2 + 1) + j] *=
					/* pocketfft produce as length of the fft transformed last axes (size[0] / 2 + 1 ) but because "i" is going till "size[0]" there are missing values, in practice this value are just the reflection, so we read decreasing*/
					((i < (sizes[0] / 2 + 1)) ? kerf_1D_row[i] : kerf_1D_row[(sizes[0] / 2) - i % (sizes[0] / 2)])
					* kerf_1D_col[j];
			}
		}

		//inverse the FFT
		pocketfft::c2r(shape, strided_out, strided_in, axes, pocketfft::BACKWARD, resf, (float*)temp[i].data, 1.f / ndata, 0);

		//just for debugging to print the FFT image
		/*
		for (int row = 0; row < sizes[0]; ++row)
			for (int col = 0; col < sizes[1]; ++col) {
				//FFTSHIFT DC - odd/even treated as in matlab
				int row_ = (row + (sizes[0] % 2 == 0 ? sizes[0] : (sizes[0] + 1)) / 2) % sizes[0];
				int col_ = (col + (sizes[1] % 2 == 0 ? sizes[1] : (sizes[1] + 1)) / 2) % sizes[1];
				//Reverse reading from end to the beginning after reached (sizes[1] / 2 + 1)
				int cval = col_ < (sizes[1] / 2 + 1) ? col_ : ((sizes[1] / 2) - col_ % (sizes[1] / 2));
				((float*)temp[i].data)[row * sizes[1] + col] = 
					20 * log10(abs(
						std::real(resf[row_ * (sizes[1] / 2 + 1) + cval])
						+ 0.01));
			}*/
		delete[] resf;
	}
	if (sizes[0] != sizes[1])
		delete[] kerf_1D_col;
	delete[] kerf_1D_row;

	printf("PocketFFT: %f\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_0).count());
	cv::merge(temp, 3, image);

	image.convertTo(image, CV_8UC3);
	cv::Rect myroi(pad[1], pad[0], sizes[1] - pad[1] * 2, sizes[0] - pad[0] * 2);
	cv::imwrite("C:/Users/Michele/Downloads/c_.png", image(myroi));

}

void pffft_(cv::Mat image, int nsmooth, bool fast = true)
{

	image.convertTo(image, CV_32FC3);
	size_t pad[2] = { nsmooth * nsmooth - 1 , nsmooth * nsmooth - 1 }; //minimal required pad to fix the circular convolution
	size_t sizes[2] = { image.size[0] + pad[0] * 2, image.size[1] + pad[1] * 2 };
	size_t ndata = sizes[0] * sizes[1];

	//if the length of the data is not decomposable in small prime numbers 2 - 3 - 5, is necessary to update the size adding more pad
	if (!pffft::Fft<float>::isValidSize(ndata)) {
		//printf("%d %d %d %d %d %d\n", sizes[0], sizes[1], pad[0], pad[1], ndata, pffft::Fft<float>::isValidSize(ndata));
		pad[0] += (pffft::Fft<float>::nearestTransformSize(sizes[0]) - sizes[0]) / 2;
		pad[1] += (pffft::Fft<float>::nearestTransformSize(sizes[1]) - sizes[1]) / 2;
		sizes[0] = pffft::Fft<float>::nearestTransformSize(sizes[0]);
		sizes[1] = pffft::Fft<float>::nearestTransformSize(sizes[1]);
		ndata = sizes[0] * sizes[1];
		//printf("%d %d %d %d %d %d\n", sizes[0], sizes[1], pad[0], pad[1], ndata, pffft::Fft<float>::isValidSize(ndata));

	}
	cv::copyMakeBorder(image, image, pad[0], pad[0], pad[1], pad[1], CV_HAL_BORDER_REFLECT);

	pffft::Fft<float> fft_kernel(ndata);
	pffft::AlignedVector<float> kernel_aligned = pffft::AlignedVector<float>(fft_kernel.getLength());

	make_kernel(kernel_aligned.data(), nsmooth * nsmooth, sizes);

	cv::Mat temp[3];
	cv::split(image, temp);

	start_0 = std::chrono::steady_clock::now();

	bool fast_convolve = fast;

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
		//if fast_convolve no z-domain reordering, so it's faster than the normal process, is written in their APIs
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
	AddPadding(tArray, rt, ct, cv::max(rk / 2, ck / 2));

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
		nreflects = std::min(nreflects, std::min(rval, cval) - 2);
		printf("%d %d %d %d %d\n", rval, cval, nrows, ncols, nreflects);

		for (int irow = rval - nreflects; irow < (rval + nrows + nreflects); irow++)
		{
			for (int c = cval - nreflects, c2 = cval + nreflects; c < cval; c++, c2--)
			{
				block[irow][c] = block[irow][c2];
				block[irow][c2 + ncols - 1] = block[irow][c + ncols - 1];
			}
		}

		for (int icol = cval - nreflects; icol < (cval + ncols + nreflects); icol++)
		{
			for (int r = rval - nreflects, r2 = rval + nreflects; r < rval; r++, r2--)
			{
				block[r][icol] = block[r2][icol];
				block[r2 + nrows - 1][icol] = block[r + nrows - 1][icol];
			}
		}
	}
}

void FFT2D(std::complex<float>(*array)[SIZE * 2], int nrows, int ncols, int sign)
{
	FFT((std::complex<float> *)array, SIZE * SIZE * 4, sign);
	printf("ex FFT2D %f\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_0).count());

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
	char* file = argv[4];
	cv::Mat noisy = cv::imread(file);
	int flag = atoi(argv[1]); //3 pffft - 2 pocketfft - 1 TJ FFT - 0 OpenCV
	int nsmooth = atoi(argv[2]);
	bool fast = atoi(argv[3]); //fast convolve flaf for pffft ( skip reordering of z domain)
	Test(noisy, flag, nsmooth, fast);
	noisy.release();

	_CrtDumpMemoryLeaks();
}