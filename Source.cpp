#include <complex>
#include <cmath>
#include <vector>
#include <stdio.h>
#include <conio.h>
#include <ppl.h>
#include <execution>
#include <iostream> 
#include <pffft.hpp>
//suppose a max size of 4080px, using a kernel 11*11 and extra padd for FFT, using cache will make faster pocketfft
#define POCKETFFT_CACHE_SIZE 4500
#include "pocketfft_hdronly.h"
#include "fast_gaussian_blur_template.h"
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

#define M_PI		3.14159265358979323846

int gaussian_window(const double sigma, const int max_width = 0) {

	//return an odd width for the kernel, if max is passed check that is not bigger than it

	const float radius = sigma * sqrt(2 * log(255)) - 1;
	int width = radius * 2 + .5f;
	if (max_width) width = std::min(width, max_width);

	if (width % 2 == 0) ++width;

	printf("sigma %f radius %f - width %d - max_width %d\n", sigma, radius, width, max_width);

	return width;
}

void getGaussian(std::vector<float>& kernel, const double sigma, int width = 0, int FFT_length = 0)
{

	if (!width) width = gaussian_window(sigma);

	kernel.resize(FFT_length ? FFT_length : width);

	const float mid_w = (width - 1) / 2.f;
	const double s = 2. * sigma * sigma;

	int i = 0;

	for (float y = -mid_w; y <= mid_w; ++y, ++i)
		kernel[i] = (exp(-(y * y) / s)) / (M_PI * s);

	const double sum = 1. / std::accumulate(&kernel[0], &kernel[width], 0.);
	for (int i = 0; i < width; ++i) kernel[i] *= sum;
	printf("\nsum %f\n", std::accumulate(&kernel[0], &kernel[width], 0.));

	//fit the kernel in the FFT_length and shift the center to avoid circular convolution
	if (FFT_length)
		std::rotate(kernel.rbegin(), kernel.rbegin() + (kernel.size() - width / 2), kernel.rend());
}


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

int isValidSize(int N) {
	const int N_min = 32;
	int R = N;
	while (R >= 5 * N_min && (R % 5) == 0)  R /= 5;
	while (R >= 3 * N_min && (R % 3) == 0)  R /= 3;
	while (R >= 2 * N_min && (R % 2) == 0)  R /= 2;
	return (R == N_min) ? 1 : 0;
}

int nearestTransformSize(int N) {
	const int N_min = 32;
	if (N < N_min) N = N_min;
	N = N_min * ((N + N_min - 1) / N_min);

	while (!isValidSize(N)) N += N_min;
	return N;
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
/*
template<typename T, int C> void flip_block(const T* in, T* out, const int w, const int h)
{
	constexpr int block = 256 / C;
#pragma omp parallel for collapse(2)
	for (int x = 0; x < w; x += block)
		for (int y = 0; y < h; y += block)
		{
			const T* p = in + y * w * C + x * C;
			T* q = out + y * C + x * h * C;

			const int blockx = std::min(w, x + block) - x;
			const int blocky = std::min(h, y + block) - y;
			for (int xx = 0; xx < blockx; xx++)
			{
				for (int yy = 0; yy < blocky; yy++)
				{
					for (int k = 0; k < C; k++)
						q[k] = p[k];
					p += w * C;
					q += C;
				}
				p += -blocky * w * C + C;
				q += -blocky * C + h * C;
			}
		}
}
*/

void pocketfft_(cv::Mat image, int nsmooth)
{
	//pocketfft can handle non-small prime numbers decomposition of ndata but becomes slower, pffft cannot handle them and force you to add more pad

	image.convertTo(image, CV_32FC3);
	int passes = 2; //we are using a kernel convolved with itself, is like running two passes of the original kernel
	int pad = (nsmooth * nsmooth - 1) / 2 * passes;
	//top - bottom - left - right margin
	int border[4] = { pad, pad, pad, pad };

	//absolute min padding to fit the kernel
	size_t sizes[2] = { image.size[0] + border[0] + border[1], image.size[1] + border[2] + border[3] };

	//if the length of the data is not decomposable in small prime numbers 2 - 3 - 5, is necessary to update the size adding more pad
	for (int i = 0; i < 2; ++i) {
		if (!isValidSize(sizes[i]))
		{
			//printf("pre %d %d %d %d %d %d %d\n", sizes[0], sizes[1], border[0], border[1], border[2], border[3], pffft::Fft<float>::isValidSize(sizes[i]));
			int new_size = nearestTransformSize(sizes[i]);
			int new_pad = (new_size - sizes[i]);
			sizes[i] = new_size;
			border[i * 2 + 0] += new_pad / 2; //floor - fix for new_pad when not even
			border[i * 2 + 1] += new_pad / 2.f + 0.5f; //ceil if odd - fix for new_pad when not even

			//printf("post %d %d %d %d %d %d %d\n", sizes[0], sizes[1], border[0], border[1], border[2], border[3], pffft::Fft<float>::isValidSize(sizes[i]));
		}
	}

	cv::copyMakeBorder(image, image, border[0], border[1], border[2], border[3], CV_HAL_BORDER_REFLECT);
	//printf("%d %d\n", image.size[0], image.size[1]);

	cv::Mat temp[3];
	cv::split(image, temp);

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
	pocketfft::shape_t shape_col{ sizes[0] };
	pocketfft::shape_t shape_row{ sizes[1] };

	std::vector<std::complex<float>> kerf_1D_col(sizes[0] / 2 + 1);
	std::complex<float>* kerf_1D_row;
	float* kernel_1D_col = new float[sizes[0]]();
	make_kernel_1D(kernel_1D_col, nsmooth * nsmooth, shape_col[0]);
	pocketfft::r2c(shape_col, strided_1D, strided_out_1D, axes_1D, pocketfft::FORWARD, kernel_1D_col, kerf_1D_col.data(), 1.f, 0);
	delete[] kernel_1D_col;

	//since col will be iterated for a length of sizes[0] but its length ends at (sizes[0]/2 + 1), we resize and reflect, with index kerf_1D_col[sizes[0] / 2 + 1] as pivot
	//explanation "In case of real (single-channel) data, the output spectrum of the forward Fourier transform or input spectrum of the inverse Fourier transform can be represented in a packed format called CCS (complex-conjugate-symmetrical)" from - OpenCV Documentation
	kerf_1D_col.resize(sizes[0]);
	std::copy_n(kerf_1D_col.rbegin() + (sizes[0] / 2.f + .5f), kerf_1D_col.size() - (sizes[0] / 2 + 1), kerf_1D_col.begin() + (sizes[0] / 2 + 1));

	//calculate kernel for row dimension and his DFT if the length of rows is not the same of cols
	if (sizes[0] != sizes[1]) {
		kerf_1D_row = new std::complex<float>[sizes[1] / 2 + 1];
		float* kernel_1D_row = new float[sizes[1]]();
		make_kernel_1D(kernel_1D_row, nsmooth * nsmooth, shape_row[0]);
		pocketfft::r2c(shape_row, strided_1D, strided_out_1D, axes_1D, pocketfft::FORWARD, kernel_1D_row, kerf_1D_row, 1.f, 0);
		delete[] kernel_1D_row;
	}
	else
		kerf_1D_row = kerf_1D_col.data();

	int ndata = sizes[0] * sizes[1];

#pragma omp parallel for
	for (int i = 0; i < 3; ++i) {
		std::complex<float>* resf = new std::complex<float>[sizes[0] * (sizes[1] / 2 + 1)];
		pocketfft::r2c(shape, strided_in, strided_out, axes, pocketfft::FORWARD, (float*)temp[i].data, resf, 1.f, 0);

		// mul image_FFT with kernel_1D_row and kernel_1D_col 
		for (int i = 0; i < sizes[0]; ++i) {
			for (int j = 0; j < (sizes[1] / 2 + 1); ++j) {
				//multiply only for the real part since the imaginary part of a centered kernel is 0, no performance improvements found
				resf[i * (sizes[1] / 2 + 1) + j] *= std::real(kerf_1D_row[j]) * std::real(kerf_1D_col[i]);
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
		delete[] kerf_1D_row;

	printf("PocketFFT: %f\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_0).count());
	cv::merge(temp, 3, image);

	image.convertTo(image, CV_8UC3);
	cv::Rect myroi(border[2], border[0], sizes[1] - border[2] - border[3], sizes[0] - border[0] - border[1]);
	cv::imwrite("C:/Users/Michele/Downloads/c_.png", image(myroi));

}


void pocketfft_1D(cv::Mat image, double nsmooth)
{
	image.convertTo(image, CV_32FC3);

	double sigma = nsmooth;
	int kSize = gaussian_window(sigma, std::max(image.size[0], image.size[1]));

	int passes = 1; //we are using 1 pass of the gaussian kernel
	int pad = (kSize - 1) / 2 * passes;
	//top - bottom - left - right margin
	int border[4] = { pad, pad, pad, pad };

	//absolute min padding to fit the kernel
	size_t sizes[2] = { image.size[0] + border[0] + border[1], image.size[1] + border[2] + border[3] };

	//if the length of the data is not decomposable in small prime numbers 2 - 3 - 5, is necessary to update the size adding more pad
	for (int i = 0; i < 2; ++i) {
		if (!isValidSize(sizes[i]))
		{

			int new_size = nearestTransformSize(sizes[i]);
			int new_pad = (new_size - sizes[i]);
			sizes[i] = new_size;
			border[i * 2 + 0] += new_pad / 2; //floor - fix for new_pad when not even
			border[i * 2 + 1] += new_pad / 2.f + 0.5f; //ceil if odd - fix for new_pad when not even

		}
	}

	cv::copyMakeBorder(image, image, border[0], border[1], border[2], border[3], CV_HAL_BORDER_REFLECT);

	cv::Mat temp[3];
	cv::split(image, temp);

	start_0 = std::chrono::steady_clock::now();

	/* arguments settings */
	//kernel 2 x 1D
	pocketfft::stride_t strided_1D{ sizeof(float) };
	pocketfft::stride_t strided_out_1D{ sizeof(std::complex<float>) };
	pocketfft::shape_t axes_1D{ 0 };
	pocketfft::shape_t shape_col{ sizes[0] };
	pocketfft::shape_t shape_row{ sizes[1] };

	std::complex<float>* kerf_1D_col = new std::complex<float>[sizes[0] / 2 + 1];
	std::complex<float>* kerf_1D_row;
	std::vector<float> kernel_1D_col;
	getGaussian(kernel_1D_col, sigma, kSize, sizes[0]);
	printf("sizes[0] %d - width %d\n", sizes[0], kSize);

	pocketfft::r2c(shape_col, strided_1D, strided_out_1D, axes_1D, pocketfft::FORWARD, kernel_1D_col.data(), kerf_1D_col, 1.f, 0);

	//calculate kernel for row dimension and his DFT if the length of rows is not the same of cols
	if (sizes[0] != sizes[1]) {
		kerf_1D_row = new std::complex<float>[sizes[1] / 2 + 1];
		//float* kernel_1D_row = new float[sizes[1]]();
		//make_kernel_1D(kernel_1D_row, nsmooth * nsmooth, shape_row[0]);
		std::vector<float> kernel_1D_row;
		getGaussian(kernel_1D_row, sigma, kSize, sizes[1]);
		pocketfft::r2c(shape_row, strided_1D, strided_out_1D, axes_1D, pocketfft::FORWARD, kernel_1D_row.data(), kerf_1D_row, 1.f, 0);
		//delete[] kernel_1D_row;
	}
	else
		kerf_1D_row = kerf_1D_col;

	int ndata = sizes[0] * sizes[1];

	for (int i = 0; i < 3; ++i) {
		std::vector<float> resf(sizes[0] * sizes[1]);
#pragma omp parallel for
		for (int j = 0; j < sizes[0]; ++j) {
			std::vector<std::complex<float>> tile(sizes[1] / 2 + 1);
			pocketfft::r2c(shape_row, strided_1D, strided_out_1D, axes_1D, pocketfft::FORWARD, (float*)temp[i].data + j * sizes[1], tile.data(), 1.f, 0);
			for (int i = 0; i < sizes[1] / 2 + 1; ++i) tile[i] *= std::real(kerf_1D_row[i]); //since the imaginary part is null, ignore it
			pocketfft::c2r(shape_row, strided_out_1D, strided_1D, axes_1D, pocketfft::BACKWARD, tile.data(), (float*)temp[i].data + j * sizes[1], 1.f / sizes[1], 0);

		}
		flip_block<float, 1>((float*)temp[i].data, resf.data(), sizes[1], sizes[0]);

#pragma omp parallel for
		for (int j = 0; j < sizes[1]; ++j) {
			std::vector<std::complex<float>> tile(sizes[0] / 2 + 1);
			pocketfft::r2c(shape_col, strided_1D, strided_out_1D, axes_1D, pocketfft::FORWARD, resf.data() + j * sizes[0], tile.data(), 1.f, 0);
			for (int i = 0; i < sizes[0] / 2 + 1; ++i) tile[i] *= std::real(kerf_1D_col[i]); //since the imaginary part is null, ignore it
			pocketfft::c2r(shape_col, strided_out_1D, strided_1D, axes_1D, pocketfft::BACKWARD, tile.data(), resf.data() + j * sizes[0], 1.f / sizes[0], 0);
		}

		flip_block<float, 1>(resf.data(), (float*)temp[i].data, sizes[0], sizes[1]);

	}
	if (sizes[0] != sizes[1])
		delete[] kerf_1D_row;
	delete[] kerf_1D_col;

	printf("PocketFFT 1D experimental: %f\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_0).count());
	cv::merge(temp, 3, image);

	image.convertTo(image, CV_8UC3);
	cv::Rect myroi(border[2], border[0], sizes[1] - border[2] - border[3], sizes[0] - border[0] - border[1]);
	cv::imwrite("C:/Users/Michele/Downloads/c_.png", image(myroi));

}


void pffft_(cv::Mat image, int nsmooth, bool fast = true)
{

	image.convertTo(image, CV_32FC3);
	int passes = 2; //we are using a kernel convolved with itself, is like running two passes of the original kernel
	int pad = (nsmooth * nsmooth - 1) / 2 * passes;
	//top - bottom - left - right margin
	int border[4] = { pad, pad, pad, pad };

	//absolute min padd
	size_t sizes[2] = { image.size[0] + border[0] + border[1], image.size[1] + border[2] + border[3] };

	//if the length of the data is not decomposable in small prime numbers 2 - 3 - 5, is necessary to update the size adding more pad
	for (int i = 0; i < 2; ++i) {
		if (!pffft::Fft<float>::isValidSize(sizes[i]))
		{
			//printf("pre %d %d %d %d %d %d %d\n", sizes[0], sizes[1], border[0], border[1], border[2], border[3], pffft::Fft<float>::isValidSize(sizes[i]));
			int new_size = pffft::Fft<float>::nearestTransformSize(sizes[i]);
			int new_pad = (new_size - sizes[i]);
			sizes[i] = new_size;
			border[i * 2 + 0] += new_pad / 2; //floor - fix for new_pad when not even
			border[i * 2 + 1] += new_pad / 2.f + 0.5f; //ceil if odd - fix for new_pad when not even
			//printf("post %d %d %d %d %d %d %d\n", sizes[0], sizes[1], border[0], border[1], border[2], border[3], pffft::Fft<float>::isValidSize(sizes[i]));
		}
	}
	cv::copyMakeBorder(image, image, border[0], border[1], border[2], border[3], CV_HAL_BORDER_REFLECT);

	cv::Mat temp[3];
	cv::split(image, temp);

	start_0 = std::chrono::steady_clock::now();

	pffft::AlignedVector<float> kerf_1D_row;
	pffft::AlignedVector<float> kerf_1D_col;
	pffft::AlignedVector<std::complex<float>> kerf_complex;
	//fast convolve by pffft, without reordering the z-domain. Thus, we perform a row by row, col by col FFT and convolution with 2x1D kernel
	if (fast) {
		pffft::Fft<float> fft_kernel_1D_row(sizes[1]);
		pffft::AlignedVector<float> kernel_aligned_1D_row = pffft::AlignedVector<float>(fft_kernel_1D_row.getLength());

		make_kernel_1D(kernel_aligned_1D_row.data(), nsmooth * nsmooth, sizes[1]);

		kerf_1D_row = fft_kernel_1D_row.internalLayoutVector();
		fft_kernel_1D_row.forwardToInternalLayout(kernel_aligned_1D_row, kerf_1D_row);

		//calculate the DFT of kernel by col if size of cols is not the same of rows
		if (sizes[0] != sizes[1]) {
			pffft::Fft<float> fft_kernel_1D_col(sizes[0]);
			pffft::AlignedVector<float> kernel_aligned_1D_col = pffft::AlignedVector<float>(fft_kernel_1D_col.getLength());

			make_kernel_1D(kernel_aligned_1D_col.data(), nsmooth * nsmooth, sizes[0]);

			kerf_1D_col = fft_kernel_1D_col.internalLayoutVector();
			fft_kernel_1D_col.forwardToInternalLayout(kernel_aligned_1D_col, kerf_1D_col);

		}
		else {
			kerf_1D_col = kerf_1D_row;
		}

	}
	else {
		//2D DFT if not fast convolution requested
		pffft::Fft<float> fft_kernel(sizes[0] * sizes[1]);
		pffft::AlignedVector<float> kernel_aligned = pffft::AlignedVector<float>(fft_kernel.getLength());
		make_kernel(kernel_aligned.data(), nsmooth * nsmooth, sizes);
		kerf_complex = fft_kernel.spectrumVector();
		fft_kernel.forward(kernel_aligned, kerf_complex);
	}

	int ndata = sizes[0] * sizes[1];
	std::vector<pffft::AlignedVector<float>> resf;
	if (!fast)
		for (int i = 0; i < 3; ++i)
			resf.push_back(pffft::AlignedVector<float>(ndata) /* same as fft_kernel.valueVector() */);

#pragma omp parallel for
	for (int i = 0; i < 3; ++i) {
		//if fast don't do z-domain reordering, so it's faster than the normal process, is written in their APIs
		if (fast) {
			pffft::AlignedVector<float> resf(ndata);

			pffft::Fft<float> fft_rows(sizes[1]);
			pffft::AlignedVector<float> tile(sizes[1]);
			pffft::AlignedVector<float> work = fft_rows.internalLayoutVector();
			for (int j = 0; j < sizes[0]; ++j) {

				std::copy_n(&((float*)temp[i].data)[j * sizes[1]], sizes[1], tile.begin());

				fft_rows.forwardToInternalLayout(tile, work);
				fft_rows.convolve(work, kerf_1D_row, work, 1.f / sizes[1]);
				fft_rows.inverseFromInternalLayout(work, tile);
				//std::transform(tile.begin(), tile.end(), tile.begin(), [&sizes](auto i) {return i * (1.f / sizes[1]); });

				//save the 1st pass row by row in the output vector
				std::copy(tile.begin(), tile.end(), resf.begin() + j * sizes[1]);
			}

			//transpose cache-friendly, took from FastBoxBlur
			flip_block<float, 1>((float*)resf.data(), ((float*)temp[i].data), (int)sizes[1], (int)sizes[0]);

			pffft::Fft<float> fft_cols(sizes[0]);
			tile.resize(sizes[0]);
			work = fft_cols.internalLayoutVector();
			for (int j = 0; j < sizes[1]; ++j) {

				std::copy_n(&((float*)temp[i].data)[j * sizes[0]], sizes[0], tile.begin());

				fft_cols.forwardToInternalLayout(tile, work);
				fft_cols.convolve(work, kerf_1D_col, work, 1.f / sizes[0]);
				fft_cols.inverseFromInternalLayout(work, tile);
				//std::transform(tile.begin(), tile.end(), tile.begin(), [&sizes](auto i) {return i * (1.f / sizes[0]); });

				//save the 2nd pass col by col in the output vector
				std::copy(tile.begin(), tile.end(), resf.begin() + j * sizes[0]);

			}
			flip_block<float, 1>((float*)resf.data(), ((float*)temp[i].data), (int)sizes[0], (int)sizes[1]);
		}
		else {
			// convolve 2D(flattened) kernel with 2D(flattened) image in the frequency domain
			std::copy_n(&((float*)temp[i].data)[0], ndata, resf[i].begin());
			pffft::Fft<float> fft(ndata);
			pffft::AlignedVector<std::complex<float>> work = fft.spectrumVector();
			fft.forward(resf[i], work);
			transform(begin(kerf_complex), end(kerf_complex), begin(work), begin(work), [&ndata](auto& n, auto& m) { return n * m * (1.f / ndata); });
			fft.inverse(work, resf[i]);
			work.clear();
			temp[i] = cv::Mat(sizes[0], sizes[1], CV_32F, resf[i].data());
		}

	}

	printf("pffft: %f\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_0).count());
	cv::merge(temp, 3, image);

	image.convertTo(image, CV_8UC3);
	cv::Rect myroi(border[2], border[0], sizes[1] - border[2] - border[3], sizes[0] - border[0] - border[1]);
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

void Test(cv::Mat image, int flag = 0, double nsmooth = 5, bool fast_ = 1)
{

	start_0 = std::chrono::steady_clock::now();

	if (flag == 5) pocketfft_1D(image, nsmooth);
	else if (flag == 4) {

		//add extra pad by reflection to keep edges details
		int border[2] = { (nsmooth * nsmooth - 1), (nsmooth * nsmooth - 1) };
		size_t sizes[2] = { image.size[0] + border[0] * 2, image.size[1] + border[1] * 2 };
		cv::copyMakeBorder(image, image, border[0], border[0], border[1], border[1], CV_HAL_BORDER_REFLECT);

		std::chrono::time_point<std::chrono::steady_clock> start_5 = std::chrono::steady_clock::now();
		//prepare temp vector
		std::vector<uchar> image_out(sizes[1] * sizes[0] * 3);
		uchar* image_out_ptr = image_out.data();

		//smooth row by row, transpose, col by col and transpose back. smooth twice, as suggested 2 passes
		int passes = 2;
		for (int i = 0; i < passes; ++i) {
			horizontal_blur_extend<uchar, 3>(image.data, image_out_ptr, sizes[1], sizes[0], nsmooth * nsmooth);
			std::swap(image.data, image_out_ptr);
		}

		std::chrono::time_point<std::chrono::steady_clock> start_1 = std::chrono::steady_clock::now();
		flip_block<uchar, 3>(image.data, image_out_ptr, image.size[1], image.size[0]);
		std::chrono::time_point<std::chrono::steady_clock> start_2 = std::chrono::steady_clock::now();

		for (int i = 0; i < passes; ++i) {
			horizontal_blur_extend<uchar, 3>(image_out_ptr, image.data, sizes[0], sizes[1], nsmooth * nsmooth);
			std::swap(image.data, image_out_ptr);
		}

		std::chrono::time_point<std::chrono::steady_clock> start_3 = std::chrono::steady_clock::now();
		flip_block<uchar, 3>(image_out_ptr, image.data, image.size[0], image.size[1]);
		std::chrono::time_point<std::chrono::steady_clock> start_4 = std::chrono::steady_clock::now();

		//printf("1st transp : %f\n", std::chrono::duration<double, std::milli>(start_2 - start_1).count());
		//printf("2nd transp : %f\n", std::chrono::duration<double, std::milli>(start_4 - start_3).count());
		printf("(horizontalblur only, tot - transp) : %f\n", std::chrono::duration<double, std::milli>(start_4 - start_5).count() - std::chrono::duration<double, std::milli>(start_2 - start_1).count() - std::chrono::duration<double, std::milli>(start_4 - start_3).count());
		//printf("fastblur only: %f\n", std::chrono::duration<double, std::milli>(start_4 - start_5).count());
		//printf("fastblur with padding: %f\n", std::chrono::duration<double, std::milli>(start_4 - start_0).count());

		//save image cropped
		cv::Rect myroi(border[1], border[0], sizes[1] - border[1] * 2, sizes[0] - border[0] * 2);
		cv::imwrite("C:/Users/Michele/Downloads/c_.png", image(myroi));

	}
	else if (flag == 3) pffft_(image, nsmooth, fast_);
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

		//nsmooth *= nsmooth;

		//cv::blur(image, image, cv::Size(nsmooth, nsmooth));
		//cv::blur(image, image, cv::Size(nsmooth, nsmooth));
		cv::GaussianBlur(image, image, cv::Size(0, 0), nsmooth);
		printf("OpenCV blur: %f\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_0).count());

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
	int flag = atoi(argv[1]); //5 pocketFFT 1D - 4 FastBoxBlur - 3 pffft - 2 pocketfft - 1 TJ FFT - 0 OpenCV
	double nsmooth = atof(argv[2]);
	bool fast_pffft = atoi(argv[3]); //only for pffft: fast convolve flag ( skip reordering of z domain)

	cv::Mat noisy = cv::imread(file);
	Test(noisy, flag, nsmooth, fast_pffft);
	noisy.release();
	_CrtDumpMemoryLeaks();
}