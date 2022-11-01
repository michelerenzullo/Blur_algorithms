#include <numeric>
// #include <iostream>
#include "pffft/pffft.h"
#include "Utils.hpp"
// #include <execution>
// suppose L2 Cache size of 256KB / sizeof(pocketfft_r<float>) --> 256KB / 24
#define POCKETFFT_CACHE_SIZE 10922
#include "pocketfft/pocketfft_hdronly.h"
#include "FastBoxBlur/fast_box_blur.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo/photo.hpp>


#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
// Replace _NORMAL_BLOCK with _CLIENT_BLOCK if you want the
// allocations to be of _CLIENT_BLOCK type
#else
#define DBG_NEW new
#endif


// #define boxblur // if defined all the tests will be performed for a Box Blur
// #define DFT_image // just for pocketfft_2D

template<typename T> using AlignedVector = typename std::vector< T, PFAlloc<T> >;

int gaussian_window(const double sigma, const int max_width = 0) {
	// calculate the width necessary for the provided sigma
	// return an odd width for the kernel, if max is passed check that is not bigger than it

	const float radius = sigma * sqrt(2 * log(255)) - 1;
	int width = radius * 2 + .5f;
	if (max_width) width = std::min(width, max_width);

	if (width % 2 == 0) ++width;

	//printf("sigma %f radius %f - width %d - max_width %d\n", sigma, radius, width, max_width);

	return width;
}

template<typename T>
void getGaussian(T& kernel, const double sigma, int width = 0, int FFT_length = 0)
{
	// create a 1D and zero padded gaussian kernel
	if (!width) width = gaussian_window(sigma);

	kernel.resize(FFT_length ? FFT_length : width);

	const float mid_w = (width - 1) / 2.f;
	const double s = 2. * sigma * sigma;

	int i = 0;

	for (float y = -mid_w; y <= mid_w; ++y, ++i)
		kernel[i] = (exp(-(y * y) / s)) / (3.14159265358979323846 * s);

	const double sum = 1. / std::accumulate(&kernel[0], &kernel[width], 0.);

	std::transform(&kernel[0], &kernel[width], &kernel[0], [&sum](auto& i) {return i * sum; });

	// fit the kernel in the FFT_length and shift the center to avoid circular convolution
	if (FFT_length) {
		// same of with raw pointers
		// std::rotate(std::reverse_iterator(&kernel[0] + FFT_length), std::reverse_iterator(&kernel[0] + width / 2), std::reverse_iterator(&kernel[0]));
		std::rotate(kernel.rbegin(), kernel.rbegin() + (kernel.size() - width / 2), kernel.rend());
	}

}

// Box kernel using Fourier Transform is not used anymore since has been replaced 
// by FastBoxBlur with a sliding accumulator, I left just for documentation. 
// In the frequency domain we use a true Gaussian kernel

void box_kernel(float* kernel, int kLen, const size_t FFT_length[2])
{
	// Create a 2D box blur kernel convolved by itself - aka. 2 passes of BoxBlur (Tent filter)
	// NOTE: This is unusued because the box kernel can be decomposed, so there is not
	// need to calculate the DFT of a big 2D kernel when we can just compute for a row
	// or a columm
	const double scale = 1. / pow(kLen, 4);
	for (int irow = -kLen + 1; irow <= (kLen - 1); irow++)
	{
		for (int icol = -kLen + 1; icol <= (kLen - 1); icol++)
		{
			const double kval = (kLen - abs(irow)) * (kLen - abs(icol));
			const int rval = (irow + FFT_length[0]) % FFT_length[0];
			const int cval = (icol + FFT_length[1]) % FFT_length[1];
			kernel[rval * FFT_length[1] + cval] += std::clamp(kval * scale, 0., 1.);
		}
	}

}


void box_kernel(float* kernel, const int kLen, const size_t FFT_length)
{
	// Create a 1D box blur kernel convolved by itself - aka. 2 passes of BoxBlur
	const double scale = 1. / pow(kLen, 4);
	for (int irow = -kLen + 1; irow <= (kLen - 1); irow++)
	{
		for (int icol = -kLen + 1; icol <= (kLen + 1); icol++) {
			const double kval = (kLen - abs(irow)) * (kLen - abs(icol));
			kernel[(icol + FFT_length) % FFT_length] += std::clamp(kval * scale, 0., 1.);
		}
	}
}


void pocketfft_2D(cv::Mat& image, double nsmooth)
{
	std::chrono::time_point<std::chrono::steady_clock> start_0 = std::chrono::steady_clock::now();
	// pocketfft can handle non-small prime numbers decomposition of ndata but becomes slower, pffft cannot handle them and force you to add more pad

	double sigma = nsmooth;
	int kSize = gaussian_window(sigma, std::max(image.size[0], image.size[1]));

	int passes = 1; // we are using 1 pass of the gaussian kernel
#ifdef boxblur
	nsmooth = sqrt(std::min((int)nsmooth * (int)nsmooth, std::min(image.size[0] - 1, image.size[1] - 1)));
	passes = 2; // we are using a the box blur kernel that is convolved by itself, so it's like we are doing 2 passes
	kSize = nsmooth * nsmooth;
#endif
	int pad = (kSize - 1) / 2 * passes;

	// top - bottom - left - right margin
	int border[4] = { pad, pad, pad, pad };

	// absolute min padding to fit the kernel
	size_t sizes[2] = { image.size[0] + border[0] + border[1], image.size[1] + border[2] + border[3] };

	// if the length of the data is not decomposable in small prime numbers 2 - 3 - 5, is necessary to update the size adding more pad
	for (int i = 0; i < 2; ++i) {
		if (!isValidSize(sizes[i]))
		{
			int new_size = nearestTransformSize(sizes[i]);
			int new_pad = (new_size - sizes[i]);
			sizes[i] = new_size;
			border[i * 2 + 0] += new_pad / 2; // floor - fix for new_pad when not even
			border[i * 2 + 1] += new_pad / 2.f + 0.5f; // ceil if odd - fix for new_pad when not even

		}
	}


	int original_size[2] = { image.size[0], image.size[1] };
	auto padded = std::make_unique<uint8_t[]>(sizes[0] * sizes[1] * 3);
	Reflect_101<uint8_t, 3>((const uint8_t* const)image.data, padded.get(), border[0], border[1], border[2], border[3], original_size);

	// cv::copyMakeBorder(image, image, border[0], border[1], border[2], border[3], CV_HAL_BORDER_REFLECT_101);
	// printf("%d %d\n", image.size[0], image.size[1]);

	std::vector<std::vector<float>> temp(3, std::vector<float>(sizes[0] * sizes[1]));
	float* BGR[3] = { temp[0].data(), temp[1].data(), temp[2].data() };
	deinterleave_BGR((const uint8_t*)padded.get(), BGR, sizes[0] * sizes[1]);


	/* arguments settings */
	pocketfft::shape_t shape{ sizes[0] , sizes[1] };
	pocketfft::stride_t strided_in{ (ptrdiff_t)(sizeof(float) * sizes[1]), sizeof(float) };
	pocketfft::stride_t strided_out{ (ptrdiff_t)(sizeof(std::complex<float>) * (sizes[1] / 2 + 1)), sizeof(std::complex<float>) };
	pocketfft::shape_t axes{ 0, 1 }; // argument setting to perform a 2D FFT so we can use 2 threads, instead of 1 . Output is the same, but faster

	// kernel 2 x 1D
	pocketfft::stride_t strided_1D{ sizeof(float) };
	pocketfft::stride_t strided_out_1D{ sizeof(std::complex<float>) };
	pocketfft::shape_t axes_1D{ 0 };
	pocketfft::shape_t shape_col{ sizes[0] };
	pocketfft::shape_t shape_row{ sizes[1] };

	auto kerf_1D_col = std::make_shared<std::complex<float>[]>(sizes[0] /* / 2 + 1 */);
	std::shared_ptr<std::complex<float>[]> kerf_1D_row;
	std::vector<float> kernel_1D_col(sizes[0]);
#ifdef boxblur
	box_kernel(kernel_1D_col.data(), nsmooth * nsmooth, sizes[0]);
#else
	getGaussian(kernel_1D_col, sigma, kSize, sizes[0]);
#endif
	pocketfft::r2c(shape_col, strided_1D, strided_out_1D, axes_1D, pocketfft::FORWARD, kernel_1D_col.data(), kerf_1D_col.get(), 1.f, 0);


	// since col will be iterated for a length of sizes[0] but its length ends at (sizes[0]/2 + 1), we resize and reflect, with index kerf_1D_col[sizes[0] / 2 + 1] as pivot (Nyquist frequency)
	// explanation "In case of real (single-channel) data, the output spectrum of the forward Fourier transform or input spectrum of the inverse Fourier transform can be represented in a packed format called CCS (complex-conjugate-symmetrical)" from - OpenCV Documentation
	// kerf_1D_col.resize(sizes[0]);
	std::copy_n(std::reverse_iterator(&kerf_1D_col.get()[(int)(sizes[0] / 2.f + .5f)]), sizes[0] - (sizes[0] / 2 + 1), kerf_1D_col.get() + (sizes[0] / 2 + 1));

	// calculate kernel for row dimension and his DFT if the length of rows is not the same of cols
	if (sizes[0] != sizes[1]) {
		kerf_1D_row = std::make_shared<std::complex<float>[]>(sizes[1] / 2 + 1);
		std::vector<float> kernel_1D_row(sizes[1]);
#ifdef boxblur
		box_kernel(kernel_1D_row.data(), nsmooth * nsmooth, sizes[1]);
#else
		getGaussian(kernel_1D_row, sigma, kSize, sizes[1]);
#endif
		pocketfft::r2c(shape_row, strided_1D, strided_out_1D, axes_1D, pocketfft::FORWARD, kernel_1D_row.data(), kerf_1D_row.get(), 1.f, 0);
	}
	else
		kerf_1D_row = kerf_1D_col;

	int ndata = sizes[0] * sizes[1];

#pragma omp parallel for
	for (int i = 0; i < 3; ++i) {
		auto resf = std::make_unique<std::complex<float>[]>(sizes[0] * (sizes[1] / 2 + 1));
		pocketfft::r2c(shape, strided_in, strided_out, axes, pocketfft::FORWARD, temp[i].data(), resf.get(), 1.f, 0);

#ifdef DFT_image
		for (int row = 0; row < sizes[0]; ++row)
			for (int col = 0; col < sizes[1]; ++col) {
				// FFTSHIFT DC - odd/even treated as in matlab
				int row_ = (row + (sizes[0] % 2 == 0 ? sizes[0] : (sizes[0] + 1)) / 2) % sizes[0];
				int col_ = (col + (sizes[1] % 2 == 0 ? sizes[1] : (sizes[1] + 1)) / 2) % sizes[1];
				// Reverse reading from end to the beginning after reached (sizes[1] / 2 + 1)
				int cval = col_ < (sizes[1] / 2 + 1) ? col_ : ((sizes[1] / 2) - col_ % (sizes[1] / 2));
				temp[i].data()[row * sizes[1] + col] =
					20 * log10(abs(
						std::real(resf[row_ * (sizes[1] / 2 + 1) + cval]))
						+ 0.00001f);
			}
#else
		// mul image_FFT with kernel_1D_row and kernel_1D_col 
		for (int i = 0; i < sizes[0]; ++i) {
			for (int j = 0; j < (sizes[1] / 2 + 1); ++j) {
				// multiply only for the real part since the imaginary part of a centered kernel is 0, no performance improvements found
				resf[i * (sizes[1] / 2 + 1) + j] *= std::real(kerf_1D_row[j]) * std::real(kerf_1D_col[i]);
			}
		}

		// inverse the FFT
		pocketfft::c2r(shape, strided_out, strided_in, axes, pocketfft::BACKWARD, resf.get(), temp[i].data(), 1.f / ndata, 0);
#endif
	}

	printf("PocketFFT 2D: %f\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_0).count());
	interleave_BGR((const float**)BGR, (uint8_t*)padded.get(), sizes[0] * sizes[1]);

	// crop the image to ignore reflected borders
#pragma omp parallel for
	for (int i = border[0]; i < sizes[0] - border[1]; ++i) {
		uint8_t* const row = image.data + (i - border[0]) * image.size[1] * image.channels();
		uint8_t* const row_padded = padded.get() + i * sizes[1] * image.channels();
		for (int j = border[2] * image.channels(); j < (sizes[1] - border[3]) * image.channels(); ++j)
			row[j - border[2] * image.channels()] = row_padded[j];
	}
}


void pocketfft_1D(cv::Mat& image, double nsmooth)
{
	std::chrono::time_point<std::chrono::steady_clock> start_0 = std::chrono::steady_clock::now();
	double sigma = nsmooth;
	int kSize = gaussian_window(sigma, std::max(image.size[0], image.size[1]));

	int passes = 1; // we are using 1 pass of the gaussian kernel
#ifdef boxblur
	nsmooth = sqrt(std::min((int)nsmooth * (int)nsmooth, std::min(image.size[0] - 1, image.size[1] - 1)));
	passes = 2; // we are using a the box blur kernel that is convolved by itself, so it's like we are doing 2 passes
	kSize = nsmooth * nsmooth;
#endif
	int pad = (kSize - 1) / 2 * passes;

	// absolute min padding to fit the kernel
	size_t sizes[2] = { image.size[0] + pad * 2, image.size[1] + pad * 2 };

	// if the length of the data is not decomposable in small prime numbers 2 - 3 - 5, is necessary to update the size adding more pad
	int trailing_zeros[2] = {};
	for (int i = 0; i < 2; ++i) {
		if (!isValidSize(sizes[i]))
		{
			int new_size = nearestTransformSize(sizes[i]);
			trailing_zeros[i] = (new_size - sizes[i]);
			sizes[i] = new_size;
		}
	}

	std::vector<std::vector<float>> temp(3, std::vector<float>(image.size[0] * image.size[1]));
	float* BGR[3] = { temp[0].data(), temp[1].data(), temp[2].data() };
	deinterleave_BGR((const uint8_t*)image.data, BGR, image.size[0] * image.size[1]);



	/* arguments settings */
	// kernel 2 x 1D
	pocketfft::stride_t strided_1D{ sizeof(float) };
	pocketfft::stride_t strided_out_1D{ sizeof(std::complex<float>) };
	pocketfft::shape_t axes_1D{ 0 };
	pocketfft::shape_t shape_col{ sizes[0] };
	pocketfft::shape_t shape_row{ sizes[1] };

	auto kerf_1D_col = std::make_shared<std::complex<float>[]>(sizes[0] / 2 + 1);
	std::shared_ptr<std::complex<float>[]> kerf_1D_row;
	std::vector<float> kernel_1D_col(sizes[0]);
#ifdef boxblur
	box_kernel(kernel_1D_col.data(), nsmooth * nsmooth, sizes[0]);
#else
	getGaussian(kernel_1D_col, sigma, kSize, sizes[0]);
#endif
	printf("sizes[0] %d - width %d\n", sizes[0], kSize);

	pocketfft::r2c(shape_col, strided_1D, strided_out_1D, axes_1D, pocketfft::FORWARD, kernel_1D_col.data(), kerf_1D_col.get(), 1.f, 0);

	// calculate kernel for row dimension and his DFT if the length of rows is not the same of cols
	if (sizes[0] != sizes[1]) {
		kerf_1D_row = std::make_shared<std::complex<float>[]>(sizes[1] / 2 + 1);
		std::vector<float> kernel_1D_row(sizes[1]);
#ifdef boxblur
		box_kernel(kernel_1D_row.data(), nsmooth * nsmooth, sizes[1]);
#else
		getGaussian(kernel_1D_row, sigma, kSize, sizes[1]);
#endif
		pocketfft::r2c(shape_row, strided_1D, strided_out_1D, axes_1D, pocketfft::FORWARD, kernel_1D_row.data(), kerf_1D_row.get(), 1.f, 0);
	}
	else
		kerf_1D_row = kerf_1D_col;

	int ndata = image.size[0] * image.size[1];

	for (int i = 0; i < 3; ++i) {
		auto resf = std::make_unique<float[]>(ndata);
#pragma omp parallel for
		for (int j = 0; j < image.size[0]; ++j) {
			auto tile = std::make_unique<float[]>(sizes[1]);
			auto work = std::make_unique<std::complex<float>[]>(sizes[1] / 2 + 1);

			std::copy_n(std::reverse_iterator(temp[i].data() + j * image.size[1] + pad + 1), pad, tile.get());
			std::copy_n(temp[i].data() + j * image.size[1], image.size[1], tile.get() + pad);
			std::copy_n(std::reverse_iterator(temp[i].data() + (j + 1) * image.size[1] - 1), pad, &(tile.get())[sizes[1]] - pad /* fft trailing 0s --> */ - trailing_zeros[1]);

			pocketfft::r2c(shape_row, strided_1D, strided_out_1D, axes_1D, pocketfft::FORWARD, tile.get(), work.get(), 1.f, 0);
			for (int i = 0; i < sizes[1] / 2 + 1; ++i) work[i] *= std::real(kerf_1D_row[i]); // since the imaginary part is null, ignore it
			pocketfft::c2r(shape_row, strided_out_1D, strided_1D, axes_1D, pocketfft::BACKWARD, work.get(), tile.get(), 1.f / sizes[1], 0);

			std::copy(tile.get() + pad, &(tile.get())[sizes[1]] - pad - trailing_zeros[1], resf.get() + j * image.size[1]);
		}
		flip_block<float, 1>(resf.get(), temp[i].data(), image.size[1], image.size[0]);

#pragma omp parallel for
		for (int j = 0; j < image.size[1]; ++j) {
			auto tile = std::make_unique<float[]>(sizes[0]);
			auto work = std::make_unique<std::complex<float>[]>(sizes[0] / 2 + 1);

			std::copy_n(std::reverse_iterator(temp[i].data() + j * image.size[0] + pad + 1), pad, tile.get());
			std::copy_n(temp[i].data() + j * image.size[0], image.size[0], tile.get() + pad);
			std::copy_n(std::reverse_iterator(temp[i].data() + (j + 1) * image.size[0] - 1), pad, &(tile.get())[sizes[0]] - pad /* fft trailing 0s --> */ - trailing_zeros[0]);

			pocketfft::r2c(shape_col, strided_1D, strided_out_1D, axes_1D, pocketfft::FORWARD, tile.get(), work.get(), 1.f, 0);
			for (int i = 0; i < sizes[0] / 2 + 1; ++i) work[i] *= std::real(kerf_1D_col[i]); // since the imaginary part is null, ignore it
			pocketfft::c2r(shape_col, strided_out_1D, strided_1D, axes_1D, pocketfft::BACKWARD, work.get(), tile.get(), 1.f / sizes[0], 0);

			std::copy(tile.get() + pad, &(tile.get())[sizes[0]] - pad - trailing_zeros[0], resf.get() + j * image.size[0]);
		}

		flip_block<float, 1>(resf.get(), temp[i].data(), image.size[0], image.size[1]);

	}


	interleave_BGR((const float**)BGR, (uint8_t*)image.data, image.size[0] * image.size[1]);
	printf("PocketFFT 1D : %f\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_0).count());

}

void pffft_(cv::Mat& image, double nsmooth)
{
	std::chrono::time_point<std::chrono::steady_clock> start_0 = std::chrono::steady_clock::now();
	double sigma = nsmooth;
	// calculate a good width of the kernel for our sigma
	int kSize = gaussian_window(sigma, std::max(image.size[0], image.size[1]));

	int passes = 1; // we are using 1 pass for the gaussian kernel
#ifdef boxblur
	nsmooth = sqrt(std::min((int)nsmooth * (int)nsmooth, std::min(image.size[0] - 1, image.size[1] - 1)));
	passes = 2; // we are using a the box blur kernel that is convolved by itself, so it's like we are doing 2 passes
	kSize = nsmooth * nsmooth;
#endif
	int pad = (kSize - 1) / 2 * passes;

	// absolute min padd
	size_t sizes[2] = { image.size[0] + pad * 2, image.size[1] + pad * 2 };

	// if the length of the data is not decomposable in small prime numbers 2 - 3 - 5, is necessary to update the size adding more pad as trailing zeros
	int trailing_zeros[2] = {};

	for (int i = 0; i < 2; ++i) {
		if (!isValidSize(sizes[i]))
		{
			int new_size = nearestTransformSize(sizes[i]);
			trailing_zeros[i] = (new_size - sizes[i]);
			sizes[i] = new_size;
		}
	}

	std::vector<std::vector<float>> temp(3, std::vector<float>(image.size[0] * image.size[1]));
	float* BGR[3] = { temp[0].data(), temp[1].data(), temp[2].data() };
	deinterleave_BGR((const uint8_t*)image.data, BGR, image.size[0] * image.size[1]);



	// fast convolve by pffft, without reordering the z-domain. Thus, we perform a row by row, col by col FFT and convolution with 2x1D kernel

	AlignedVector<float> kernel_aligned_1D_row(sizes[1]);

	// create a gaussian 1D kernel with the specified sigma and kernel size, and center it in a length of FFT_length
#ifdef boxblur
	box_kernel(kernel_aligned_1D_row.data(), nsmooth * nsmooth, sizes[1]);
#else
	getGaussian(kernel_aligned_1D_row, sigma, kSize, sizes[1]);
#endif

	AlignedVector<float> kerf_1D_row(sizes[1]);
	AlignedVector<float> kerf_1D_col;

	PFFFT_Setup* rows = pffft_new_setup(sizes[1], PFFFT_REAL);
	PFFFT_Setup* cols = pffft_new_setup(sizes[0], PFFFT_REAL);

	AlignedVector<float> tmp;
	const int maxsize = std::max(sizes[0], sizes[1]);
	tmp.reserve(maxsize);
	tmp.resize(sizes[1]);

	pffft_transform(rows, kernel_aligned_1D_row.data(), kerf_1D_row.data(), tmp.data(), PFFFT_FORWARD);

	// calculate the DFT of kernel by col if size of cols is not the same of rows
	if (sizes[0] != sizes[1]) {

		AlignedVector<float> kernel_aligned_1D_col(sizes[0]);
#ifdef boxblur
		box_kernel(kernel_aligned_1D_col.data(), nsmooth * nsmooth, sizes[0]);
#else
		getGaussian(kernel_aligned_1D_col, sigma, kSize, sizes[0]);
#endif

		kerf_1D_col.resize(sizes[0]);
		tmp.resize(sizes[0]);
		pffft_transform(cols, kernel_aligned_1D_col.data(), kerf_1D_col.data(), tmp.data(), PFFFT_FORWARD);

	}
	else
		kerf_1D_col = kerf_1D_row;

	const int ndata = image.size[0] * image.size[1];
	const float divisor_row = 1.f / sizes[1];
	const float divisor_col = 1.f / sizes[0];


	for (int i = 0; i < 3; ++i) {
		AlignedVector<float> resf(ndata);
		tmp.resize(sizes[1]);

		AlignedVector<float> tile;
		tile.reserve(maxsize);
		tile.resize(sizes[1]);

		AlignedVector<float> work;
		work.reserve(maxsize);
		work.resize(sizes[1]);

#pragma omp parallel for firstprivate(tile, work, tmp)
		for (int j = 0; j < image.size[0]; ++j) {

			// copy the row and pad by reflection in the aligned vector
			// left reflected pad
			std::copy_n(std::reverse_iterator(temp[i].data() + j * image.size[1] + pad + 1), pad, tile.begin());
			// middle
			std::copy_n(temp[i].data() + j * image.size[1], image.size[1], tile.begin() + pad);
			// right reflected pad
			std::copy_n(std::reverse_iterator(temp[i].data() + (j + 1) * image.size[1] - 1), pad, tile.end() - pad /* fft trailing 0s --> */ - trailing_zeros[1]);


			pffft_transform(rows, tile.data(), work.data(), tmp.data(), PFFFT_FORWARD);

			// When executing pffft_zconvolve_no_accu inside a parallel for something strange happens internally,
			// there a few pixels different of 1 compared to when omp parallel is commented out
			pffft_zconvolve_no_accu(rows, work.data(), kerf_1D_row.data(), work.data(), divisor_row);
			pffft_transform(rows, work.data(), tile.data(), tmp.data(), PFFFT_BACKWARD);

			// save the 1st pass row by row in the output vector
			std::copy_n(tile.begin() + pad, image.size[1], resf.begin() + j * image.size[1]);
		}

		// transpose cache-friendly, took from FastBoxBlur
		flip_block<float, 1>(resf.data(), temp[i].data(), image.size[1], image.size[0]);

		tmp.resize(sizes[0]);
		tile.resize(sizes[0]);
		work.resize(sizes[0]);

#pragma omp parallel for firstprivate(tile, work, tmp)
		for (int j = 0; j < image.size[1]; ++j) {

			std::copy_n(std::reverse_iterator(temp[i].data() + j * image.size[0] + pad + 1), pad, tile.begin());
			std::copy_n(temp[i].data() + j * image.size[0], image.size[0], tile.begin() + pad);
			std::copy_n(std::reverse_iterator(temp[i].data() + (j + 1) * image.size[0] - 1), pad, tile.end() - pad /* fft trailing 0s --> */ - trailing_zeros[0]);


			pffft_transform(cols, tile.data(), work.data(), tmp.data(), PFFFT_FORWARD);
			pffft_zconvolve_no_accu(cols, work.data(), kerf_1D_col.data(), work.data(), divisor_col);
			pffft_transform(cols, work.data(), tile.data(), tmp.data(), PFFFT_BACKWARD);

			// save the 2nd pass col by col in the output vector 
			std::copy_n(tile.begin() + pad, image.size[0], resf.begin() + j * image.size[0]);

		}

		flip_block<float, 1>(resf.data(), temp[i].data(), image.size[0], image.size[1]);

	}
	pffft_destroy_setup(cols);
	pffft_destroy_setup(rows);
	interleave_BGR((const float**)BGR, (uint8_t*)image.data, image.size[0] * image.size[1]);
	printf("pffft: %f\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_0).count());

}



void Test(cv::Mat& image, int flag = 0, double nsmooth = 5)
{

	std::chrono::time_point<std::chrono::steady_clock> start_0 = std::chrono::steady_clock::now();

	if (flag == 5) pocketfft_1D(image, nsmooth);
	else if (flag == 4) {

		size_t sizes[2] = { image.size[0], image.size[1] };

		std::chrono::time_point<std::chrono::steady_clock> start_5 = std::chrono::steady_clock::now();

		uint8_t* in = image.data;
		fastboxblur(in, sizes[1], sizes[0], image.channels(), nsmooth * nsmooth, 2);

		printf("fastboxblur: %f\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_0).count());

	}
	else if (flag == 3) pffft_(image, nsmooth);
	else if (flag == 2) pocketfft_2D(image, nsmooth);
	else
	{
#ifdef boxblur
		nsmooth = (int)nsmooth * (int)nsmooth;

		cv::blur(image, image, cv::Size(nsmooth, nsmooth));
		cv::blur(image, image, cv::Size(nsmooth, nsmooth));
#else
		cv::GaussianBlur(image, image, cv::Size(0, 0), nsmooth);
#endif
		printf("OpenCV blur: %f\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_0).count());


	}
}


int main(int argc, char* argv[]) {
	char* file = argv[3];
	/*
	5 pocketFFT 1D
	4 FastBoxBlur
	3 pffft 1D
	2 pocketfft 2D
	1 OpenCV
	*/
	int flag = atoi(argv[1]);
	double nsmooth = atof(argv[2]);

	cv::Mat test_image = cv::imread(file);
	Test(test_image, flag, nsmooth);
	/*
	// Benchmark test
	int x = 1500, y = 1000;
	for (int i = 0; i < 45; ++i) {
		cv::resize(test_image, test_image, cv::Size(y, x));
		Test(test_image, flag, sqrt(x));
		x += 225, y += 150;
		if (i == 0) cv::imwrite("C:/Users/miki/Downloads/c_.png", test_image);
	}*/
	cv::imwrite("C:/Users/miki/Downloads/c_.png", test_image);
	test_image.release();

#ifdef _DEBUG
	_CrtDumpMemoryLeaks();
#endif
}