#include <numeric>
// #include <iostream> 
#include <pffft.hpp>
// suppose L2 Cache size of 256KB / sizeof(pocketfft_r<float>) --> 256KB / 24
#define POCKETFFT_CACHE_SIZE 10922
#include "pocketfft_hdronly.h"
#include "fast_box_blur.h"
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

int gaussian_window(const double sigma, const int max_width = 0) {
	// calculate the width necessary for the provided sigma
	// return an odd width for the kernel, if max is passed check that is not bigger than it

	const float radius = sigma * sqrt(2 * log(255)) - 1;
	int width = radius * 2 + .5f;
	if (max_width) width = std::min(width, max_width);

	if (width % 2 == 0) ++width;

	printf("sigma %f radius %f - width %d - max_width %d\n", sigma, radius, width, max_width);

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

void box_kernel(float* kernel, int kLen, const size_t iFTsize[])
{
	// Create a 2D box blur kernel convolved by itself - aka. 2 passes of BoxBlur
	// NOTE: This is unusued because the box kernel can be decomposed, so there is not
	// need to calculate the DFT of a big 2D kernel when we can just compute for a row
	// or a columm
	const double scale = 1. / pow(kLen, 4);
	for (int irow = -kLen + 1; irow <= (kLen - 1); irow++)
	{
		for (int icol = -kLen + 1; icol <= (kLen - 1); icol++)
		{
			double kval = ((kLen - abs(irow)) * (kLen - abs(icol)));
			int rval = (irow + iFTsize[0]) % iFTsize[0];
			int cval = (icol + iFTsize[1]) % iFTsize[1];
			kernel[rval * iFTsize[1] + cval] += std::clamp(kval * scale, 0., 1.);
		}
	}

}

void box_kernel_1D(float* kernel, const int kLen, const size_t iFTsize)
{
	// Create a 1D box blur kernel convolved by itself - aka. 2 passes of BoxBlur
	const double scale = 1. / pow(kLen, 4);
	for (int irow = -kLen + 1; irow <= (kLen - 1); irow++)
	{
		for (int icol = -kLen + 1; icol <= (kLen + 1); icol++) {
			double kval = (kLen - abs(irow)) * (kLen - abs(icol));
			kernel[(icol + iFTsize) % iFTsize] += std::clamp(kval * scale, 0., 1.);
		}
	}
}

template<typename T, int C>
void Reflect_101(const T* const input, T* output, int pad_top, int pad_bottom, int pad_left, int pad_right, const int* original_size) {

	// This function padd a 2D matrix or a multichannel image with the specified top,bottom,left,right pad and it applies
	// a reflect 101 like cv::copyMakeBorder, the main (and only) difference is the following constraint to prevent out of buffer reading
	pad_top = std::min(pad_top, original_size[0] - 1);
	pad_bottom = std::min(pad_bottom, original_size[0] - 1);
	pad_left = std::min(pad_left, original_size[1] - 1);
	pad_right = std::min(pad_right, original_size[1] - 1);

	const int stride[2] = { original_size[0], original_size[1] * C };
	const int padded[2] = { stride[0] + pad_top + pad_bottom, stride[1] + (pad_left + pad_right) * C };
	const int right_offset = (original_size[1] + pad_left - 1) * 2 * C;
	const int left_offset = pad_left * 2 * C;
	const int bottom_offset = 2 * stride[0] + pad_top - 2;


#pragma omp parallel for
	for (int i = 0; i < padded[0]; ++i) {
		T* const row = output + i * padded[1];

		if (i < padded[0] - pad_bottom)
			std::copy_n(&input[stride[1] * abs(i - pad_top)], stride[1], &row[pad_left * C]);
		else
			std::copy_n(&input[stride[1] * (bottom_offset - i)], stride[1], &row[pad_left * C]);

		for (int j = 0; j < pad_left * C; j += C)
			std::copy_n(row + left_offset - j, C, row + j);

		for (int j = padded[1] - pad_right * C; j < padded[1]; j += C)
			std::copy_n(row + right_offset - j, C, row + j);
	}

}



// Utils from pffft to check the nearest efficient transform size of FFT
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

template<typename T, typename U>
void deinterleave_BGR(const T* const interleaved_BGR, U** const deinterleaved_BGR, const uint32_t nsize) {

	// Cache-friendly deinterleave BGR, splitting for blocks of 256 KB, inspired by flip-block
	constexpr float round = std::is_integral_v<U> ? std::is_integral_v<T> ? 0 : 0.5f : 0;
	const uint32_t block = 262144 / (3 * std::max(sizeof(T), sizeof(U)));

#pragma omp parallel for
	for (int32_t x = 0; x < nsize; x += block)
	{
		U* const B = deinterleaved_BGR[0] + x;
		U* const G = deinterleaved_BGR[1] + x;
		U* const R = deinterleaved_BGR[2] + x;
		const T* const interleaved_ptr = interleaved_BGR + x * 3;

		const int blockx = std::min(nsize, x + block) - x;
		for (int xx = 0; xx < blockx; ++xx)
		{
			B[xx] = interleaved_ptr[xx * 3 + 0] + round;
			G[xx] = interleaved_ptr[xx * 3 + 1] + round;
			R[xx] = interleaved_ptr[xx * 3 + 2] + round;
		}
	}

}

template<typename T, typename U>
void interleave_BGR(const U** const deinterleaved_BGR, T* const interleaved_BGR, const uint32_t nsize) {

	constexpr float round = std::is_integral_v<T> ? std::is_integral_v<U> ? 0 : 0.5f : 0;
	const uint32_t block = 262144 / (3 * std::max(sizeof(T), sizeof(U)));

#pragma omp parallel for
	for (int32_t x = 0; x < nsize; x += block)
	{
		const U* const B = deinterleaved_BGR[0] + x;
		const U* const G = deinterleaved_BGR[1] + x;
		const U* const R = deinterleaved_BGR[2] + x;
		T* const interleaved_ptr = interleaved_BGR + x * 3;

		const int blockx = std::min(nsize, x + block) - x;
		for (int xx = 0; xx < blockx; ++xx)
		{
			interleaved_ptr[xx * 3 + 0] = B[xx] + round;
			interleaved_ptr[xx * 3 + 1] = G[xx] + round;
			interleaved_ptr[xx * 3 + 2] = R[xx] + round;
		}
	}

}

void pocketfft_2D(cv::Mat& image, double nsmooth)
{
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
			// printf("pre %d %d %d %d %d %d %d\n", sizes[0], sizes[1], border[0], border[1], border[2], border[3], pffft::Fft<float>::isValidSize(sizes[i]));
			int new_size = nearestTransformSize(sizes[i]);
			int new_pad = (new_size - sizes[i]);
			sizes[i] = new_size;
			border[i * 2 + 0] += new_pad / 2; // floor - fix for new_pad when not even
			border[i * 2 + 1] += new_pad / 2.f + 0.5f; // ceil if odd - fix for new_pad when not even

			// printf("post %d %d %d %d %d %d %d\n", sizes[0], sizes[1], border[0], border[1], border[2], border[3], pffft::Fft<float>::isValidSize(sizes[i]));
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


	std::chrono::time_point<std::chrono::steady_clock> start_0 = std::chrono::steady_clock::now();

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
	box_kernel_1D(kernel_1D_col.data(), nsmooth * nsmooth, sizes[0]);
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
		box_kernel_1D(kernel_1D_row.data(), nsmooth * nsmooth, sizes[1]);
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
						std::real(resf[row_ * (sizes[1] / 2 + 1) + cval])
						+ 0.01));
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

	std::chrono::time_point<std::chrono::steady_clock> start_0 = std::chrono::steady_clock::now();

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
	box_kernel_1D(kernel_1D_col.data(), nsmooth * nsmooth, sizes[0]);
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
		box_kernel_1D(kernel_1D_row.data(), nsmooth * nsmooth, sizes[1]);
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

	printf("PocketFFT 1D : %f\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_0).count());
	interleave_BGR((const float**)BGR, (uint8_t*)image.data, image.size[0] * image.size[1]);


}


void pffft_(cv::Mat& image, double nsmooth)
{

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
		if (!pffft::Fft<float>::isValidSize(sizes[i]))
		{
			int new_size = pffft::Fft<float>::nearestTransformSize(sizes[i]);
			trailing_zeros[i] = (new_size - sizes[i]);
			sizes[i] = new_size;
		}
	}

	std::vector<std::vector<float>> temp(3, std::vector<float>(image.size[0] * image.size[1]));
	float* BGR[3] = { temp[0].data(), temp[1].data(), temp[2].data() };
	deinterleave_BGR((const uint8_t*)image.data, BGR, image.size[0] * image.size[1]);

	std::chrono::time_point<std::chrono::steady_clock> start_0 = std::chrono::steady_clock::now();

	// fast convolve by pffft, without reordering the z-domain. Thus, we perform a row by row, col by col FFT and convolution with 2x1D kernel

	pffft::AlignedVector<float> kerf_1D_row;
	pffft::AlignedVector<float> kerf_1D_col;

	pffft::Fft<float> fft_kernel_1D_row(sizes[1]);
	pffft::AlignedVector<float> kernel_aligned_1D_row = pffft::AlignedVector<float>(fft_kernel_1D_row.getLength());

	// create a gaussian 1D kernel with the specified sigma and kernel size, and center it in a length of FFT_length
#ifdef boxblur
	box_kernel_1D(kernel_aligned_1D_row.data(), nsmooth * nsmooth, sizes[1]);
#else
	getGaussian(kernel_aligned_1D_row, sigma, kSize, sizes[1]);
#endif

	kerf_1D_row = fft_kernel_1D_row.internalLayoutVector();
	fft_kernel_1D_row.forwardToInternalLayout(kernel_aligned_1D_row, kerf_1D_row);

	// calculate the DFT of kernel by col if size of cols is not the same of rows
	if (sizes[0] != sizes[1]) {
		pffft::Fft<float> fft_kernel_1D_col(sizes[0]);
		pffft::AlignedVector<float> kernel_aligned_1D_col = pffft::AlignedVector<float>(fft_kernel_1D_col.getLength());
#ifdef boxblur
		box_kernel_1D(kernel_aligned_1D_col.data(), nsmooth * nsmooth, sizes[0]);
#else
		getGaussian(kernel_aligned_1D_col, sigma, kSize, sizes[0]);
#endif

		kerf_1D_col = fft_kernel_1D_col.internalLayoutVector();
		fft_kernel_1D_col.forwardToInternalLayout(kernel_aligned_1D_col, kerf_1D_col);

	}
	else
		kerf_1D_col = kerf_1D_row;

	int ndata = image.size[0] * image.size[1];

#pragma omp parallel for
	for (int i = 0; i < 3; ++i) {
		// if fast don't do z-domain reordering, so it's faster than the normal process, is written in their APIs

		pffft::AlignedVector<float> resf(ndata);

		pffft::Fft<float> fft_rows(sizes[1]);
		pffft::AlignedVector<float> tile(sizes[1]);
		pffft::AlignedVector<float> work = fft_rows.internalLayoutVector();

		for (int j = 0; j < image.size[0]; ++j) {

			// copy the row and pad by reflection in the aligned vector
			// left reflected pad
			std::copy_n(std::reverse_iterator(temp[i].data() + j * image.size[1] + pad + 1), pad, tile.begin());
			// middle
			std::copy_n(temp[i].data() + j * image.size[1], image.size[1], tile.begin() + pad);
			// right reflected pad
			std::copy_n(std::reverse_iterator(temp[i].data() + (j + 1) * image.size[1] - 1), pad, tile.end() - pad /* fft trailing 0s --> */ - trailing_zeros[1]);


			fft_rows.forwardToInternalLayout(tile, work);
			fft_rows.convolve(work, kerf_1D_row, work, 1.f / sizes[1]);
			fft_rows.inverseFromInternalLayout(work, tile);

			// save the 1st pass row by row in the output vector
			std::copy(tile.begin() + pad, tile.end() - pad - trailing_zeros[1], resf.begin() + j * image.size[1]);
		}

		// transpose cache-friendly, took from FastBoxBlur
		flip_block<float, 1>(resf.data(), temp[i].data(), image.size[1], image.size[0]);


		pffft::Fft<float> fft_cols(sizes[0]);
		tile.resize(sizes[0]);
		work = fft_cols.internalLayoutVector();
		for (int j = 0; j < image.size[1]; ++j) {

			std::copy_n(std::reverse_iterator(temp[i].data() + j * image.size[0] + pad + 1), pad, tile.begin());
			std::copy_n(temp[i].data() + j * image.size[0], image.size[0], tile.begin() + pad);
			std::copy_n(std::reverse_iterator(temp[i].data() + (j + 1) * image.size[0] - 1), pad, tile.end() - pad /* fft trailing 0s --> */ - trailing_zeros[0]);


			fft_cols.forwardToInternalLayout(tile, work);
			fft_cols.convolve(work, kerf_1D_col, work, 1.f / sizes[0]);
			fft_cols.inverseFromInternalLayout(work, tile);

			// save the 2nd pass col by col in the output vector 
			std::copy(tile.begin() + pad, tile.end() - pad - trailing_zeros[0], resf.begin() + j * image.size[0]);

		}

		flip_block<float, 1>(resf.data(), temp[i].data(), image.size[0], image.size[1]);

	}

	printf("pffft: %f\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_0).count());

	interleave_BGR((const float**)BGR, (uint8_t*)image.data, image.size[0] * image.size[1]);

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
	cv::imwrite("C:/Users/miki/Downloads/c_.png", test_image);
	test_image.release();

#ifdef _DEBUG
	_CrtDumpMemoryLeaks();
#endif
}