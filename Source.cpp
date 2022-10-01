#include <numeric>
#include <iostream> 
#include <pffft.hpp>
//suppose a max size of 4080px, using a kernel 11*11 and extra padd for FFT, using cache will make faster pocketfft
#define POCKETFFT_CACHE_SIZE 4500
#include "pocketfft_hdronly.h"
#include "fast_box_blur.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo/photo.hpp>

std::chrono::time_point<std::chrono::steady_clock> start_0;


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
template<typename T>
void getGaussian(T& kernel, const double sigma, int width = 0, int FFT_length = 0)
{

	if (!width) width = gaussian_window(sigma);

	kernel.resize(FFT_length ? FFT_length : width);

	const float mid_w = (width - 1) / 2.f;
	const double s = 2. * sigma * sigma;

	int i = 0;

	for (float y = -mid_w; y <= mid_w; ++y, ++i)
		kernel[i] = (exp(-(y * y) / s)) / (M_PI * s);

	const double sum = 1. / std::accumulate(&kernel[0], &kernel[width], 0.);

	std::transform(&kernel[0], &kernel[width], &kernel[0], [&sum](auto& i) {return i * sum; });

	//fit the kernel in the FFT_length and shift the center to avoid circular convolution
	if (FFT_length)
		std::rotate(kernel.rbegin(), kernel.rbegin() + (kernel.size() - width / 2), kernel.rend());
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
	printf("rank %.3f\n", rank);
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

}


void pocketfft_2D(cv::Mat image, int nsmooth)
{
	//pocketfft can handle non-small prime numbers decomposition of ndata but becomes slower, pffft cannot handle them and force you to add more pad

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
	std::vector<std::complex<float>> kerf_1D_row;
	std::vector<float> kernel_1D_col(sizes[0]);
	//make_kernel_1D(kernel_1D_col, nsmooth * nsmooth, sizes[0]);
	getGaussian(kernel_1D_col, sigma, kSize, sizes[0]);
	pocketfft::r2c(shape_col, strided_1D, strided_out_1D, axes_1D, pocketfft::FORWARD, kernel_1D_col.data(), kerf_1D_col.data(), 1.f, 0);


	//since col will be iterated for a length of sizes[0] but its length ends at (sizes[0]/2 + 1), we resize and reflect, with index kerf_1D_col[sizes[0] / 2 + 1] as pivot (Nyquist frequency)
	//explanation "In case of real (single-channel) data, the output spectrum of the forward Fourier transform or input spectrum of the inverse Fourier transform can be represented in a packed format called CCS (complex-conjugate-symmetrical)" from - OpenCV Documentation
	kerf_1D_col.resize(sizes[0]);
	std::copy_n(kerf_1D_col.rbegin() + (sizes[0] / 2.f + .5f), kerf_1D_col.size() - (sizes[0] / 2 + 1), kerf_1D_col.begin() + (sizes[0] / 2 + 1));

	//calculate kernel for row dimension and his DFT if the length of rows is not the same of cols
	if (sizes[0] != sizes[1]) {
		kerf_1D_row.resize(sizes[1] / 2 + 1);
		std::vector<float> kernel_1D_row(sizes[1]);
		//make_kernel_1D(kernel_1D_row, nsmooth * nsmooth, sizes[1]);
		getGaussian(kernel_1D_row, sigma, kSize, sizes[1]);
		pocketfft::r2c(shape_row, strided_1D, strided_out_1D, axes_1D, pocketfft::FORWARD, kernel_1D_row.data(), kerf_1D_row.data(), 1.f, 0);
	}
	else
		kerf_1D_row = kerf_1D_col;

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

	printf("PocketFFT 2D: %f\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_0).count());
	cv::merge(temp, 3, image);

	image.convertTo(image, CV_8UC3);
	cv::Rect myroi(border[2], border[0], sizes[1] - border[2] - border[3], sizes[0] - border[0] - border[1]);
	cv::imwrite("C:/Users/miki/Downloads/c_.png", image(myroi));

}


void pocketfft_1D(cv::Mat image, double nsmooth)
{
	image.convertTo(image, CV_32FC3);

	double sigma = nsmooth;
	int kSize = gaussian_window(sigma, std::max(image.size[0], image.size[1]));

	int passes = 1; //we are using 1 pass of the gaussian kernel
	int pad = (kSize - 1) / 2 * passes;

	//absolute min padding to fit the kernel
	size_t sizes[2] = { image.size[0] + pad * 2, image.size[1] + pad * 2 };

	//if the length of the data is not decomposable in small prime numbers 2 - 3 - 5, is necessary to update the size adding more pad
	int trailing_zeros[2] = {};
	for (int i = 0; i < 2; ++i) {
		if (!isValidSize(sizes[i]))
		{
			int new_size = nearestTransformSize(sizes[i]);
			trailing_zeros[i] = (new_size - sizes[i]);
			sizes[i] = new_size;
		}
	}

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

	std::vector<std::complex<float>> kerf_1D_col(sizes[0] / 2 + 1);
	std::vector<std::complex<float>> kerf_1D_row;
	std::vector<float> kernel_1D_col;
	getGaussian(kernel_1D_col, sigma, kSize, sizes[0]);
	printf("sizes[0] %d - width %d\n", sizes[0], kSize);

	pocketfft::r2c(shape_col, strided_1D, strided_out_1D, axes_1D, pocketfft::FORWARD, kernel_1D_col.data(), kerf_1D_col.data(), 1.f, 0);

	//calculate kernel for row dimension and his DFT if the length of rows is not the same of cols
	if (sizes[0] != sizes[1]) {
		kerf_1D_row.resize(sizes[1] / 2 + 1);
		//make_kernel_1D(kernel_1D_row, nsmooth * nsmooth, shape_row[0]);
		std::vector<float> kernel_1D_row;
		getGaussian(kernel_1D_row, sigma, kSize, sizes[1]);
		pocketfft::r2c(shape_row, strided_1D, strided_out_1D, axes_1D, pocketfft::FORWARD, kernel_1D_row.data(), kerf_1D_row.data(), 1.f, 0);
	}
	else
		kerf_1D_row = kerf_1D_col;

	int ndata = image.size[0] * image.size[1];

	for (int i = 0; i < 3; ++i) {
		std::vector<float> resf(ndata);
#pragma omp parallel for
		for (int j = 0; j < image.size[0]; ++j) {
			std::vector<float> tile(sizes[1]);
			std::vector<std::complex<float>> work(sizes[1] / 2 + 1);

			std::copy_n(std::reverse_iterator(&((float*)temp[i].data)[(j + 1) * image.size[1]]) + image.size[1] - pad - 1, pad, tile.begin());
			std::copy_n(&((float*)temp[i].data)[j * image.size[1]], image.size[1], tile.begin() + pad);
			std::copy_n(std::reverse_iterator(&((float*)temp[i].data)[(j + 1) * image.size[1]]) + 1, pad, tile.end() - pad /* fft trailing 0s --> */ - trailing_zeros[1]);

			pocketfft::r2c(shape_row, strided_1D, strided_out_1D, axes_1D, pocketfft::FORWARD, tile.data(), work.data(), 1.f, 0);
			for (int i = 0; i < sizes[1] / 2 + 1; ++i) work[i] *= std::real(kerf_1D_row[i]); //since the imaginary part is null, ignore it
			pocketfft::c2r(shape_row, strided_out_1D, strided_1D, axes_1D, pocketfft::BACKWARD, work.data(), tile.data(), 1.f / sizes[1], 0);

			std::copy(tile.begin() + pad, tile.end() - pad - trailing_zeros[1], resf.begin() + j * image.size[1]);
		}
		flip_block<float, 1>(resf.data(), (float*)temp[i].data, image.size[1], image.size[0]);

#pragma omp parallel for
		for (int j = 0; j < image.size[1]; ++j) {
			std::vector<float> tile(sizes[0]);
			std::vector<std::complex<float>> work(sizes[0] / 2 + 1);

			std::copy_n(std::reverse_iterator(&((float*)temp[i].data)[(j + 1) * image.size[0]]) + image.size[0] - pad - 1, pad, tile.begin());
			std::copy_n(&((float*)temp[i].data)[j * image.size[0]], image.size[0], tile.begin() + pad);
			std::copy_n(std::reverse_iterator(&((float*)temp[i].data)[(j + 1) * image.size[0]]) + 1, pad, tile.end() - pad /* fft trailing 0s --> */ - trailing_zeros[0]);

			pocketfft::r2c(shape_col, strided_1D, strided_out_1D, axes_1D, pocketfft::FORWARD, tile.data(), work.data(), 1.f, 0);
			for (int i = 0; i < sizes[0] / 2 + 1; ++i) work[i] *= std::real(kerf_1D_col[i]); //since the imaginary part is null, ignore it
			pocketfft::c2r(shape_col, strided_out_1D, strided_1D, axes_1D, pocketfft::BACKWARD, work.data(), tile.data(), 1.f / sizes[0], 0);

			std::copy(tile.begin() + pad, tile.end() - pad - trailing_zeros[0], resf.begin() + j * image.size[0]);
		}

		flip_block<float, 1>(resf.data(), (float*)temp[i].data, image.size[0], image.size[1]);

	}

	printf("PocketFFT 1D : %f\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_0).count());
	cv::merge(temp, 3, image);

	image.convertTo(image, CV_8UC3);

	cv::imwrite("C:/Users/miki/Downloads/c_.png", image);

}


void pffft_(cv::Mat image, int nsmooth)
{

	image.convertTo(image, CV_32FC3);
	double sigma = nsmooth;
	//calculate a good width of the kernel for our sigma
	int kSize = gaussian_window(sigma, std::max(image.size[0], image.size[1]));

	int passes = 1; //we are using 1 pass for the gaussian kernel
	int pad = (kSize - 1) / 2 * passes;

	//absolute min padd
	size_t sizes[2] = { image.size[0] + pad * 2, image.size[1] + pad * 2 };

	//if the length of the data is not decomposable in small prime numbers 2 - 3 - 5, is necessary to update the size adding more pad as trailing zeros
	int trailing_zeros[2] = {};

	for (int i = 0; i < 2; ++i) {
		if (!pffft::Fft<float>::isValidSize(sizes[i]))
		{
			int new_size = pffft::Fft<float>::nearestTransformSize(sizes[i]);
			trailing_zeros[i] = (new_size - sizes[i]);
			sizes[i] = new_size;
		}
	}

	cv::Mat temp[3];
	cv::split(image, temp);

	start_0 = std::chrono::steady_clock::now();

	//fast convolve by pffft, without reordering the z-domain. Thus, we perform a row by row, col by col FFT and convolution with 2x1D kernel

	pffft::AlignedVector<float> kerf_1D_row;
	pffft::AlignedVector<float> kerf_1D_col;

	pffft::Fft<float> fft_kernel_1D_row(sizes[1]);
	pffft::AlignedVector<float> kernel_aligned_1D_row = pffft::AlignedVector<float>(fft_kernel_1D_row.getLength());

	//create a gaussian 1D kernel with the specified sigma and kernel size, and center it in a length of FFT_length
	getGaussian(kernel_aligned_1D_row, sigma, kSize, sizes[1]);

	kerf_1D_row = fft_kernel_1D_row.internalLayoutVector();
	fft_kernel_1D_row.forwardToInternalLayout(kernel_aligned_1D_row, kerf_1D_row);

	//calculate the DFT of kernel by col if size of cols is not the same of rows
	if (sizes[0] != sizes[1]) {
		pffft::Fft<float> fft_kernel_1D_col(sizes[0]);
		pffft::AlignedVector<float> kernel_aligned_1D_col = pffft::AlignedVector<float>(fft_kernel_1D_col.getLength());

		getGaussian(kernel_aligned_1D_col, sigma, kSize, sizes[0]);

		kerf_1D_col = fft_kernel_1D_col.internalLayoutVector();
		fft_kernel_1D_col.forwardToInternalLayout(kernel_aligned_1D_col, kerf_1D_col);

	}
	else {
		kerf_1D_col = kerf_1D_row;
	}

	int ndata = image.size[0] * image.size[1];

#pragma omp parallel for
	for (int i = 0; i < 3; ++i) {
		//if fast don't do z-domain reordering, so it's faster than the normal process, is written in their APIs

		pffft::AlignedVector<float> resf(ndata);

		pffft::Fft<float> fft_rows(sizes[1]);
		pffft::AlignedVector<float> tile(sizes[1]);
		pffft::AlignedVector<float> work = fft_rows.internalLayoutVector();

		for (int j = 0; j < image.size[0]; ++j) {

			//copy the row and pad by reflection in the aligned vector
			//left reflected pad
			std::copy_n(std::reverse_iterator(&((float*)temp[i].data)[(j + 1) * image.size[1]]) + image.size[1] - pad - 1, pad, tile.begin());
			//middle
			std::copy_n(&((float*)temp[i].data)[j * image.size[1]], image.size[1], tile.begin() + pad);
			//right reflected pad
			std::copy_n(std::reverse_iterator(&((float*)temp[i].data)[(j + 1) * image.size[1]]) + 1, pad, tile.end() - pad /* fft trailing 0s --> */ - trailing_zeros[1]);

			fft_rows.forwardToInternalLayout(tile, work);
			fft_rows.convolve(work, kerf_1D_row, work, 1.f / sizes[1]);
			fft_rows.inverseFromInternalLayout(work, tile);

			//save the 1st pass row by row in the output vector
			std::copy(tile.begin() + pad, tile.end() - pad - trailing_zeros[1], resf.begin() + j * image.size[1]);
		}

		//transpose cache-friendly, took from FastBoxBlur
		flip_block<float, 1>(resf.data(), ((float*)temp[i].data), image.size[1], image.size[0]);

		pffft::Fft<float> fft_cols(sizes[0]);
		tile.resize(sizes[0]);
		work = fft_cols.internalLayoutVector();
		for (int j = 0; j < image.size[1]; ++j) {

			std::copy_n(std::reverse_iterator(&((float*)temp[i].data)[(j + 1) * image.size[0]]) + image.size[0] - pad - 1, pad, tile.begin());
			std::copy_n(&((float*)temp[i].data)[j * image.size[0]], image.size[0], tile.begin() + pad);
			std::copy_n(std::reverse_iterator(&((float*)temp[i].data)[(j + 1) * image.size[0]]) + 1, pad, tile.end() - pad /* fft trailing 0s --> */ - trailing_zeros[0]);

			fft_cols.forwardToInternalLayout(tile, work);
			fft_cols.convolve(work, kerf_1D_col, work, 1.f / sizes[0]);
			fft_cols.inverseFromInternalLayout(work, tile);

			//save the 2nd pass col by col in the output vector 
			std::copy(tile.begin() + pad, tile.end() - pad - trailing_zeros[0], resf.begin() + j * image.size[0]);

		}
		flip_block<float, 1>(resf.data(), ((float*)temp[i].data), image.size[0], image.size[1]);

	}

	printf("pffft: %f\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_0).count());
	cv::merge(temp, 3, image);

	image.convertTo(image, CV_8UC3);
	cv::imwrite("C:/Users/miki/Downloads/c_.png", image);
}



void Test(cv::Mat image, int flag = 0, double nsmooth = 5, bool fast_ = 1)
{

	start_0 = std::chrono::steady_clock::now();

	if (flag == 6) pocketfft_1D(image, nsmooth);
	else if (flag == 5) {

		size_t sizes[2] = { image.size[0], image.size[1] };

		std::chrono::time_point<std::chrono::steady_clock> start_5 = std::chrono::steady_clock::now();
		//prepare temp vector
		std::vector<uchar> image_out(sizes[1] * sizes[0] * 3);
		uchar* image_out_ptr = image_out.data();

		horizontal_blur_kernel_reflect_double<uchar, 3>(image.data, image_out_ptr, sizes[1], sizes[0], nsmooth * nsmooth);


		std::chrono::time_point<std::chrono::steady_clock> start_1 = std::chrono::steady_clock::now();
		flip_block<uchar, 3>(image_out_ptr, image.data, image.size[1], image.size[0]);
		std::chrono::time_point<std::chrono::steady_clock> start_2 = std::chrono::steady_clock::now();


		horizontal_blur_kernel_reflect_double<uchar, 3>(image.data, image_out_ptr, sizes[0], sizes[1], nsmooth * nsmooth);


		std::chrono::time_point<std::chrono::steady_clock> start_3 = std::chrono::steady_clock::now();
		flip_block<uchar, 3>(image_out_ptr, image.data, image.size[0], image.size[1]);
		std::chrono::time_point<std::chrono::steady_clock> start_4 = std::chrono::steady_clock::now();

		printf("1st transp : %f\n", std::chrono::duration<double, std::milli>(start_2 - start_1).count());
		printf("2nd transp : %f\n", std::chrono::duration<double, std::milli>(start_4 - start_3).count());
		printf("(double accumulation, horizontalblur only, tot - transp) : %f\n", std::chrono::duration<double, std::milli>(start_4 - start_5).count() - std::chrono::duration<double, std::milli>(start_2 - start_1).count() - std::chrono::duration<double, std::milli>(start_4 - start_3).count());
		printf("fastblur only: %f\n", std::chrono::duration<double, std::milli>(start_4 - start_5).count());
		printf("fastblur with padding: %f\n", std::chrono::duration<double, std::milli>(start_4 - start_0).count());

		//save image
		cv::imwrite("C:/Users/miki/Downloads/c_.png", image);

	}
	else if (flag == 4) {

		size_t sizes[2] = { image.size[0], image.size[1] };

		std::chrono::time_point<std::chrono::steady_clock> start_5 = std::chrono::steady_clock::now();
		//prepare temp vector
		std::vector<uchar> image_out(sizes[1] * sizes[0] * image.channels());
		
		uchar* in = image.data;
		//uchar* out = image_out.data();
		fastboxblur(in, sizes[1], sizes[0], image.channels(), nsmooth * nsmooth, 2);
		/*
		uchar* image_out_ptr = image_out.data();
		//smooth row by row, transpose, col by col and transpose back. smooth twice, as suggested 2 passes
		int passes = 2;
		for (int i = 0; i < passes; ++i) {
			horizontal_blur_kernel_reflect<uchar, 3>(image.data, image_out_ptr, sizes[1], sizes[0], nsmooth * nsmooth);
			std::swap(image.data, image_out_ptr);
		}

		std::chrono::time_point<std::chrono::steady_clock> start_1 = std::chrono::steady_clock::now();
		flip_block<uchar, 3>(image.data, image_out_ptr, image.size[1], image.size[0]);
		std::chrono::time_point<std::chrono::steady_clock> start_2 = std::chrono::steady_clock::now();

		for (int i = 0; i < passes; ++i) {
			horizontal_blur_kernel_reflect<uchar, 3>(image_out_ptr, image.data, sizes[0], sizes[1], nsmooth * nsmooth);
			std::swap(image.data, image_out_ptr);
		}

		std::chrono::time_point<std::chrono::steady_clock> start_3 = std::chrono::steady_clock::now();
		flip_block<uchar, 3>(image_out_ptr, image.data, image.size[0], image.size[1]);
		std::chrono::time_point<std::chrono::steady_clock> start_4 = std::chrono::steady_clock::now();

		printf("1st transp : %f\n", std::chrono::duration<double, std::milli>(start_2 - start_1).count());
		printf("2nd transp : %f\n", std::chrono::duration<double, std::milli>(start_4 - start_3).count());
		printf("(horizontalblur only, tot - transp) : %f\n", std::chrono::duration<double, std::milli>(start_4 - start_5).count() - std::chrono::duration<double, std::milli>(start_2 - start_1).count() - std::chrono::duration<double, std::milli>(start_4 - start_3).count());
		printf("fastblur only: %f\n", std::chrono::duration<double, std::milli>(start_4 - start_5).count());
		printf("fastblur with padding: %f\n", std::chrono::duration<double, std::milli>(start_4 - start_0).count());
		*/

		printf("fastboxblur: %f\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_0).count());
		cv::imwrite("C:/Users/miki/Downloads/c_.png", image);

	}
	else if (flag == 3) pffft_(image, nsmooth);
	else if (flag == 2) pocketfft_2D(image, nsmooth);
	else
	{
		nsmooth *= nsmooth;

		cv::blur(image, image, cv::Size(nsmooth, nsmooth));
		cv::blur(image, image, cv::Size(nsmooth, nsmooth));
		//cv::GaussianBlur(image, image, cv::Size(0, 0), nsmooth);
		printf("OpenCV blur: %f\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_0).count());

		cv::imwrite("C:/Users/miki/Downloads/c_.png", image);
	}
}


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