#pragma once


template <typename T, int C>
void horizontal_blur_kernel_reflect(const T* in, T* out, const int w, const int h, int r)
{
	// change the local variable types depending on the template type for faster calculations
	using calc_type = std::conditional_t<std::is_integral_v<T>, int, float>;

	r = 0.5f * (r - 1);
	r = std::min(r, w - 1);

	const float iarr = 1.f / (r + r + 1);

#pragma omp parallel for
	for (int i = 0; i < h; i++)
	{
		const int begin = i * w, end = begin + w, max_end = end - 1;
		int li = begin + r, ri = begin + r + 1; // left index(mirrored in the beginning), right index(mirrored at the end)
		calc_type acc[C] = {};

		// for ksize = 7, and r = 3, and array length = 11
		// array is [ a b c d e f g h i j k ]
		// emulated array is [d c b _ a b c d e f g h i j k _ j i h]

		// emulating the left padd: the initial accumulation is (d + c + b + a + b + c + d) --> 2 * (a + b + c + d) - a

		for (int ch = 0; ch < C; ++ch)
		{
			for (int j = 0; j <= r; j++)
				acc[ch] += 2 * in[(begin + j) * C + ch];
			acc[ch] -= in[begin * C + ch]; // remove extra pivot value

			// calculated first value
			out[begin * C + ch] = acc[ch] * iarr + (std::is_integral_v<T> ? 0.5f : 0);
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////

		for (int j = begin + 1; j < begin + r + 1; ++j)
		{
			for (int ch = 0; ch < C; ++ch)
			{
				// ri < end ? ri : max_end - ri % max_end   <--  reading in a reverse way
				// when reached the end of the row buffer and starting to read the "emulated" right pad
				acc[ch] += in[(ri < end ? ri : max_end - ri % max_end) * C + ch] - in[li * C + ch];
				out[j * C + ch] = acc[ch] * iarr + (std::is_integral_v<T> ? 0.5f : 0);
			}
			--li, ++ri;
		}

		// this loop won't be executed when r > w / 2 - 2 therefore the end of the image buffer will never be reached
		for (int j = begin + r + 1; j < end - r - 1; ++j)
		{
			for (int ch = 0; ch < C; ++ch)
			{
				acc[ch] += in[ri * C + ch] - in[li * C + ch];
				out[j * C + ch] = acc[ch] * iarr + (std::is_integral_v<T> ? 0.5f : 0);
			}
			++li, ++ri;
		}

		for (int j = end - r - 1; j < end; ++j)
		{
			for (int ch = 0; ch < C; ++ch)
			{
				acc[ch] += in[(ri < end ? ri : max_end - ri % max_end) * C + ch] - in[li * C + ch];
				out[j * C + ch] = acc[ch] * iarr + (std::is_integral_v<T> ? 0.5f : 0);
			}
			++li, --ri;
		}
	}
}

//!
//! \brief This function performs a 2D tranposition of an image.
//!
//! The transposition is done per
//! block to reduce the number of cache misses and improve cache coherency for large image buffers.
//! Templated by buffer data type T and buffer number of channels C.
//!
//! \param[in] in           source buffer
//! \param[in,out] out      target buffer
//! \param[in] w            image width
//! \param[in] h            image height
//!
template <typename T, int C>
void flip_block(const T* in, T* out, const int w, const int h)
{
	const int block = sqrt(262144.0 / (C * sizeof(T)));
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

template <typename T>
void flip_block(const T* in, T* out, const int w, const int h, const int c)
{
	switch (c)
	{
	case 1:
		flip_block<T, 1>(in, out, w, h);
		break;
	case 2:
		flip_block<T, 2>(in, out, w, h);
		break;
	case 3:
		flip_block<T, 3>(in, out, w, h);
		break;
	case 4:
		flip_block<T, 4>(in, out, w, h);
		break;
	default:
		printf("flip_block over %d channels is not supported yet. Add a specific case if possible or fall back to the generic version.\n", c);
		break;
	}
}

template <typename T>
void horizontal_blur(const T* in, T* out, const int w, const int h, const int c, const int r)
{
	switch (c)
	{
	case 1:
		horizontal_blur_kernel_reflect<T, 1>(in, out, w, h, r);
		break;
	case 2:
		horizontal_blur_kernel_reflect<T, 2>(in, out, w, h, r);
		break;
	case 3:
		horizontal_blur_kernel_reflect<T, 3>(in, out, w, h, r);
		break;
	case 4:
		horizontal_blur_kernel_reflect<T, 4>(in, out, w, h, r);
		break;
	default:
		printf("horizontal_blur over %d channels is not supported yet. Add a specific case if possible or fall back to the generic version.\n", c);
		break;
	}
}

template <typename T>
void fastboxblur(T* in, const int w, const int h, const int channels, const int ksize, int passes = 1)
{
	std::vector<T> tmp(w * h * channels);
	T* out = tmp.data();
	for (int i = 0; i < passes; ++i)
	{
		horizontal_blur<T>(in, out, w, h, channels, ksize);
		std::swap(in, out);
	}

	flip_block<T>(in, out, w, h, channels);
	std::swap(in, out);

	for (int i = 0; i < passes; ++i)
	{
		horizontal_blur<T>(in, out, h, w, channels, ksize);
		std::swap(in, out);
	}

	flip_block<T>(in, out, h, w, channels);
}