#pragma once
#include <stdlib.h>
//#include <emscripten/bind.h>
//#include <emscripten/val.h>
#ifdef OMP
#include <omp.h>
#endif
#include <memory>
#include <thread>
#include <iostream>
//#include <sanitizer/lsan_interface.h>

//typedef emscripten::val em_val;

void generate_table(uint32_t *crc_table)
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

uint32_t crc32c(uint8_t *data, size_t bytes, uint8_t *data1 = nullptr, size_t bytes1 = 0)
{
  auto crc_table = std::make_unique<uint32_t[]>(256);
  generate_table(crc_table.get());
  uint32_t crc = 0xFFFFFFFF;
  while (bytes--)
  {
    int i = (crc ^ *data++) & 0xFF;
    crc = (crc_table[i] ^ (crc >> 8));
  }
  if (data1)
  {
    while (bytes1--)
    {
      int i = (crc ^ *data1++) & 0xFF;
      crc = (crc_table[i] ^ (crc >> 8));
    }
  }
  return crc ^ 0xFFFFFFFF;
}

template <typename T, typename op>
void hybrid_loop(T end, op operation)
{
	auto operation_wrapper = [&](T i, int tid = 0)
	{
		if constexpr (std::is_invocable_v<op, T>) operation(i);
		else operation(i, tid);
	};
#ifdef SINGLE
	for (T i = 0; i < end; ++i) operation_wrapper(i);
#elif OMP
#pragma omp parallel for
	for (T i = 0; i < end; ++i) operation_wrapper(i, omp_get_thread_num());
#elif defined __EMSCRIPTEN_THREADS__ || defined MYLOOP
	const int num_threads = std::thread::hardware_concurrency();

	// Split in block equally for each thread. ex: 3 threads, start = 0, end = 8
    // Thread 0: 0,1,2
    // Thread 1: 3,4,5
    // Thread 2: 6,7
	// Also don't spawn more threads than needed
	// ex: 4 threads, start = 0, end = 3
    // Thread 0: 0
    // Thread 1: 1
    // Thread 2: 2
    // Thread 3: NOT SPAWNED
	const T block_size = (end + num_threads - 1) / num_threads;
	std::vector<std::thread> threads;
	const int threads_needed = std::min(num_threads, (int)std::ceil(end / (float)block_size));
	for (int tid = 0; tid < threads_needed; ++tid)
	{
		threads.emplace_back([=]() {
			T block_start = tid * block_size;
            T block_end = (tid == threads_needed - 1) ? end : block_start + block_size;

            for (T i = block_start; i < block_end; ++i) operation_wrapper(i, tid);});
	}
	for (auto &thread : threads) thread.join();
#endif
}

#define MALLOC_V4SF_ALIGNMENT 64

static void* Valigned_malloc(size_t nb_bytes) {
	void* p, * p0 = malloc(nb_bytes + MALLOC_V4SF_ALIGNMENT);
	if (!p0) return (void*)0;
	p = (void*)(((size_t)p0 + MALLOC_V4SF_ALIGNMENT) & (~((size_t)(MALLOC_V4SF_ALIGNMENT - 1))));
	*((void**)p - 1) = p0;
	return p;
}

static void Valigned_free(void* p) {
	if (p) free(*((void**)p - 1));
}

template <class T>
class PFAlloc {
public:
	// type definitions
	typedef T        value_type;
	typedef T* pointer;
	typedef const T* const_pointer;
	typedef T& reference;
	typedef const T& const_reference;
	typedef std::size_t    size_type;
	typedef std::ptrdiff_t difference_type;

	// rebind allocator to type U
	template <class U>
	struct rebind {
		typedef PFAlloc<U> other;
	};

	// return address of values
	pointer address(reference value) const {
		return &value;
	}
	const_pointer address(const_reference value) const {
		return &value;
	}

	/* constructors and destructor
	 * - nothing to do because the allocator has no state
	 */
	PFAlloc() throw() {
	}
	PFAlloc(const PFAlloc&) throw() {
	}
	template <class U>
	PFAlloc(const PFAlloc<U>&) throw() {
	}
	~PFAlloc() throw() {
	}

	// return maximum number of elements that can be allocated
	size_type max_size() const throw() {
		return std::numeric_limits<std::size_t>::max() / sizeof(T);
	}

	// allocate but don't initialize num elements of type T
	pointer allocate(size_type num, const void* = 0) {
		pointer ret = (pointer)Valigned_malloc(int(num) * sizeof(T));
		return ret;
	}

	// initialize elements of allocated storage p with value value
	void construct(pointer p, const T& value) {
		// initialize memory with placement new
		new((void*)p)T(value);
	}

	// destroy elements of initialized storage p
	void destroy(pointer p) {
		// destroy objects by calling their destructor
		p->~T();
	}

	// deallocate storage p of deleted elements
	void deallocate(pointer p, size_type num) {
		// deallocate memory with pffft
		Valigned_free((void*)p);
	}
};

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
void deinterleave_BGR(const T* const interleaved_BGR, U** const deinterleaved_BGR, const uint32_t total_size) {

	// Cache-friendly deinterleave BGR, splitting for blocks of 256 KB, inspired by flip-block
	constexpr float round = std::is_integral_v<U> ? std::is_integral_v<T> ? 0 : 0.5f : 0;
	constexpr uint32_t block = 262144 / (3 * std::max(sizeof(T), sizeof(U)));
    const uint32_t num_blocks = std::ceil(total_size / (float)block);
	const uint32_t last_block_size = total_size % block == 0 ? block : total_size % block;

    hybrid_loop(num_blocks, [&](auto n) {
		const uint32_t x = n * block;
		U* const B = deinterleaved_BGR[0] + x;
		U* const G = deinterleaved_BGR[1] + x;
		U* const R = deinterleaved_BGR[2] + x;
		const T* const interleaved_ptr = interleaved_BGR + x * 3;

		const int blockx = (n == num_blocks - 1) ? last_block_size : block;
		for (int xx = 0; xx < blockx; ++xx)
		{
			B[xx] = interleaved_ptr[xx * 3 + 0] + round;
			G[xx] = interleaved_ptr[xx * 3 + 1] + round;
			R[xx] = interleaved_ptr[xx * 3 + 2] + round;
		}
	});

}

template<typename T, typename U>
void interleave_BGR(const U** const deinterleaved_BGR, T* const interleaved_BGR, const uint32_t total_size) {

	constexpr float round = std::is_integral_v<T> ? std::is_integral_v<U> ? 0 : 0.5f : 0;
	constexpr uint32_t block = 262144 / (3 * std::max(sizeof(T), sizeof(U)));
    const uint32_t num_blocks = std::ceil(total_size / (float)block);
	const uint32_t last_block_size = total_size % block == 0 ? block : total_size % block;
	
    hybrid_loop(num_blocks, [&](auto n) {
		const uint32_t x = n * block;
		const U* const B = deinterleaved_BGR[0] + x;
		const U* const G = deinterleaved_BGR[1] + x;
		const U* const R = deinterleaved_BGR[2] + x;
		T* const interleaved_ptr = interleaved_BGR + x * 3;

		const int blockx = (n == num_blocks - 1) ? last_block_size : block;
		for (int xx = 0; xx < blockx; ++xx)
		{
			interleaved_ptr[xx * 3 + 0] = B[xx] + round;
			interleaved_ptr[xx * 3 + 1] = G[xx] + round;
			interleaved_ptr[xx * 3 + 2] = R[xx] + round;
		}
	});

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
	const int right_offset = (pad_left + original_size[1] - 1) * 2 * C;
	const int left_offset = pad_left * 2 * C;
	const int bottom_offset = 2 * (stride[0] - 1) + pad_top;

    hybrid_loop(padded[0], [&](auto i){
		T* const row = output + i * padded[1];

		if (i < padded[0] - pad_bottom)
			std::copy_n(&input[stride[1] * abs(i - pad_top)], stride[1], &row[pad_left * C]);
		else
			std::copy_n(&input[stride[1] * (bottom_offset - i)], stride[1], &row[pad_left * C]);

		for (int j = 0; j < pad_left * C; j += C)
			std::copy_n(row + left_offset - j, C, row + j);

		for (int j = padded[1] - pad_right * C; j < padded[1]; j += C)
			std::copy_n(row + right_offset - j, C, row + j);
	});

}