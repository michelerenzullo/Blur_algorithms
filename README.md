

﻿
# Fast Fourier Convolution - image processing

**DRAFT** - Refactoring and documentation in progress

In this repository we explore a
-  Fast Fourier Convolution algorithm using Fast Fourier Transform
- The application of a separable kernel to a 2D image 
- A true Gaussian blur and box blur

Thus this is an alternative to cv::blur and cv::GaussianBlur when OpenCV isn't needed

 - cv::blur has been replaced with FastBoxBlur ([repo](https://github.com/michelerenzullo/FastBoxBlur)), using the
   sliding accumulator and padding by reflection as default without
   increasing the memory usage

- cv::GaussianBlur has been replaced using two libraries to have a Fast Fourier Convolution

The libraries used are [pffft](https://bitbucket.org/jpommier/pffft/src/master/) and [pocketFFT](https://github.com/mreineck/pocketfft)
The choice has been made regards to performance, easy to implement and portability, while some libraries might perform better like [fftw](fftw.org/) or Intel MKL, they have many dependencies and the approach is not so straight-forward. 
The best performance, in order of speed, are obtained trough pffft, pocketfft_2D and least pocketfft_1D convolution with a setted cache of 16MB

## Common utils and procedures among the implementations:

### GaussianBlur sigma
 Given a sigma we calculate the expected window size to contain the distribution, clamping the tails, so that  
 $$radius = {σ\sqrt{2log(255)} -1} $$  
 $$width = {round(2 radius)} $$  
 I setted a costraint so that the width (aka gaussian_window) has to be inside the biggest dimension, this means that over that we will crop more the tails reducing the accuracy, but it won't make any difference since is just a big color blurred.  
   This limit might be increased further till: 
   ```c++
    (width - 1 ) / 2 <=std::min(rows - 1, cols - 1);
   ```
   in order to prevent out-of-buffer reading when reflecting 101 the borders.

### Padding
- To have a natural fading at borders I implemented a reflection 101, so that
	`pad = (kernel_width - 1) / 2` 
	 in any case has to be 
	 ```c++
	 pad <= std::min(rows - 1, cols - 1);
	 ```
	Example
	 ```
	 array_length = 7;
	 kernel_width = 13;
	 pad = 6; //maximum allowed to prevent out-of-buffer
	 g f e d c b | A B C D E F G | f e d c b a
	 ```
	 To achieve this behaviour, when processing:
	 - a 2D image (pocketfft_2d), I implemented a function `Reflect_101` similar to `cv::copyMakeBorder`
	 - 1 dimension, for tiles, I use `std::copy_n` and a `reverse_iterator`

- To calculate a DFT faster, the lenght of the array has to be decomposable in small prime factors like 2, 3, 5, therefore using the functions `isValidSize` and `nearestTransformSize` we find the closest "preferred" length, the increase of length will be covered adding **trailing zeros** at the end of each dimension, this has no effect in terms of artifacts or quality of the output

### Reflect_101
This function padd an image by reflection mono channel or multi channel with the specified pad_top, bottom, left, right, [gist here](https://gist.github.com/michelerenzullo/8dc5af393aa581c72b6dd7f640c8e406) .

### Deinterleave and Interleave RGB
This is important because the image is processed as it was single channel.
It's similar to `cv::split` and `cv::merge`, with 24bit images (RGB), and it performs the operation for sub-blocks, cache friendly, considering a setted L2 cache of 16MB and can either be compiled with OpenMP or with standard library threads for parallel computing. 


## Differences in the implementations:
- 1D (pffft and pocketfft_1D) vs 2D (pocketfft_2D)  
	 - processing 1 dimension at time first row by row, then tranpose, second col by col, then transpose back, is more memory and cache friendly since the padding to reflect the borders is done time by time, and trailing zeros are added at the end of the row, thus the plan informations are reused.
	 
	 - The big 2D "approach" requires more memory to padd prior each dimension and it's not cache friendly, so it's usually slower, but it's needed when we want the save the output spectrum of the image, I added a macro `#define DFT_image` that skips the convolution and just save the spectrum image  
	 $$spectrum = 20 log_{10}(| real | + 0.00001) $$ 

 - pffft
	  -  doesn't support N dimensions, only 1D, so even if you pass a 2D image, it will be processed as a flattened big row, therefore I adopted the process "for tiles"
	  - There is an option to speed-up further, skipping the z-domain reordering, called `forwardToInternalLayout` since we are just doing a convolution and we don't need that the spectrum is sorted.
 - pocketfft 
	  - supports N dimensions, the output spectrum complex array size will be `(size[0]) * (size[1] / 2 + 1)`, this means that just the last dimension will be shorter since the output spectrum of the forward transform is symmtetrical
- Parallel
	 - pocketfft_2D is executed in parallel for each channel image
	 - pocketfft_1D and pffft are executed in parallel for each tiles (row or col), the plans informations are safely shared between threads

## Kernel
The convolution / pointwise multiplication can be done with any kernel but in this repository I implemented two kernels, Gaussian and Box Blur, that have both the separability property, this is good since we just have to calculate the DFT of a row and (eventually if it's a different size) a col rather than `rows * cols` and multiply each point of the forward spectrum of the kernel with the one from the image tile.

Note: the Box Blur kernel using Fourier Transform is unusued by default since I implemented a different and faster algorithm called `fastboxblur` , but I left it for documentation purposes, you could use it defining the macro `#define boxblur` 

In order to apply the kernel, we are going to do a point wise mul in the frequency domain and we need the same length for the kernel and the image, so once generated a kernel of size N, and given an FFT length of X, we have to padd the kernel adding extra zeros of `X - N` and shift the center element of the kernel to the left-most top corner in order to avoid the circular convolution, this has to be done no matter how many dimensions we have.

    Example of 1D padding and centering
    Box car kernel 3
    1/3 1/3 1/3
    
    FFT Length (for tile) = 8
    extra padd = 5
    1/3 1/3 1/3 0 0 0 0 0
    
    Shifting and centering
    1/3 1/3 0 0 0 0 0 1/3
    The above can be achieved easily with std::rotate
    
    Example of 2D padding and centering
    Box car kernel 3x3
    1/9 1/9 1/9
    1/9 1/9 1/9
    1/9 1/9 1/9
    
    FFT Length 64
    extra padd = 55
    1/9 1/9 1/9 0 0 0 0 0
    1/9 1/9 1/9 0 0 0 0 0
    1/9 1/9 1/9 0 0 0 0 0
    0    0   0  0 0 0 0 0
    0    0   0  0 0 0 0 0
    0    0   0  0 0 0 0 0
    0    0   0  0 0 0 0 0
    0    0   0  0 0 0 0 0
    
    Shifting and centering
    1/9 1/9 0 0 0 0 0 1/9
    1/9 1/9 0 0 0 0 0 1/9
    0    0  0 0 0 0 0  0
    0    0  0 0 0 0 0  0
    0    0  0 0 0 0 0  0
    0    0  0 0 0 0 0  0
    0    0  0 0 0 0 0  0
    1/9 1/9 0 0 0 0 0 1/9
  
    


When the kernel is centered you will notice that his spectrum has the imaginary part 0, therefore when doing the mul with the image spectrum, we will use just the real part of the complex number.

## Benchmark
Using an M3 Pro 12 cores, with 45 images 3 channels from 1500 x 1000 px to 11400 x 7600 px , true Gaussian blur with a sigma of
$$sigma = \sqrt{width}$$
and a setted cache of 16MB (M3 Pro) for PocketFFT.

We can notice **how surprisingly fast** is the 1D implementation in pffft compared to OpenCV that is the standard market reference, also, at each input size increase, the trend is linear for the first, while exponential for the latter.

PocketFFT 1D and 2D also performs quite well, but the 2D version starts to struggle when having to deal with large amount of memory at time, while the 1D version still remains the most memory friendly and use the full amount of processors available.

Detailed timings are in a jupyter notebook file  py / performance.ipynb
![](py/bench.png)

Note: the above test using the Fast Fourier Convolution is made just for the Gaussian kernel, since the Box kernel has been left for documentation and it has been implemented in a different way, through a faster and simpler algorithm called "sliding accumulator", benchmark and details are in its repository [here](https://github.com/michelerenzullo/FastBoxBlur#performance).


## Usage and APIs coming soon
I'm thinking to create a "wrapper" library or maybe just simplify the function calls, since this repository is designed for academic research and probably used by other developers just the pffft implementation.

PocketFFT is header only, so you don't have to compile any library, while pffft implementation has to be compiled first, I compiled the "float" precision version and it's quite easy and intuitive with CMake-GUI. 
There are the respective git submodules to the repositories in this project.
