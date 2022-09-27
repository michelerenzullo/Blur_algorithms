DRAFT

This repository has high performant algorithm to replace cv::blur and cv::GaussianBlur

cv::blur has been replaced with FastBoxBlur, using the accumulation algorithm and padding by reflection as default without increasing the memory usage

cv::GaussianBlur has been replaced using 2 light and fast libraries,

the fastest implementation is through pffft 1D 

The gaussian kernel is separable, therefore we generate 2x1D kernels, one for rows and one for cols, and we process the image for chunks, row by row, col by col, this is more cache friendly and with less memory usage.

the second implementation is through pocketfft 1D
same algorithm as pffft 1D, and with a setted cache size of 4500

the third implementation is through pocketfft 2D
in this case the image is padded prior, with more memory usage, and the DFT generated is a true 2D DFT


