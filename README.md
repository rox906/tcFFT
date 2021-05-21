# tcFFT
Accelerating FFT with Tensor Cores. It has been tested on NVIDIA GPU V100 and A100. The following packages are required: FFTW v3.3.8 or higher; CUDA v11.0 or higher.
## Structure
Core files:
- tcfft_half.h
- tcfft_half.cu
- tcfft_half_2d.h
- tcfft_half_2d.cu

Testing framework:
- all others
## Usage
make

./test.sh (details in ./test.sh)
