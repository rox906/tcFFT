echo 'tcfft/cufft_half_accuracy:'
echo ' -b: batchsize (should bigger than 32 for short sizes)'
echo ' -n: FFT size'
echo ' -s: specific random seed'
echo ''
echo 'tcfft/cufft_half_2d_accuracy:'
echo ' -b: batchsize (should bigger than 32 for short sizes)'
echo ' -x: size of the first dimension'
echo ' -y: size of the second dimension'
echo ' -s: specific random seed'
echo ''
echo 'tcfft/cufft_half_speed:'
echo ' -b: batchsize (should bigger than 32 for short sizes)'
echo ' -n: FFT size'
echo ' -s: specific random seed'
echo ' -m: max repeated times'
echo ''
echo 'tcfft/cufft_half_2d_speed:'
echo ' -b: batchsize (should bigger than 32 for short sizes)'
echo ' -x: size of the first dimension'
echo ' -y: size of the second dimension'
echo ' -s: specific random seed'
echo ' -m: max repeated times'

# 1D Accuracy ------------------------------------------------------
for i in 256 512 1024
do
    echo "Accuracy test: cuFFT, FFT size: $i batchsize:32"
    ./cufft_half_accuracy -b 32 -n $i
done
for i in 131072 262144 524288 16777216 33554432 67108864 134217728
do
    echo "Accuracy test: cuFFT, FFT size: $i batchsize:1"
    ./cufft_half_accuracy -n $i
done

for i in 256 512 1024
do
    echo "Accuracy test: tcFFT, FFT size: $i batchsize:32"
    ./tcfft_half_accuracy -b 32 -n $i
done
for i in 131072 262144 524288 16777216 33554432 67108864 134217728
do
    echo "Accuracy test: tcFFT, FFT size: $i batchsize:1"
    ./tcfft_half_accuracy -n $i
done

# 2D Accuracy ------------------------------------------------------
for i in 256 512
do
    for j in 256 512 1024
    do
        echo "Accuracy test: cuFFT, FFT size: ${i}x${j} batchsize:1"
        ./cufft_half_2d_accuracy -b 1 -x $i -y $j
    done
done

for i in 256 512
do
    for j in 256 512 1024
    do
        echo "Accuracy test: tcFFT, FFT size: ${i}x${j} batchsize:1"
        ./tcfft_half_2d_accuracy -b 1 -x $i -y $j
    done
done

# 1D Speed ------------------------------------------------------
for i in 256 512 1024
do
    echo "Speed test: cuFFT, FFT size: $i batchsize:1048576"
    ./cufft_half_speed -b 1048576 -n $i
done
for i in 131072 262144 524288
do
    echo "Speed test: cuFFT, FFT size: $i batchsize:2048"
    ./cufft_half_speed -b 2048 -n $i
done
for i in 16777216 33554432 67108864 134217728
do
    echo "Speed test: cuFFT, FFT size: $i batchsize:8"
    ./cufft_half_speed -b 8 -n $i
done

for i in 256 512 1024
do
    echo "Speed test: tcFFT, FFT size: $i batchsize:1048576"
    ./tcfft_half_speed -b 1048576 -n $i
done
for i in 131072 262144 524288
do
    echo "Speed test: tcFFT, FFT size: $i batchsize:2048"
    ./tcfft_half_speed -b 2048 -n $i
done
for i in 16777216 33554432 67108864 134217728
do
    echo "Speed test: tcFFT, FFT size: $i batchsize:8"
    ./tcfft_half_speed -b 8 -n $i
done

# 2D Speed ------------------------------------------------------
for i in 256 512
do
    for j in 256 512 1024
    do
        echo "Speed test: cuFFT, FFT size: ${i}x${j} batchsize:2048"
        ./cufft_half_2d_speed -b 2048 -x $i -y $j
    done
done

for i in 256 512
do
    for j in 256 512 1024
    do
        echo "Speed test: tcFFT, FFT size: ${i}x${j} batchsize:2048"
        ./tcfft_half_2d_speed -b 2048 -x $i -y $j
    done
done