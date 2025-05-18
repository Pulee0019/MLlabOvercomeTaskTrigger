# Software and plugin download
1. Download nvidia driver: **`Nvidia app`**
2. Download **[`cuda 11.2`](https://developer.nvidia.com/cuda-downloads)**
3. Download **[`cudnn 8.1.1`](https://developer.nvidia.com/rdp/cudnn-archive)**
4. Copy `.lib` file in `cudnn/lib/x64` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib\x64`
5. Copy `.h` file in `cudnn/include` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include`
6. Copy `.dll` file in `cudnn/bin` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin`
7. Download **[`Video Codec SDK`](https://developer.nvidia.com/video-codec-sdk-source-code)**
8. Copy `.lib` file in `Video_Codec_SDK_13.0.19.zip\Video_Codec_SDK_13.0.19\Lib\x64` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib\x64`
9. Copy `.h` file in `Video_Codec_SDK_13.0.19.zip\Video_Codec_SDK_13.0.19\Interface` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include`
10. Download **[`opencv contrib`](https://github.com/opencv/opencv_contrib/releases/tag/4.5.5)**
11. Download **[`opencv`](https://github.com/opencv/opencv/releases/tag/4.5.5https://github.com/opencv/opencv/releases/tag/4.5.5)**
12. Extract to your custom address such as `D:Lib`
13. Install **[`Visual Studio 16 2019`](https://learn.microsoft.com/en-us/visualstudio/releases/2019/release-notes#16.11.46)**(with c++ desktop development)
14. Install **`Anaconda`**
# Add environment varibles
- `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin`
- `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp`
- `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib\x64`
- `D:\APP\Anaconda`
- `others`
# All of the following can run in Anaconda Prompt:
## Environmment create and activate
- `conda create -n dlc310 python=3.10 -y`
  `conda env remove --name dlc310`
- `conda activate dlc310`
  `conda deactivate`
## Configure image source
- `pip config set install.trusted-host mirrors.aliyun.com`
  `pip config unset global.index-ur`
## Install deeplabcut and uninstall the opencv that does not support CUDA acceleration
- `pip install numpy==1.24.0`
- `pip install tensorflow-gpu==2.10.0`
- `pip install matplotlib pandas scikit-learn scikit-image h5py ipython jupyter tqdm ruamel.yaml moviepy`
- `pip install deeplabcut==2.3.10`
- `pip install deeplabcut[gui,tf]`
- `pip uninstall opencv-python opencv-contrib-python -y`
- `pip uninstall opencv-python-headless -y`
- `pip list | findstr opencv`
## Work need be done before compiling opencv
- `cd /d D:\Lib\opencv-4.5.5`
- `rmdir /s /q build`(Not required for first time)
- `mkdir build && cd build`
## Notice: `CUDA_ARCH_BIN` is according to your GPU(https://developer.nvidia.com/cuda-gpus)
## Compiling opencv(with all cuda function)
```
cmake .. ^
 -G "Visual Studio 16 2019" ^
 -A x64 ^
 -D CMAKE_BUILD_TYPE=Release ^
 -D CMAKE_INSTALL_PREFIX=D:/APP/Anaconda/envs/dlc310 ^
 -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.5.5/modules ^
 -D BUILD_opencv_python3=ON ^
 -D INSTALL_PYTHON_EXAMPLES=OFF ^
 -D BUILD_TESTS=OFF ^
 -D BUILD_PERF_TESTS=OFF ^
 -D BUILD_EXAMPLES=OFF ^
 -D WITH_CUDA=ON ^
 -D WITH_CUDNN=ON ^
 -D CUDA_NVCC_FLAGS="--Wno-deprecated-gpu-targets" ^
 -D ENABLE_FAST_MATH=ON ^
 -D CUDA_FAST_MATH=ON ^
 -D OPENCV_DNN_CUDA=ON ^
 -D CUDA_ARCH_BIN=8.9 ^
 -D CUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2" ^
 -D PYTHON3_EXECUTABLE=D:/APP/Anaconda/envs/dlc310/python.exe ^
 -D PYTHON3_INCLUDE_DIR=D:/APP/Anaconda/envs/dlc310/include ^
 -D PYTHON3_LIBRARY=D:/APP/Anaconda/envs/dlc310/libs/python310.lib ^
 -D PYTHON3_PACKAGES_PATH=D:/APP/Anaconda/envs/dlc310/Lib/site-packages
```
## Compiling opencv(with core cuda function)
```
cmake .. ^
 -G "Visual Studio 16 2019" ^
 -A x64 ^
 -D CMAKE_BUILD_TYPE=Release ^
 -D CMAKE_INSTALL_PREFIX=D:/APP/Anaconda/envs/dlc310 ^
 -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.5.5/modules ^ 
 -D BUILD_opencv_python3=ON ^
 -D INSTALL_PYTHON_EXAMPLES=OFF ^
 -D BUILD_TESTS=OFF ^
 -D BUILD_PERF_TESTS=OFF ^
 -D BUILD_EXAMPLES=OFF ^
 -D WITH_CUDA=ON ^
 -D WITH_CUDNN=ON ^
 -D CUDA_NVCC_FLAGS="--Wno-deprecated-gpu-targets" ^
 -D ENABLE_FAST_MATH=1 ^
 -D CUDA_FAST_MATH=1 ^
 -D OPENCV_DNN_CUDA=ON ^
 -D CUDA_NVCC_FLAGS="--expt-relaxed-constexpr;-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH;-D_ITERATOR_DEBUG_LEVEL=0" ^
 -D PYTHON3_EXECUTABLE=D:/APP/Anaconda/envs/dlc310/python.exe ^
 -D PYTHON3_INCLUDE_DIR=D:/APP/Anaconda/envs/dlc310/include ^
 -D PYTHON3_LIBRARY=D:/APP/Anaconda/envs/dlc310/libs/python310.lib ^
 -D PYTHON3_PACKAGES_PATH=D:/APP/Anaconda/envs/dlc310/Lib/site-packages ^
 -D BUILD_opencv_cudacodec=ON ^
 -D BUILD_opencv_cudaarithm=OFF ^
 -D BUILD_opencv_cudafilters=OFF ^
 -D BUILD_opencv_cudawarping=OFF ^
 -D BUILD_opencv_cudalegacy=OFF ^
 -D BUILD_opencv_cudabgsegm=OFF ^
 -D BUILD_opencv_cudaoptflow=OFF
```
## Release and install opencv
- `cmake --build . --config Release --target INSTALL`
## Verification if no error at release
- `python -c "import cv2; print(cv2.getBuildInformation())"`
## Other packages
- `pip install pyserial`
## Commend to get network training log
- `E: && cd E:\PCW\tube-ycyy-hes-2025-04-28`
- `tensorboard --logdir=dlc-models\iteration-0\tube-ycyyApr28-trainset95shuffle1\train\log\`
- `tensorboard --logdir=dlc-models\iteration-0\close-loopMay16-trainset95shuffle1\train\log\`
- `conda activate FLIR`
- `python -m ensurepip`
- `python -m pip install --upgrade pip`
- `python -m pip install numpy matplotlib Pillow==9.2.0`
- `D: & cd D:\APP\Spinnaker`
- `python -m pip install spinnaker_python-4.2.0.83-cp310-cp310-win_amd64.whl`
