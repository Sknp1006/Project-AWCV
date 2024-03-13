# Project-AWCV
为了练习Cmake和C++所造的opencv轮子🤔

## 环境
> 不要在嵌入式平台上使用，部分opencv特性不支持arm64平台，改为在wsl下开发
- ~~硬件：Friendly NanoPi-R6S~~
- ~~固件：rk3588-usb-debian-bullseye-minimal-6.1-arm64-20240131~~


## 第三方依赖
- opencv (需要contrib、nonfree、VTK模块)
参考 [https://sknp.top/posts/cross-compile-opencv](https://sknp.top/posts/cross-compile-opencv) 自行编译
  - 安装VTK依赖：`sudo apt install libgoogle-glog-dev` 
  - 安装VTK：`sudo apt install libvtk9-dev` 

```shell
#!/bin/bash
set -e

# 获取绝对路径
ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )

# 创建build目录
BUILD_DIR=${ROOT_PWD}/build
if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

cd ${BUILD_DIR}
cmake -B . -S ${ROOT_PWD} \
-DCMAKE_INSTALL_PREFIX=/opt/opencv \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_C_COMPILER=gcc \
-DCMAKE_CXX_COMPILER=g++ \
-DOPENCV_EXTRA_MODULES_PATH=${ROOT_PWD}/opencv_contrib/modules \
-DOPENCV_ENABLE_NONFREE=ON \
-DBUILD_TESTS=OFF \
-DBUILD_PERF_TESTS=OFF \
-DWITH_VTK=ON \
-DWITH_EIGEN=ON

```

- NumCpp + boost-1.68.0
参考 [https://sknp.top/posts/cross-compile-numcpp](https://sknp.top/posts/cross-compile-numcpp) 自行编译

- spdlog (通过apt安装 1.8.1)
```shell
sudo apt install libspdlog-dev
```

## 致谢

<div align="center">
<image src="https://resources.jetbrains.com/storage/products/company/brand/logos/jb_beam.svg" />
<div>
感谢 <a href=https://jb.gg/OpenSourceSupport>JetBrains</a> 为本项目提供的大力支持
</div>
</div>
