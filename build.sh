#!/bin/bash
set -e

TARGET_SOC="RK3568"

# 设置环境变量
compiler_root=/opt/nvr

export LD_LIBRARY_PATH=${compiler_root}/lib:$LD_LIBRARY_PATH

# 获取绝对路径
ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )

# 创建build目录
BUILD_DIR=${ROOT_PWD}/build/build_linux_aarch64
if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

cd ${BUILD_DIR}
cmake -B . -S ${ROOT_PWD} \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=${compiler_root}/bin/aarch64-buildroot-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=${compiler_root}/bin/aarch64-buildroot-linux-gnu-g++
make -j8
make install
