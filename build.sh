#!/bin/bash
set -e

usage() {
    echo -e "build.sh - Build Project-AWCV binary packages"
    echo -e "Usage:"
    echo -e "  $0 -t/--type <BUILD_TYPE> "Release or Debug""
}

TARGET_SOC="RK3588"

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -t|--type)
        BUILD_TYPE="$2"
        shift # past argument
        shift # past value
        ;;
        -h|--help)
        usage
        exit 0
        ;;
        *)    # unknown option
        echo "Unknown option $1"
        usage
        exit 1
        ;;
    esac
done

# 获取绝对路径
ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )

# 创建build目录
BUILD_DIR=${ROOT_PWD}/build/build_linux_aarch64
if [[ -d "${BUILD_DIR}" ]]; then
  rm -rf ${BUILD_DIR}
fi
mkdir -p ${BUILD_DIR}

# if [[ ! -d "${BUILD_DIR}" ]]; then
#   mkdir -p ${BUILD_DIR}
# fi

cd ${BUILD_DIR}
cmake -B . -S ${ROOT_PWD} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DSDK_BUILD_TYPE=${SDK_BUILD_TYPE} \
    -DTARGET_SOC=${TARGET_SOC} \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_MAKE_PROGRAM=ninja \
    -GNinja \
    -DBUILD_STATIC=OFF

ninja
ninja install
