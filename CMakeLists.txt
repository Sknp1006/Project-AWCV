cmake_minimum_required(VERSION 3.5)
project(Project-AWCV VERSION 1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(WIN32)
message("------[We are working on Windows now!]------")
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
elseif(UNIX)
message("------[We are working on Unix now!]------")
set(CMAKE_TOOLCHAIN_FILE "/Users/zhe/Github/vcpkg/buildtrees/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
endif()
# set(CMAKE_TOOLCHAIN_FILE "/Users/zhe/Github/vcpkg/buildtrees/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/publish)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/runtime)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/library)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/runtime)
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/path)

message("CMAKE_INSTALL_PREFIX: " ${CMAKE_INSTALL_PREFIX})
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

if(UNIX)
set(OpenCV_DIR "${VCPKG_INSTALLED_DIR}/arm64-osx/share/opencv4")
endif(UNIX)

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)
message("CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR})    # 项目路径

# 获取src下的所有文件
file(GLOB_RECURSE AWCV_SRCS src/*.cpp)
# message("AWCV_SRCS" ${AWCV_SRCS})
file(GLOB_RECURSE AWCV_INCLUDE_HEADERS include/*.hpp)
# message("AWCV_INCLUDE_HEADERS" ${AWCV_INCLUDE_HEADERS})

# add_executable(${PROJECT_NAME} 
#     main.cc
#     examples/CoreTest.hpp 
#     ${AWCV_INCLUDE_HEADERS}
#     ${AWCV_SRCS}
# )

include_directories(
    ${PROJECT_SOURCE_DIR}/include/
    ${PROJECT_SOURCE_DIR}/examples/
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

# target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_library(awcv SHARED ${AWCV_SRCS})
target_link_libraries(awcv PRIVATE ${OpenCV_LIBS} Boost::filesystem)
add_library(awcv_static STATIC ${AWCV_SRCS})
target_link_libraries(awcv_static PRIVATE ${OpenCV_LIBS} Boost::filesystem)
