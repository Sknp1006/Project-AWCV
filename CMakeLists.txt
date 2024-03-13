cmake_minimum_required(VERSION 3.5)
project(Project-AWCV VERSION 1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE STRING "Install path prefix, prepended onto install directories." FORCE)
message("CMAKE_INSTALL_PREFIX: " ${CMAKE_INSTALL_PREFIX})

set(3RD_PARTY_DIR "${CMAKE_SOURCE_DIR}/3rdparty")
set(CMAKE_PREFIX_PATH ${3RD_PARTY_DIR})

# boost
set(BOOST_ROOT ${3RD_PARTY_DIR}/boost)
include_directories(${BOOST_ROOT}/include)
link_directories(${BOOST_ROOT}/lib)

# opencv4
set(OpenCV_DIR /usr/lib/aarch64-linux-gnu/cmake/opencv4)
find_package(OpenCV 4 REQUIRED)

# eigen3
find_package(Eigen3 REQUIRED PATH "/usr/share/eigen3/cmake")
include_directories(${EIGEN3_INCLUDE_DIR})

# numcpp
set(NUMCPP_INCLUDE_DIR ${3RD_PARTY_DIR}/numcpp/include)
include_directories(${NUMCPP_INCLUDE_DIR})

# spdlog
find_package(spdlog REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/src)
file(GLOB_RECURSE AWCV_SRCS ${CMAKE_SOURCE_DIR}/src/*.cpp)

option(BUILD_STATIC "Build as SHARE or STATIC library, default STATIC" ON)
message(STATUS "BUILD_STATIC:  ${BUILD_STATIC}")
message(STATUS "BUILD_TYPE:  ${CMAKE_BUILD_TYPE}")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(
        -O0     #编译优化等级0 不优化
        -Wall   #开启所有警告
        -g      #包含调试信息
        -fPIC   #位置无关代码
    )
    add_compile_definitions(SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(
        -O3     #编译优化等级3
        -fPIC   #位置无关代码
    )
    add_compile_definitions(SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO)
else()
    message(FATAL_ERROR "CMAKE_BUILD_TYPE must be Debug or Release")
endif()

if (BUILD_STATIC)
    message("------[Build as STATIC library]------")
    add_library(awcv STATIC ${AWCV_SRCS})
    target_link_libraries(awcv PRIVATE ${OpenCV_LIBS})
else()
    message("------[Build as SHARE library]------")
    add_library(awcv SHARED ${AWCV_SRCS})
    target_link_libraries(awcv PRIVATE ${OpenCV_LIBS})
endif()

install(TARGETS awcv
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)
install(FILES ${AWCV_INCLUDE_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)
