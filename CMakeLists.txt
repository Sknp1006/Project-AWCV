cmake_minimum_required(VERSION 3.5)
project(Project-AWCV VERSION 1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
message("------[We are working on Windows now!]------")
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
execute_process(COMMAND cmd /c "echo %VCPKG_ROOT%" OUTPUT_VARIABLE VCPKG_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
elseif(UNIX)
message("------[We are working on Unix now!]------")
execute_process(COMMAND sh -c "echo $VCPKG_ROOT" OUTPUT_VARIABLE VCPKG_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
set(VCPKG_INSTALLED_DIR "${VCPKG_ROOT}/installed/")
message(${VCPKG_INSTALLED_DIR})
set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildtrees/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
endif()
message(STATUS "Using CMAKE_TOOLCHAIN_FILE:  ${CMAKE_TOOLCHAIN_FILE}")

set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install" CACHE STRING "Install path prefix, prepended onto install directories." FORCE)
message("CMAKE_INSTALL_PREFIX: " ${CMAKE_INSTALL_PREFIX})
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

if(UNIX)
set(OpenCV_DIR "${VCPKG_INSTALLED_DIR}/arm64-osx/share/opencv4")
endif(UNIX)

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)
find_package(NumCpp REQUIRED)
message("CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR})    # 项目路径

# # 获取src下的所有文件
# file(GLOB_RECURSE AWCV_SRCS src/*.cpp)
# # message("AWCV_SRCS" ${AWCV_SRCS})
# file(GLOB_RECURSE AWCV_INCLUDE_HEADERS include/awcv.hpp)
# # message("AWCV_INCLUDE_HEADERS" ${AWCV_INCLUDE_HEADERS})

set(AWCV_INCLUDE_HEADERS 
    include/core.hpp
    include/region.hpp
    include/file.hpp)

set(AWCV_SRCS 
    src/core.cpp
    src/region.cpp
    src/file.cpp)

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

option(USE_STATIC "Build as SHARE or STATIC library, default STATIC" ON)
message(STATUS "USE_STATIC:  ${USE_STATIC}")
message(STATUS "BUILD_TYPE:  ${CMAKE_BUILD_TYPE}")

if (USE_STATIC)
    message("------[Build as STATIC library]------")
    add_library(awcv STATIC ${AWCV_SRCS})
    target_link_libraries(awcv PRIVATE ${OpenCV_LIBS} Boost::filesystem)
    # set_target_properties(awcv
    #     PROPERTIES
    #         OUTPUT_NAME "awcv"
    #     )
else()
    message("------[Build as SHARE library]------")
    add_library(awcv SHARED ${AWCV_SRCS})
    target_link_libraries(awcv PRIVATE ${OpenCV_LIBS} Boost::filesystem)
    # set_target_properties(awcv
    #     PROPERTIES
    #         OUTPUT_NAME "awcv"
    #     )
endif()

install(TARGETS awcv
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)
install(FILES ${AWCV_INCLUDE_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)
