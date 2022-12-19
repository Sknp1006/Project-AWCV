cmake_minimum_required(VERSION 3.5)

project(Project-AWCV VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS regex system filesystem REQUIRED)

add_executable(${PROJECT_NAME} 
    main.cc
    include/core.hpp
    include/dft.hpp
    include/face.hpp
    include/file.hpp
    include/opticalFlow.hpp
    include/region.hpp
    include/sfm.hpp
    include/moire/moire.hpp
    src/core.cpp
    src/dft.cpp
    src/face.cpp
    src/file.cpp
    src/opticalFlow.cpp
    src/region.cpp
    src/sfm.cpp
    src/moire/moire.cpp
    examples/CoreTest.hpp
)
include_directories(
    ${PROJECT_SOURCE_DIR}/include/
    ${PROJECT_SOURCE_DIR}/examples/
    ${OpenCV_INCLUDE_DIRS} 
    ${Boost_INCLUDE_DIRS}
    )
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${Boost_LIBRARIES})
