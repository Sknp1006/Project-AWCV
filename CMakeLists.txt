cmake_minimum_required(VERSION 3.5)

project(Project-AWCV VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR})

message("CMAKE_INSTALL_PREFIX: " ${CMAKE_INSTALL_PREFIX}/publish)

# 设置可执行程序输出目录
set(publish_bin_debug 			${CMAKE_CURRENT_SOURCE_DIR}/publish/x64/bin/debug64)
set(publish_bin_release 		${CMAKE_CURRENT_SOURCE_DIR}/publish/x64/bin/release64)
# 设置库文件输出目录
set(publish_lib_debug 			${CMAKE_CURRENT_SOURCE_DIR}/publish/x64/lib/debug64)
set(publish_lib_release 		${CMAKE_CURRENT_SOURCE_DIR}/publish/x64/lib/release64)
# 指定可执行程序输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG 	${publish_bin_debug})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE 	${publish_bin_release})
# 指定 库文件输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG 	${publish_lib_debug})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RLEASE 	${publish_lib_release})


add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)

message("CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR})    # 项目路径

# 获取src下的所有文件
file(GLOB_RECURSE AWCV_SRCS src/*.cpp)
# message("AWCV_SRCS" ${AWCV_SRCS})
file(GLOB_RECURSE AWCV_INCLUDE_HEADERS include/*.hpp)
# message("AWCV_INCLUDE_HEADERS" ${AWCV_INCLUDE_HEADERS})

#add_library(awcv_dll SHARED ${AWCV_SRCS})
add_library(awcv_lib STATIC ${AWCV_SRCS})

add_executable(${PROJECT_NAME} 
    main.cc
    examples/CoreTest.hpp 
    ${AWCV_INCLUDE_HEADERS}
    ${AWCV_SRCS}
)

include_directories(
    ${PROJECT_SOURCE_DIR}/include/
    ${PROJECT_SOURCE_DIR}/examples/
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE awcv_lib ${OpenCV_LIBS} ${Boost_LIBRARIES})

# install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RLEASE}/lib)
# install(FILES "awcv.hpp" DESTINATION ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RLEASE}/include COMPONENT dev)
