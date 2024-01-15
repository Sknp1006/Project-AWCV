cmake_minimum_required(VERSION 3.5)
project(Project-AWCV VERSION 1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# fuck windows, never use

set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE STRING "Install path prefix, prepended onto install directories." FORCE)
message("CMAKE_INSTALL_PREFIX: " ${CMAKE_INSTALL_PREFIX})

if(UNIX)
set(OpenCV_DIR "${VCPKG_INSTALLED_DIR}/arm64-osx/share/opencv4")
endif(UNIX)

set(3RD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty")
set(CMAKE_PREFIX_PATH ${3RD_PARTY_DIR})

# opencv4
# set(OpenCV_DIR "${3RD_PARTY_DIR}/opencv/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
message()

# find_package(Boost REQUIRED COMPONENTS filesystem system)
find_package(NumCpp REQUIRED)

# numcpp
find_package(NumCpp REQUIRED)
message()

include_directories(${CMAKE_SOURCE_DIR}/src)
file(GLOB_RECURSE AWCV_SRCS ${CMAKE_SOURCE_DIR}/src/*.cpp)

option(BUILD_STATIC "Build as SHARE or STATIC library, default STATIC" ON)
message(STATUS "BUILD_STATIC:  ${BUILD_STATIC}")
message(STATUS "BUILD_TYPE:  ${CMAKE_BUILD_TYPE}")

if (BUILD_STATIC)
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
