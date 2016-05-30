project("Mnist")
cmake_minimum_required(VERSION 2.8)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# turn on c++11
add_definitions(-std=c++11)

find_package(Caffe)
find_package(OpenCV)
find_package(Boost COMPONENTS system)
include_directories(${Caffe_INCLUDE_DIRS})
add_executable(go
    go.cpp)

target_link_libraries(
    go
    ${OpenCV_LIBS}
    "${Caffe_DIR}/lib/libcaffe.so"
    /usr/lib/x86_64-linux-gnu/libglog.so
    ${Boost_SYSTEM_LIBRARY_RELEASE}
)

add_executable(test
    test.cpp)

target_link_libraries(
    test
    ${OpenCV_LIBS}
    "${Caffe_DIR}/lib/libcaffe.so"
    /usr/lib/x86_64-linux-gnu/libglog.so
    ${Boost_SYSTEM_LIBRARY_RELEASE}
)
