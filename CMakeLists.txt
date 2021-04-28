project(rgbd_mapping)
cmake_minimum_required(VERSION 3.0)

add_definitions(-std=c++17)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


find_package(OpenCV REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})


add_library(params src/params.cpp)

add_executable(main src/main.cpp)
target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(main ${PCL_LIBRARIES})
target_link_libraries(main params)

