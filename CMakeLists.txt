project(rgbd_mapping)
cmake_minimum_required(VERSION 3.0)

add_definitions(-std=c++14)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})


LIST(APPEND CMAKE_MODULE_PATH /home/yuwei/g2o/cmake_modules)
#set (G2O_ROOT /home/yuwei/g2o)
#SET( G2O_ROOT /usr/local/include/g2o )

find_package(G2O REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(CSparse REQUIRED)
include_directories(${CSPARSE_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})


include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${EIGEN_INCLUDE_DIRS}
        ${CSPARSE_INCLUDE_DIR}
        ${Cholmod_INCLUDE_DIR}
        ${G2O_INCLUDE_DIR}
        ${G2O_INCLUDE_DIRS}
        /usr/include/suitesparse
)

SET(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only g2o_types_sba g2o_types_slam3d g2o_csparse_extension g2o_opengl_helper g2o_solver_dense g2o_stuff g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse )

link_directories(${G2O_LIBRARY_DIRS})
link_libraries(${G2O_LIBRARIES})


find_package(CSparse REQUIRED)
FIND_PACKAGE(Cholmod REQUIRED)



#add_library(rgbd_lib src/params.cpp src/utils.cpp src/rgbd_slam.cpp)
#target_link_libraries(rgbd_lib ${G2O_LIBS})

add_executable(main src/main.cpp src/rgbd_slam.cpp src/params.cpp src/utils.cpp)
target_link_libraries(main
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        ${G2O_LIBS})


