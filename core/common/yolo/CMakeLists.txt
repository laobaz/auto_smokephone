
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)




# FIND_PACKAGE(OpenMP)
# if(OPENMP_FOUND)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#     set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# endif()


find_package(xtensor REQUIRED)

message("xtensor_INCLUDE_DIRS: ${xtensor_INCLUDE_DIRS}")

include_directories( ${xtensor_INCLUDE_DIRS})

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories("/usr/local/cuda/include")

# 盒子:cudnn,tensorrt  In direct lib and include: /usr/include/aarch64-linux-gnu/,/usr/lib/aarch64-linux-gnu/
# 服务器：都放在/usr/local/cuda/
link_directories("/usr/local/cuda/lib64")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
#-pedantic ：配合使用 -ansi ，这告诉编译器要严格遵守ANSI标准
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3 -ffast-math -std=c++14 -Wall -pedantic -Wextra -Ofast -Wfatal-errors ")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fopenmp -ffast-math -std=c++14 -Wall -pedantic -Wextra -Ofast -Wfatal-errors ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -ffast-math -std=c++14 -Wall -pedantic -Wextra -Ofast -Wfatal-errors ")

include_directories(${OPENCV_INCLUDE_DIRS})
include_directories( ${Boost_INCLUDE_DIR} )



set(SRC_LIST yolo_common.cpp )
add_library(yolo STATIC ${SRC_LIST} )


target_link_libraries(yolo ${INFER_DEP_LIBS})
target_link_libraries(yolo utils)
target_link_libraries(yolo ${OpenCV_LIBS})
target_link_libraries(yolo ${Boost_LIBRARIES})

add_definitions( -pthread )

