


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)



# cudnn,tensorrt  In direct lib and include: /usr/include/aarch64-linux-gnu/,/usr/lib/aarch64-linux-gnu/

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall  -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")




include_directories("/usr/local/cuda/include")
include_directories(${OPENCV_INCLUDE_DIRS})
include_directories(${CMAKE_BINARY_DIR})

# set(SRC_LIST SmokePhone.cpp  smoke_classification.cpp phone.cpp)
set(SRC_LIST SmokePhone_new.cpp  smoke_classification.cpp )
add_library(smoke STATIC ${SRC_LIST} )

target_link_libraries(smoke yolo  personbase  ppyoloe)
target_link_libraries(smoke utils)
target_link_libraries(smoke  ${INFER_DEP_LIBS})
target_link_libraries(smoke ${OpenCV_LIBS})


 #add_definitions(-O2 -pthread)

