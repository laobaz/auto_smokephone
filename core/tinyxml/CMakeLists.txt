

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)




# cudnn,tensorrt  In direct lib and include: /usr/include/aarch64-linux-gnu/,/usr/lib/aarch64-linux-gnu/


# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall  -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")


#include_directories(${OPENCV_INCLUDE_DIRS})
link_directories("/usr/local/lib")

set(SRC_LIST tinyxml.cpp tinystr.cpp tinyxmlerror.cpp tinyxmlparser.cpp )
add_library(tinyxml STATIC ${SRC_LIST} )

# target_link_libraries(personfall yolo)
# target_link_libraries(personfall utils)
# target_link_libraries(personfall  ${INFER_DEP_LIBS})
# target_link_libraries(personfall ${OpenCV_LIBS})

# add_definitions(-O2 -pthread)


