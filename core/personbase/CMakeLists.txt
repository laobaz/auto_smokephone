

set(CMAKE_BUILD_TYPE Debug)




# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall  -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")


include_directories(
    ${OPENCV_INCLUDE_DIRS} 
    /usr/include/eigen3/
    )

find_package( Boost 1.52 COMPONENTS system thread REQUIRED )
include_directories( ${Boost_INCLUDE_DIR} )
    

include_directories("/usr/local/cuda/include")
include_directories("/usr/local/include/")
link_directories("/usr/local/lib")

set(SRC_LIST basePerson.cpp  )
add_library(personbase STATIC ${SRC_LIST} )


target_link_libraries(personbase deepsort)
target_link_libraries(personbase utils)
target_link_libraries(personbase yolo)
target_link_libraries(personbase ppyoloe)



#for paddle, and opencv also neccessary for paddle
target_link_libraries(personbase ${INFER_DEP_LIBS})
target_link_libraries(personbase ${OpenCV_LIBS})
target_link_libraries(personbase ${Boost_LIBRARIES})


add_definitions( -pthread)

