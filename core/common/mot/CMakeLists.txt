

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_BUILD_TYPE Debug)


find_package(OpenCV REQUIRED)


include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ./include
    /usr/include/eigen3/
)

#不需要？
# link_directories("/opt/protobuf/lib")

aux_source_directory(./src SRC_DIR)

# ===== deepsort =====
add_library(deepsort ${SRC_DIR})
target_link_libraries(deepsort 
     ${OpenCV_LIBS} 
     uuid
     ${INFER_DEP_LIBS}
)


# # ===== demo =====
# add_executable(demo ./demo.cpp)
# target_link_libraries(demo 
#      ${OpenCV_LIBS} 
#      deepsort
# )



