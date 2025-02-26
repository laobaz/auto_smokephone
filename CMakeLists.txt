cmake_minimum_required(VERSION 2.8)
project (auto_smokePhone_labeler)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++14 -pedantic -Wextra")


if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("this is aarch64 cpu")
    set(ISAarch64    ON)  
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
	message("this is x86_64 cpu")
    set(ISX86_64    ON)  

endif()


find_package(Threads)

#why below not working
# find_package(Eigen3 REQUIRED)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

message("OpenCV INCLUDE:" ${OPENCV_INCLUDE_DIRS})
#add_definitions(-O2 -pthread)

# FIND_PACKAGE(OpenMP)
# if(OPENMP_FOUND)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#     set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# endif()

find_package( Boost 1.52 COMPONENTS system thread REQUIRED )


include_directories("/usr/local/cuda/include")

#盒子上面
#tensorrt头文件:/usr/include/aarch64-linux-gnu/NvInfer.h
#tensorrt库放在/usr/lib/aarch64-linux-gnu里边


#x86_64: 
#/usr/cuda/lib64/




message("CUDA_CUDART_LIBRARY: ${CUDA_CUDART_LIBRARY}")

if(ISAarch64)
    set(INFER_DEP_LIBS
        libnvinfer.so
        ${CUDA_CUDART_LIBRARY}
    )
else()
    set(INFER_DEP_LIBS
        /usr/local/cuda/lib64/libnvinfer.so
        ${CUDA_CUDART_LIBRARY}
    )
endif()






##util需要 cryptopp for cryptopp

#./x86_64-linux-gnu/libcryptopp.a
# message("cryptopp lib:/opt/cryptopp/lib")
# link_directories("/opt/cryptopp/lib")


#add_subdirectory(core/safeErea) 

#add_subdirectory(core/helmet) 

add_subdirectory(core/personbase) 

#add_subdirectory(core/window) 
#add_subdirectory(core/uptruck)

#add_subdirectory(core/firesmog) 
add_subdirectory(core/smoke) 
#add_subdirectory(core/xunjian) 
#add_subdirectory(core/zhuangxieyou) 
add_subdirectory(core/utils) 
#add_subdirectory(core/personfall) 
add_subdirectory(core/common/yolo) 

add_subdirectory(core/common/mot) 

add_subdirectory(core/tinyxml) 

add_subdirectory(core/common/ppyoloe)
#add_subdirectory(core/common/rtdetrs)

if(ISX86_64)
# add_subdirectory(core/convery) 
endif()


#add_subdirectory(core/chepai) 
#add_subdirectory(core/mask) 


#add_subdirectory(framing) 


message("LOCAL_STATIC_LIBS: ${LOCAL_STATIC_LIBS}")
message("CMAKE_THREAD_LIBS_INIT: ${CMAKE_THREAD_LIBS_INIT}")

############################################################

add_executable(Auto_SmokePhone auto_smokephone.cc core/common/logging.cpp core/LogRotateAnniwo.cpp)

target_link_libraries(Auto_SmokePhone ${LOCAL_STATIC_LIBS})



target_link_libraries(Auto_SmokePhone  smoke personbase)








