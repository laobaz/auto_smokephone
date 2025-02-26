


option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)


#crypto?
# find_library(CRYPTO_LIB libcrypto++.a REQUIRED)



# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Ofast -Wfatal-errors ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall  -Wfatal-errors ")


include_directories(${OPENCV_INCLUDE_DIRS})

#for cryptopp:/usr/include
#./x86_64-linux-gnu/libcryptopp.a
# message("cryptopp include:/opt/cryptopp/include")
# include_directories("/opt/cryptopp/include")

# message("cryptopp lib:/opt/cryptopp/lib")
# link_directories("/opt/cryptopp/lib")

#for g3log
include_directories("/usr/local/include/")
link_directories("/usr/local/lib")


#add_subdirectory(tinyxml) 

set(SRC_LIST utils_intersection.cpp httpUtil.cpp cycrptUtil.cpp  subUtils.cpp  )
add_library(utils STATIC ${SRC_LIST} )
target_link_libraries(utils curlpp curl z tinyxml)

# target_link_libraries(utils cryptopp)
# target_link_libraries(utils /opt/cryptopp/lib/libcryptopp.a  g3log g3logrotate)
# target_link_libraries(utils /opt/cryptopp/lib/libcryptopp.a  g3log )
target_link_libraries(utils cryptopp  g3log )

target_link_libraries(utils  ${OpenCV_LIBS})

# add_definitions(-O2 )

