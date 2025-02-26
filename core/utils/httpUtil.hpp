#ifndef __UTILS_HTTPUTIL_HPP__
#define __UTILS_HTTPUTIL_HPP__

#include <string>    // std::string
#include <chrono>
#include <opencv2/opencv.hpp>
#include "../tinyxml/tinystr.h"
#include "../tinyxml/tinyxml.h"
#include "../common/logging.h"

class HttpUtil{
    public:
        HttpUtil() {}
        ~HttpUtil() {}

        void  post( const char* str2send, size_t sendlength, const std::string& url);
};

void saveImgAndPost(int camID, const std::string taskId, const std::string imgPath, 
            cv::Mat image, std::chrono::system_clock::time_point eventStartTP,
            const std::string& jsonStrIn, size_t jsonlength, const std::string submitUrl);
void save_imgXmL(const cv::Mat& bgr,const std::vector<Object>& objects,std::string &f,std::vector<std::string> &class_names);
extern const std::string ANNIWO_LOG_IMAGES_PATH;

#endif //
