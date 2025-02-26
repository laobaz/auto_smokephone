#ifndef YOLOX_SMOKEPHONE_H
#define YOLOX_SMOKEPHONE_H
#include <opencv2/opencv.hpp>
#include "../utils/utils_intersection.hpp"
#include "../common/yolo/yolo_common.hpp"
#include "../common/ppyoloe/object_detector.h"

#include <unordered_set>
#include <unordered_map>



class SmokePhoneDetection  {
    public:
        SmokePhoneDetection() ;
        ~SmokePhoneDetection() ;

        void initTracks();
        //todo:polygonSafeArea
        static void detect(  int camID,int instanceID, cv::Mat img, const Polygon* polygonSafeArea_ptr,const std::vector<Object>& in_person_results,std::string &f) ;

    private:
        static  std::unordered_map<int, std::vector<float> >  m_input_datas; 
        static PaddleDetection::ObjectDetector *m_facedetptr;

        

};

#endif 

