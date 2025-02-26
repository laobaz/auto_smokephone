#ifndef YOLOX_SMOKE_CLASSICATION_H
#define YOLOX_SMOKE_CLASSICATION_H
#include <opencv2/opencv.hpp>
#include "../utils/utils_intersection.hpp"
#include "../common/yolo/yolo_common.hpp"
#include <unordered_set>
#include <unordered_map>



class SmokeClassification  {
    public:
        SmokeClassification() ;
        ~SmokeClassification() ;

        void initTracks();
        //todo:polygonSafeArea
        static void detect(  int camID, int instanceID,cv::Mat face_img, std::vector<float>& outresults );

    private:
        static  std::unordered_map<int, std::vector<float> >  m_input_datas;
        
};

#endif 

