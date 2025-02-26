#ifndef YOLOX_BASEPERSON_H
#define YOLOX_BASEPERSON_H
#include <opencv2/opencv.hpp>
#include <unordered_set>
#include <unordered_map>
#include "../utils/utils_intersection.hpp"
#include "../utils/subUtils.hpp"

#include "../common/ppyoloe/object_detector.h"

#include "../common/yolo/yolo_common.hpp"

extern const std::vector<std::string> person_base_class_namesJYZ ;
extern const std::vector<std::string> person_base_class_namesCOCO ;

inline bool isVehicle(int inlabel)
{
    if(globalINICONFObj.domain_config == ANNIWO_DOMANI_LIANGKU)
    {
        if(
           person_base_class_namesCOCO[inlabel] == std::string("bicycle")
        || person_base_class_namesCOCO[inlabel] == std::string("car")
        || person_base_class_namesCOCO[inlabel] == std::string("motorcycle")
        || person_base_class_namesCOCO[inlabel] == std::string("bus")
        || person_base_class_namesCOCO[inlabel] == std::string("truck")
        )
        {
            return true;
        }
    }
    else if(globalINICONFObj.domain_config == ANNIWO_DOMANI_JIAYOUZHAN)
    {

        if(  person_base_class_namesJYZ[inlabel] == std::string("car")
            || person_base_class_namesJYZ[inlabel] == std::string("tank_truck")
            || person_base_class_namesJYZ[inlabel] == std::string("truck")
            || person_base_class_namesJYZ[inlabel] == std::string("motor")
            || person_base_class_namesJYZ[inlabel] == std::string("unloader")
            || person_base_class_namesJYZ[inlabel] == std::string("cement_truck")
        )
        {
            return true;
        }


    }
    else
    {
        ANNIWOCHECK(false);
    }

    return false;

}

inline bool isPerson(int inlabel)
{
    if(globalINICONFObj.domain_config == ANNIWO_DOMANI_LIANGKU)
    {
        if(person_base_class_namesCOCO[inlabel]  == std::string("person"))
        {
            return true;
        }
    }
    else if(globalINICONFObj.domain_config == ANNIWO_DOMANI_JIAYOUZHAN)
    {
        if(person_base_class_namesJYZ[inlabel]  == std::string("person"))
        {
            return true;
        }
    }
    else
    {
        ANNIWOCHECK(false);
    }
    return false;

}

inline std::string getPersonCarbaseClassName(int inlabel)
{
    if(globalINICONFObj.domain_config == ANNIWO_DOMANI_LIANGKU)
    {
        return person_base_class_namesCOCO[inlabel];
    }
    else if(globalINICONFObj.domain_config == ANNIWO_DOMANI_JIAYOUZHAN)
    {
        return person_base_class_namesJYZ[inlabel];

    }
    else
    {
        ANNIWOCHECK(false);
    }
    return std::string("");
}

class BasePersonDetection  {
    public:
        BasePersonDetection() ;
        ~BasePersonDetection() ;

        void initTracks();

        //todo:polygonSafeArea
        static int detect(  int camID, int instanceID, cv::Mat img, std::vector<Object>& objects);

    private:
        static  std::unordered_map<int, std::vector<float> >  m_input_datas;

        
};

//临时记录结构，用于检查该trackid对应的目标停留时间是否已经达到阈值
// {camID,{trackid,stayStart}}
struct AnniwoTrackRecord
{
    Object detresult; //
    std::chrono::system_clock::time_point startPoint;  //stayStart时间
};

#endif // YOLOX_DEMO_H

