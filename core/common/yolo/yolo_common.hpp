#ifndef _YOLO4_COMMON_H_
#define _YOLO4_COMMON_H_



#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <uuid/uuid.h>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include "../logging.h"





#define NMS_THRESH 0.45



extern const float color_list[80][3];


extern const int INPUT_W ;
extern const int INPUT_H ;
extern const int CHANNELS;
extern const int NUM_ANCHORS;
extern const char* INPUT_BLOB_NAME;
extern const char* OUTPUT_BLOB_NAME;


void TrtGPUInfer( nvinfer1::IExecutionContext& context, int gpuNum, std::mutex& mutexlock, float* input, float* output, int output_size,
                int input_size= 1 * CHANNELS * INPUT_H * INPUT_W, 
                 const char* inputblobname=INPUT_BLOB_NAME,const char* ouputblobname=OUTPUT_BLOB_NAME,std::string logstr="" );




//CUBIC
cv::Mat static_resize(cv::Mat img,int inINPUT_W, int inINPUT_H);
//Linear
cv::Mat static_resizeLINEAR(cv::Mat img,int inINPUT_W, int inINPUT_H);
//nearest
cv::Mat static_resizeNEAREST(cv::Mat img,int inINPUT_W, int inINPUT_H);
cv::Mat static_resizex(cv::Mat img,int xINPUT_W, int xINPUT_H,float& alpha );






struct ObjectHisSave
{
    Object detObj;
    cv::Mat cutpic;
};

struct ObjectTimeSave
{
    Object detObj;
    std::chrono::system_clock::time_point  reportedtime;
};



struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

void generate_grids_and_stride(std::vector<int>& strides, std::vector<GridAndStride>& grid_strides);

inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

inline float intersection_area(const ObjectHisSave& a, const ObjectHisSave& b)
{
    cv::Rect_<float> inter = a.detObj.rect & b.detObj.rect;
    return inter.area();
}

//结果IOU去重判断
inline bool isResultDuplicated(const std::vector<Object>& lastObjects, const std::vector<Object>& filteredObjects, bool considerLabel=true)
{
    float inter_area =0.0;
    float union_area =0.0;
    int duplicatedboxCnt = 0;
    

    for (size_t i = 0; i < filteredObjects.size(); i++)
    {
        const Object& obj = filteredObjects[i];

        for (auto& last_det : lastObjects) 
        {
                //与box_det求ioa
                // intersection over union
                inter_area = intersection_area(obj, last_det);
                union_area = obj.rect.area() + last_det.rect.area() - inter_area;

                if(considerLabel)
                {
                    //iou>0.8 found duplicated box
                    if(inter_area / union_area > 0.8 &&  (last_det.label == obj.label) )
                    {
                        duplicatedboxCnt+=1;
                        break;
                    }   
                }else
                {
                    //iou>0.8 found duplicated box
                    if(inter_area / union_area > 0.8  )
                    {
                        duplicatedboxCnt+=1;
                        break;
                    }  
                }


        }

    }
    if((int)filteredObjects.size() == duplicatedboxCnt )
    {
        //all duplicated!
        return true;
    }
    return false;

}

void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right);

void qsort_descent_inplace(std::vector<Object>& objects);

void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold);


void blobFromImageAndNormalize(cv::Mat img,std::vector<float>& arrvectorOut);
void blobFromImageAndNorm_resnetv2style(cv::Mat img,std::vector<float>& arrvectorOut);

inline void blobFromImageAndNorm_paddlestyle(cv::Mat img,std::vector<float>& arrvectorOut){
    int img_h = img.rows;
    int img_w = img.cols;

    //paddle mobilenet
    //python: mean=[125.31, 122.95, 113.86], std=[62.99, 62.08, 66.7]
    // std::vector<float> means={125.31, 122.95, 113.86};
    // std::vector<float> stds={62.99, 62.08, 66.7};

    //paddle resnet
    std::vector<float> means={127.5,127.5,127.5};
    std::vector<float> stds={63.0,63.0,63.0};


    // //对应tf模型为h,w,c.而paddle/pytorch的模型输入是c,h,w,此处应该注意一下顺序！
    // //有关系，i*(列长*对长)+j*对长+k 修改如下：
    // paddle同pytorch 
    for (size_t c = 0; c < CHANNELS; c++) 
    {
        for (size_t  h = 0; h < img_h; h++) 
        {
            for (size_t w = 0; w < img_w; w++) 
            {
                // mean = np.float32(np.array(mean).reshape(1, 1, -1))
                // std = np.float32(np.array(std).reshape(1, 1, -1))
                // img = (img - mean) / std
                arrvectorOut[c*img_h*img_w  + h*img_w + w] =
                    ( ((float)img.at<cv::Vec3b>(h, w)[c]) - means[c] ) / stds[c];
            }

        }
    }
    return ;
}


void decode_outputs(std::vector<float>& prob, std::vector<Object>& objects, float scale, const int img_w, const int img_h,
    float BBOX_CONF_THRESH,int NUM_CLASSES, std::string& logtitleSTRin, bool isDebug=false);





void yolov4_detection_staff(std::unordered_map<int, std::vector<float> >& m_input_datas,
    int camID,int instanceID, cv::Mat img,
    nvinfer1::IRuntime* runtime,
    nvinfer1::ICudaEngine* engine,
    TrtSampleUniquePtr<nvinfer1::IExecutionContext>& context,
    int gpuNum,
    std::unique_ptr<std::mutex>&  lockptr,
    int YOLO4_OUTPUT_SIZE, int INPUT_W, int INPUT_H, 
    /*out*/std::vector<Object>& objects,
    float BBOX_CONF_THRESH,
    int NUM_CLASSES,
    std::string logstring );


void anniwo_debug_draw_objects(const cv::Mat bgr, const std::vector<Object>& objects, std::string output_path, bool isDrawResult);


#endif