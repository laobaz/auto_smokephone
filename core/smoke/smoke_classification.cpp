#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>


#include "smoke_classification.hpp"


#include "../utils/httpUtil.hpp"

#include "../utils/rapidjson/writer.h"
#include "../utils/rapidjson/stringbuffer.h"
#include "../utils/subUtils.hpp"


#include "../common/yolo/yolo_common.hpp"
#include <xtensor.hpp>



// static const char* class_names[] = {
//     "normal",
//     "phone", 
//     "smoke"
// };


const static std::string model_file={"../models/smoke/kailei/smoke_phone_mbC_sim.trt"};


const int batch_size = 1;
const int RES_INPUT_HW=299;  //mobile net改
static std::vector<int> input_shape = {batch_size, RES_INPUT_HW, RES_INPUT_HW, 3};    //yolov4 keras/tf 

// {0: 'normal', 1: 'phone', 2: 'smoke'}
static int resnetoutputsize=3;

static nvinfer1::IRuntime* runtime{nullptr};
static nvinfer1::ICudaEngine* engine{nullptr};

static std::unordered_map<int, TrtSampleUniquePtr<nvinfer1::IExecutionContext>> executionContexts;
static std::unordered_map<int, std::unique_ptr<std::mutex> > contextlocks;

static int gpuNum=0;

std::unordered_map<int, std::vector<float> >  SmokeClassification::m_input_datas;

// Default constructor
SmokeClassification::SmokeClassification () { 

    ANNIWOLOG(INFO) << "SmokeClassification(): call initInferContext!" ;

    gpuNum = initInferContext(
                    model_file.c_str(), 
                    &runtime,
                    &engine);

    

    ANNIWOLOG(INFO) << "SmokeClassification(): Success initialized!" ;
}

// Destructor
SmokeClassification::~SmokeClassification () {
    // destroy the engine
    for(int i=0;i<globalINICONFObj.ANNIWO_NUM_INSTANCE_SMOKEPHONE;i++)
    {
        executionContexts[i]->destroy();
    }

    engine->destroy();
    runtime->destroy();

}



void SmokeClassification::initTracks() 
{
    cudaSetDevice(gpuNum);

    int cntID=0;
    //只生成ANNIWO_NUM_INSTANCE_FIRE个实例
    while(cntID < globalINICONFObj.ANNIWO_NUM_THREAD_SMOKEPHONE)
    {
        ANNIWOLOG(INFO) << "SmokeClassification::initTracks insert instance" <<"cntID:"<<cntID;

        std::vector<float> input_data(batch_size * CHANNELS * RES_INPUT_HW * RES_INPUT_HW,1.0);
        std::pair<int, std::vector<float> > itempair2(cntID,std::move(input_data));
        m_input_datas.insert( std::move(itempair2) );

////////////////////////

        cntID++;
    }

    executionContexts.clear();
    contextlocks.clear();
    for(int i=0;i<globalINICONFObj.ANNIWO_NUM_INSTANCE_SMOKEPHONE;i++)
    {
        TrtSampleUniquePtr<nvinfer1::IExecutionContext>  context4thisCam(engine->createExecutionContext());
        std::pair<int, TrtSampleUniquePtr<nvinfer1::IExecutionContext> > tmpitem{i,std::move(context4thisCam)};

        executionContexts.insert(std::move(tmpitem));

        std::unique_ptr<std::mutex>    newmutexptr(new std::mutex);
        std::pair<int, std::unique_ptr<std::mutex> > tmplockitem{i,std::move(newmutexptr)};

        contextlocks.insert(std::move(tmplockitem));
    }

}

//todo:polygonSafeArea
void SmokeClassification::detect(  int camID,int instanceID,  cv::Mat face_img, std::vector<float>& outresults ) 
{    

    int img_w = face_img.cols;
    int img_h = face_img.rows;

    // std::vector<float> input_data(batch_size * CHANNELS * RES_INPUT_HW * RES_INPUT_HW,1.0);


    // const std::string input_image_path {"../images/23enlargeface_img_8.jpg"};//test
    // cv::Mat img = cv::imread(input_image_path);
    // cv::Mat pr_img = static_resizeLINEAR(img,RES_INPUT_HW,RES_INPUT_HW);

    cv::Mat pr_img = static_resizeLINEAR(face_img,RES_INPUT_HW,RES_INPUT_HW);
    ANNIWOLOG(INFO)  << "SmokeClassification static_resized image" <<"camID:"<<camID ;

    //python: bgr_cropimg = preprocess_inputInception(bgr_cropimg)
    //python代码中用的bgr输入!
    blobFromImageAndNorm_resnetv2style(pr_img,m_input_datas[instanceID]);

    ANNIWOLOG(INFO)  << "SmokeClassification blob ok." <<"camID:"<<camID ;

    // {0: 'normal', 1: 'phone', 2: 'smoke'}
    std::vector<float> out_data(resnetoutputsize,0.0);
    cudaSetDevice(gpuNum);


    int choiceIntVal = randIntWithinScale(globalINICONFObj.ANNIWO_NUM_INSTANCE_SMOKEPHONE);
    std::unordered_map<int, TrtSampleUniquePtr<nvinfer1::IExecutionContext> >::iterator iterCamInstance =  executionContexts.find(choiceIntVal);
    std::unordered_map<int, std::unique_ptr<std::mutex> >::iterator iterCamInstancelock =  contextlocks.find(choiceIntVal);
    ANNIWOCHECK(iterCamInstance != executionContexts.end()) ;

    // run inference
    auto start = std::chrono::system_clock::now();
    //:uniq_ptr不能转移，转移之后原来的就没有了！！！
    ANNIWOLOG(INFO)  << "SmokeClassification call TrtGPUInfer." <<"camID:"<<camID ;

///////////////////////////////
//  std::vector<std::size_t> shape = { 1,256,256,3 };
//  auto a1 = xt::adapt(input_data, shape);
// //  xt::dump_npy("inputimage_xtensor.npy", a1);
//  std::cout <<"input_xtensor:"<< a1 << std::endl;

///////////////////////////////

    int inputStreamSize = batch_size * CHANNELS * RES_INPUT_HW * RES_INPUT_HW;

    TrtGPUInfer(*iterCamInstance->second,gpuNum, *iterCamInstancelock->second, m_input_datas[instanceID].data(), out_data.data(), resnetoutputsize,
                inputStreamSize,  "input_1", "dense_2","SmokeClassification:" );

    auto end = std::chrono::system_clock::now();

    ANNIWOLOG(INFO) <<"SmokeClassification:infer time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms,"<<"camID:"<<camID ;

    outresults=out_data;

    std::stringstream buffer;  
    for(float item:outresults)
    {
        buffer <<item<<",";  
    }
    std::string contents(buffer.str());

    ANNIWOLOG(INFO) << "SmokeClassification:exit detect()" <<"camID:"<<camID<<"res result:"<<contents<<std::endl;

    return;
}
