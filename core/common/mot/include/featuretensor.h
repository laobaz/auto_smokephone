#ifndef FEATURETENSOR_H
#define FEATURETENSOR_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>


#include "model.hpp"
#include "datatype.h"
#include "../../yolo/yolo_common.hpp"



class FeatureTensor {
public:
    FeatureTensor(const int maxBatchSize, const cv::Size imgShape, const std::vector<int>& camIDs);
    ~FeatureTensor();

public:
    bool getRectsFeature(const cv::Mat img, DETECTIONS& det,int camID,int instanceID);
    
    void loadEngine(const std::string& model_file,const std::string* model_file2) ;

    //whichPredictor:0 for m_main_predictor,1 for m_main_predictor2
    void doInference(std::vector<cv::Mat>& imgMats,int camID,int whichPredictor,int instanceID);

private:

    int mat2stream(std::vector<cv::Mat>& imgMats, std::vector<float>& stream);
    //image/255. 的预处理
    int mat2stream2(std::vector<cv::Mat>& imgMats, std::vector<float>& stream);

    void stream2det(std::vector<float>& stream, DETECTIONS& det,std::vector<int>& detIndxlist);
    void stream2det2(std::vector<float>& stream, DETECTIONS& det,std::vector<int>& detIndxlist);

private:
    const int maxBatchSize;
    const cv::Size imgShape;
    const cv::Size imgShape2;

private:
    const int inputStreamSize, outputStreamSize;
    const int inputStreamSize2, outputStreamSize2;
    bool initFlag;
    std::unordered_map<int, std::vector<float> >  m_input_datas;
    std::unordered_map<int, std::vector<float> >  m_input_datas2;
    std::unordered_map<int, std::vector<float> >  m_output_datas;
    std::unordered_map<int, std::vector<float> >  m_output_datas2;

    // BGR format
    float means[3], std[3];
    const std::vector<int>& m_camIDs;


private:
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    std::unordered_map<int, TrtSampleUniquePtr<nvinfer1::IExecutionContext>> executionContexts;
    std::unordered_map<int, std::unique_ptr<std::mutex> > contextlocks;
    int gpuNum;


    nvinfer1::IRuntime* runtime2;
    nvinfer1::ICudaEngine* engine2;

    std::unordered_map<int, TrtSampleUniquePtr<nvinfer1::IExecutionContext>> executionContexts2;
    std::unordered_map<int, std::unique_ptr<std::mutex> > contextlocks2;
    int gpuNum2;

};

#endif
