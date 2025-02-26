#include "featuretensor.h"
#include <fstream>
#include <chrono>
#include <iostream>
#include <memory>
#include <vector>
#include <numeric>

#include "../../../utils/subUtils.hpp"




#define INPUTSTREAM_SIZE (maxBatchSize*3*imgShape.area())
#define OUTPUTSTREAM_SIZE (maxBatchSize*512)

#define INPUTSTREAM_SIZE2 (maxBatchSize*3*imgShape2.area())
#define OUTPUTSTREAM_SIZE2 (maxBatchSize*256)

FeatureTensor::FeatureTensor(const int maxBatchSize, const cv::Size imgShape, const std::vector<int>& camIDs) 
        : maxBatchSize(maxBatchSize), imgShape(imgShape),imgShape2(cv::Size(125, 250)),  
        inputStreamSize(INPUTSTREAM_SIZE), outputStreamSize(OUTPUTSTREAM_SIZE),
        inputStreamSize2(INPUTSTREAM_SIZE2), outputStreamSize2(OUTPUTSTREAM_SIZE2),
        m_camIDs(camIDs),
        runtime(nullptr),engine(nullptr),gpuNum(0),runtime2(nullptr),engine2(nullptr),gpuNum2(0)
{
    

    means[0] = 0.485, means[1] = 0.456, means[2] = 0.406;
    std[0] = 0.229, std[1] = 0.224, std[2] = 0.225;

    initFlag = false;
    ANNIWOLOG(INFO) <<"FeatureTensor:maxBatchSize:"<<maxBatchSize<<" featureDim:"<<256;

}

FeatureTensor::~FeatureTensor() {

    //vector不必手动释放
    if (initFlag) {


        contextlocks.clear();
		contextlocks2.clear();

        if(engine)
        {
            // destroy the engine
                    
            for(int i=0;i<globalINICONFObj.ANNIWO_NUM_INSTANCE_PERSONBASE;i++)
            {
                executionContexts[i]->destroy();
            }

            engine->destroy();
            runtime->destroy();
        }
        if(engine2)
        {
            // destroy the engine
            
            for(int i=0;i<globalINICONFObj.ANNIWO_NUM_INSTANCE_PERSONBASE;i++)
            {
                executionContexts2[i]->destroy();
            }

            engine2->destroy();
            runtime2->destroy();
        }
        
    }
}

//多线程
bool FeatureTensor::getRectsFeature(const cv::Mat img, DETECTIONS& det,int camID,int instanceID) {
    std::vector<cv::Mat> mats,matsVehicle;
    int indx=0,cnt=0;
    std::vector<int> mats2det_indexlist;
    std::vector<int> matsVehicle2det_indexlist;
    for (auto& dbox : det) {
        cv::Rect rect = cv::Rect(int(dbox.tlwh(0)), int(dbox.tlwh(1)),
                                 int(dbox.tlwh(2)), int(dbox.tlwh(3)));
        rect.x -= (rect.height * 0.5 - rect.width) * 0.5;
        rect.width = rect.height * 0.5;
        rect.x = (rect.x >= 0 ? rect.x : 0);
        rect.y = (rect.y >= 0 ? rect.y : 0);
        rect.width = (rect.x + rect.width <= img.cols ? rect.width : (img.cols - rect.x));
        rect.height = (rect.y + rect.height <= img.rows ? rect.height : (img.rows - rect.y));

        cv::Mat tempMatCut = img(rect).clone();
        cv::Mat tempMat;

        if(globalINICONFObj.domain_config == ANNIWO_DOMANI_JIAYOUZHAN)//from subutils
        {
            if(dbox.classID == 0)//person
            {
                cv::resize(tempMatCut, tempMat, imgShape);

                mats.push_back(tempMat);
                mats2det_indexlist.push_back(indx);
            }else if(dbox.classID == 1)//car
            {
                cv::resize(tempMatCut, tempMat, imgShape2);
                matsVehicle.push_back(tempMat);
                matsVehicle2det_indexlist.push_back(indx);
            }
        }else
        {
            cv::resize(tempMatCut, tempMat, imgShape);

            mats.push_back(tempMat);
            mats2det_indexlist.push_back(indx);
        }

        indx++;
        cnt++;


        if(cnt >= maxBatchSize)
        {//检测一次
            /////////////代码与下同。
            if(mats2det_indexlist.size() > 0)
            {
                //whichPredictor:0 for m_main_predictor,1 for m_main_predictor2
                doInference(mats,camID,0,instanceID);
                // decode output to det
                stream2det(m_output_datas[instanceID], det, mats2det_indexlist);
            }


            if(globalINICONFObj.domain_config == ANNIWO_DOMANI_JIAYOUZHAN)
            {
                if(matsVehicle2det_indexlist.size() > 0)
                {
                    //whichPredictor:0 for m_main_predictor,1 for m_main_predictor2
                    doInference(matsVehicle,camID,1,instanceID);
                    // decode output to det
                    stream2det2(m_output_datas2[instanceID], det, matsVehicle2det_indexlist);
                }

            }
            ///////////////
            mats2det_indexlist.clear();
            matsVehicle2det_indexlist.clear();
            cnt=0;
            mats.clear();
            matsVehicle.clear();

        }

    }

    if(cnt > 0)
    {
        /////////////
        //whichPredictor:0 for m_main_predictor,1 for m_main_predictor2
        if(mats2det_indexlist.size() > 0)
        {
            //whichPredictor:0 for m_main_predictor,1 for m_main_predictor2
            doInference(mats,camID,0,instanceID);
            // decode output to det
            stream2det(m_output_datas[instanceID], det, mats2det_indexlist);
        }


        if(globalINICONFObj.domain_config == ANNIWO_DOMANI_JIAYOUZHAN)
        {
            if(matsVehicle2det_indexlist.size() > 0)
            {
                //whichPredictor:0 for m_main_predictor,1 for m_main_predictor2
                doInference(matsVehicle,camID,1,instanceID);
                // decode output to det
                stream2det2(m_output_datas2[instanceID], det, matsVehicle2det_indexlist);
            }

        }
        ///////////////
    }

    #ifdef ANNIWO_INTERNAL_DEBUG
    ANNIWOLOG(INFO) << "getRectsFeature Succeed !"<<"camID:"<<camID<<"instanceID:"<<instanceID ;
    #endif

    return true;
}



void FeatureTensor::loadEngine(const std::string& model_file,const std::string* model_file2)
{

    int cntID=0;
    //只生成ANNIWO_NUM_INSTANCE_PERSONBASE个实例
    while(cntID < globalINICONFObj.ANNIWO_NUM_THREAD_PERSONBASE)
    {
        ANNIWOLOG(INFO) << "FeatureTensor::loadEngine: insert instance" <<"cntID:"<<cntID<<" ";


        std::vector<float> input_data(inputStreamSize);
        std::pair<int, std::vector<float> > itempair1(cntID,std::move(input_data));
        m_input_datas.insert( std::move(itempair1) );  

        std::vector<float> output_data(outputStreamSize);
        std::pair<int, std::vector<float> > itempair2(cntID,std::move(output_data));
        m_output_datas.insert( std::move(itempair2) );    

        cntID++;
    }


    gpuNum = initInferContext(
                model_file.c_str(), 
                &runtime,
                &engine);
    

    executionContexts.clear();
    contextlocks.clear();
    for(int i=0;i<globalINICONFObj.ANNIWO_NUM_INSTANCE_PERSONBASE;i++)
    {
        TrtSampleUniquePtr<nvinfer1::IExecutionContext>  context4thisCam(engine->createExecutionContext());
        std::pair<int, TrtSampleUniquePtr<nvinfer1::IExecutionContext> > tmpitem{i,std::move(context4thisCam)};

        executionContexts.insert(std::move(tmpitem));


        std::unique_ptr<std::mutex>    newmutexptr(new std::mutex);
        std::pair<int, std::unique_ptr<std::mutex> > tmplockitem{i,std::move(newmutexptr)};

        contextlocks.insert(std::move(tmplockitem));
    }



    //for vehicle encoder
    if(model_file2)
    {

        int cntID=0;
        //只生成ANNIWO_NUM_INSTANCE_PERSONBASE个实例
        while(cntID < globalINICONFObj.ANNIWO_NUM_THREAD_PERSONBASE)
        {
            ANNIWOLOG(INFO) << "FeatureTensor::loadEngine : model_file2 insert instance" <<"cntID:"<<cntID;


            std::vector<float> input_data(inputStreamSize2);
            std::pair<int, std::vector<float> > itempair1(cntID,std::move(input_data));
            m_input_datas2.insert( std::move(itempair1) );  

            std::vector<float> output_data(outputStreamSize2);
            std::pair<int, std::vector<float> > itempair2(cntID,std::move(output_data));
            m_output_datas2.insert( std::move(itempair2) );  

            cntID++;
        }

        gpuNum2 = initInferContext(
            model_file2->c_str(), 
            &runtime2,
            &engine2);


        executionContexts2.clear();
		contextlocks2.clear();
        for(int i=0;i<globalINICONFObj.ANNIWO_NUM_INSTANCE_PERSONBASE;i++)
		{
			TrtSampleUniquePtr<nvinfer1::IExecutionContext>  context4thisCam(engine2->createExecutionContext());
			std::pair<int, TrtSampleUniquePtr<nvinfer1::IExecutionContext> > tmpitem{i,std::move(context4thisCam)};

			executionContexts2.insert(std::move(tmpitem));


			std::unique_ptr<std::mutex>    newmutexptr(new std::mutex);
			std::pair<int, std::unique_ptr<std::mutex> > tmplockitem{i,std::move(newmutexptr)};

			contextlocks2.insert(std::move(tmplockitem));
		}


    }



    initFlag = true;

} 



//多线程
//whichPredictor:0 用m_id_predictors , 1：用m_id_predictors2:用于车辆
void FeatureTensor::doInference(std::vector<cv::Mat>& imgMats, int camID, int whichPredictor,int instanceID) 
{

    ANNIWOLOG(INFO) <<"FeatureTensor:: doInference:imgMats.size" << imgMats.size() << "instanceID:"<< instanceID <<" camID:"<<camID;
  
    std::unordered_map<int, std::vector<float> >::iterator got_m_input_datas ;
    if(whichPredictor == 0)
    {
        got_m_input_datas = m_input_datas.find(instanceID);
        if (got_m_input_datas == m_input_datas.end())
        {
            ANNIWOLOG(INFO) <<"FeatureTensor::doInference "<<"instanceID:"<<instanceID <<"NOT in planned map!";
            ANNIWOCHECK(false);
            return ;
        }

    }
    else if(whichPredictor == 1)
    {
        got_m_input_datas = m_input_datas2.find(instanceID);
        if (got_m_input_datas == m_input_datas2.end())
        {
            ANNIWOLOG(INFO) <<"FeatureTensor::doInference "<<"instanceID:"<<instanceID <<"NOT in predictors map2!";
            ANNIWOCHECK(false);

            return ;
        }

    }



    /////////////////////////////////////////
    int curBatchSize=0;
    if(whichPredictor == 0)
    {
        curBatchSize = mat2stream(imgMats, m_input_datas[instanceID]);
    }
    else if(whichPredictor == 1)
    {
        curBatchSize = mat2stream2(imgMats, m_input_datas2[instanceID]);
    }

    //curBatchSize在mat2stream中初始化。
    ANNIWOCHECK(curBatchSize>0);
    //   std::vector<int> input_shape = {curBatchSize, 3, imgShape.height, imgShape.width};
    //因为是静态模型，输入必须按照模型转换时候的。64,3,128,64
    
    cv::Size thximgShape;
    if(whichPredictor == 0)
    {
        thximgShape=imgShape;
    }else
    {
        thximgShape=imgShape2;
    }


/////////////////////////////////////////////////////////////////////////////////

    if(whichPredictor == 0)
    {
        ANNIWOCHECK(runtime != nullptr);
        ANNIWOCHECK(engine != nullptr); 
        
        cudaSetDevice(gpuNum);
        

        int choiceIntVal = randIntWithinScale(globalINICONFObj.ANNIWO_NUM_INSTANCE_PERSONBASE);
	    std::unordered_map<int, TrtSampleUniquePtr<nvinfer1::IExecutionContext> >::iterator iterCamInstance =  executionContexts.find(choiceIntVal);
        std::unordered_map<int, std::unique_ptr<std::mutex> >::iterator iterCamInstancelock =  contextlocks.find(choiceIntVal);
        ANNIWOCHECK(iterCamInstance != executionContexts.end()) ;

        // run inference
        auto start = std::chrono::system_clock::now();        

        TrtGPUInfer(*iterCamInstance->second, gpuNum, *iterCamInstancelock->second, m_input_datas[instanceID].data(), m_output_datas[instanceID].data(), outputStreamSize,
                    inputStreamSize, "input", "output", "FeatureTensor::predictor0" );


        auto end = std::chrono::system_clock::now();


        ANNIWOLOG(INFO) <<"FeatureTensor::predictor0 doInference:infer time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms," <<"camID:"<<camID;

    }
    else if(whichPredictor == 1)
    {

        ANNIWOCHECK(runtime2 != nullptr);
        ANNIWOCHECK(engine2 != nullptr); 
        
        cudaSetDevice(gpuNum2);


        int choiceIntVal = randIntWithinScale(globalINICONFObj.ANNIWO_NUM_INSTANCE_PERSONBASE);
	    std::unordered_map<int, TrtSampleUniquePtr<nvinfer1::IExecutionContext> >::iterator iterCamInstance =  executionContexts2.find(choiceIntVal);
        std::unordered_map<int, std::unique_ptr<std::mutex> >::iterator iterCamInstancelock =  contextlocks2.find(choiceIntVal);

        ANNIWOCHECK(iterCamInstance != executionContexts2.end()) ;

        // run inference
        auto start = std::chrono::system_clock::now();

        TrtGPUInfer(*iterCamInstance->second, gpuNum2, *iterCamInstancelock->second, m_input_datas2[instanceID].data(), m_output_datas2[instanceID].data(), outputStreamSize2,
                    inputStreamSize2,  "input_1", "fc10","FeatureTensor::predictor1" );

        auto end = std::chrono::system_clock::now();


        ANNIWOLOG(INFO) <<"FeatureTensor::predictor1 doInference:infer time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms," <<"camID:"<<camID;



    }



}


//正则化不同,使用imgShape2
int FeatureTensor::mat2stream2(std::vector<cv::Mat>& imgMats, std::vector<float>& stream) {
    int imgArea = imgShape2.area();
    int curBatchSize = imgMats.size();
    if (curBatchSize > maxBatchSize) {
        ANNIWOLOG(INFO) << "[WARNING]::Batch size overflow, input will be truncated!" ;
        curBatchSize = maxBatchSize;
    }
    for (int batch = 0; batch < curBatchSize; ++batch) {
        cv::Mat tempMat = imgMats[batch];

        //b,h,w,c
        for (size_t  h = 0; h < imgShape2.height; h++) 
        {
            for (size_t w = 0; w < imgShape2.width; w++) 
            {
                for (size_t c = 0; c < 3; c++) 
                {
                    //python:img.astype(np.float32) / 255.
                    stream[batch*h*imgShape2.width*3 + h*imgShape2.width*3  + w*3 + c] =
                        ((float)tempMat.at<cv::Vec3b>(h, w)[c]) / 255.0;
                }

            }
        }
    }




    return curBatchSize;
}

int FeatureTensor::mat2stream(std::vector<cv::Mat>& imgMats, std::vector<float>& stream) {
    int imgArea = imgShape.area();
    int curBatchSize = imgMats.size();
    if (curBatchSize > maxBatchSize) {
        ANNIWOLOG(INFO) << "[WARNING]::Batch size overflow, input will be truncated!" ;
        curBatchSize = maxBatchSize;
    }
    for (int batch = 0; batch < curBatchSize; ++batch) {
        cv::Mat tempMat = imgMats[batch];
        int i = 0; 
        for (int row = 0; row < imgShape.height; ++row) {
            uchar* uc_pixel = tempMat.data + row * tempMat.step;
            //[b,c,h,w]  pytorch过来的，输入是rgb图形,channel first
            for (int col = 0; col < imgShape.width; ++col) {
                stream[batch * 3 * imgArea + i] = ((float)uc_pixel[0] / 255.0 - means[0]) / std[0];
                stream[batch * 3 * imgArea + i + imgArea] = ((float)uc_pixel[1] / 255.0 - means[1]) / std[1];
                stream[batch * 3 * imgArea + i + 2 * imgArea] = ((float)uc_pixel[2] / 255.0 - means[2]) / std[2];
                uc_pixel += 3;
                ++i;
            }
        }
    }
    return curBatchSize;
}

//实际有几个box就拷贝几个出来，即使有尾巴上多余的feature也不会被拷贝.
void FeatureTensor::stream2det(std::vector<float>& stream, DETECTIONS& det,std::vector<int>& detIndxlist) {
    int i = 0;
    // for (DETECTION_ROW& dbox : det) {
    for(int detIndx:detIndxlist)
    {
        DETECTION_ROW& dbox=det[detIndx];
        for (int j = 0; j < 256; ++j)
        {
            dbox.feature[j] = stream[i * 512 + j];
        }

        ++i;
    }
}

void FeatureTensor::stream2det2(std::vector<float>& stream, DETECTIONS& det,std::vector<int>& detIndxlist) {
    int i = 0;
    // for (DETECTION_ROW& dbox : det) {
    for(int detIndx:detIndxlist)
    {
        DETECTION_ROW& dbox=det[detIndx];
        for (int j = 0; j < 256; ++j)
        {
            dbox.feature[j] = stream[i * 256 + j];
        }

        ++i;
    }
}