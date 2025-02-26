#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <list>   
#include<set>
#include <opencv2/opencv.hpp>
#include <dirent.h>


#include "basePerson.hpp"


#include "../utils/httpUtil.hpp"

#include "../utils/rapidjson/writer.h"
#include "../utils/rapidjson/stringbuffer.h"
#include "../utils/subUtils.hpp"

#include "../common/yolo/yolo_common.hpp"

#include <uuid/uuid.h>


#include "../common/mot/include/deepsort.h"



static const int NUM_CLASSESCOCO = 80;

// static const float BBOX_CONF_THRESHCOCO = 0.3;  //yolox
static const float BBOX_CONF_THRESHCOCO = 0.5999;   //coco pretrain 

const std::vector<std::string> person_base_class_namesCOCO = {
    "person", //0
    "bicycle",//1
     "car",//2
     "motorcycle",//3
     "airplane", 
     "bus", //5
     "train",
     "truck",//7
     "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};



static const int NUM_CLASSESJYZCAR = 11;

static const float BBOX_CONF_THRESHJYZCAR = 0.5999;

const std::vector<std::string> person_base_class_namesJYZ = {
"work_clothe_blue",
"person",
"car",
"work_clothe_yellow",
"tank_truck",
"truck",
"motor",
"reflective_vest",
"rider",
"cement_truck",
"reflective_vest_half"
};


const static std::string model_file_jyz_weilan={"../models/safearea/jyz_car/jyz_weilan_y4_sim.trt"};
// const static std::string ppyoloe_config_file_path_in = {"../models/safearea/jyz_car/infer_cfg.yml"};


static DeepSort* DS = nullptr;


static nvinfer1::IRuntime* runtime{nullptr};
static nvinfer1::ICudaEngine* engine{nullptr};


static std::unordered_map<int, TrtSampleUniquePtr<nvinfer1::IExecutionContext>> executionContexts;
static std::unordered_map<int, std::unique_ptr<std::mutex> > contextlocks;
        
static int gpuNum=0;



//yolov4 ouput flat size 
static const int YOLO4_OUTPUT_SIZE_COCO = 1*7581*NUM_ANCHORS* (NUM_CLASSESCOCO + 5);


const static std::string model_file={"../models/safearea/personbase/yolov4personbase_sim.trt"};


const static std::string deepsort_model_file_path_model={"../models/safearea/dstk_model/deepsort_sim.trt"};


//todo:用自己训练的加油站车辆reid
const static std::string deepsort_model_file_path_model2={"../models/safearea/dstk_model2/carRreid_sim.trt"};

const int reidbatchsize=8;


const int batch_size = 1;

static std::vector<int> input_shape = {batch_size, INPUT_H, INPUT_W, 3};    //yolov4 keras/tf

std::unordered_map<int, std::vector<float> >  BasePersonDetection::m_input_datas;

BasePersonDetection::BasePersonDetection () { 
    ////////////////////////
    enum anniwo_domain_type { ANNIWO_DOMANI_LIANGKU,ANNIWO_DOMANI_JIAYOUZHAN  };
    extern anniwo_domain_type domain_config;

    if(globalINICONFObj.domain_config == ANNIWO_DOMANI_LIANGKU)
    {
//model_file
        gpuNum = initInferContext(
                        globalINICONFObj.personbase_model_Path.c_str(), 
                        &runtime,
                        &engine);
        



    }
    else if(globalINICONFObj.domain_config == ANNIWO_DOMANI_JIAYOUZHAN)
    {
//model_file_jyz_weilan
        gpuNum = initInferContext(
                globalINICONFObj.personbase_model_Path2.c_str(), 
                &runtime,
                &engine);

  
    }

    //  gpuNum = initInferContext(
    //                     model_file.c_str(), 
    //                     &runtime,
    //                     &engine);


   ANNIWOLOG(INFO) << "BasePersonDetection(): initilized!";
   std::cout<<"BasePersonDetection(): initilized!"<<std::endl;

}

// Destructor
BasePersonDetection::~BasePersonDetection () 
{

    for(int i=0;i<globalINICONFObj.ANNIWO_NUM_INSTANCE_PERSONBASE;i++)
    {
        executionContexts[i]->destroy();
    }

    engine->destroy();
    runtime->destroy();
    ANNIWOLOG(INFO) << "BasePersonDetection(): destroyed successfully";
}


//objects:是遵循person_base_class_namesJYZ的
static int mainfunc2(int camID,int instanceID, cv::Mat bgr, std::vector<Object>& objects )
{

    std::vector<DetectBox> track_results;
    cv::Mat image = bgr;
    bgr.release();
    

    int jsonObjCnt=0;
    int img_w = image.cols;
    int img_h = image.rows;




    //过滤有用类别与准备track结构体
    for (size_t i = 0; i < objects.size(); i++)
    // for(auto iter=objects.begin(); iter!=objects.end();iter++)  //list
    {
        Object& obj = objects[i];
        obj.trackID=-1;

        ANNIWOLOG(INFO) <<
        "BasePersonDetection: mainfunc2. input box:"<<obj.rect.x<<","<< obj.rect.y<<","
        <<obj.rect.width<<","<<obj.rect.height<<","
        << "score:"<<obj.prob<<"class:"<<person_base_class_namesJYZ[obj.label].c_str()<<",camID:"<<camID;



// - work_clothe_blue
// - person
// - car
// - work_clothe_yellow
// - tank_truck
// - truck
// - motor
// - unloader
// - reflective_vest
// - work_clothe_wathet
// - rider
// - cement_truck
        if(    person_base_class_namesJYZ[obj.label]  == std::string("person")
            || person_base_class_namesJYZ[obj.label] == std::string("car")
            || person_base_class_namesJYZ[obj.label] == std::string("tank_truck")
            || person_base_class_namesJYZ[obj.label] == std::string("truck")
            || person_base_class_namesJYZ[obj.label] == std::string("motor")
            || person_base_class_namesJYZ[obj.label] == std::string("rider")
            || person_base_class_namesJYZ[obj.label] == std::string("cement_truck")
            || person_base_class_namesJYZ[obj.label] == std::string("work_clothe_blue")
            || person_base_class_namesJYZ[obj.label] == std::string("work_clothe_yellow")
            || person_base_class_namesJYZ[obj.label] == std::string("reflective_vest")
            || person_base_class_namesJYZ[obj.label] == std::string("work_clothe_wathet")
        )
        {
            ANNIWOLOG(INFO) <<
            "BasePersonDetection: mainfunc2. box:"<<obj.rect.x<<","<< obj.rect.y<<","
            <<obj.rect.width<<","<<obj.rect.height<<","
            << "score:"<<obj.prob<<"class:"<<person_base_class_namesJYZ[obj.label].c_str()<<",camID:"<<camID;
            
            if(person_base_class_namesJYZ[obj.label] == std::string("car")
            || person_base_class_namesJYZ[obj.label] == std::string("tank_truck")
            || person_base_class_namesJYZ[obj.label] == std::string("truck")
            || person_base_class_namesJYZ[obj.label] == std::string("motor")
            || person_base_class_namesJYZ[obj.label] == std::string("rider")
            || person_base_class_namesJYZ[obj.label] == std::string("cement_truck")
            )
            {
                if(obj.rect.width < 80. || obj.rect.height < 80.)
                {
                    ANNIWOLOG(INFO) <<"Ignore small truck/car";    
                    continue;    
                }
            }

            float x1=obj.rect.x;
            float y1 = obj.rect.y;
            float x2=(obj.rect.x+obj.rect.width) > img_w ? img_w : (obj.rect.x+obj.rect.width) ;
            float y2 =(obj.rect.y+obj.rect.height) > img_h ? img_h: (obj.rect.y+obj.rect.height);





            DetectBox detBoxObj;
            detBoxObj.x1 = x1; 
            detBoxObj.y1= y1; 
            detBoxObj.x2= x2; 
            detBoxObj.y2= y2;
            detBoxObj.confidence = obj.prob;

            //区分人，车。因为需要用不同的reid encoder.
            //0:为人，1为车
            //工作服无法跟踪，在各自功能自行处理
            //0:"person"
            if(    person_base_class_namesJYZ[obj.label]  == std::string("person")
                || person_base_class_namesJYZ[obj.label] == std::string("work_clothe_blue")
                || person_base_class_namesJYZ[obj.label] == std::string("work_clothe_yellow")
                || person_base_class_namesJYZ[obj.label] == std::string("reflective_vest")
                || person_base_class_namesJYZ[obj.label] == std::string("work_clothe_wathet")
            )
            {
                detBoxObj.classID = 0;
            }
            else
            {
                detBoxObj.classID = 1;
            }
            
            uuid_generate_random(detBoxObj.uuid);
            track_results.push_back(detBoxObj);


            obj.trackID=-1;

            strncpy((char*)obj.uuid, (char*)detBoxObj.uuid, sizeof(uuid_t));


            jsonObjCnt++;                   

        }

    }




/////////////TRACKING PART//////////////////////////////////

    if(track_results.size() > 0)
    {
        if(DS)
        {
            cv::Mat img_rgb;
            cv::cvtColor(image, img_rgb, cv::COLOR_BGR2RGB);

            DS->sort(img_rgb, track_results,camID,instanceID);
        }else
        {
            ANNIWOLOG(INFO) <<"BasePersonDetection:mainfunc2 FATAL ERROR! DS is null!\n";
        }

    }

    //匹配track与det框。
    for (auto & personvehicel_det : objects) {
        int trackID = -1;

// "work_clothe_blue",
// "person",
// "car",
// "work_clothe_yellow",
// "tank_truck",
// "truck",
// "motor",
// "reflective_vest",
// "rider",
// "cement_truck"
        if(     person_base_class_namesJYZ[personvehicel_det.label]  == std::string("person")
            || person_base_class_namesJYZ[personvehicel_det.label] == std::string("car")
            || person_base_class_namesJYZ[personvehicel_det.label] == std::string("motor")
            || person_base_class_namesJYZ[personvehicel_det.label] == std::string("tank_truck")
            || person_base_class_namesJYZ[personvehicel_det.label] == std::string("truck")
            || person_base_class_namesJYZ[personvehicel_det.label] == std::string("rider")
            || person_base_class_namesJYZ[personvehicel_det.label] == std::string("cement_truck")
            || person_base_class_namesJYZ[personvehicel_det.label] == std::string("work_clothe_blue")
            || person_base_class_namesJYZ[personvehicel_det.label] == std::string("work_clothe_yellow")
            || person_base_class_namesJYZ[personvehicel_det.label] == std::string("reflective_vest")
            || person_base_class_namesJYZ[personvehicel_det.label] == std::string("work_clothe_wathet")
        )
        {
            for (auto & box : track_results ) {
                // if(box.uuid == personvehicel_det.uuid)
                if(0 == strncmp( (char *)box.uuid,  (char *)personvehicel_det.uuid, sizeof(uuid_t)))
                {
                    //trackID = (int)box.trackID;
                   personvehicel_det.trackID = trackID;
                    
                }
            }

            ANNIWOLOG(INFO) <<"BasePersonDetection:mainfunc2:"<<person_base_class_namesJYZ[personvehicel_det.label].c_str()<<","<< personvehicel_det.prob<<","<< "tid:"<<trackID<<  "  x, y, w,  h:"<<personvehicel_det.rect.x<<","<<personvehicel_det.rect.y<<","<<personvehicel_det.rect.width<<","<<personvehicel_det.rect.height<<","<<" camID:"<<camID;
              

        }

    }



    return jsonObjCnt;

}




static int mainfunc(int camID,int instanceID, cv::Mat bgr, std::vector<Object>& objects )
{

    std::vector<DetectBox> track_results;
    cv::Mat image = bgr;
    bgr.release();
    

    int jsonObjCnt=0;
    int img_w = image.cols;
    int img_h = image.rows;

    //过滤有用类别与准备track结构体
    for (size_t i = 0; i < objects.size(); i++)
    {
        Object& obj = objects[i];
        obj.trackID=-1;


        if(person_base_class_namesCOCO[obj.label]  == std::string("person")
        || person_base_class_namesCOCO[obj.label] == std::string("bicycle")
        || person_base_class_namesCOCO[obj.label] == std::string("car")
        || person_base_class_namesCOCO[obj.label] == std::string("motorcycle")
        || person_base_class_namesCOCO[obj.label] == std::string("bus")
        || person_base_class_namesCOCO[obj.label] == std::string("truck")
        )
        {

            ANNIWOLOG(INFO) <<
            "BasePersonDetection: detect. box:"<<obj.rect.x<<","<< obj.rect.y<<","
            <<obj.rect.width<<","<<obj.rect.height<<","
            << "score:"<<obj.prob<<"class:"<<person_base_class_namesCOCO[obj.label].c_str()<<" camID:"<<camID;


            if(obj.rect.width < 5 || obj.rect.height < 5 || (obj.rect.x+obj.rect.width) > img_w || (obj.rect.y+obj.rect.height) > img_h)
            {
                ANNIWOLOG(INFO) <<"Ignore invalid box, to check!"<<" camID:"<<camID;        
                continue;    
            }

            if(person_base_class_namesCOCO[obj.label] == std::string("bus")
            || person_base_class_namesCOCO[obj.label] == std::string("truck"))
            {
                if(obj.rect.width < 80 || obj.rect.height < 80)
                {
                    ANNIWOLOG(INFO) <<"Ignore small truck"<<" camID:"<<camID;        
                    continue;    

                }
            }


            float x1=obj.rect.x;
            float y1 = obj.rect.y;
            float x2=(obj.rect.x+obj.rect.width) > img_w ? img_w : (obj.rect.x+obj.rect.width) ;
            float y2 =(obj.rect.y+obj.rect.height) > img_h ? img_h: (obj.rect.y+obj.rect.height);

            //用于户外环境后无此必要.
            // if(person_base_class_namesCOCO[obj.label]  == std::string("person"))
            // {
            //     // if(abs(y2 - y1) < 2.0*abs(x2 - x1))
            //     if(abs(y2 - y1) < abs(x2 - x1))
            //     {
            //         loggerObj<<INFO<<"BasePersonDetection:Ignore person:h < 2.0*w"<<"camID:"<<camID<<std::endl;
            //         continue;
            //     }
            // }



            DetectBox detBoxObj;
            detBoxObj.x1 = x1; 
            detBoxObj.y1= y1; 
            detBoxObj.x2= x2; 
            detBoxObj.y2= y2;
            detBoxObj.confidence = obj.prob;
            detBoxObj.trackID = -1;

            //todo:区分人，车。因为需要用不同的reid encoder.
            //0:为人，1为车
            if(person_base_class_namesCOCO[obj.label]  == std::string("person"))
            {
                detBoxObj.classID = 0;
            }
            else
            {
                detBoxObj.classID = 1;
            }
            
            uuid_generate_random(detBoxObj.uuid);
            track_results.push_back(detBoxObj);


            obj.trackID=-1;

            strncpy((char*)obj.uuid, (char*)detBoxObj.uuid, sizeof(uuid_t));


            jsonObjCnt++;                   

        }

    }




/////////////TRACKING PART//////////////////////////////////

    if(track_results.size() > 0)
    {
        //如下使用cpu
        if(DS)
        {
            cv::Mat img_rgb;
            cv::cvtColor(image, img_rgb, cv::COLOR_BGR2RGB);

            DS->sort(img_rgb, track_results,camID,instanceID);
        }else
        {
           ANNIWOLOG(INFO) <<"BasePersonDetection:FATAL ERROR! DS is null!"<<","<<" camID:"<<camID;
        }

    }

    //匹配track与det框。
    for (auto & personvehicel_det : objects) {
        int trackID = -1;

        if(person_base_class_namesCOCO[personvehicel_det.label] == std::string("person")
        || person_base_class_namesCOCO[personvehicel_det.label] == std::string("bicycle")
        || person_base_class_namesCOCO[personvehicel_det.label] == std::string("car")
        || person_base_class_namesCOCO[personvehicel_det.label] == std::string("motorcycle")
        || person_base_class_namesCOCO[personvehicel_det.label] == std::string("bus")
        || person_base_class_namesCOCO[personvehicel_det.label] == std::string("truck")
        )
        {
            bool isInTrackResults=false;
            ANNIWOLOG(INFO) <<"BasePersonDetection:track_results.size:"<<track_results.size()<<","<<" camID:"<<camID;

            for (auto & box : track_results ) {
                // if(box.uuid == personvehicel_det.uuid)
                if(0 == strncmp( (char *)box.uuid,  (char *)personvehicel_det.uuid, sizeof(uuid_t)))
                {
                    //trackID = (int)box.trackID;
                    personvehicel_det.trackID = trackID;
                    isInTrackResults=true;
                    
                }
            }

            if(!isInTrackResults)
            {
                ANNIWOLOG(INFO) <<"BasePersonDetection: NOT IN track Resuslts:"<<person_base_class_namesCOCO[personvehicel_det.label].c_str()<<","<< personvehicel_det.prob<<","<< "tid:"<<trackID<<  "x, y, w,  h:"<<personvehicel_det.rect.x<<","<<personvehicel_det.rect.y<<","<<personvehicel_det.rect.width<<","<<personvehicel_det.rect.height<<","<<" camID:"<<camID;
            }else
            {
                ANNIWOLOG(INFO) <<"BasePersonDetection:"<<person_base_class_namesCOCO[personvehicel_det.label].c_str()<<","<< personvehicel_det.prob<<","<< "tid:"<<trackID<<  "x, y, w,  h:"<<personvehicel_det.rect.x<<","<<personvehicel_det.rect.y<<","<<personvehicel_det.rect.width<<","<<personvehicel_det.rect.height<<","<<" camID:"<<camID;

            }

        }

    }



    return jsonObjCnt;

}




//person_base_functionsIn:所有需要先检测人、车的其他功能列表
void BasePersonDetection::initTracks()
{

    std::unordered_set<int> setcamIDs ;
    setcamIDs.insert(0);

    executionContexts.clear();
    contextlocks.clear();

    cudaSetDevice(gpuNum);


    if(DS)
    {
        delete DS;
    }

    std::vector<int> camIDs;

    int cntID=0;
    //只生成ANNIWO_NUM_INSTANCE_PERSONBASE个实例
    while(cntID < globalINICONFObj.ANNIWO_NUM_THREAD_PERSONBASE)
    {
        ANNIWOLOG(INFO) << "BasePersonDetection::initTracks: insert instance" <<"cntID:"<<cntID<<" ";

        std::vector<float> input_data(batch_size * CHANNELS * INPUT_H * INPUT_W,0);
        std::pair<int, std::vector<float> > itempair2(cntID,std::move(input_data));
        m_input_datas.insert( std::move(itempair2) );

        cntID++;
    }


    for (auto& camID : setcamIDs)
    {
        camIDs.push_back(camID);
    }

    for(int i=0;i<globalINICONFObj.ANNIWO_NUM_INSTANCE_PERSONBASE;i++)
    {
        TrtSampleUniquePtr<nvinfer1::IExecutionContext>  context4thisCam(engine->createExecutionContext());
        std::pair<int, TrtSampleUniquePtr<nvinfer1::IExecutionContext> > tmpitem{i,std::move(context4thisCam)};

        executionContexts.insert(std::move(tmpitem));


        std::unique_ptr<std::mutex>    newmutexptr(new std::mutex);
        std::pair<int, std::unique_ptr<std::mutex> > tmplockitem{i,std::move(newmutexptr)};

        contextlocks.insert(std::move(tmplockitem));
    }

    //人和车分开用reid,例如加油站情况。
    if(globalINICONFObj.domain_config == ANNIWO_DOMANI_JIAYOUZHAN)
    {

        DS = new DeepSort(deepsort_model_file_path_model,&deepsort_model_file_path_model2, reidbatchsize, camIDs,globalINICONFObj.ANNIWO_NUM_THREAD_PERSONBASE);

    }else
    {

        //人和车混用一个reid,车的效果差。
        DS = new DeepSort(deepsort_model_file_path_model,NULL, reidbatchsize,  camIDs,globalINICONFObj.ANNIWO_NUM_THREAD_PERSONBASE);
    }

}


int BasePersonDetection::detect(  int camID, int instanceID, cv::Mat img, /*out*/std::vector<Object>& objects ) 
{    
    int objCnt = 0;
    cudaSetDevice(gpuNum);


    if(globalINICONFObj.domain_config == ANNIWO_DOMANI_LIANGKU)
    {


        int img_w = img.cols;
        int img_h = img.rows;

        int choiceIntVal = randIntWithinScale(globalINICONFObj.ANNIWO_NUM_INSTANCE_PERSONBASE);
        std::unordered_map<int, std::unique_ptr<std::mutex> >::iterator iterCamInstancelock =  contextlocks.find(choiceIntVal);
        std::unordered_map<int, TrtSampleUniquePtr<nvinfer1::IExecutionContext> >::iterator iterCamInstance =  executionContexts.find(choiceIntVal);
        
        if (iterCamInstance != executionContexts.end()) 
        {

            yolov4_detection_staff(m_input_datas,camID,instanceID,img,
                runtime,engine,
                iterCamInstance->second,//smart pointer context for this func-cam
                gpuNum,
                iterCamInstancelock->second,//smart pointer context LOCK for this func-cam
                YOLO4_OUTPUT_SIZE_COCO,INPUT_W,INPUT_H,objects,BBOX_CONF_THRESHCOCO,NUM_CLASSESCOCO,
                "BasePersonDetection");

        }else
        {
            ANNIWOLOG(INFO) <<"Not found the context for camId:"<<camID;
            ANNIWOCHECK(false);
        }




            

        //先删除无用类别
        std::vector<Object>::iterator it=objects.begin();
        while(it != objects.end() )
        {
            bool isDel=false;
            // ANNIWOLOG(INFO)  << "BasePersonDetection debug 1:size: "<<objects.size() <<",camID:"<<camID ;

            if(person_base_class_namesCOCO[it->label] == std::string("bus")
            || person_base_class_namesCOCO[it->label] == std::string("truck"))
            {
                if(it->rect.width < 80 || it->rect.height < 80)
                {
                    isDel=true;        

                }else
                {
                    isDel=false;
                }
            }
            else  if(person_base_class_namesCOCO[it->label]  == std::string("person")
            || person_base_class_namesCOCO[it->label] == std::string("bicycle")
            || person_base_class_namesCOCO[it->label] == std::string("car")
            || person_base_class_namesCOCO[it->label] == std::string("motorcycle")
            )
            {

                if(it->rect.width < 5 || it->rect.height < 5 || (it->rect.x+it->rect.width) > img_w || (it->rect.y+it->rect.height) > img_h)
                {
                    ANNIWOLOG(INFO) <<"BasePersonDetection:Ignore invalid box, to check!";        
                    isDel=true;
                }else
                {
                    isDel=false;
                }

            }else
            {
                isDel=true;
            }
            
            if (isDel)
            {
                it = objects.erase(it); //vector的erase会返回一个自动指向下一个元素，必须赋值！
            }
            else
            {
                it++;
            }

        }

        ANNIWOLOG(INFO)  << "BasePersonDetection debug 2:size:"<<objects.size() <<",camID:"<<camID ;


        if(objects.size() > 0)
        {
            objCnt = mainfunc(camID,instanceID, img, /*out*/objects);
        }
        else
        {
            ANNIWOLOG(INFO) << "BasePersonDetection:no objects." <<"camID:"<<camID;
        }
    }

    //加油站的人车,目前就用coco的结果。等新模型好了再说。
    if(globalINICONFObj.domain_config == ANNIWO_DOMANI_JIAYOUZHAN)
    {

        //yolov4 ouput flat size 
        static const int YOLO4_OUTPUT_SIZE = 1*7581*NUM_ANCHORS* (NUM_CLASSESJYZCAR + 5);
        
        
        int choiceIntVal = randIntWithinScale(globalINICONFObj.ANNIWO_NUM_INSTANCE_PERSONBASE);
	    std::unordered_map<int, TrtSampleUniquePtr<nvinfer1::IExecutionContext> >::iterator iterCamInstance =  executionContexts.find(choiceIntVal);
        std::unordered_map<int, std::unique_ptr<std::mutex> >::iterator iterCamInstancelock =  contextlocks.find(choiceIntVal);

        if (iterCamInstance != executionContexts.end()) 
        {
            yolov4_detection_staff(m_input_datas,camID,instanceID,img,
                runtime,engine,
                iterCamInstance->second,//smart pointer context for this cam
                gpuNum,
                iterCamInstancelock->second,//smart pointer context LOCK for this func-cam
                YOLO4_OUTPUT_SIZE,INPUT_W,INPUT_H,objects,BBOX_CONF_THRESHJYZCAR,NUM_CLASSESJYZCAR,
                "BasePersonDetection");
        }else
        {
            ANNIWOLOG(INFO) <<"Not found the context for camId:"<<camID;
            ANNIWOCHECK(false);
        }



        ////////////////////////////////////////////////

        if(objects.size() > 0)
        {
            objCnt = mainfunc2(camID,instanceID, img, /*out*/objects);
            if(objCnt > 0)
            {
                ANNIWOLOG(INFO) << "BasePersonDetection: objects cnt:"<<objCnt <<"camID:"<<camID;
            }else
            {
                ANNIWOLOG(INFO) << "BasePersonDetection:no objects." <<"camID:"<<camID;
            }
        }
        else
        {
            ANNIWOLOG(INFO) << "BasePersonDetection:no objects." <<"camID:"<<camID;
        }
    }

    ANNIWOLOG(INFO) << "BasePersonDetection: objects size."<<objects.size() <<"camID:"<<camID;
    ANNIWOLOG(INFO) << "BasePersonDetection:exit detect()" <<"camID:"<<camID ;
    
    
    return objCnt;
}
