#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <xtensor.hpp>

// #include "smoke.hpp"
// #include "phone.hpp"
#include "smoke_classification.hpp"
#include "SmokePhone.hpp"

#include "../utils/httpUtil.hpp"

#include "../utils/rapidjson/writer.h"
#include "../utils/rapidjson/stringbuffer.h"
#include "../utils/subUtils.hpp"

#include "../common/yolo/yolo_common.hpp"
#include "../personbase/basePerson.hpp"

static const ANNIWO_JSON_CONF_CLASS *globalJsonConfObjPtr;

/////////////////////////////////////////////////////////////yolo to grab the face

static const int NUM_CLASSES = 3; //
static const int NUM_CLASSES_INUSE = 3;

static float BBOX_CONF_THRESH = 0.35; // ini可设置
static const int YOLO4_OUTPUT_SIZE = 1 * 7581 * NUM_ANCHORS * (NUM_CLASSES + 5);

static const int MAX_CLASS_NUM = 2;

static std::vector<std::string> PHONEclass_names = {
    "background",
    "phone",
    "smoke"};
static std::vector<std::string> Faceclass_names = {
    "face_smoke",
    "face_phone",
    "noraml_face",
    "background",
    "phone",
    "smoke"};


static std::unordered_map<int, std::unordered_map<std::string, std::unordered_map<int, int>>> reporthistoryArray;

// const static std::string model_file={"../models/phone/bestsmokephone_sim.trt"};

static nvinfer1::IRuntime *runtime{nullptr};
static nvinfer1::ICudaEngine *engine{nullptr};

static std::unordered_map<int, TrtSampleUniquePtr<nvinfer1::IExecutionContext>> executionContexts;
static std::unordered_map<int, std::unique_ptr<std::mutex>> contextlocks;
static int gpuNum = 0;

const int batch_size = 1;
// static std::vector<int> input_shape = {batch_size, 3, INPUT_H, INPUT_W};  //yolox pytorch
static std::vector<int> input_shape = {batch_size, INPUT_H, INPUT_W, 3}; // yolov4 keras/tf

std::unordered_map<int, std::vector<float>> SmokePhoneDetection::m_input_datas;

/////////////////////////////////////////////////////////////////////////////
// const static std::string face_det_model_file={"../models/smoke/face/face_det.trt"};
// const static std::string face_det_config_path={"../models/smoke/face/infer_cfg.yml"};
const int FACE_DET_CLASS_NUM = 1;
static const float FACE_DET_BBOX_CONF_THRESH = 0.4;
PaddleDetection::ObjectDetector *SmokePhoneDetection::m_facedetptr = nullptr;
/////////////////////////////////////////////////////////////////////////////

static SmokeClassification *smkclassifier = nullptr;

// Default constructor
SmokePhoneDetection::SmokePhoneDetection()
{
    smkclassifier = new SmokeClassification();

    gpuNum = initInferContext(
        globalINICONFObj.smokePhonethings_Path.c_str(),
        &runtime,
        &engine);

    // m_facedetptr=new PaddleDetection::ObjectDetector(face_det_config_path,face_det_model_file,globalINICONFObj.ANNIWO_NUM_INSTANCE_SMOKEPHONE);
    m_facedetptr = new PaddleDetection::ObjectDetector(globalINICONFObj.smokePhoneconfig_Path, globalINICONFObj.smokePhoneface_Path, globalINICONFObj.ANNIWO_NUM_INSTANCE_SMOKEPHONE);

    ANNIWOLOG(INFO) << "SmokePhoneDetection(): Success initialized!";
}

// Destructor
SmokePhoneDetection::~SmokePhoneDetection()
{

    // destroy the engine for yolov4

    for (int i = 0; i < globalINICONFObj.ANNIWO_NUM_INSTANCE_SMOKEPHONE; i++)
    {
        executionContexts[i]->destroy();
    }

    engine->destroy();
    runtime->destroy();

    delete smkclassifier;

    if (m_facedetptr)
    {
        delete m_facedetptr;
    }
    ANNIWOLOG(INFO) << "SmokePhoneDetection destroyed successfully";
}

// static void PostProcessResults(int camID, cv::Mat bgr, const std::vector<Object>& objects, const Polygon* polygonSafeArea_ptr, const std::vector<std::string>& class_names)
// {

//     cv::Mat image = bgr;
//     bgr.release();
// 	Polygon _inter;
//     Polygon box_poly;

//     rapidjson::StringBuffer jsonstrbuf;
//     rapidjson::Writer<rapidjson::StringBuffer> writer(jsonstrbuf);

//     int jsonObjCnt=0;
//     writer.StartArray();
//     std::string reportType;

//     for (size_t i = 0; i < objects.size(); i++)
//     {
//         const Object& obj = objects[i];

//         ANNIWOLOGF(INFO, "SmokePhoneDetection post:camID:%d, %s = confidence:%.5f at x:%.2f y:%.2f w:%.2f  h:%.2f\n",camID, class_names[obj.label].c_str(), obj.prob,
//                 obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

//         if(obj.label > MAX_CLASS_NUM)
//         {
//             continue;
//         }

//         //todo:DEBUG
//         if( ( class_names[obj.label] == std::string("smoke") || class_names[obj.label] == std::string("phone") )  && obj.prob > 0.12)
//         {
//             reportType = class_names[obj.label] ;

//             int x1=obj.rect.x;
//             int y1 = obj.rect.y;
//             int x2=(obj.rect.x+obj.rect.width) > image.cols ? image.cols : (obj.rect.x+obj.rect.width) ;
//             int y2 =(obj.rect.y+obj.rect.height) > image.rows ? image.rows : (obj.rect.y+obj.rect.height);

//             writer.StartObject();               // Between StartObject()/EndObject(),

//             writer.Key("y1");
//             writer.Int(y1);
//             writer.Key("x1");
//             writer.Int(x1);
//             writer.Key("y2");
//             writer.Int(y2);
//             writer.Key("x2");
//             writer.Int(x2);
//             writer.Key("classItem");                // output a key,
//             writer.String(class_names[obj.label].c_str());             // follow by a value.

//             writer.EndObject();

//             if(obj.trackID == -1)
//             {
//                 //此时报警但不记录
//                 jsonObjCnt++;

//             }else
//             {
//                 //记录已报警过的trackID
//                 std::unordered_map<int, std::unordered_map<std::string, std::unordered_map<int, int > > >::iterator got_it = reporthistoryArray.find(camID);

//                 std::unordered_map<std::string, std::unordered_map<int, int > >& perfuncCamIDhistory = got_it->second;
//                 for(auto& kv:perfuncCamIDhistory )
//                 {
//                     if(class_names[obj.label] == kv.first)
//                     {
//                         std::unordered_map<int, int >& perCamIDhistory = kv.second;

//                         std::unordered_map<int, int >::iterator got_it2 = perCamIDhistory.find(obj.trackID);

//                         if (got_it2 == perCamIDhistory.end())//new to this camID
//                         {
//                             perCamIDhistory.insert(std::pair<int,int>(obj.trackID,1) );
//                             jsonObjCnt++;

//                         }
//                         else
//                         {
//                             got_it2->second++;
//                             ANNIWOLOG(INFO) <<"SmokePhoneDetection: Warning:found reported.Ignored.trackID:"<<obj.trackID<<"hit:"<<got_it2->second<<"camID:"<<camID<<std::endl;
//                         }
//                     }

//                 }

//                 ///

//             }

//         }

// #ifdef ANNIWO_INTERNAL_DEBUG

//         //todo:Below is leave for debugging! 描绘部分!
//         cv::Scalar color = cv::Scalar(color_list[obj.label][0], color_list[obj.label][1], color_list[obj.label][2]);
//         float c_mean = cv::mean(color)[0];
//         cv::Scalar txt_color;
//         if (c_mean > 0.5){
//             txt_color = cv::Scalar(0, 0, 0);
//         }else{
//             txt_color = cv::Scalar(255, 255, 255);
//         }

//         cv::rectangle(image, obj.rect, color * 255, 2);

//         char text[256];
//         sprintf(text, "%s %.1f%%", class_names[obj.label].c_str(), obj.prob * 100);

//         int baseLine = 0;
//         cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

//         cv::Scalar txt_bk_color = color * 0.7 * 255;

//         int x = obj.rect.x;
//         int y = obj.rect.y + 1;
//         if (y > image.rows)
//             y = image.rows;

//         cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
//                       txt_bk_color, -1);

//         cv::putText(image, text, cv::Point(x, y + label_size.height),
//                     cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
// #endif
//     }

//     writer.EndArray();

//     if(jsonObjCnt > 0)
//     {

//         std::string imagename=getRandomName();
//         std::string imgPath = ANNIWO_LOG_IMAGES_PATH + "/" + reportType + "/" + imagename;

//         std::string taskIdstr={"00000"};
//         std::string submitUrl={"http://localhost:7008/safety-event-local/socketEvent/"+reportType};

//         getTaskId(globalJsonConfObjPtr,camID,reportType,taskIdstr);
//         getEventUrl(globalJsonConfObjPtr,camID,reportType,"/"+reportType,submitUrl);

//         ANNIWOLOG(INFO) <<"SmokePhoneDetection:save file name  is:"<<"camID:"<<camID<<" "<<imgPath<<std::endl;
//         pool->enqueue(saveImgAndPost,camID,taskIdstr,imgPath,image,std::chrono::system_clock::from_time_t(0),
//         std::string(jsonstrbuf.GetString()),jsonstrbuf.GetLength(), submitUrl);

//     }

// }

void SmokePhoneDetection::initTracks()
{

    reporthistoryArray.clear();

    executionContexts.clear();
    contextlocks.clear();

    cudaSetDevice(gpuNum);

    int cntID = 0;
    // 只生成ANNIWO_NUM_THREAD_SMOKEPHONE个
    while (cntID < globalINICONFObj.ANNIWO_NUM_THREAD_SMOKEPHONE)
    {
        ANNIWOLOG(INFO) << "SmokePhoneDetection::initTracks insert thread"
                        << "cntID:" << cntID;

        std::vector<float> input_data(batch_size * CHANNELS * INPUT_H * INPUT_W, 1.0);
        std::pair<int, std::vector<float>> itempair2(cntID, std::move(input_data));
        m_input_datas.insert(std::move(itempair2));

        ////////////////////////

        cntID++;
    }

    for (int i = 0; i < globalINICONFObj.ANNIWO_NUM_INSTANCE_SMOKEPHONE; i++)
    {

        TrtSampleUniquePtr<nvinfer1::IExecutionContext> context4thisCam(engine->createExecutionContext());
        std::pair<int, TrtSampleUniquePtr<nvinfer1::IExecutionContext>> tmpitem{i, std::move(context4thisCam)};

        executionContexts.insert(std::move(tmpitem));

        std::unique_ptr<std::mutex> newmutexptr(new std::mutex);
        std::pair<int, std::unique_ptr<std::mutex>> tmplockitem{i, std::move(newmutexptr)};

        contextlocks.insert(std::move(tmplockitem));
    }
    smkclassifier->initTracks();

    // for (auto iter = globalJsonConfObj.id_func_cap.begin(); iter != globalJsonConfObj.id_func_cap.end(); ++iter) {
    //     int camID=0;
    //     for(auto& f : iter->second)
    //     {
    //         if ( f == std::string("smoke") || f == std::string("phone"))
    //         {
    //             std::unordered_map<int, int > trackidhits;
    //             ANNIWOLOG(INFO) << "SmokePhoneDetection::initTracks:history insert" <<" subfunc:"<<f<<" camID:"<<camID<<" ";

    //             std::unordered_map<int, std::unordered_map<std::string, std::unordered_map<int, int > > >::iterator got_it = reporthistoryArray.find(camID);

    //             if (got_it == reporthistoryArray.end())
    //             {
    //                 std::unordered_map<std::string, std::unordered_map<int, int > > functh;
    //                 functh.insert(std::pair<std::string, std::unordered_map<int, int >  >(f,trackidhits) );
    //                 reporthistoryArray.insert(std::pair<int, std::unordered_map<std::string, std::unordered_map<int, int >>  >(camID,functh) );
    //             }else
    //             {
    //                 std::unordered_map<std::string, std::unordered_map<int, int > >& perCamIDfunchistory = got_it->second;
    //                 perCamIDfunchistory.insert(std::pair<std::string, std::unordered_map<int, int > >(f,trackidhits));

    //             }

    //         }
    //         else
    //         {
    //             continue;
    //         }
    //     }
    // }

    // smkclassifier->initTracks(globalJsonConfObj);

    // globalJsonConfObjPtr=&globalJsonConfObj;
}

// todo:polygonSafeArea
void SmokePhoneDetection::detect(int camID, int instanceID, cv::Mat img, const Polygon *polygonSafeArea_ptr, const std::vector<Object> &in_person_results, std::string &f)
{
    int newpersonCnt = 0;
    Polygon _inter;
    Polygon box_poly;
    std::vector<Object> person_det_resultsInside;
    ANNIWOLOG(INFO) << "SmokePhoneDetection: entered."
                    << "camID:" << camID << std::endl;

    cv::Mat image = img;

    for (auto &obj : in_person_results)
    {

        if (!isPerson(obj.label))
            continue;

        int x1 = obj.rect.x;
        int y1 = obj.rect.y;
        int x2 = (obj.rect.x + obj.rect.width) > image.cols ? image.cols : (obj.rect.x + obj.rect.width);
        int y2up = (obj.rect.y + obj.rect.height / 4) > image.rows ? image.rows : (obj.rect.y + obj.rect.height / 4);

        if (polygonSafeArea_ptr && polygonSafeArea_ptr->size() >= 3)
        {
            box_poly.clear();
            box_poly.add(cv::Point(int(x1), int(y1)));
            box_poly.add(cv::Point(int(x2), int(y1)));
            box_poly.add(cv::Point(int(x2), int(y2up)));
            box_poly.add(cv::Point(int(x1), int(y2up)));
            _inter.clear();
            intersectPolygonSHPC(box_poly, *polygonSafeArea_ptr, _inter);
            if (_inter.size())
            {
                float area = _inter.area();
                // cv::Point center = _inter.getCenter();
                // ANNIWOLOGF(INFO,"HelmetDetection: Area intersected = %0.1f \n",area);

                if (area <= 10.0)
                {
                    ANNIWOLOG(INFO) << "SmokePhoneDetection: detect.Ignored as not in valid area box:" << obj.rect.x << "," << obj.rect.y << ","
                                    << obj.rect.width << "," << obj.rect.height << ","
                                    << "score:" << obj.prob << "class:" << getPersonCarbaseClassName(obj.label) << ","
                                    << "area:" << area << "trackid:" << obj.trackID << " camID:" << camID;
                    continue;
                }
                else
                {
                    person_det_resultsInside.push_back(obj);
                }
            }
            else
            {
                ANNIWOLOG(INFO) << "SmokePhoneDetection: inter size none"
                                << "camID:" << camID << std::endl;

                break;
            }
        }
        else // use all
        {
            person_det_resultsInside.push_back(obj);
        }

        // int trackID = (int)obj.trackID;
        // if(trackID == -1)
        // {
        //     newpersonCnt++;
        // }else
        // {
        //     std::unordered_map<int, std::unordered_map<std::string, std::unordered_map<int, int > > >::iterator got_it = reporthistoryArray.find(camID);

        //     if (got_it == reporthistoryArray.end())
        //     {
        //         ANNIWOLOG(INFO) <<"SmokePhoneDetection: Not in history map!!!"<<"camID:"<<camID<<std::endl;
        //         ANNIWOCHECK(false);
        //     }

        //     std::unordered_map<std::string, std::unordered_map<int, int > >& perCamIDfunchistory = got_it->second;
        //     for(auto& kv:perCamIDfunchistory )
        //     {
        //         std::unordered_map<int, int >& perCamIDhistory = kv.second;

        //         std::unordered_map<int, int >::iterator got_it2 = perCamIDhistory.find(obj.trackID);

        //         if (got_it2 == perCamIDhistory.end())//new to this camID
        //         {
        //             newpersonCnt++;
        //             break;
        //         }
        //         else
        //         {
        //             ANNIWOLOG(INFO) <<"SmokePhoneDetection: found tracked&reported..trackID:"<<trackID<<"hit:"<<got_it2->second<<"camID:"<<camID<<" func:"<<kv.first<<std::endl;
        //         }

        //     }

        // }
    }

    if (person_det_resultsInside.size() <= 0)
    {
        ANNIWOLOG(INFO) << "SmokePhoneDetection:exit no new person detect()"
                        << "camID:" << camID;
        return;
    }

    /////////////////////////////////////////////////////////////////////////////////
    // 先进行yolo人脸的抓取
    std::vector<Object> facedetobjects;
    cudaSetDevice(gpuNum);

    ppyoloe_det_stuff(m_facedetptr, camID, instanceID, img, /*out*/ facedetobjects, FACE_DET_BBOX_CONF_THRESH, FACE_DET_CLASS_NUM, "SmokePhoneDetection");

    //////////////////////////////////////////////////////////////////////////////////

    int orig_img_w = image.cols;
    int orig_img_h = image.rows;

    // std::vector<Object> overall_objectsSMK;
    // std::vector<Object> overall_objectsPhone;
    std::vector<Object> overall_objects;

    for (auto &obj : person_det_resultsInside)
    {
        ANNIWOLOGF(INFO, "SmokePhoneDetection::detect:camID:%d person_det_resultsInside %s = confidence:%.5f at x:%.2f y:%.2f w:%.2f  h:%.2f tid:%d\n", camID, getPersonCarbaseClassName(obj.label).c_str(), obj.prob,
                   obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height, obj.trackID);
    }
    for (auto &obj : facedetobjects)
    {
        ANNIWOLOGF(INFO, "FaceDetection::detect:camID:%d person_det_resultsInside %s = confidence:%.5f at x:%.2f y:%.2f w:%.2f  h:%.2f tid:%d\n", camID, getPersonCarbaseClassName(obj.label).c_str(), obj.prob,
                   obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height, obj.trackID);
    }

    for (size_t i = 0; i < person_det_resultsInside.size(); i++)
    {
        const Object &obj = person_det_resultsInside[i];

        if (!isPerson(obj.label))
            continue;

        ANNIWOLOGF(INFO, "SmokePhoneDetection::detect:camID:%d person_det_resultsInside %s = confidence:%.5f at x:%.2f y:%.2f w:%.2f  h:%.2f tid:%d\n", camID, getPersonCarbaseClassName(obj.label).c_str(), obj.prob,
                   obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height, obj.trackID);

        int person_x = int(std::max(.0f, obj.rect.x));
        int person_y = int(std::max(.0f, obj.rect.y));
        int person_w = int(std::min(obj.rect.width, orig_img_w * 1.0f));
        int person_h = int(std::min(obj.rect.height, orig_img_h * 1.0f));
        int person_trackID = obj.trackID;

        Object personUp;
        personUp.rect.x = person_x;
        personUp.rect.y = person_y;
        personUp.rect.width = person_w;
        personUp.rect.height = int(std::min(obj.rect.height / 4, orig_img_h * 1.0f));
        ;

        bool isInsideP = false;
        Object faceobj;

        for (int idx = 0; idx < facedetobjects.size(); idx++) // 找一个在人内部的人脸目标
        {
            faceobj = facedetobjects[idx];
            ANNIWOLOGF(INFO, "FaceDetection::detect:camID:%d person_det_resultsInside %s = confidence:%.5f at x:%.2f y:%.2f w:%.2f  h:%.2f tid:%d\n", camID, getPersonCarbaseClassName(faceobj.label).c_str(), faceobj.prob,
                       faceobj.rect.x, faceobj.rect.y, faceobj.rect.width, faceobj.rect.height, faceobj.trackID);
            // 看看与人重合.
            //  intersection over union
            float inter_area = intersection_area(faceobj, personUp);
            float chepaiobj_area = faceobj.rect.area();

            double IOU = inter_area / chepaiobj_area;

            ANNIWOLOGF(INFO, "inter_ared: %.2f chepaiobj_area: %.2f face:%d IOU is %.3f", inter_area, chepaiobj_area, idx, IOU);

            if (inter_area / chepaiobj_area > 0.9)
            {
                isInsideP = true;
                break;
            }
        }

        if (isInsideP)
        {

            // 找到一个face在person内的

            int face_x = faceobj.rect.x;
            int face_y = faceobj.rect.y;
            int face_w = faceobj.rect.width;
            int face_h = faceobj.rect.height;

            ANNIWOLOGF(INFO, "SmokePhoneDetection:camID:%d   x:%d y:%d w:%d  h:%d\n", camID,
                       face_x, face_y, face_w, face_h);

            // 扩大些
            //  face_x = int(std::max(0.0, face_x - 0.6 * face_w));
            face_x = int(std::max(0.0, face_x - 1.5 * face_w));
            face_y = int(std::max(0.0, face_y - 0.6 * face_h));

            // face_w = int(std::min(face_w * 2.2, orig_img_w*1.0));
            face_w = int(std::min(face_w * 4.0, orig_img_w * 1.0));
            face_h = int(std::min(face_h * 2.2, orig_img_h * 1.0)); // 此处会造成h>w很多?为啥python上面ok

            if (face_x + face_w > orig_img_w)
            {
                face_w = orig_img_w - face_x;
            }
            if (face_y + face_h > orig_img_h)
            {
                face_h = orig_img_h - face_y;
            }

            ANNIWOLOGF(INFO, "SmokePhoneDetection enlarged face on p:camID:%d   x:%d y:%d w:%d  h:%d\n", camID,
                       face_x, face_y, face_w, face_h);

            // 对脸部区域进行扩大处理
            int face_expend = 50;

            face_x = std::max((face_x - face_expend), 1);
            face_y = std::max((face_y - face_expend), 1);
            face_h = face_h + face_expend;
            face_w = face_w + face_expend;

            cv::Mat face_img(face_h, face_w, CV_8UC3, cv::Scalar(114, 114, 114));
            image(cv::Rect(face_x, face_y, face_w, face_h)).copyTo(face_img(cv::Rect(0, 0, face_w, face_h)));

            ANNIWOCHECK(face_img.data != nullptr);

            ////////////////////////////////////
            std::vector<float> resnetResults;
            cv::Mat faceimg2 = face_img;

            /////////////////////////////yolo detect on face_img for smoke/////////////////
            // std::unordered_map<int, std::unordered_map<std::string, std::unordered_map<int, int > > >::iterator got_it = reporthistoryArray.find(camID);

            // if (got_it == reporthistoryArray.end())
            // {
            //     ANNIWOLOG(INFO) <<"SmokePhoneDetection: Not in history map!!!"<<"camID:"<<camID<<std::endl;
            //     ANNIWOCHECK(false);
            // }

            // std::unordered_map<std::string, std::unordered_map<int, int > >& perCamIDfunchistory = got_it->second;

            // ///////////////////////////////////////DEBUG visualize class input image
            // std::stringstream buffer;
            // static int debug_outputImgCnt=0;

            // buffer <<camID<<"_efaceimg_"<<debug_outputImgCnt++<<".jpg";

            // std::string text(buffer.str());

            // std::string imgPath(text);
            // cv::imwrite(imgPath,faceimg2);
            // ////////////////////////////////////////

            std::vector<Object> objects;

            int choiceIntVal = randIntWithinScale(globalINICONFObj.ANNIWO_NUM_INSTANCE_SMOKEPHONE);
            std::unordered_map<int, TrtSampleUniquePtr<nvinfer1::IExecutionContext>>::iterator iterCamInstance = executionContexts.find(choiceIntVal);
            std::unordered_map<int, std::unique_ptr<std::mutex>>::iterator iterCamInstancelock = contextlocks.find(choiceIntVal);

            if (iterCamInstance != executionContexts.end())
            {
                yolov4_detection_staff(m_input_datas, camID, instanceID, faceimg2,
                                       runtime, engine,
                                       iterCamInstance->second, // smart pointer context for this cam
                                       gpuNum,
                                       iterCamInstancelock->second, // smart pointer context LOCK for this func-cam
                                       YOLO4_OUTPUT_SIZE, INPUT_W, INPUT_H, objects, BBOX_CONF_THRESH, NUM_CLASSES,
                                       "SmokePhoneDetection_sp");
            }
            else
            {
                ANNIWOLOG(INFO) << "Not found the context for camId:" << camID;
                ANNIWOCHECK(false);
            }

            // static std::vector<std::string> PHONEclass_names = {
            //     "background",
            //     "phone",
            //     "smoke"
            //     };
            // static std::vector<std::string> Faceclass_names={
            //      "face_smoke",
            //      "favce_phone",
            //      "noraml_face"
            //      "background",
            //      "phone",
            //      "smoke"
            // };

            //    }
            bool found=false;
            if(globalINICONFObj.ANNIWO_CHANGE_SMOKE2PHONE==1){
            found = false;
            }
            else{
                found =true;

            }
            for (auto kv : globalINICONFObj.in_use_conf_functions)
            {
                // ANNIWOLOG(INFO) <<"SmokePhoneDetection: kv.first:"<<kv.first<<" predictcls:"<<predictcls<<" camID:"<<camID<<std::endl;
                ANNIWOLOG(INFO) << "SmokePhoneDetection: kv.first:" << kv << " camID:" << camID << std::endl;
                

                //分类模型检测
                //classWeight = {0: 'normal', 1: 'phone', 2: 'smoke'}
                 //if(kv.first=="smoke" && predictcls==2 )
                if (kv == "smoke")
                {
                    // smkObjDet->detect(camID,face_img,objects);

                    // 转换为原图上的坐标
                    for (size_t i = 0; i < objects.size(); i++)
                    {
                        Object obj = objects[i];

                        ANNIWOLOGF(INFO, "SmokePhoneDetection smoke on face:camID:%d  %s = confidence:%.5f at x:%.2f y:%.2f w:%.2f  h:%.2f\n", camID, PHONEclass_names[obj.label].c_str(), obj.prob,
                                   obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

                        if (obj.label == 0 || obj.label == 1) // 类别为：background phone smoke
                        {
                            continue;
                        }

                        // 在face_img上面再次抠图
                        int smoke_x = obj.rect.x;
                        int smoke_y = obj.rect.y;
                        int smoke_w = obj.rect.width;
                        int smoke_h = obj.rect.height;

                        // 抠图
                        smoke_x = int(std::max(0, smoke_x));
                        smoke_y = int(std::max(0, smoke_y));
                        smoke_w = int(std::min(smoke_w, face_w));
                        smoke_h = int(std::min(smoke_h, face_h)); // 此处会造成h>w很多?为啥python上面ok?
                        if (smoke_x + smoke_w > face_w)
                        {
                            smoke_w = face_w - smoke_x;
                        }
                        if (face_y + smoke_h > face_h)
                        {
                            smoke_h = face_h - smoke_y;
                        }

                        cv::Mat smoke_img(smoke_h, smoke_w, CV_8UC3, cv::Scalar(114, 114, 114));
                        face_img(cv::Rect(smoke_x, smoke_y, smoke_w, smoke_h)).copyTo(smoke_img(cv::Rect(0, 0, smoke_w, smoke_h)));

                        ANNIWOCHECK(smoke_img.data != nullptr);

                        ///////////////////////////////////////DEBUG visualize class input image
                        // std::stringstream bufferx;
                        // static int debug_outputImgCntx=0;

                        // bufferx <<camID<<"enlargesp_img_"<<debug_outputImgCntx++<<".jpg";

                        // std::string textx(bufferx.str());

                        // std::string imgPathx(textx);
                        // cv::imwrite(imgPathx,smoke_img);

                        ////////////////////////////////////////
                        smkclassifier->detect(camID, instanceID, smoke_img, resnetResults);
                        float predictcls = -2.0;
                        if (resnetResults.size() >= 3)
                        {
                            // //2:对应resnet的smoke分类
                            // if( resnetResults[2] <= 0.8 )
                            // {
                            //     ANNIWOLOG(INFO) << "SmokePhoneDetection:detect "<<"camID:"<<camID <<"Res NOT 0.8,ignore.";
                            //     continue;
                            // }

                            std::vector<std::size_t> gridwhshape = {3};
                            // todo:注意！ auto 类型 xtensor adapt来的，reshape不能用！
                            //                arrange来的，reshape也不能用!!!
                            auto probs = xt::adapt(resnetResults, gridwhshape);
                            // classWeight = {0: 'normal', 1: 'phone', 2: 'smoke'}
                            //  auto mobilenetclass=xt::argmax(probs);
                            ANNIWOLOG(INFO) << "SmokePhoneDetection: clsnet:" << probs << " camID:" << camID;
                            predictcls = probs(2);
                            if (predictcls < 0.5)
                            {
                                ANNIWOLOG(INFO) << "SmokePhoneDetection:detect "
                                                << "camID:" << camID << "Res NOT ok,ignore:" << predictcls;
                                continue;
                            }
                            else
                            {
                                ANNIWOLOG(INFO) << "SmokePhoneDetection:detect "
                                                << "camID:" << camID << "Res OK prob:" << predictcls;
                            }
                        }

                        ////////////////////////////////////

                        {

                            int x1 = (obj.rect.x + face_x);
                            int y1 = (obj.rect.y + face_y);

                            x1 = x1 > person_x + person_w ? (person_x + person_w) : x1;
                            y1 = y1 > person_y + person_h ? (person_y + person_h) : y1;

                            obj.rect.x = x1;
                            obj.rect.y = y1;
                            obj.rect.width = obj.rect.width;
                            obj.rect.height = obj.rect.height;

                            obj.rect.width = obj.rect.width > person_w ? person_w : obj.rect.width;
                            obj.rect.height = obj.rect.height > person_h ? person_h : obj.rect.height;
                            obj.trackID = person_trackID;
                            obj.label = 5;
                        }
                        overall_objects.push_back(obj);
                        Object face_objects;

                        face_objects.rect.x = face_x+face_expend;
                        face_objects.rect.y = face_y+face_expend;
                        face_objects.rect.height = face_h-face_expend;
                        face_objects.rect.width = face_w-face_expend;
                        face_objects.prob = obj.prob;
                        face_objects.label = 0;

                        overall_objects.push_back(face_objects);
                        found=true;
                        // }else if(kv.first=="phone" && predictcls==1 )
                    }
                }
                else if (kv == "phone")
                {

                    // 转换为原图上的坐标
                    for (size_t i = 0; i < objects.size(); i++)
                    {
                        Object obj = objects[i];

                        ANNIWOLOGF(INFO, "SmokePhoneDetection phone on face:camID:%d  %s = confidence:%.5f at x:%.2f y:%.2f w:%.2f  h:%.2f\n", camID, PHONEclass_names[obj.label].c_str(), obj.prob,
                                   obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

                        if (obj.label == 0 || obj.label == 2) // 类别为：background phone smoke
                        {
                            continue;
                        }

                        //////////////////////////////////
                        // 在face_img上面再次抠图
                        int phone_x = obj.rect.x;
                        int phone_y = obj.rect.y;
                        int phone_w = obj.rect.width;
                        int phone_h = obj.rect.height;

                        // 抠图
                        phone_x = int(std::max(0, phone_x));
                        phone_y = int(std::max(0, phone_y));
                        phone_w = int(std::min(phone_w, face_w));
                        phone_h = int(std::min(phone_h, face_h)); // 此处会造成h>w很多?为啥python上面ok?
                        if (phone_x + phone_w > face_w)
                        {
                            phone_w = face_w - phone_x;
                        }
                        if (face_y + phone_h > face_h)
                        {
                            phone_h = face_h - phone_y;
                        }

                        cv::Mat phone_img(phone_h, phone_w, CV_8UC3, cv::Scalar(114, 114, 114));
                        face_img(cv::Rect(phone_x, phone_y, phone_w, phone_h)).copyTo(phone_img(cv::Rect(0, 0, phone_w, phone_h)));

                        ANNIWOCHECK(phone_img.data != nullptr);

                        ///////////////////////////////////////DEBUG visualize class input image
                        // std::stringstream bufferx;
                        // static int debug_outputImgCntx=0;

                        // bufferx <<camID<<"enlargesp_img_"<<debug_outputImgCntx++<<".jpg";

                        // std::string textx(bufferx.str());

                        // std::string imgPathx(textx);
                        // cv::imwrite(imgPathx,phone_img);

                        ////////////////////////////////////////

                        ////////////////////////////////////////
                        smkclassifier->detect(camID, instanceID, phone_img, resnetResults);
                        float predictcls = -0.1;
                        if (resnetResults.size() >= 3)
                        {
                            // //2:对应resnet的smoke分类
                            // if( resnetResults[2] <= 0.8 )
                            // {
                            //     ANNIWOLOG(INFO) << "SmokePhoneDetection:detect "<<"camID:"<<camID <<"Res NOT 0.8,ignore.";
                            //     continue;
                            // }

                            std::vector<std::size_t> gridwhshape = {3};
                            // todo:注意！ auto 类型 xtensor adapt来的，reshape不能用！
                            //                arrange来的，reshape也不能用!!!
                            auto probs = xt::adapt(resnetResults, gridwhshape);
                            // classWeight = {0: 'normal', 1: 'phone', 2: 'smoke'}
                            //  auto mobilenetclass=xt::argmax(probs);
                            ANNIWOLOG(INFO) << "SmokePhoneDetection: clsnet:" << probs << " camID:" << camID;
                            predictcls = probs(1);
                            if (predictcls < 0.5)
                            {
                                ANNIWOLOG(INFO) << "SmokePhoneDetection:detect "
                                                << "camID:" << camID << "Res NOT ok,ignore.";
                                continue;
                            }
                            else
                            {
                                ANNIWOLOG(INFO) << "SmokePhoneDetection:detect "
                                                << "camID:" << camID << "Res OK prob:" << predictcls;
                            }
                        }

                        ////////////////////////////////////

                        {
                            int x1 = (obj.rect.x + face_x);
                            int y1 = (obj.rect.y + face_y);

                            x1 = x1 > person_x + person_w ? (person_x + person_w) : x1;
                            y1 = y1 > person_y + person_h ? (person_y + person_h) : y1;

                            obj.rect.x = x1;
                            obj.rect.y = y1;
                            obj.rect.width = obj.rect.width;
                            obj.rect.height = obj.rect.height;

                            obj.rect.width = obj.rect.width > person_w ? person_w : obj.rect.width;
                            obj.rect.height = obj.rect.height > person_h ? person_h : obj.rect.height;
                            obj.trackID = person_trackID;
                            obj.label= 4;

                        }
                        overall_objects.push_back(obj);
                        Object face_objects;
                        face_objects.rect.x = face_x+face_expend;
                        face_objects.rect.y = face_y+face_expend;
                        face_objects.rect.height = face_h-face_expend;
                        face_objects.rect.width = face_w-face_expend;
                        face_objects.prob = obj.prob;
                        face_objects.label = 1;

                        overall_objects.push_back(face_objects);
                        found=true;
                    }
                }

                
            }
            //啥也不是
            if (found == false)
                {
                    for(auto obj:objects){
                            int x1 = (obj.rect.x + face_x);
                            int y1 = (obj.rect.y + face_y);

                            x1 = x1 > person_x + person_w ? (person_x + person_w) : x1;
                            y1 = y1 > person_y + person_h ? (person_y + person_h) : y1;

                            obj.rect.x = x1;
                            obj.rect.y = y1;
                            obj.rect.width = obj.rect.width;
                            obj.rect.height = obj.rect.height;

                            obj.rect.width = obj.rect.width > person_w ? person_w : obj.rect.width;
                            obj.rect.height = obj.rect.height > person_h ? person_h : obj.rect.height;
                            obj.trackID = person_trackID;
                            obj.label= 3;
                            overall_objects.push_back(obj);
                    }
                    Object face_objects;
                        face_objects.rect.x = face_x+face_expend;
                        face_objects.rect.y = face_y+face_expend;
                        face_objects.rect.height = face_h-face_expend;
                        face_objects.rect.width = face_w-face_expend;
                    face_objects.prob = obj.prob;
                    face_objects.label = 2;
                    overall_objects.push_back(face_objects);

                }

            ANNIWOLOG(INFO) << "SmokePhoneDetection:det on enlarged face entered"
                            << "camID:" << camID;

            /////////////////////////////yolo detect on face_img for smoke end/////////////////
        }
    }

    if (overall_objects.size() > 0)
    {
        // ostProcessResults(camID, image, overall_objectsSMK,polygonSafeArea_ptr,PHONEclass_names);

        save_imgXmL(img,overall_objects,f, Faceclass_names);
        ANNIWOLOG(INFO) << "save_img_xml successfully!";
    }
    else
    {
        ANNIWOLOG(INFO) << "SmokePhoneDetection:no objects for smoke"
                        << "camID:" << camID;
    }

    // if(overall_objectsPhone.size() > 0)
    // {
    //     //PostProcessResults(camID, image, overall_objectsPhone,polygonSafeArea_ptr,PHONEclass_names);
    //     save_imgXmL(img,overall_objectsPhone,f,PHONEclass_names,"phone");

    // }
    // else
    // {
    //     ANNIWOLOG(INFO) << "SmokePhoneDetection:no objects for phone"<<"camID:"<<camID;
    // }

    ANNIWOLOG(INFO) << "SmokePhoneDetection:exit detect()"
                    << "camID:" << camID;
    image.release();

    return;
}
