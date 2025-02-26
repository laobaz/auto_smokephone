#include <chrono>
#include <iostream>
#include <memory>
#include <vector>
#include <numeric>

///////////////from yolox/////////////
#include <fstream>
#include <sstream>

#include <dirent.h>
#include <cstddef>
#include <vector>
//////////////////////////////////////
#include "core/smoke/SmokePhone.hpp"
#include "core/personbase/basePerson.hpp"
#include <xtensor.hpp>
#include <xtensor/xnpy.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>

#include "core/common/logging.h"
#include <uuid/uuid.h>
#include <unordered_set>
#include "core/utils/ini.h"



#define DEVICE 0


#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

using namespace nvinfer1;





///////////////from yolox/////////////


//#define NMS_THRESH 0.45
// #define BBOX_CONF_THRESH  0.45
// #define BBOX_CONF_THRESH  0.35
// #define BBOX_CONF_THRESH  0.2
//#define BBOX_CONF_THRESH  0.35

 //const int INPUT_W = 608;
 //const int INPUT_H = 608;
// static const int NUM_CLASSES = 10;  //yolov4 jiayouzhan 人车

// static const int NUM_CLASSES = 80;  //yolov4 coco pretrain/personbase
// static const int NUM_CLASSES = 2;  //smoke
// static const int NUM_CLASSES = 4;  //window
//const int NUM_CLASSES = 4;  //JIAYOUJI
// static const int NUM_CLASSES = 1;  //反光衣
// static const int NUM_CLASSES = 12;  //卸油区
// static const int NUM_CLASSES = 4;  //人孔井

// static const int NUM_CLASSES = 9;//helmet
// static const int NUM_CLASSES = 4;//zhuangxieyou

// static const int NUM_CLASSES = 3;//xiaohuang somke phone
//const int CHANNELS = 3;

//std::vector<int> input_image_shape={608, 608};


//////////////////////////////////////



static BasePersonDetection* basePersonDetect = NULL;
static SmokePhoneDetection* smokePhoneDetect=NULL;

struct ANNIWO_INI_CONF_CLASS globalINICONFObj;








// void Stringsplit(const std::string str,const  char split,std::vector<std::string>& rst)
// {
// 	std::istringstream iss(str);	// 输入流
// 	std::string token;			// 接收缓冲区
// 	while (getline(iss, token, split))	// 以split为分隔符
// 	{
// 		rst.push_back(token);
// 	}
// }


// void save_imgXmL(const cv::Mat& bgr,const std::vector<Object>& objects, std::string &f,std::string &type) {

//     //新建一个Xml文件
//     TiXmlDocument* imgXML = new TiXmlDocument();

//     //TiXmlDeclaration* img = new TiXmlDeclaration("1.0", "utf-8", "");

//     //imgXML->LinkEndChild(img);



//     std::string imgWidth = std::to_string(bgr.cols);
//     std::string imgHeight = std::to_string(bgr.rows);
//     std::string imgDepth = std::to_string(bgr.channels());


//     std::string filenames=f;
//     std::vector<std::string> strlist;
//     Stringsplit(filenames,'/',strlist);

//     filenames=strlist.back();



//     //printf("成功了！");

//     //第一层
//     TiXmlElement* annotation = new TiXmlElement("annotation");
//     //把根节点加到文档类

//     imgXML->LinkEndChild(annotation);


//     //第二层
//     TiXmlElement* folder = new TiXmlElement("folder");
//     TiXmlText* folderText = new TiXmlText("folder");
//     folder->LinkEndChild(folderText);

//     TiXmlElement* filename = new TiXmlElement("filename");
//     TiXmlText* filename_name = new TiXmlText(filenames.c_str());
//     filename->LinkEndChild(filename_name);

//     TiXmlElement* path = new TiXmlElement("path");
//     TiXmlText* pathText = new TiXmlText(f.c_str());
//     path->LinkEndChild(pathText);

//     TiXmlElement* source = new TiXmlElement("source");
//     //第三层
//     TiXmlElement* database = new TiXmlElement("datebase");
//     TiXmlText* databaseText = new TiXmlText("Unknown");
//     database->LinkEndChild(databaseText);
//     source->LinkEndChild(database);
//     //end//
//     TiXmlElement* size = new TiXmlElement("size");

//     //第三层
//     TiXmlElement* width = new TiXmlElement("width");
//     TiXmlText* widthText = new TiXmlText(imgWidth.c_str() );
//     width->LinkEndChild(widthText);
//     TiXmlElement* height = new TiXmlElement("height");
//     TiXmlText* heightText = new TiXmlText(imgHeight.c_str());
//     height->LinkEndChild(heightText);
//     TiXmlElement* depth = new TiXmlElement("depth");
//     TiXmlText* depthText = new TiXmlText(imgDepth.c_str());
//     depth->LinkEndChild(depthText);

//     size->LinkEndChild(width);
//     size->LinkEndChild(height);
//     size->LinkEndChild(depth);
//     //end//

//     TiXmlElement* segmented = new TiXmlElement("segmented");
//     TiXmlText* segmentedText = new TiXmlText("0");
//     segmented->LinkEndChild(segmentedText);



//     annotation->LinkEndChild(folder);
//     annotation->LinkEndChild(filename);
//     annotation->LinkEndChild(path);
//     annotation->LinkEndChild(source);
//     annotation->LinkEndChild(size);
//     annotation->LinkEndChild(segmented);
    


//     for (size_t i = 0; i < objects.size(); i++) 
//     {
//         const Object& obj = objects[i];

//         std::string x0 = std::to_string((int)obj.rect.x);
//         std::string y0 = std::to_string((int)obj.rect.y);
//         std::string x1 = std::to_string((int)(obj.rect.x + obj.rect.width));
//         std::string y1 = std::to_string((int)(obj.rect.y + obj.rect.height));

//         //第二层
//         TiXmlElement* object = new TiXmlElement("object");
//         //第三层
//         TiXmlElement* name = new TiXmlElement("name");
//         TiXmlText* nameText = new TiXmlText(type.c_str());
//         name->LinkEndChild(nameText);

//         TiXmlElement* pose = new TiXmlElement("pose");
//         TiXmlText* poseText = new TiXmlText("Unspecified");
//         pose->LinkEndChild(poseText);

//         TiXmlElement* truncated = new TiXmlElement("truncated");
//         TiXmlText* truncatedText = new TiXmlText("0");
//         truncated->LinkEndChild(truncatedText);

//         TiXmlElement* difficult = new TiXmlElement("difficult");
//         TiXmlText* difficultText = new TiXmlText("0");
//         difficult->LinkEndChild(difficultText);

//         TiXmlElement* bndbox = new TiXmlElement("bndbox");
//         //第四层
//         TiXmlElement* xmin = new TiXmlElement("xmin");
//         TiXmlText* xminText = new TiXmlText(x0.c_str());
//         xmin->LinkEndChild(xminText);

//         TiXmlElement* ymin = new TiXmlElement("ymin");
//         TiXmlText* yminText = new TiXmlText(y0.c_str());
//         ymin->LinkEndChild(yminText);

//         TiXmlElement* xmax = new TiXmlElement("xmax");
//         TiXmlText* xmaxText = new TiXmlText(x1.c_str());
//         xmax->LinkEndChild(xmaxText);

//         TiXmlElement* ymax = new TiXmlElement("ymax");
//         TiXmlText* ymaxText = new TiXmlText(y1.c_str());
//         ymax->LinkEndChild(ymaxText);

//         bndbox->LinkEndChild(xmin);
//         bndbox->LinkEndChild(ymin);
//         bndbox->LinkEndChild(xmax);
//         bndbox->LinkEndChild(ymax);
//         //....//

//         object->LinkEndChild(name);
//         object->LinkEndChild(pose);
//         object->LinkEndChild(truncated);
//         object->LinkEndChild(difficult);
//         object->LinkEndChild(bndbox);

//         annotation->LinkEndChild(object);
//     }




//     std::stringstream buffer;  
//     // buffer << "det_res_"<<saveCnt<<".xml";  

//     std::string filenamestub;
//     if(filenames.rfind(".") != std::string::npos)
//     {
//         int atpos=filenames.rfind(".");
//         filenamestub=filenames.substr(0,atpos);
//     }else
//     {
//         printf("%s Enter a file name without suffix!!\n", f);
//         exit(-1);
//     }

//     buffer <<"../labels_gen/"<<filenamestub<<".xml";  

//     bool result = imgXML->SaveFile(buffer.str().c_str());
//     if (result)
//         printf("%s Write complete!\n", filenamestub.c_str());
//     else
//         printf("%s Write failed\n", filenamestub.c_str());
// }



void Load_config(){
     mINI::INIFile file("../config.ini");
    mINI::INIStructure ini;

    if(!file.read(ini))
    {
        ANNIWOLOG(INFO) << "Error no ini file!" ;
          ANNIWOCHECK(false);
         exit(-1);
    }

    if(ini.has("config"))
    {
        if(ini["config"].has("inusefunctions"))
        {
            std::string strvalue = ini.get("config").get("inusefunctions");
            std::vector<std::string> strVector = stringSplit(strvalue, ' '); 
            for (auto f : strVector)
            {
                    globalINICONFObj.in_use_conf_functions.emplace(f);
            }

        }else
        {
            ANNIWOLOG(INFO) << "Error no inusefunctions setting in ini file!" ;
            ANNIWOCHECK(false);
            exit(-1);

        }
        

        globalINICONFObj.domain_config=ANNIWO_DOMANI_LIANGKU;
        if(ini["config"].has("domain"))
        {
            std::string strvalue = ini.get("config").get("domain");
            std::vector<std::string> strVector = stringSplit(strvalue, ' '); 
            if(strVector[0]=="jiayouzhan")
            {
                globalINICONFObj.domain_config=ANNIWO_DOMANI_JIAYOUZHAN;
                ANNIWOLOG(INFO) <<"domain_config:  "<<"jiayouzhan"<<std::endl;

            }else
            {
                ANNIWOLOG(INFO) <<"domain_config:  "<<"jiayouzhan"<<std::endl;
            }

        }



        
    }
    if(ini.has("path"))
    {
        //输入图片目录
        if(ini["path"].has("img_input_Path")){
            std::string strvalue = ini.get("path").get("img_input_Path");
            std::vector<std::string> strVector = stringSplit(strvalue, ' ');
            globalINICONFObj.img_input_Path=strVector[0];

        }
        if(ini["path"].has("img_output_Path")){
            std::string strvalue = ini.get("path").get("img_output_Path");
            std::vector<std::string> strVector = stringSplit(strvalue, ' ');
            globalINICONFObj.img_output_Path=strVector[0];
            if(globalINICONFObj.img_output_Path.length() > 5)
            {
                //标识目录
                if(globalINICONFObj.img_output_Path.back() != '/')
                {
                    globalINICONFObj.img_output_Path.push_back('/');
                }


                struct stat st = {0};
                //创建目录
                if (stat(globalINICONFObj.img_output_Path.c_str(), &st) == -1) {
                    mkdir(globalINICONFObj.img_output_Path.c_str(), 0700);
                }
                //检查是否创建成功
                if (stat(globalINICONFObj.img_output_Path.c_str(), &st) == -1) {
                    ANNIWOLOG(INFO) <<"Unable to create:"<<globalINICONFObj.img_output_Path<<std::endl;
                    ANNIWOCHECK(false);
                    exit(-1);
                }
            }
        }
        //#人物基础模型输入地址
        if(ini["path"].has("personbase_model_Path")){
            std::string strvalue = ini.get("path").get("personbase_model_Path");
            std::vector<std::string> strVector = stringSplit(strvalue, ' ');
            globalINICONFObj.personbase_model_Path=strVector[0];
        }
        if(ini["path"].has("personbase_model_Path2")){
            std::string strvalue = ini.get("path").get("personbase_model_Path2");
            std::vector<std::string> strVector = stringSplit(strvalue, ' ');
            globalINICONFObj.personbase_model_Path2=strVector[0];
        }
        //#脸部模型输入地址
        if(ini["path"].has("smokePhoneface_Path")){
            std::string strvalue = ini.get("path").get("smokePhoneface_Path");
            std::vector<std::string> strVector = stringSplit(strvalue, ' ');
            globalINICONFObj.smokePhoneface_Path=strVector[0];
        }
        //#脸部模型配置文件
        if(ini["path"].has("smokePhoneconfig_Path")){
            std::string strvalue = ini.get("path").get("smokePhoneconfig_Path");
            std::vector<std::string> strVector = stringSplit(strvalue, ' ');
            globalINICONFObj.smokePhoneconfig_Path=strVector[0];
        }
        //#smoke or phone模型地址
        if(ini["path"].has("smokePhonethings_Path")){
            std::string strvalue = ini.get("path").get("smokePhonethings_Path");
            std::vector<std::string> strVector = stringSplit(strvalue, ' ');
            globalINICONFObj.smokePhonethings_Path=strVector[0];
        }

        

        
        //创建根图片输出目录
        //std::string tmpPath=std::string("/var/anniwo/");

        // struct stat st = {0};
        // //创建目录
        // if (stat(tmpPath.c_str(), &st) == -1) {
        //     mkdir(tmpPath.c_str(), 0700);
        // }
        // //检查是否创建成功
        // if (stat(tmpPath.c_str(), &st) == -1) {
        //     ANNIWOLOG(INFO) <<"Unable to create:"<<tmpPath<<std::endl;
        //     ANNIWOCHECK(false);
        //     exit(-1);
        // }
        //////////////////////////////////////

        // if(ini["path"].has("onlineCollectionPath"))
        // {
        //     std::string strvalue = ini.get("path").get("onlineCollectionPath");
        //     std::vector<std::string> strVector = stringSplit(strvalue, ' '); 

        //     //todo
        //     globalINICONFObj.onlineCollectionPath=strVector[0];
        //     if(globalINICONFObj.onlineCollectionPath.length() > 5)
        //     {
        //         //标识目录
        //         if(globalINICONFObj.onlineCollectionPath.back() != '/')
        //         {
        //             globalINICONFObj.onlineCollectionPath.push_back('/');
        //         }


        //         struct stat st = {0};
        //         //创建目录
        //         if (stat(globalINICONFObj.onlineCollectionPath.c_str(), &st) == -1) {
        //             mkdir(globalINICONFObj.onlineCollectionPath.c_str(), 0700);
        //         }
        //         //检查是否创建成功
        //         if (stat(globalINICONFObj.onlineCollectionPath.c_str(), &st) == -1) {
        //             ANNIWOLOG(INFO) <<"Unable to create:"<<globalINICONFObj.onlineCollectionPath<<std::endl;
        //             ANNIWOCHECK(false);
        //             exit(-1);
        //         }



        //     }

        // }
        
    }
 //改动12.4
    if(ini.has("change")){
        globalINICONFObj.ANNIWO_CHANGE_SMOKE2PHONE=0;
        if(ini["change"].has("smoke2phone"))
        {
            std::string strvalue = ini.get("change").get("smoke2phone");
            std::istringstream isb_str(strvalue);
            int ivalue = 0;
	        isb_str >> ivalue;
            if(ivalue==1||ivalue==0)
            {
                globalINICONFObj.ANNIWO_CHANGE_SMOKE2PHONE=ivalue;
            }
            else
            {
                ANNIWOLOG(INFO) <<"ini:performance.smoke2phone The value should be 0 or 1, please check the value";
            }
        }
    }
    else{
        //默认关闭
        globalINICONFObj.ANNIWO_CHANGE_SMOKE2PHONE=0;
    }


}



int successful_img_xml=0;
int all_img=0;

int main(int argc, char *argv[]) {

    const std::string path_to_log_file = "../";
    const std::string log_file = "";

    std::unique_ptr<g3::LogWorker> logworker {g3::LogWorker::createLogWorker()};

    
    // auto handle = logworker->addDefaultLogger(log_file, path_to_log_file);
    // g3::initializeLogging(logworker.get());


   auto sinkHandle = logworker->addSink(std::make_unique<LogRotate>(log_file,path_to_log_file),
                                          &LogRotate::save);
   
   // initialize the logger before it can receive LOG calls
   initializeLogging(logworker.get());                        


    Load_config();
    //std::cout << "load config sucessful" << std::endl;

    basePersonDetect = new BasePersonDetection();
    
    smokePhoneDetect = new SmokePhoneDetection();


    smokePhoneDetect->initTracks();
    basePersonDetect->initTracks();


    //无用的参数
    int camID=0;
    int instanceID=0;
    const Polygon* polygonSafeArea_ptr={nullptr};



    //basePersonDetect->initTracks(globalJsonConfObj,person_base_functions);


    //int dims = num_anchors* (NUM_CLASSES + 5);
    //int output_size = 1*7581*dims;

///////////////////////


    std::vector<std::string> all_img_paths5;
    if(all_img_paths5.size() <= 0 )
    {
        std::vector<cv::String> cv_all_img_paths;
        // cv::glob("../testchepai/", cv_all_img_paths); //仅有两个错误
        cv::glob(globalINICONFObj.img_input_Path, cv_all_img_paths); //
        // cv::glob("../testjyj_fanguangyi_imgs/", cv_all_img_paths); //
        // cv::glob("../testjyj_xieyou_imgs/", cv_all_img_paths); //
        // cv::glob("../test_jyz_rkj_xunjian_imgs/", cv_all_img_paths); //
        
        // cv::glob("../jingzhichepai/", cv_all_img_paths);//ok
        // cv::glob("../cutchepai/", cv_all_img_paths); //都被判断为模糊

        for (const auto& img_path : cv_all_img_paths) {
            all_img_paths5.push_back(img_path);
        }
    }
    std::vector<Object> basePersonResult;

    for(int i = 0; i<all_img_paths5.size(); i++)
    {
        // std::vector<std::size_t> shape = { 1,7581,dims };

        //std::vector<float> out_data( output_size, 1.0);
        
        //  std::vector<std::size_t> shape = { 1,608,608,3 };
        //  auto a1 = xt::adapt(input_data, shape);
        //  xt::dump_npy("inputimage_xtensor.npy", a1);
        //  exit(0);

        std::string input_image_path = all_img_paths5[i];
        all_img++;

        cv::Mat img = cv::imread(input_image_path);

        int img_w = img.cols;
        int img_h = img.rows;

        if(img.empty()||img_w<0||img_h<0){
            std::cout<<input_image_path<<"may is no input image"<<std::endl;
            continue;
        }
        ANNIWOLOG(INFO)<<input_image_path<<" starting";


        //std::cout << "static_resized image" << std::endl;
        //python: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        //   run(predictor, input_data, input_shape,out_data);
                // run inference
        
        auto start = std::chrono::system_clock::now();
        basePersonResult.clear();
        basePersonDetect->detect(camID,instanceID,img,basePersonResult);
        if(!basePersonResult.empty()){
            ANNIWOLOG(INFO)<<"basePersonResult Detection successful";
            smokePhoneDetect->detect(camID,instanceID,img,polygonSafeArea_ptr,basePersonResult,input_image_path); 
            ANNIWOLOG(INFO)<<"SmokePhoneDetect sucessful";
            
        }
        else{
            ANNIWOLOG(INFO)<<"Not Detect person!";
            continue;
        }
        

        auto end = std::chrono::system_clock::now();

        ANNIWOLOG(INFO)<< "infer time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        ANNIWOLOG(INFO)<<input_image_path<<" successfully";


    }

    std::cout<<"All pictures are done"<<std::endl;
    double rate=1.0*successful_img_xml/all_img;
    std::cout<<"Success img is "<<successful_img_xml<<"all img is "<<all_img<<"Detection rate is "<<rate<<std::endl;
    ANNIWOLOG(INFO)<<"All pictures are done"<<std::endl;
    


    if(basePersonDetect)
        delete basePersonDetect;
    if(smokePhoneDetect)
        delete smokePhoneDetect;

    ANNIWOLOG(INFO)<<"Detection destory successfully"<<std::endl;

    sinkHandle.release();
    logworker.release();

    



  return 0;
}

//结束时出现的错误


// ***** FATAL SIGNAL RECEIVED ******* 
// Received fatal signal: SIGSEGV(11)	PID: 9950

// ***** SIGNAL SIGSEGV(11)

// *******	STACKDUMP *******
// 	stack dump [1]  /usr/local/lib/libg3log.so.2.1.0-0+0x161aa [0x7f0c215161aa]
// 	stack dump [2]  /lib/x86_64-linux-gnu/libpthread.so.0+0x14420 [0x7f0c21628420]
// 	stack dump [3]  /lib/x86_64-linux-gnu/libgcc_s.so.1+0x1c328 [0x7f0c11cd3328]
// 	stack dump [4]  /lib/x86_64-linux-gnu/libgcc_s.so.1_Unwind_Find_FDE+0xd1 [0x7f0c11cd3fd1]
// 	stack dump [5]  /lib/x86_64-linux-gnu/libgcc_s.so.1+0x1860a [0x7f0c11ccf60a]
// 	stack dump [6]  /lib/x86_64-linux-gnu/libgcc_s.so.1_Unwind_RaiseException+0x1cd [0x7f0c11cd107d]
// 	stack dump [7]  /lib/x86_64-linux-gnu/libstdc++.so.6__cxa_throw+0x3b [0x7f0c11f1024b]
// 	stack dump [8]  /usr/local/cuda/lib64/libnvinfer.so.8+0xe772d5 [0x7f0c13bd92d5]

// Exiting after fatal event  (FATAL_SIGNAL). Fatal type:  SIGSEGV
// Log content flushed sucessfully to sink



// Exiting, log  

// exitWithDefaultSignalHandler:229. Exiting due to FATAL_SIGNAL, 11   

// 段错误 (核心已转储)


//