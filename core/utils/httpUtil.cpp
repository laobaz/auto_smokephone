
#include "httpUtil.hpp"

#include <sstream>

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <chrono>

#include <iostream>
#include <ctime>
#include <iomanip>
#include <string>
#include <sstream>
#include <stdio.h>

#include <curlpp/cURLpp.hpp>
#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>
#include <curlpp/Exception.hpp>
#include "subUtils.hpp"

#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"



const std::string ANNIWO_LOG_IMAGES_PATH={"/var/anniwo/images/"};

void  HttpUtil::post( const char* str2send, size_t sendlength, const std::string& url)
{
    // std::istringstream myStream(str2send);
    // int size = myStream.str().size();
        
    char buf[50];
    try
    {
        // curlpp::Cleanup cleaner;
        // curlpp::Easy request;

        // std::list< std::string > headers;
        // headers.push_back("User-Agent: Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko"); 
        // headers.push_back("Content-Type: application/json"); 
        // sprintf(buf, "Content-Length: %d", size); 
        // headers.push_back(buf);
        
        // using namespace curlpp::Options;
        // request.setOpt(new Verbose(true));
        // request.setOpt(new ReadStream(&myStream));
        // request.setOpt(new InfileSize(size));
        // request.setOpt(new Upload(true));
        // request.setOpt(new HttpHeader(headers));
        // request.setOpt(new Url(url.c_str()));
        
        // request.perform();


        curlpp::Cleanup cleaner;
        curlpp::Easy request;
        
        request.setOpt(new curlpp::options::Url(url.c_str())); 
        request.setOpt(new curlpp::options::Verbose(true)); 
        
        std::list< std::string > headers;
        headers.push_back("User-Agent: Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko"); 
        headers.push_back("Content-Type: application/json"); 
        sprintf(buf, "Content-Length: %d", sendlength); 
        headers.push_back(std::string(buf));
        
        request.setOpt(new curlpp::options::HttpHeader(headers)); 
        
        request.setOpt(new curlpp::options::PostFields(str2send));
        request.setOpt(new curlpp::options::PostFieldSize( sendlength ) );
        
        request.perform(); 


    // ANNIWOLOG(INFO) <<"debug:json lenght:"<<sendlength<<"to send json:"<< str2send ;
    }
    catch ( curlpp::LogicError & e )
    {
        ANNIWOLOG(INFO) << e.what() ;
    }

}

void saveImgAndPost(int camID, const std::string taskId, const std::string imgPath, 
                    cv::Mat image, std::chrono::system_clock::time_point eventStartTP,
                    const std::string& jsonStrIn, size_t jsonlength, const std::string submitUrl)
{

        //todo:WHY?CV_IMWRITE_PNG_COMPRESSION not defined???
        // std::vector<int> compression_params;
        // compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION); //PNG格式图片的压缩级别  
        // compression_params.push_back(8);  //这里设置保存的图像质量级别

        // cv::imwrite(imgPath,image,compression_params);


        int imgWidth = image.cols;
        int imgHeight = image.rows;
        int imgDepth = image.channels();

        ANNIWOCHECK(imgWidth>0 && imgHeight>0);

        cv::imwrite(imgPath,image);

        //      res = {
        //     "camID": camID,
        //     "imgPath": imgPath,
        //     "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        //     'info': jsonStrBuf
        // }


        rapidjson::StringBuffer jsonstrbuf;
        rapidjson::Writer<rapidjson::StringBuffer> writer(jsonstrbuf);

        writer.StartObject();               // Between StartObject()/EndObject(), 

        writer.Key("camId");                
        writer.Int(camID); 

        writer.Key("taskId");                
        writer.String(taskId.c_str());  

        writer.Key("imgWidth");                
        writer.Int(imgWidth); 
        
        writer.Key("imgHeight");                
        writer.Int(imgHeight); 

        writer.Key("imgDepth");                
        writer.Int(imgDepth); 

        writer.Key("imgPath");                
        writer.String(imgPath.c_str());  


        writer.Key("startTime");                
        //
        // time2Str
        //
        std::time_t tt = std::chrono::system_clock::to_time_t(eventStartTP); 
        std::ostringstream oss;
        oss << std::put_time(std::localtime(&tt), "%Y-%m-%d %H:%M:%S");
        std::string strTime{oss.str()};
        // std::cout << "strTime:" << strTime << '\n';
        writer.String(strTime.c_str());  



        writer.Key("time");                
        //
        // time2Str
        //
        std::time_t tt2{std::time(nullptr)};
        std::ostringstream oss2;
        oss2 << std::put_time(std::localtime(&tt2), "%Y-%m-%d %H:%M:%S");
        std::string strTime2{oss2.str()};
        // std::cout << "strTime:" << strTime << '\n';
        writer.String(strTime2.c_str());  


        writer.Key("info");         
        writer.RawValue(jsonStrIn.c_str(), jsonlength,  rapidjson::Type::kArrayType);       
        //Wrong here.
        // writer.String(jsonStrIn);  



        writer.EndObject();




        HttpUtil httputilObj;
        ANNIWOLOG(INFO) <<"MESSAGE:"<< jsonstrbuf.GetString() ;
        httputilObj.post(jsonstrbuf.GetString(), jsonstrbuf.GetLength(),submitUrl);
        
        image.release();
}

extern int successful_img_xml;

void Stringsplit(const std::string str,const  char split,std::vector<std::string>& rst)
{
	std::istringstream iss(str);	// 输入流
	std::string token;			// 接收缓冲区
	while (getline(iss, token, split))	// 以split为分隔符
	{
		rst.push_back(token);
	}
}
void save_imgXmL(const cv::Mat& bgr,const std::vector<Object>& objects,std::string &f,std::vector<std::string> &class_names) {

    //新建一个Xml文件
    TiXmlDocument* imgXML = new TiXmlDocument();

    //TiXmlDeclaration* img = new TiXmlDeclaration("1.0", "utf-8", "");

    //imgXML->LinkEndChild(img);



    std::string imgWidth = std::to_string(bgr.cols);
    std::string imgHeight = std::to_string(bgr.rows);
    std::string imgDepth = std::to_string(bgr.channels());


    std::string filenames=f;
    std::vector<std::string> strlist;
    Stringsplit(filenames,'/',strlist);

    filenames=strlist.back();



    //printf("成功了！");

    //第一层
    TiXmlElement* annotation = new TiXmlElement("annotation");
    //把根节点加到文档类

    imgXML->LinkEndChild(annotation);


    //第二层
    TiXmlElement* folder = new TiXmlElement("folder");
    TiXmlText* folderText = new TiXmlText("folder");
    folder->LinkEndChild(folderText);

    TiXmlElement* filename = new TiXmlElement("filename");
    TiXmlText* filename_name = new TiXmlText(filenames.c_str());
    filename->LinkEndChild(filename_name);

    TiXmlElement* path = new TiXmlElement("path");
    TiXmlText* pathText = new TiXmlText(f.c_str());
    path->LinkEndChild(pathText);

    TiXmlElement* source = new TiXmlElement("source");
    //第三层
    TiXmlElement* database = new TiXmlElement("datebase");
    TiXmlText* databaseText = new TiXmlText("Unknown");
    database->LinkEndChild(databaseText);
    source->LinkEndChild(database);
    //end//
    TiXmlElement* size = new TiXmlElement("size");

    //第三层
    TiXmlElement* width = new TiXmlElement("width");
    TiXmlText* widthText = new TiXmlText(imgWidth.c_str() );
    width->LinkEndChild(widthText);
    TiXmlElement* height = new TiXmlElement("height");
    TiXmlText* heightText = new TiXmlText(imgHeight.c_str());
    height->LinkEndChild(heightText);
    TiXmlElement* depth = new TiXmlElement("depth");
    TiXmlText* depthText = new TiXmlText(imgDepth.c_str());
    depth->LinkEndChild(depthText);

    size->LinkEndChild(width);
    size->LinkEndChild(height);
    size->LinkEndChild(depth);
    //end//

    TiXmlElement* segmented = new TiXmlElement("segmented");
    TiXmlText* segmentedText = new TiXmlText("0");
    segmented->LinkEndChild(segmentedText);



    annotation->LinkEndChild(folder);
    annotation->LinkEndChild(filename);
    annotation->LinkEndChild(path);
    annotation->LinkEndChild(source);
    annotation->LinkEndChild(size);
    annotation->LinkEndChild(segmented);
    

    for (size_t i = 0; i < objects.size(); i++) 
    {
        const Object& obj = objects[i];

        std::string x0 = std::to_string((int)obj.rect.x);
        std::string y0 = std::to_string((int)obj.rect.y);
        std::string x1 = std::to_string((int)(obj.rect.x + obj.rect.width));
        std::string y1 = std::to_string((int)(obj.rect.y + obj.rect.height));

        //第二层
        TiXmlElement* object = new TiXmlElement("object");
        //第三层
        TiXmlElement* name = new TiXmlElement("name");
        TiXmlText* nameText = new TiXmlText(class_names[obj.label].c_str());
        name->LinkEndChild(nameText);

        TiXmlElement* pose = new TiXmlElement("pose");
        TiXmlText* poseText = new TiXmlText("Unspecified");
        pose->LinkEndChild(poseText);

        TiXmlElement* truncated = new TiXmlElement("truncated");
        TiXmlText* truncatedText = new TiXmlText("0");
        truncated->LinkEndChild(truncatedText);

        TiXmlElement* difficult = new TiXmlElement("difficult");
        TiXmlText* difficultText = new TiXmlText("0");
        difficult->LinkEndChild(difficultText);

        TiXmlElement* bndbox = new TiXmlElement("bndbox");
        //第四层
        TiXmlElement* xmin = new TiXmlElement("xmin");
        TiXmlText* xminText = new TiXmlText(x0.c_str());
        xmin->LinkEndChild(xminText);

        TiXmlElement* ymin = new TiXmlElement("ymin");
        TiXmlText* yminText = new TiXmlText(y0.c_str());
        ymin->LinkEndChild(yminText);

        TiXmlElement* xmax = new TiXmlElement("xmax");
        TiXmlText* xmaxText = new TiXmlText(x1.c_str());
        xmax->LinkEndChild(xmaxText);

        TiXmlElement* ymax = new TiXmlElement("ymax");
        TiXmlText* ymaxText = new TiXmlText(y1.c_str());
        ymax->LinkEndChild(ymaxText);

        bndbox->LinkEndChild(xmin);
        bndbox->LinkEndChild(ymin);
        bndbox->LinkEndChild(xmax);
        bndbox->LinkEndChild(ymax);
        //....//

        object->LinkEndChild(name);
        object->LinkEndChild(pose);
        object->LinkEndChild(truncated);
        object->LinkEndChild(difficult);
        object->LinkEndChild(bndbox);

        annotation->LinkEndChild(object);
    }
    




    std::stringstream buffer;  
    // buffer << "det_res_"<<saveCnt<<".xml";  

    std::string filenamestub;
    if(filenames.rfind(".") != std::string::npos)
    {
        int atpos=filenames.rfind(".");
        filenamestub=filenames.substr(0,atpos);
    }else
    {
        printf("%s Enter a file name without suffix!!\n", f);
        exit(-1);
    }

    buffer <<globalINICONFObj.img_output_Path<<filenamestub<<".xml";  

    bool result = imgXML->SaveFile(buffer.str().c_str());
    if (result)
    {
        printf("%s Write complete!\n", filenamestub.c_str());
        successful_img_xml ++;
    }
    else
        printf("%s Write failed\n", filenamestub.c_str());
}