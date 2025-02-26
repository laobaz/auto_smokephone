#ifndef __ANNIWO_SUB_UTIL_HPP__
#define __ANNIWO_SUB_UTIL_HPP__

#include<ctime>
#include<vector>
#include<regex>
#include<iostream>
#include<ctime>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h> // struct stat

#include<iomanip>
#include<fstream>



#include <g3log/g3log.hpp>
#include <g3log/logworker.hpp>
#include <g3log/std2_make_unique.hpp>
#include <g3log/logmessage.hpp>
#include "../LogRotateAnniwo.h"

#include <memory>

#include <iomanip>
#include <thread>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <string>  


#include "../ThreadPool.h"


#include <mutex>
#include <thread>


extern std::mutex g_mtx4GpuQueue;  //todo:只在convery里边有调用，无用？

enum anniwo_domain_type { ANNIWO_DOMANI_LIANGKU,ANNIWO_DOMANI_JIAYOUZHAN  };


// #define ANNIWO_INTERNAL_DEBUG  1
//全局ini读入值
struct ANNIWO_INI_CONF_CLASS
{

    int ANNIWO_NUM_INSTANCE_SMOKEPHONE=1;
    int ANNIWO_NUM_THREAD_SMOKEPHONE=1;
    int ANNIWO_NUM_INSTANCE_PERSONBASE=1;
    int ANNIWO_NUM_THREAD_PERSONBASE=1;
    int ANNIWO_CHANGE_SMOKE2PHONE=0;

    anniwo_domain_type domain_config;//地点
    std::unordered_set<std::string> in_use_conf_functions;//功能
    std::string img_input_Path;
    std::string img_output_Path;
    std::string personbase_model_Path;
    std::string personbase_model_Path2;
    std::string smokePhoneface_Path;
    std::string smokePhoneconfig_Path;
    std::string smokePhonethings_Path;
    
};

extern struct ANNIWO_INI_CONF_CLASS globalINICONFObj;





extern ThreadPool *pool;

inline int randIntWithinScale(int range)
{
    if(range <= 0)
    {
        return 0;
    }
    // srand((unsigned)time(NULL));
    int value = rand()%range;
    // std::cout << "randIntWithinScale:value: " << value << std::endl;
    return value;
}


inline const char *rand_str(char *str,const int len)
{
    int i;
    for(i=0;i<len;++i)
        str[i]='A'+rand()%26;
    str[i]='\0';
    return str;
}



inline std::string getRandomName(std::string suffix=".jpg")
{
    static const int LEN_NAME=10;
    // srand((unsigned)time(NULL));
    char name[LEN_NAME+1];
    std::string out = rand_str(name,LEN_NAME);
    return out+suffix;
}

inline std::vector<std::string> stringSplit(const std::string& str, char delim) {
    std::string s;
    s.append(1, delim);
    std::regex reg(s);
    std::vector<std::string> elems(std::sregex_token_iterator(str.begin(), str.end(), reg, -1),
                                   std::sregex_token_iterator());
    return elems;
}



 
inline int getCurWeekDayXB()
{
    // int day,month,year,wday;
    int wday;
    time_t t;
    struct tm * timeinfo;
    time(&t);
    timeinfo = localtime(&t);

    // day = timeinfo->tm_mday;
    // month = timeinfo->tm_mon + 1;
    // year = timeinfo->tm_year + 1900;
 
    wday = timeinfo->tm_wday;

    //0->7
    if(wday == 0)
    {
        wday=7;
    }
 
    // cout<<year<<"  "<<month<<"  "<<day<<"  "<<wday<<endl;
    return wday;
}

//curMinutetime = datetime.datetime.now().hour * 60 + datetime.datetime.now().minute
inline int getCurMinuteDaytimeXB()
{
    time_t t;
    struct tm * timeinfo;
    time(&t);
    timeinfo = localtime(&t);

    //    int tm_sec;         /* 秒，范围从 0 到 59*/
    //    int tm_min;         /* 分，范围从 0 到 59*/
    //    int tm_hour;        /* 小时，范围从 0 到 23 */

    return timeinfo->tm_hour*60+timeinfo->tm_min;

}

 
/**
 * 拷贝文件
 * @param src 原文件
 * @param des 目标文件
 * @return ture 拷贝成功, false 拷贝失败
 */
inline bool CopyFile(const char *src, const char *des)
{
    FILE * fSrc = fopen(src, "rb");
    if(!fSrc)
    {
        printf("打开文件`%s`失败", src);
        return false;
    }
    FILE * fDes = fopen(des, "wb");
    if(!fDes)
    {
        printf("创建文件`%s`失败", des);
        return false;
    }
 
    unsigned char * buf;
    unsigned int length;
    fseek(fSrc, 0, SEEK_END);
    length = ftell(fSrc);
    buf = new unsigned char[length+1];
    memset(buf, 0, length+1);
    fseek(fSrc, 0, SEEK_SET);
    fread(buf, length, 1, fSrc);
 
    fwrite(buf, length, 1, fDes);
 
    fclose(fSrc);
    fclose(fDes);
    delete [] buf;
    return true;
}

// 通过stat结构体 获得文件大小，单位字节
inline size_t getFileSize1(const char *fileName) {

	if (fileName == NULL) {
		return -1;
	}
	
	// 这是一个存储文件(夹)信息的结构体，其中有文件大小和创建时间、访问时间、修改时间等
	struct stat statbuf;

	// 提供文件名字符串，获得文件属性结构体
	stat(fileName, &statbuf);
	
	// 获取文件大小
	size_t filesize = statbuf.st_size;

	return filesize;
}


 // writing on a text file
inline bool TrunckFile(const char *fileName) {
    std::ofstream out(fileName,std::ios::out|std::ios::binary);
    if (out.is_open()) 
    {
        out << "Backup and Cleaned log.\n";
        out.close();
    }else
    {
        return false;
    }
    return true;
}


 // create and write a text file
inline bool dumpToTxtFile(const char *fileName, const std::string& strcontent) {
    std::ofstream out(fileName,std::ios::out);
    if (out.is_open()) 
    {
        out << strcontent;
        out.close();
    }else
    {
        return false;
    }
    return true;
}

 

#endif //