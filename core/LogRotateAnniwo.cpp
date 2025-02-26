

#include "LogRotateAnniwo.h"

#include <iostream>
#include <unistd.h>
#include<fstream>
#include <sstream>  


static size_t cntByteSize=0;
static int log_suffix_cnt=0;
static std::ofstream *pOutStream=nullptr;
static std::stringstream buffer;  //output filename

/// @param log_prefix to use for the file
LogRotate::LogRotate(const std::string& log_prefix, const std::string& log_directory)
{}


LogRotate::~LogRotate() {
   if(pOutStream)
   {
      if (pOutStream->is_open()) 
      {
         pOutStream->close();
      }else
      {
         std::cerr << "\n LogRotate::save2, log NOT ABLE TO CLOSE!!!";
      }
      delete pOutStream;
      pOutStream=nullptr;
   }
   std::cerr << "\nExiting, log  ";
}


/// @param logEntry to write to file
void LogRotate::save(std::string logEntry) 
{

   if(log_suffix_cnt==0)
   {
      buffer <<"../anniwo.log"; 
      pOutStream = new std::ofstream(buffer.str().c_str(),std::ios::out);
      if (pOutStream->is_open()) 
      {
      }else
      {
         std::cerr << "\n LogRotate::save, log NOT ABLE TO CREATE!!!:"<<buffer.str();
      }
      log_suffix_cnt++;
   }

   if(cntByteSize > 500*1024*1024)//500M
   {

      buffer.clear(); //清不了
      buffer.str(""); //清空
      buffer <<"../anniwo.log_"<<log_suffix_cnt;  

   
      if(pOutStream)
      {
         if (pOutStream->is_open()) 
         {
            pOutStream->close();
         }else
         {
            std::cerr << "\n LogRotate::save2, log NOT ABLE TO CLOSE!!!";
         }
         delete pOutStream;
         pOutStream=nullptr;
      }
      pOutStream = new std::ofstream(buffer.str().c_str(),std::ios::out);
      if (pOutStream->is_open()) 
      {
      }else
      {
         std::cerr << "\n LogRotate::save3, log NOT ABLE TO CREATE!!!:"<<buffer.str();
      }
      cntByteSize=0;
      log_suffix_cnt += 1;

   }
   cntByteSize+=logEntry.size();
   *pOutStream << logEntry;



}