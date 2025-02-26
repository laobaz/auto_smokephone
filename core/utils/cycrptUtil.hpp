#ifndef __CYCRPT_HTTPUTIL_HPP__
#define __CYCRPT_HTTPUTIL_HPP__


#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>

#include <cryptopp/aes.h>
#include <cryptopp/base64.h>
#include <cryptopp/modes.h>    // CFB_Mode  

#include <cryptopp/hex.h>      // StreamTransformationFilter  
#include <sstream>   // std::stringstream    
#include <string>  
#include <stddef.h>

std::string getMachineID(const char* rawcmd)
{
        FILE* fp = NULL;
        char cmd[512];
        memset(cmd,0x00,512);

        sprintf(cmd, rawcmd);
        if ((fp = popen(cmd, "r")) != NULL)
        {
            memset(cmd,0x00,512);
            fgets(cmd, sizeof(cmd), fp);
            pclose(fp);
        }else
        {
            memset(cmd,0x00,512);
        }


        return std::string(cmd);

}


std::string CBC_AESDecryptStr(std::string sKey, std::string sIV, const char *cipherText)
{
    std::string outstr;

    //填key    
    CryptoPP::SecByteBlock key(CryptoPP::AES::MAX_KEYLENGTH);
    memset(key, 0x30, key.size());
    sKey.size() <= CryptoPP::AES::MAX_KEYLENGTH ? memcpy(key, sKey.c_str(), sKey.size()) : memcpy(key, sKey.c_str(), CryptoPP::AES::MAX_KEYLENGTH);

    //填iv    
    byte iv[CryptoPP::AES::BLOCKSIZE];
    memset(iv, 0x30, CryptoPP::AES::BLOCKSIZE);
    sIV.size() <= CryptoPP::AES::BLOCKSIZE ? memcpy(iv, sIV.c_str(), sIV.size()) : memcpy(iv, sIV.c_str(), CryptoPP::AES::BLOCKSIZE);


    CryptoPP::CBC_Mode<CryptoPP::AES >::Decryption cbcDecryption((byte*)key, CryptoPP::AES::MAX_KEYLENGTH, iv);

    CryptoPP::HexDecoder decryptor(new CryptoPP::StreamTransformationFilter(cbcDecryption, new CryptoPP::StringSink(outstr)));
    decryptor.Put((byte*)cipherText, strlen(cipherText));
    decryptor.MessageEnd();

    return outstr;
}




bool verifyID_CBC_AES(const std::string& plainText, const std::string& CBC_EncryptedText2)
{
    // ANNIWOLOGF(INFO,"plainText is %s\n", plainText.c_str());

    static std::string aesKey = "0123456789ABCDanniwo456789ABCDEF";//256bits, also can be 128 bits or 192bits  
    static std::string aesIV = "ABCDanniwo456789";//128 bits  


    std::string CBC_DecryptedText;

    try
    {
        //CBC  
        CBC_DecryptedText = CBC_AESDecryptStr(aesKey, aesIV, CBC_EncryptedText2.c_str());//CBC解密  
        if(plainText == CBC_DecryptedText) 
        {
            return true;
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }


    std::cout<<"Mac Failed with:"<<plainText<<" "<<CBC_DecryptedText<<std::endl;

    return false;
}
/////////////////////////////////////////////////////////////




bool VerifyKey(const std::string& CBC_EncryptedText2)
{
#ifdef __aarch64__
   std::string cpuid = "";
   std::string serialnum = getMachineID("lshw|grep \"serial\"|tr -d [:blank:] |head -n 3|tr -d \"\\r\\n\"");

   if(serialnum.length() != 0)
   {
        std::cout<<"serialnum.length():"<<serialnum.length()<<" :"<<serialnum<<std::endl;
        return verifyID_CBC_AES(cpuid+serialnum,CBC_EncryptedText2);
   }
#else
   std::string cpuid = getMachineID("dmidecode -t 4 | grep ID|tr -d [:blank:] |head -n 1|tr -d \"\\r\\n\"|tr -d \"ID:\"");
   //motherboard serial
   std::string serialnum = getMachineID("dmidecode -t 2 | grep Serial |tr -d [:blank:] |head -n 1|tr -d \"\\r\\n\"|tr -d \"SerialNumber:\"");
   //GPU serial
   std::string serialnumGpu = getMachineID("lspci -vnn | grep VGA -A 12|head -n 1|grep -o \"\\[.........\\]\" |tr -d \"\\[\\]\\n\\r \"");

   if(serialnum.length() != 0)
   {
        std::cout<<"serialnum.length():"<<serialnum.length()<<" :"<<serialnum<<std::endl;
        return verifyID_CBC_AES(cpuid+serialnum,CBC_EncryptedText2);
   }else
   {
        std::cout<<"serialnum.length():"<<serialnum.length()<<" :"<<serialnum<<std::endl;
        return verifyID_CBC_AES(cpuid+serialnumGpu,CBC_EncryptedText2);
   }
#endif


}

#endif //
