#ifndef DEEPSORT_H
#define DEEPSORT_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "featuretensor.h"
#include "tracker.h"
#include "datatype.h"
#include <vector>
#include <unordered_set>
#include <unordered_map>




class DeepSort {
public:    
    DeepSort(const std::string& modelpath,const std::string* modelpath2, int batchSize, const std::vector<int>& camIDs,int instanceCnt);
    ~DeepSort();

public:
    void sort(cv::Mat frame, vector<DetectBox>& dets,int camID,int instanceID);

private:
    void sort_priv(cv::Mat frame, DETECTIONSV2& detectionsv2,int camID,int instanceID);    


private:
    int batchSize;
    cv::Size imgShape;
    int maxBudget;
    float maxCosineDist;

private:
    
    std::unordered_map<int, std::vector<RESULT_DATA> > m_result;
    std::unordered_map<int, std::vector<std::pair<CLSCONF, DETECTBOX>> > m_results;

    std::unordered_map<int, tracker* > objTrackers;
    FeatureTensor* featureExtractor;

};

#endif  //deepsort.h
