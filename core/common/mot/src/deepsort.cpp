#include "deepsort.h"
#include "../../../utils/subUtils.hpp"

DeepSort::DeepSort(const std::string& modelpath,const std::string* modelpath2, int batchSize,   const std::vector<int>& camIDs, int instanceCnt) 
{

    this->batchSize = batchSize; //8

    //width=64,height=128
    this->imgShape = cv::Size(64, 128);
    this->maxBudget = 100;
    this->maxCosineDist = 0.2;
    featureExtractor=nullptr;
    objTrackers.clear();


    ANNIWOCHECK(featureExtractor==nullptr);
    ANNIWOCHECK(objTrackers.size()==0);

    //一个DeepSort对象管理多个摄像头的跟踪。一个tracker有多个检测结果目标
    for(int camID: camIDs)
    {
        ANNIWOLOG(INFO) << "DS:: insert tracker for camID" <<camID;

        tracker* objTracker = new tracker(maxCosineDist, maxBudget);
        objTrackers.insert(std::pair<int, tracker*>(camID,objTracker) );
    }


    int cntID=0;
    //只生成ANNIWO_NUM_INSTANCE_PERSONBASE个实例
    while(cntID < instanceCnt)
    {
        ANNIWOLOG(INFO) << "DS:: insert result instance" <<"cntID:"<<cntID<<" ";

        // std::unordered_map<int, vector<RESULT_DATA> > m_result;
        std::pair<int,std::vector<RESULT_DATA> > tmpitem;
        m_result.insert(std::move(tmpitem));

        // std::unordered_map<int, vector<std::pair<CLSCONF, DETECTBOX>> > m_results;
        std::pair<int,std::vector<std::pair<CLSCONF, DETECTBOX>> > tmpme;
        m_results.insert(std::move(tmpme));

        cntID++;
    }



    //一个FeatureTensor对应多个摄像头的特征提取。
    featureExtractor = new FeatureTensor(batchSize, imgShape,camIDs);
    featureExtractor->loadEngine(modelpath,modelpath2);

    ANNIWOLOG(INFO) <<"DeepSort:batchSize:"<<batchSize<<" featureDim:512,256->256";
}





DeepSort::~DeepSort() {
    for (auto iter = objTrackers.begin(); iter != objTrackers.end(); ++iter) {
        int camID= iter->first ;
        tracker* ptr=iter->second ;
        if(ptr)
        {
            delete ptr;
            ptr=nullptr;
        }
    }
    if(featureExtractor)
    {
        delete featureExtractor;
        featureExtractor=nullptr;
    }


}

//将被多线程调用！
void DeepSort::sort(cv::Mat frame, vector<DetectBox>& dets,int camID,int instanceID) {
    // preprocess Mat -> DETECTION
    DETECTIONS detections;
    vector<CLSCONF> clsConf;
    
    for (DetectBox i : dets) {
        DETECTBOX box(i.x1, i.y1, i.x2-i.x1, i.y2-i.y1);
        DETECTION_ROW d;
        d.tlwh = box;
        d.confidence = i.confidence;
        // d.uuid = i.uuid;
        strncpy((char*)d.uuid, (char*)i.uuid, sizeof(uuid_t));
        d.classID = i.classID;
        detections.push_back(d);
        clsConf.push_back(CLSCONF((int)i.classID, i.confidence,i.uuid));
    }
    m_result[instanceID].clear();
    m_results[instanceID].clear();
    if (detections.size() > 0) {
        DETECTIONSV2 detectionsv2 = make_pair(clsConf, detections);
        sort_priv(frame, detectionsv2,camID,instanceID);
    }

    #ifdef ANNIWO_INTERNAL_DEBUG
    ANNIWOLOG(INFO) << "sort_priv returned !"<<"camID:"<<camID<<"instanceID:"<<instanceID ;
    #endif

    // postprocess DETECTION -> Mat
    dets.clear();
    for (auto r : m_result[instanceID]) {
        DETECTBOX i = r.second;
        DetectBox b(i(0), i(1), i(2)+i(0), i(3)+i(1), 1.);
        b.trackID = (float)r.first;

        dets.push_back(b);
    }
    for (int i = 0; i < m_results[instanceID].size(); ++i) {
        CLSCONF c = m_results[instanceID][i].first;
        dets[i].classID = c.cls;
        dets[i].confidence = c.conf;
        // dets[i].uuid = c.uuid;
        strncpy((char*)dets[i].uuid, (char*)c.uuid, sizeof(uuid_t));

    }

    #ifdef ANNIWO_INTERNAL_DEBUG
    ANNIWOLOG(INFO) << "DS::sort returned !"<<"camID:"<<camID<<"instanceID:"<<instanceID ;
    #endif

}




//将被多线程调用！
void DeepSort::sort_priv(cv::Mat frame, DETECTIONSV2& detectionsv2,int camID,int instanceID) {
    std::vector<CLSCONF>& clsConf = detectionsv2.first;
    DETECTIONS& detections = detectionsv2.second;
    bool flag = featureExtractor->getRectsFeature(frame, detections,camID,instanceID);
    if (flag) {
        objTrackers[camID]->predict();

    #ifdef ANNIWO_INTERNAL_DEBUG
        ANNIWOLOG(INFO) << "sort_priv predict Succeed !detections.size"<<detections.size()<<"camID:"<<camID<<"instanceID:"<<instanceID ;
    #endif

        objTrackers[camID]->update(detectionsv2,camID);  //只有一个车的目标的时候,偶尔会卡在这个函数！！！

    #ifdef ANNIWO_INTERNAL_DEBUG
        ANNIWOLOG(INFO) << "sort_priv update Succeed !objTrackers[camID]->tracks.size:"<<objTrackers[camID]->tracks.size()<<"camID:"<<camID<<"instanceID:"<<instanceID ;
    #endif

        
        m_result[instanceID].clear();
        m_results[instanceID].clear();
        for (Track& track : objTrackers[camID]->tracks) {
            if (!track.is_confirmed() || track.time_since_update > 1)
                continue;
            m_result[instanceID].push_back(make_pair(track.track_id, track.to_tlwh()));
            m_results[instanceID].push_back(make_pair(CLSCONF(track.cls, track.conf,track.uuid) ,track.to_tlwh()));
        }
    }

    #ifdef ANNIWO_INTERNAL_DEBUG
    ANNIWOLOG(INFO) << "sort_priv returned ! m_results[instanceID].size():"<<m_results[instanceID].size()<<"camID:"<<camID<<"instanceID:"<<instanceID ;
    #endif

}

