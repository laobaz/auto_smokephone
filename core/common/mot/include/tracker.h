#ifndef TRACKER_H
#define TRACKER_H


#include <vector>

#include "kalmanfilter.h"
#include "track.h"
#include "model.hpp"

using namespace std;

class NearNeighborDisMetric;

class tracker
{
public:
    NearNeighborDisMetric* metric;
    float max_iou_distance;
    int max_age;
    int n_init;

    KalmanFilter* kf;

    int _next_idx;
public:
    std::vector<Track> tracks;
    // tracker(/*NearNeighborDisMetric* metric,*/
    // 		float max_cosine_distance, int nn_budget,
    //         float max_iou_distance = 0.7,
    //         int max_age = 70, int n_init=3);

    tracker(/*NearNeighborDisMetric* metric,*/
    		float max_cosine_distance, int nn_budget,
            // float max_iou_distance = 0.7,
            float max_iou_distance = 1.0,  //
            // int max_age = 1800, int n_init=2);
            int max_age = 70, int n_init=3);

    void predict();
    void update(const DETECTIONS& detections,int camID);
    void update(const DETECTIONSV2& detectionsv2,int camID);
    typedef DYNAMICM (tracker::* GATED_METRIC_FUNC)(
            std::vector<Track>& tracks,
            const DETECTIONS& dets,
            const std::vector<int>& track_indices,
            const std::vector<int>& detection_indices);
private:    
    void _match(const DETECTIONS& detections, TRACHER_MATCHD& res,int camID);
    void _initiate_track(const DETECTION_ROW& detection);
    void _initiate_track(const DETECTION_ROW& detection, CLSCONF clsConf);
public:
    DYNAMICM gated_matric(
            std::vector<Track>& tracks,
            const DETECTIONS& dets,
            const std::vector<int>& track_indices,
            const std::vector<int>& detection_indices);
    DYNAMICM iou_cost(
            std::vector<Track>& tracks,
            const DETECTIONS& dets,
            const std::vector<int>& track_indices,
            const std::vector<int>& detection_indices);
    Eigen::VectorXf iou(DETECTBOX& bbox,
            DETECTBOXSS &candidates);
};

#endif // TRACKER_H
