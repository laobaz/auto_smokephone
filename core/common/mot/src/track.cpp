#include "track.h"
#include <iostream>

Track::Track(KAL_MEAN & mean, KAL_COVA & covariance, int track_id, int n_init, int max_age, const FEATURE & feature)
{
    this->mean = mean;
    this->covariance = covariance;
    this->track_id = track_id;
    this->hits = 1;
    this->age = 1;
    this->time_since_update = 0;
    this->state = TrackState::Tentative;
    features = FEATURESS(1, 256);
    features.row(0) = feature;  //features.rows() must = 0;

    this->_n_init = n_init;
    this->_max_age = max_age;
}

Track::Track(KAL_MEAN & mean, KAL_COVA & covariance, int track_id, int n_init, int max_age, const FEATURE & feature, int cls, float conf,const uuid_t& uuid)
{
    this->mean = mean;
    this->covariance = covariance;
    this->track_id = track_id;
    this->hits = 1;
    this->age = 1;
    this->time_since_update = 0;
    this->state = TrackState::Tentative;
    features = FEATURESS(1, 256);
    features.row(0) = feature;  //features.rows() must = 0;

    this->_n_init = n_init;
    this->_max_age = max_age;

    this->cls = cls;
    this->conf = conf;
    // this->uuid = uuid;
    strncpy((char*)this->uuid, (char*)uuid, sizeof(uuid_t));
    
}

void Track::predit(KalmanFilter * kf)
{
    /*Propagate the state distribution to the current time step using a
       Kalman filter prediction step.

       Parameters
       ----------
       kf : kalman_filter.KalmanFilterd
       The Kalman filter.
     */

    kf->predict(this->mean, this->covariance);


    this->age += 1;
    this->time_since_update += 1;
}

void Track::update(KalmanFilter * const kf, const DETECTION_ROW & detection)
{
    // this->uuid = detection.uuid;
    strncpy((char*)this->uuid, (char*)detection.uuid, sizeof(uuid_t));


    KAL_DATA pa = kf->update(this->mean, this->covariance, detection.to_xyah());
    this->mean = pa.first;
    this->covariance = pa.second;

    featuresAppendOne(detection.feature);
    //    this->features.row(features.rows()) = detection.feature;
    this->hits += 1;
    this->time_since_update = 0;
    if (this->state == TrackState::Tentative && this->hits >= this->_n_init) {
        this->state = TrackState::Confirmed;
    }
}

void Track::update(KalmanFilter * const kf, const DETECTION_ROW & detection, CLSCONF pair_det)
{

    KAL_DATA pa = kf->update(this->mean, this->covariance, detection.to_xyah());
    this->mean = pa.first;
    this->covariance = pa.second;

    featuresAppendOne(detection.feature);
    //    this->features.row(features.rows()) = detection.feature;
    this->hits += 1;
    this->time_since_update = 0;
    if (this->state == TrackState::Tentative && this->hits >= this->_n_init) {
        this->state = TrackState::Confirmed;
    }
    this->cls = pair_det.cls;
    this->conf = pair_det.conf;
    // this->uuid = detection.uuid;
    strncpy((char*)this->uuid, (char*)detection.uuid, sizeof(uuid_t));


}

void Track::mark_missed()
{
    if (this->state == TrackState::Tentative) {
        this->state = TrackState::Deleted;
    } else if (this->time_since_update > this->_max_age) {
        this->state = TrackState::Deleted;
    }
}

bool Track::is_confirmed()
{
    return this->state == TrackState::Confirmed;
}

bool Track::is_deleted()
{
    return this->state == TrackState::Deleted;
}

bool Track::is_tentative()
{
    return this->state == TrackState::Tentative;
}

DETECTBOX Track::to_tlwh()
{
    DETECTBOX ret = mean.leftCols(4);
    ret(2) *= ret(3);
    ret.leftCols(2) -= (ret.rightCols(2) / 2);
    return ret;
}

void Track::featuresAppendOne(const FEATURE & f)
{
    int size = this->features.rows();
    FEATURESS newfeatures = FEATURESS(size + 1, 256);
    newfeatures.block(0, 0, size, 256) = this->features;
    newfeatures.row(size) = f;
    features = newfeatures;
}
