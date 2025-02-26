

#include <iomanip>

#include <memory>


#include <dirent.h>
#include <cstddef>
#include <vector>


#include <xtensor.hpp>
#include <xtensor/xnpy.hpp>

#include <fstream>
#include <cmath> //isinfinite
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>

#include <dirent.h>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <uuid/uuid.h>

#include "object_detector.h"
#include "../..//utils/subUtils.hpp"





#define NMS_THRESH 0.45



namespace PaddleDetection {


static const int INPUT_W = 640;
static const int INPUT_H = 640;




static const int CHANNELS = 3;

std::vector<int> input_image_shape={640, 640};


//---------------------------------------------------------------------
struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static inline void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static inline void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static inline void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}



///////////////////////////////////////////////////////////////////////////

static const char* INPUT_BLOB_NAME1 = "image";       //float32[-1,3,640,640]
static const char* INPUT_BLOB_NAME2 = "scale_factor";//float32[-1,2]


static const char* OUTPUT_BLOB_NAME1 = "tmp_84";            //float32[-1,8400,4]
static const char* OUTPUT_BLOB_NAME2 = "concat_14.tmp_0";   //float32[-1,NUM_CLASSES,8400]



void doInference(nvinfer1::IExecutionContext& context,int gpuNum, std::mutex& mutexlock,  float* input1,  float* input2, 
                    int camID,
                    float* output1, float* output2, 
                    const int output_size_boxes,const int  output_size_scores)
{
    cudaSetDevice(gpuNum);

    const nvinfer1::ICudaEngine& engine= context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    ANNIWOCHECK(engine.getNbBindings() == 4);
    void* buffers[4];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex1 = engine.getBindingIndex(INPUT_BLOB_NAME1);
    const int inputIndex2 = engine.getBindingIndex(INPUT_BLOB_NAME2);


    ANNIWOCHECK(engine.getBindingDataType(inputIndex1) == nvinfer1::DataType::kFLOAT);
    ANNIWOCHECK(engine.getBindingDataType(inputIndex2) == nvinfer1::DataType::kFLOAT);


    const int outputIndex1 = engine.getBindingIndex(OUTPUT_BLOB_NAME1); //boxes
    const int outputIndex2 = engine.getBindingIndex(OUTPUT_BLOB_NAME2); //scores
    ANNIWOCHECK(engine.getBindingDataType(outputIndex1) == nvinfer1::DataType::kFLOAT);
    ANNIWOCHECK(engine.getBindingDataType(outputIndex2) == nvinfer1::DataType::kFLOAT);



    

    // Create GPU buffers on device

    //  "image";       //float32[-1,3,640,640]
    CHECK(cudaMalloc(&buffers[inputIndex1], 3 * 640 * 640 * sizeof(float)));
    // "scale_factor";//float32[-1,2]
    CHECK(cudaMalloc(&buffers[inputIndex2], 2 * sizeof(float)));


    // "tmp_84";            //float32[-1,8400,4]
    ANNIWOCHECK(output_size_boxes==8400*4);
    CHECK(cudaMalloc(&buffers[outputIndex1], output_size_boxes*sizeof(float)));
    // "concat_14.tmp_0";   //float32[-1,NUM_CLASSES,8400]
    CHECK(cudaMalloc(&buffers[outputIndex2], output_size_scores*sizeof(float)));
    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));



    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex1], input1, 3*640*640*sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(buffers[inputIndex2], input2, 2*sizeof(float), cudaMemcpyHostToDevice, stream));

    std::unique_lock<std::mutex> gpuQueueLock(mutexlock, std::defer_lock);
    ANNIWOLOG(INFO)  << "inference: Wait lock...camID:"<<camID ;
    gpuQueueLock.lock();

    context.enqueueV2((void**)buffers, stream, nullptr);
    

    CHECK(cudaMemcpyAsync(output1, buffers[outputIndex1], output_size_boxes*sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(output2, buffers[outputIndex2], output_size_scores*sizeof(float), cudaMemcpyDeviceToHost, stream));

    
    cudaStreamSynchronize(stream);

    gpuQueueLock.unlock();
    ANNIWOLOG(INFO)  << "inference: Out lock...camID:"<<camID ;

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex1]));
    CHECK(cudaFree(buffers[inputIndex2]));
    CHECK(cudaFree(buffers[outputIndex1]));
    CHECK(cudaFree(buffers[outputIndex2]));
}


//////////////////////////////////////////////////////////////////////////

ObjectDetector::~ObjectDetector()
{

  // destroy the engine
    for(int i=0;i<executionContexts.size();i++)
    {
        executionContexts[i]->destroy();
    }

    engine->destroy();
    runtime->destroy();
}

// Load Model and create model predictor
void ObjectDetector::LoadModel(const std::string& model_filename)
{

    gpuNum = initInferContext(
                    model_filename.c_str(), 
                    &runtime,
                    &engine);
      
    for(int i=0;i<m_contextCnt;i++)
		{
			TrtSampleUniquePtr<nvinfer1::IExecutionContext>  context4thisCam(engine->createExecutionContext());
			std::pair<int, TrtSampleUniquePtr<nvinfer1::IExecutionContext> > tmpitem{i,std::move(context4thisCam)};

			executionContexts.insert(std::move(tmpitem));


			std::unique_ptr<std::mutex>    newmutexptr(new std::mutex);
			std::pair<int, std::unique_ptr<std::mutex> > tmplockitem{i,std::move(newmutexptr)};

			contextlocks.insert(std::move(tmplockitem));
		}

}

// Visualiztion MaskDetector results
cv::Mat VisualizeResult(
    const cv::Mat img,
    const std::vector<PaddleDetection::ObjectResult>& results,
    const std::vector<std::string>& lables,
    const std::vector<int>& colormap,
    const bool is_rbox = false) {
  cv::Mat vis_img = img.clone();
  int img_h = vis_img.rows;
  int img_w = vis_img.cols;
  for (int i = 0; i < results.size(); ++i) {
    // Configure color and text size
    std::ostringstream oss;
    oss << std::setiosflags(std::ios::fixed) << std::setprecision(4);
    oss << lables[results[i].class_id] << " ";
    oss << results[i].confidence;
    std::string text = oss.str();
    int c1 = colormap[3 * results[i].class_id + 0];
    int c2 = colormap[3 * results[i].class_id + 1];
    int c3 = colormap[3 * results[i].class_id + 2];
    cv::Scalar roi_color = cv::Scalar(c1, c2, c3);
    int font_face = cv::FONT_HERSHEY_COMPLEX_SMALL;
    double font_scale = 0.5f;
    float thickness = 0.5;
    cv::Size text_size =
        cv::getTextSize(text, font_face, font_scale, thickness, nullptr);
    cv::Point origin;

    if (is_rbox) {
      // Draw object, text, and background
      for (int k = 0; k < 4; k++) {
        cv::Point pt1 = cv::Point(results[i].rect[(k * 2) % 8],
                                  results[i].rect[(k * 2 + 1) % 8]);
        cv::Point pt2 = cv::Point(results[i].rect[(k * 2 + 2) % 8],
                                  results[i].rect[(k * 2 + 3) % 8]);
        cv::line(vis_img, pt1, pt2, roi_color, 2);
      }
    } else {
      int w = results[i].rect[2] - results[i].rect[0];
      int h = results[i].rect[3] - results[i].rect[1];
      cv::Rect roi = cv::Rect(results[i].rect[0], results[i].rect[1], w, h);
      // Draw roi object, text, and background
      cv::rectangle(vis_img, roi, roi_color, 2);

      // Draw mask
      std::vector<int> mask_v = results[i].mask;
      if (mask_v.size() > 0) {
        cv::Mat mask = cv::Mat(img_h, img_w, CV_32S);
        std::memcpy(mask.data, mask_v.data(), mask_v.size() * sizeof(int));

        cv::Mat colored_img = vis_img.clone();

        std::vector<cv::Mat> contours;
        cv::Mat hierarchy;
        mask.convertTo(mask, CV_8U);
        cv::findContours(
            mask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(colored_img,
                         contours,
                         -1,
                         roi_color,
                         -1,
                         cv::LINE_8,
                         hierarchy,
                         100);

        cv::Mat debug_roi = vis_img;
        colored_img = 0.4 * colored_img + 0.6 * vis_img;
        colored_img.copyTo(vis_img, mask);
      }
    }

    origin.x = results[i].rect[0];
    origin.y = results[i].rect[1];

    // Configure text background
    cv::Rect text_back = cv::Rect(results[i].rect[0],
                                  results[i].rect[1] - text_size.height,
                                  text_size.width,
                                  text_size.height);
    // Draw text, and background
    cv::rectangle(vis_img, text_back, roi_color, -1);
    cv::putText(vis_img,
                text,
                origin,
                font_face,
                font_scale,
                cv::Scalar(255, 255, 255),
                thickness);
  }
  return vis_img;
}

void ObjectDetector::Preprocess(const cv::Mat ori_im) {
  // Clone the image : keep the original mat for postprocess
  cv::Mat im = ori_im.clone();
  cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
  preprocessor_.Run(&im, &inputs_);
}

void ObjectDetector::Postprocess(
    const std::vector<cv::Mat> mats,
    std::vector<PaddleDetection::ObjectResult>* result,
    std::vector<int> bbox_num,
    std::vector<float> output_data_,
    std::vector<int> output_mask_data_,
    bool is_rbox = false) {
  result->clear();
  int start_idx = 0;
  int total_num = std::accumulate(bbox_num.begin(), bbox_num.end(), 0);
  int out_mask_dim = -1;
  if (config_.mask_) {
    out_mask_dim = output_mask_data_.size() / total_num;
  }

  for (int im_id = 0; im_id < mats.size(); im_id++) {
    cv::Mat raw_mat = mats[im_id];
    int rh = 1;
    int rw = 1;
    // if (config_.arch_ == "Face") {
    //   rh = raw_mat.rows;
    //   rw = raw_mat.cols;
    // }
    for (int j = start_idx; j < start_idx + bbox_num[im_id]; j++) {
      if (is_rbox) {
        // // Class id
        // int class_id = static_cast<int>(round(output_data_[0 + j * 10]));
        // // Confidence score
        // float score = output_data_[1 + j * 10];
        // int x1 = (output_data_[2 + j * 10] * rw);
        // int y1 = (output_data_[3 + j * 10] * rh);
        // int x2 = (output_data_[4 + j * 10] * rw);
        // int y2 = (output_data_[5 + j * 10] * rh);
        // int x3 = (output_data_[6 + j * 10] * rw);
        // int y3 = (output_data_[7 + j * 10] * rh);
        // int x4 = (output_data_[8 + j * 10] * rw);
        // int y4 = (output_data_[9 + j * 10] * rh);

        // PaddleDetection::ObjectResult result_item;
        // result_item.rect = {x1, y1, x2, y2, x3, y3, x4, y4};
        // result_item.class_id = class_id;
        // result_item.confidence = score;
        // result->push_back(result_item);
      } else {
        // Class id
        int class_id = static_cast<int>(round(output_data_[0 + j * 6]));
        // Confidence score
        float score = output_data_[1 + j * 6];
        int xmin = (output_data_[2 + j * 6] * rw);
        int ymin = (output_data_[3 + j * 6] * rh);
        int xmax = (output_data_[4 + j * 6] * rw);
        int ymax = (output_data_[5 + j * 6] * rh);
        int wd = xmax - xmin;
        int hd = ymax - ymin;

        PaddleDetection::ObjectResult result_item;
        result_item.rect = {xmin, ymin, xmax, ymax};
        result_item.class_id = class_id;
        result_item.confidence = score;

        if (config_.mask_) {
          std::vector<int> mask;
          for (int k = 0; k < out_mask_dim; ++k) {
            if (output_mask_data_[k + j * out_mask_dim] > -1) {
              mask.push_back(output_mask_data_[k + j * out_mask_dim]);
            }
          }
          result_item.mask = mask;
        }

        result->push_back(result_item);
      }
    }
    start_idx += bbox_num[im_id];
  }
}

void ObjectDetector::Predict(const std::vector<cv::Mat> imgs,
    const double threshold,
    const int numClass,
    std::vector<PaddleDetection::ObjectResult>* result,
    std::vector<int>* bbox_num,
    std::vector<double>* times,
    int instanceID,
    int camID) {

    ANNIWOLOG(INFO) << "ObjectDetector::Predict entered" << "camID:" << camID;
    double BBOX_CONF_THRESH = threshold;
    int NUM_CLASSES = numClass;
    const int warmup = 0;
    const int repeats = 1;

    auto preprocess_start = std::chrono::steady_clock::now();
    int batch_size = imgs.size();

    // in_data_batch
    std::vector<float> in_data_all;
    std::vector<float> im_shape_all(batch_size * 2);
    std::vector<float> scale_factor_all(batch_size * 2);
    std::vector<const float*> output_data_list_;
    std::vector<int> out_bbox_num_data_;
    std::vector<int> out_mask_data_;

    // in_net img for each batch
    std::vector<cv::Mat> in_net_img_all(batch_size);

    // Preprocess image
    for (int bs_idx = 0; bs_idx < batch_size; bs_idx++) {
        cv::Mat im = imgs.at(bs_idx);
        ANNIWOLOG(INFO) << "ObjectDetector::Predict before Preprocess" << "context size:" << executionContexts.size() << ",camID:" << camID;

        Preprocess(im);
        ANNIWOLOG(INFO) << "ObjectDetector::Predict after Preprocess" << "context size:" << executionContexts.size() << ",camID:" << camID;

        im_shape_all[bs_idx * 2] = inputs_.im_shape_[0];
        im_shape_all[bs_idx * 2 + 1] = inputs_.im_shape_[1];

        scale_factor_all[bs_idx * 2] = inputs_.scale_factor_[0];
        scale_factor_all[bs_idx * 2 + 1] = inputs_.scale_factor_[1];

        // TODO: reduce cost time
        in_data_all.insert(
            in_data_all.end(), inputs_.im_data_.begin(), inputs_.im_data_.end());

        // collect in_net img
        in_net_img_all[bs_idx] = inputs_.in_net_im_;
    }

    // Pad Batch if batch size > 1 xb deleted!!!


    auto preprocess_end = std::chrono::steady_clock::now();


    // Run predictor
    std::vector<std::vector<float>> out_tensor_list;
    std::vector<std::vector<int>> output_shape_list;
    bool is_rbox = false;
    int reg_max = 7;

    // warmup xb deleted

    int output_size_boxes = 1 * 8400 * 4;
    int output_size_scores = 1 * NUM_CLASSES * 8400;


    std::vector<float> out_data_boxes(output_size_boxes, 1.0);
    std::vector<float> out_data_scores(output_size_scores, 1.0);

    auto inference_start = std::chrono::steady_clock::now();

    ANNIWOLOG(INFO) << "ObjectDetector::Predict before cudaSetDevice" << "context size:" << executionContexts.size() << ",camID:" << camID;

    cudaSetDevice(gpuNum);
    ANNIWOLOG(INFO) << "ObjectDetector::Predict after cudaSetDevice" << "context size:" << executionContexts.size() << ",camID:" << camID;


    int choiceIntVal = randIntWithinScale(m_contextCnt);
    std::unordered_map<int, TrtSampleUniquePtr<nvinfer1::IExecutionContext> >::iterator iterCamInstance = executionContexts.find(choiceIntVal);
    std::unordered_map<int, std::unique_ptr<std::mutex> >::iterator iterCamInstancelock = contextlocks.find(choiceIntVal);

    ANNIWOLOG(INFO) << "ObjectDetector::Predict before assert" << "context size:" << executionContexts.size() << ",choiceIntVal:" << choiceIntVal << ",camID:" << camID;

    ANNIWOCHECK(iterCamInstance != executionContexts.end());

    ANNIWOLOG(INFO) << "ObjectDetector::Predict before doInference" << "camID:" << camID;


    doInference(*iterCamInstance->second, gpuNum, *iterCamInstancelock->second, in_data_all.data(), scale_factor_all.data(), camID, out_data_boxes.data(), out_data_scores.data(), output_size_boxes, output_size_scores);

    ANNIWOLOG(INFO) << "ObjectDetector::Predict after doInference" << "camID:" << camID;


    //此处做nms,输出到out_tensor_list
  // ANNIWOLOG(INFO) << "out_data_boxes num is " << out_data_boxes.size() <<"camID:"<<camID ;

  //////////////////////////////////////////////
    // std::cout<<" out_data_boxes:";
    // int printCnt=0;
    // for(auto item:out_data_boxes)
    // {
    //     printCnt++;
    //     std::cout<<item<<",";   
    //     if(printCnt > 24)
    //     {
    //         break;
    //     }

    // }
    // std::cout<<std::endl;

    // std::cout<< "output_size_scores num is " << out_data_scores.size();

    // std::cout<<" out_data_scores:";
    // printCnt=0;
    // for(auto item:out_data_scores)
    // {
    //     printCnt++;
    //     std::cout<<item<<",";   
    //     if(printCnt > 24)
    //     {
    //         break;
    //     }

    // }
    // std::cout<<std::endl;

    /////////////////////////////////////////
    std::vector<std::size_t> tmpshape = { 8400,4 };//x1,y1,x2,y2
    auto boxes = xt::adapt(out_data_boxes, tmpshape);


    std::vector<std::size_t> tmpshape2 = { NUM_CLASSES, 8400 };
    auto scores_probs = xt::adapt(out_data_scores, tmpshape2);

    //python: classes = np.argmax(box_scores, axis=-1)
  //   xt::xtensor<int,8400> classes=xt::argmax(scores_probs, 0);
    auto classes = xt::argmax(scores_probs, 0);
    // std::cout << "classes shape:" <<xt::adapt(classes.shape()) <<std::endl;  

    //python: scores = np.max(box_scores, axis=-1)
  //   xt::xtensor<float,8400> scores = xt::amax(scores_probs,0);
    auto scores = xt::amax(scores_probs, 0);
    // std::cout << "scores shape:" <<xt::adapt(scores.shape()) <<std::endl;  
    // std::cout << "scores:" <<scores <<std::endl;  


    std::vector<Object> proposals;
    for (size_t i = 0; i < classes.size(); i++)
    {
        Object obj;
        obj.rect.x = boxes(i, 0);
        obj.rect.y = boxes(i, 1);
        obj.rect.width = boxes(i, 2) - obj.rect.x;   //x2-x1
        obj.rect.height = boxes(i, 3) - obj.rect.y;  //y2-y1
        obj.label = classes(i);
        obj.prob = scores(i);

        // if(i < 50)
        // {
        //     std::cout << " score:"  << obj.prob
        //             <<" label:"  << obj.label
        //             <<" x:"  << obj.rect.x 
        //             <<" y:"  << obj.rect.y
        //             <<" width:"  << obj.rect.width
        //             <<" height:"  << obj.rect.height 
        //             <<std::endl; 
        // }


        if (obj.rect.width < 1 || obj.rect.height < 1)
        {
            // std::cout << "this box ignored for h,w < 1"<< std::endl;
            continue;
        }
        if (obj.rect.x < 0 || obj.rect.y < 0)
        {
            // std::cout << "this box ignored for - x,y"<< std::endl;
            continue;
        }
        if (obj.prob < BBOX_CONF_THRESH)
        {
            // std::cout << "this box ignored for score"<< std::endl;
            continue;
        }

        proposals.push_back(obj);
    }


    // std::cout << "num of boxes before nms: " << proposals.size() << std::endl;

    qsort_descent_inplace(proposals);

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, NMS_THRESH);


    int count = picked.size();

    // std::cout << "num of boxes after nms: " << count << std::endl;
    ANNIWOLOG(INFO) << "ObjectDetector::Predict num of boxes after nms:" << count << "camID:" << camID;

    //picked convert to -1,6 -> plain vector
    std::vector<float> out_data(6 * count);

    for (size_t i = 0; i < picked.size(); i++)
    {
        out_data[i * 6 + 0] = proposals[picked[i]].label; //Class id
        out_data[i * 6 + 1] = proposals[picked[i]].prob; //score
        out_data[i * 6 + 2] = proposals[picked[i]].rect.x; //x1
        out_data[i * 6 + 3] = proposals[picked[i]].rect.y; //y1
        out_data[i * 6 + 4] = proposals[picked[i]].rect.x + proposals[picked[i]].rect.width; //x2
        out_data[i * 6 + 5] = proposals[picked[i]].rect.y + proposals[picked[i]].rect.height; //y2

    }


    ///////////////////////////////////////////////////////////
    // Get output tensor
    out_tensor_list.clear();
    out_tensor_list.push_back(out_data); //out_tensor_list只有一个值
    out_bbox_num_data_.clear();
    out_bbox_num_data_.push_back(count);
    // }
    auto inference_end = std::chrono::steady_clock::now();
    auto postprocess_start = std::chrono::steady_clock::now();
    // Postprocessing result
    result->clear();
    bbox_num->clear();
    if (config_.arch_ == "PicoDet") {
    }
    else {
        // is_rbox = output_shape_list[0][output_shape_list[0].size() - 1] % 10 == 0;
        is_rbox = false;
        Postprocess(imgs,
            result,
            out_bbox_num_data_,
            out_tensor_list[0],
            out_mask_data_,
            is_rbox);
        for (int k = 0; k < out_bbox_num_data_.size(); k++) {
            int tmp = out_bbox_num_data_[k];
            bbox_num->push_back(tmp);
        }
    }

    auto postprocess_end = std::chrono::steady_clock::now();

    std::chrono::duration<float> preprocess_diff =
        preprocess_end - preprocess_start;
    times->push_back(static_cast<double>(preprocess_diff.count() * 1000));
    std::chrono::duration<float> inference_diff = inference_end - inference_start;
    times->push_back(
        static_cast<double>(inference_diff.count() / repeats * 1000));
    std::chrono::duration<float> postprocess_diff =
        postprocess_end - postprocess_start;
    times->push_back(static_cast<double>(postprocess_diff.count() * 1000));
}

std::vector<int> GenerateColorMap(int num_class) {
  auto colormap = std::vector<int>(3 * num_class, 0);
  for (int i = 0; i < num_class; ++i) {
    int j = 0;
    int lab = i;
    while (lab) {
      colormap[i * 3] |= (((lab >> 0) & 1) << (7 - j));
      colormap[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j));
      colormap[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j));
      ++j;
      lab >>= 3;
    }
  }
  return colormap;
}

}  // namespace PaddleDetection


void ppyoloe_det_stuff(PaddleDetection::ObjectDetector* m_detptr, int camID, int instanceID, cv::Mat img, /*out*/std::vector<Object>& objects, const double threshold, const int numClass, std::string logstr)
{
    std::vector<cv::Mat> batch_imgs;
    batch_imgs.insert(batch_imgs.end(), img);


    // Store all detected result
    std::vector<PaddleDetection::ObjectResult> result;
    std::vector<int> bbox_num;
    std::vector<double> det_times;

    bool is_rbox = false;

    // run inference
    auto start = std::chrono::system_clock::now();


    m_detptr->Predict(batch_imgs, threshold, numClass, &result, &bbox_num, &det_times, instanceID, camID);

    auto end = std::chrono::system_clock::now();
    ANNIWOLOG(INFO) << logstr << ":infer time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms," << "camID:" << camID;
    ANNIWOLOG(INFO) << logstr << ":det_times:" << det_times[0] << " bbox_num:" << bbox_num[0];

    if (bbox_num[0] <= 0)
    {
        ANNIWOLOG(INFO) << logstr << ":det_stuff no object.";
        // goto detectEndflag;
        ANNIWOLOG(INFO) << logstr << ":exit det_stuff()" << " camID:" << camID;
        return;
    }
    else
    {
        // if(bbox_num[0] >= 50)
        // {
        //   ANNIWOLOG(INFO) <<logstr<<":unexpected number of box!!" ;
        //   ANNIWOCHECK(false);
        // }
    }

    //将输出的PaddleDetection::ObjectResult转成Object
    objects.clear();
    for (int j = 0; j < bbox_num[0]; j++)
    {
        PaddleDetection::ObjectResult item = result[j];


        if (!std::isfinite(item.confidence))
        {
            ANNIWOLOG(INFO) << logstr << ":get unexpected infinite value,ignored!!";
            continue;
        }



        if (item.confidence < threshold || item.class_id == -1)
        {
            continue;
        }

        ANNIWOLOG(INFO) << logstr << ":confidence:" << item.confidence << ",id:" << item.class_id;


        Object obj;
        obj.trackID = -1;
        obj.label = item.class_id;
        obj.prob = item.confidence;
        obj.rect.x = item.rect[0]; //x1
        obj.rect.y = item.rect[1]; //y1
        obj.rect.width = item.rect[2] - item.rect[0]; //x2-x1
        obj.rect.height = item.rect[3] - item.rect[1];//y2-y1

        if (obj.rect.width <= 0 || obj.rect.height <= 0)
        {
            ANNIWOLOG(INFO) << "BasePersonDetection:obj.rect.width <= 0. Ignored";
            continue;
        }


        objects.emplace_back(obj);


    }


}
