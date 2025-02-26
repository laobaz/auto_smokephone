#include "yolo_common.hpp"

#include "../../utils/subUtils.hpp"

#include <xtensor.hpp>
#include <xtensor/xnpy.hpp>

#include <dirent.h>
#include <thread>
#include <omp.h>
#include <unordered_set>


// using namespace nvinfer1;

#define NMS_THRESH 0.45




const int INPUT_W = 608;
const int INPUT_H = 608;
const int CHANNELS = 3;
const int NUM_ANCHORS = 3;

std::vector<int> input_image_shape={608, 608};

const char* INPUT_BLOB_NAME = "input_1";
const char* OUTPUT_BLOB_NAME = "lambda_1";


const float color_list[80][3] =
{
    {0.000, 0.447, 0.741},
    {0.850, 0.325, 0.098},
    {0.929, 0.694, 0.125},
    {0.494, 0.184, 0.556},
    {0.466, 0.674, 0.188},
    {0.301, 0.745, 0.933},
    {0.635, 0.078, 0.184},
    {0.300, 0.300, 0.300},
    {0.600, 0.600, 0.600},
    {1.000, 0.000, 0.000},
    {1.000, 0.500, 0.000},
    {0.749, 0.749, 0.000},
    {0.000, 1.000, 0.000},
    {0.000, 0.000, 1.000},
    {0.667, 0.000, 1.000},
    {0.333, 0.333, 0.000},
    {0.333, 0.667, 0.000},
    {0.333, 1.000, 0.000},
    {0.667, 0.333, 0.000},
    {0.667, 0.667, 0.000},
    {0.667, 1.000, 0.000},
    {1.000, 0.333, 0.000},
    {1.000, 0.667, 0.000},
    {1.000, 1.000, 0.000},
    {0.000, 0.333, 0.500},
    {0.000, 0.667, 0.500},
    {0.000, 1.000, 0.500},
    {0.333, 0.000, 0.500},
    {0.333, 0.333, 0.500},
    {0.333, 0.667, 0.500},
    {0.333, 1.000, 0.500},
    {0.667, 0.000, 0.500},
    {0.667, 0.333, 0.500},
    {0.667, 0.667, 0.500},
    {0.667, 1.000, 0.500},
    {1.000, 0.000, 0.500},
    {1.000, 0.333, 0.500},
    {1.000, 0.667, 0.500},
    {1.000, 1.000, 0.500},
    {0.000, 0.333, 1.000},
    {0.000, 0.667, 1.000},
    {0.000, 1.000, 1.000},
    {0.333, 0.000, 1.000},
    {0.333, 0.333, 1.000},
    {0.333, 0.667, 1.000},
    {0.333, 1.000, 1.000},
    {0.667, 0.000, 1.000},
    {0.667, 0.333, 1.000},
    {0.667, 0.667, 1.000},
    {0.667, 1.000, 1.000},
    {1.000, 0.000, 1.000},
    {1.000, 0.333, 1.000},
    {1.000, 0.667, 1.000},
    {0.333, 0.000, 0.000},
    {0.500, 0.000, 0.000},
    {0.667, 0.000, 0.000},
    {0.833, 0.000, 0.000},
    {1.000, 0.000, 0.000},
    {0.000, 0.167, 0.000},
    {0.000, 0.333, 0.000},
    {0.000, 0.500, 0.000},
    {0.000, 0.667, 0.000},
    {0.000, 0.833, 0.000},
    {0.000, 1.000, 0.000},
    {0.000, 0.000, 0.167},
    {0.000, 0.000, 0.333},
    {0.000, 0.000, 0.500},
    {0.000, 0.000, 0.667},
    {0.000, 0.000, 0.833},
    {0.000, 0.000, 1.000},
    {0.000, 0.000, 0.000},
    {0.143, 0.143, 0.143},
    {0.286, 0.286, 0.286},
    {0.429, 0.429, 0.429},
    {0.571, 0.571, 0.571},
    {0.714, 0.714, 0.714},
    {0.857, 0.857, 0.857},
    {0.000, 0.447, 0.741},
    {0.314, 0.717, 0.741},
    {0.50, 0.5, 0}
};





void TrtGPUInfer(nvinfer1::IExecutionContext& context, int gpuNum, std::mutex& mutexlock, float* input, float* output,
 int output_size,int input_size,const char* inputblobname,const char* ouputblobname,std::string logstr) 
{
    //Each IExecutionContext is bound to the same GPU as the engine from which it was created. 
    //When calling execute() or enqueue(), ensure that the thread is associated with the correct device by calling cudaSetDevice().
    cudaSetDevice(gpuNum);

    const nvinfer1::ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    ANNIWOCHECK(engine.getNbBindings() == 2);

    ANNIWOCHECK(input != nullptr);
    ANNIWOCHECK(output != nullptr);
    ANNIWOCHECK(output_size > 0);
    ANNIWOCHECK(input_size > 0);


    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(inputblobname);

    ANNIWOCHECK(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    const int outputIndex = engine.getBindingIndex(ouputblobname);
    ANNIWOCHECK(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
    


    
    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], input_size*sizeof(float) ));
    CHECK(cudaMalloc(&buffers[outputIndex], output_size*sizeof(float)));

    // Create stream
    // nvinfer1::cudaStream_t stream;
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    std::unique_lock<std::mutex> gpuQueueLock(mutexlock, std::defer_lock);


    // ANNIWOLOG(INFO) <<logstr << "TrtGPUInfer: cudaMemcpyAsync cudaMemcpyHostToDevice..." ;
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host    
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, input_size*sizeof(float), cudaMemcpyHostToDevice, stream));

    // ANNIWOLOG(INFO) <<logstr << "TrtGPUInfer: Wait lock..." ;
    gpuQueueLock.lock();
    
    if( ! context.enqueueV2((void**)buffers, stream, nullptr) )
    {
        ANNIWOLOG(INFO) <<logstr << "TrtGPUInfer: enqueue failed!!!" ;
    }


    // ANNIWOLOG(INFO) <<logstr << "TrtGPUInfer: cudaMemcpyAsync cudaMemcpyDeviceToHost..." ;
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));

    // ANNIWOLOG(INFO) <<logstr << "TrtGPUInfer: Out lock..." ;

    CHECK( cudaStreamSynchronize(stream) );

    gpuQueueLock.unlock();  //必须到此处，因为同一个模型的同一个context必须串行！否则unpredictable


    // Release stream and buffers
    CHECK( cudaStreamDestroy(stream) );
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}






cv::Mat     static_resizeLINEAR(cv::Mat img,int inINPUT_W, int inINPUT_H)
{
    float scale_x = float(inINPUT_W) / (img.cols*1.0);
    float scale_y = float(inINPUT_H) / (img.rows*1.0);
        
    // r = std::min(r, 1.0f);
    int unpad_w = scale_x * img.cols;
    int unpad_h = scale_y * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    //python:cv2.resize(img, None, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
    cv::resize(img, re, re.size(),0,0,cv::INTER_LINEAR);

    cv::Mat out(inINPUT_H, inINPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

cv::Mat static_resize(cv::Mat img,int inINPUT_W, int inINPUT_H)
{
    float scale_x = float(inINPUT_W) / (img.cols*1.0);
    float scale_y = float(inINPUT_H) / (img.rows*1.0);
        
    // r = std::min(r, 1.0f);
    int unpad_w = scale_x * img.cols;
    int unpad_h = scale_y * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    //python:cv2.resize(img, None, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)
    cv::resize(img, re, re.size(),0,0,cv::INTER_CUBIC);

    cv::Mat out(inINPUT_H, inINPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

cv::Mat static_resizeNEAREST(cv::Mat img,int inINPUT_W, int inINPUT_H)
{
    float scale_x = float(inINPUT_W) / (img.cols*1.0);
    float scale_y = float(inINPUT_H) / (img.rows*1.0);
        
    // r = std::min(r, 1.0f);
    int unpad_w = scale_x * img.cols;
    int unpad_h = scale_y * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    //python:cv2.resize(img, None, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)
    cv::resize(img, re, re.size(),0,0,cv::INTER_NEAREST);

    cv::Mat out(inINPUT_H, inINPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

//仅仅在抽烟中扣人的图用
cv::Mat static_resizex(cv::Mat img,int xINPUT_W, int xINPUT_H,float& alpha  ) {
    // float r = std::min(xINPUT_W / (img.cols*1.0), xINPUT_H / (img.rows*1.0));
    int r = xINPUT_W / (img.rows*1.0) + 0.5;

    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    // cv::Mat out(xINPUT_H, xINPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
    // re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));

    alpha=r;
    // return out;
    return re;
}

void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
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

    // #pragma omp parallel sections
    {
        // #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        // #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
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



void blobFromImageAndNormalize(cv::Mat img,std::vector<float>& arrvectorOut){
    int img_h = img.rows;
    int img_w = img.cols;
    // //对应tf模型为h,w,c.而pytorch的模型输入是c,h,w,此处应该改一下顺序！
    // //有关系，i*(列长*对长)+j*对长+k 修改如下：
    for (size_t  h = 0; h < img_h; h++) 
    {
        for (size_t w = 0; w < img_w; w++) 
        {
                for (size_t c = 0; c < CHANNELS; c++) 
                {
                    //python:img.astype(np.float32) / 255.
                    arrvectorOut[h*img_w*CHANNELS  + w*CHANNELS + c] =
                        ((float)img.at<cv::Vec3b>(h, w)[c]) / 255.0;
                }

        }
    }
    return ;
}


void blobFromImageAndNorm_resnetv2style(cv::Mat img,std::vector<float>& arrvectorOut){
    int img_h = img.rows;
    int img_w = img.cols;
    // //对应tf模型为h,w,c.而pytorch的模型输入是c,h,w,此处应该改一下顺序！
    // //有关系，i*(列长*对长)+j*对长+k 修改如下：
    for (size_t  h = 0; h < img_h; h++) 
    {
        for (size_t w = 0; w < img_w; w++) 
        {
                for (size_t c = 0; c < CHANNELS; c++) 
                {
                    //python:    if mode == 'tf':
                    // x /= 127.5
                    // x -= 1.
                    // return x
                    arrvectorOut[h*img_w*CHANNELS  + w*CHANNELS + c] =
                        ((float)img.at<cv::Vec3b>(h, w)[c]) / 127.5 -1.0;
                }

        }
    }
    return ;
}



//todo:inline
void process_feats(xt::xarray<float>& outIn, xt::xarray<int>&  anchors,
/*out*/xt::xtensor<float,4>& box_confidence,
/*out*/xt::xtensor<float,4>& box_class_probs,
/*out*/xt::xtensor<float,4>& boxes
)
{
    //python:grid_h, grid_w, num_boxes = map(int, out.shape[1: 4])
    auto out_shape =xt::adapt(outIn.shape());
    int grid_h=out_shape(1);
    int grid_w=out_shape(2);
    int num_boxes=out_shape(3);
    std::vector<int> vecgridwh={grid_w, grid_h};

    // ANNIWOLOG(INFO) <<  "grid_h:" <<grid_h<< "grid_w:"<<grid_w<< "num_boxes:"<<num_boxes <<std::endl;  


    // ANNIWOLOG(INFO) <<  "anchors:"<<anchors;

    //python:anchors_tensor = np.array(anchors).reshape(1, 1, len(anchors), 2)
    anchors.reshape({1, 1, 3, 2});

    //python:out = out[0]
    xt::xarray<float> out = xt::view(outIn,0);
    // ANNIWOLOG(INFO) <<  "out shape:" <<xt::adapt(out.shape()) <<std::endl;  


    //python:box_xy = self._sigmoid(out[..., :2])
    //out[..., :2] 等价于 out[:,:,:, :2]
    xt::xarray<float> box_xy = xt::view(out, xt::all(),xt::all(),xt::all(),xt::range(0, 2)) ;
    // ANNIWOLOG(INFO) <<  "box_xy shape:" <<xt::adapt(box_xy.shape()) <<std::endl;  
    // ANNIWOLOG(INFO) <<  "box_xy :" <<box_xy <<std::endl;  

    //python: _sigmoid(self, x):1 / (1 + np.exp(-x))
    auto tmpbox_xy = (1/(1.0+xt::exp(box_xy*(-1.0))) );
    box_xy = tmpbox_xy;
    // ANNIWOLOG(INFO) <<  "box_xy after sigmoid shape:" <<xt::adapt(box_xy.shape()) <<std::endl;  
    // ANNIWOLOG(INFO) <<  "box_xy after sigmoid:" <<box_xy <<std::endl;  
    // if(grid_h==38)
    // { 
    //     xt::dump_npy("box_xy_aftersigmoid_xtensor.npy", box_xy); 
    // } 


// python:box_wh = np.exp(out[..., 2:4])
    // ANNIWOLOG(INFO) <<  "out shape:" <<xt::adapt(out.shape()) <<std::endl;  
    xt::xarray<float> box_wh = xt::view(out, xt::all(),xt::all(),xt::all(),xt::range(2, 4)) ;
    auto tmpbox_wh = xt::exp(box_wh);
// python:box_wh = box_wh * anchors_tensor
    box_wh=tmpbox_wh*anchors;
    // ANNIWOLOG(INFO) <<  "box_wh shape:" <<xt::adapt(box_wh.shape()) <<std::endl;  
    // ANNIWOLOG(INFO) <<  "box_wh:" <<box_wh <<std::endl;  
    // if(grid_h==38)
    // { 
    //     xt::dump_npy("box_wh_afteranchors_xtensor.npy", box_wh); 
         
    // } 


    
// python:box_confidence = self._sigmoid(out[..., 4])
    auto tmp0box_confidence = xt::view(out, xt::all(),xt::all(),xt::all(),4) ;
//python: _sigmoid(self, x):1 / (1 + np.exp(-x))
    auto tmpbox_confidence = (1/( 1.0 + xt::exp( tmp0box_confidence * (-1.0) ) ) );
// python:box_confidence = np.expand_dims(box_confidence, axis=-1)
    box_confidence = xt::expand_dims(tmpbox_confidence,3);   //shape:{19, 19,  3,  1}
    // ANNIWOLOG(INFO) <<  "box_confidence shape:" <<xt::adapt(box_confidence.shape()) <<std::endl;  

// python:box_class_probs = self._sigmoid(out[..., 5:])
    box_class_probs = xt::view(out, xt::all(),xt::all(),xt::all(),xt::range(5, _));
    auto tmpbox_class_probs = (1/(1.0+xt::exp(box_class_probs*(-1.0)) ) );
    box_class_probs=tmpbox_class_probs;
    // ANNIWOLOG(INFO) <<  "box_class_probs shape:" <<xt::adapt(box_class_probs.shape()) <<std::endl;  


//此函数为扩展函数，data为要扩展的数据，类型为np类型数组，x,扩展行数，y扩展列数，如下代码测试 
// python:col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    xt::xarray<float> col = xt::tile(xt::arange(0, grid_w), grid_w);
    col.reshape({-1, grid_w});


// python:row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    //todo:以auto 类型为返回  adapt来的，reshape不能用！
    //               arrange来的，reshape也不能用!!!
    //因为默认返回的是tensor类型,不能改变维度
    xt::xarray<float> rowtmp0 = xt::arange(0, grid_h);
    rowtmp0.reshape({-1, 1});
    //当行为-1时候，numpy默认tile列。xtensor需要指定！    
    xt::xarray<float> row =xt::tile(rowtmp0,{1,grid_h}); //tensor


// python:col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    col.reshape({grid_h, grid_w, 1, 1}); //{19, 19,  1,  1}
    // ANNIWOLOG(INFO) <<  "col shape before repeat:" <<xt::adapt(col.shape()) <<std::endl;  
    // ANNIWOLOG(INFO) <<  "col before repeat:" <<col <<std::endl;  

    auto tmpcol = xt::repeat(col,3,2);
    col = tmpcol;
    // ANNIWOLOG(INFO) <<  "col shape:" <<xt::adapt(col.shape()) <<std::endl;  
    // ANNIWOLOG(INFO) <<  "col :" <<col <<std::endl;  



// python:row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row.reshape({grid_h, grid_w, 1, 1});//{19, 19,  1,  1}
    auto tmprow = xt::repeat(row,3,2);
    row=tmprow;

    // ANNIWOLOG(INFO) <<  "row shape:" <<xt::adapt(row.shape()) <<std::endl; 
    // ANNIWOLOG(INFO) <<  "row :" <<row <<std::endl;  


// python:grid = np.concatenate((col, row), axis=-1)
    xt::xarray<float> grid = xt::concatenate(xtuple(col, row), 3);

    // ANNIWOLOG(INFO) <<  "grid shape:" <<xt::adapt(grid.shape()) <<std::endl;  
    // ANNIWOLOG(INFO) <<  "grid :" <<grid <<std::endl;  

    // if(grid_h==38)
    // { 
    //     // xt::dump_npy("box_xy_gridwh_xtensor.npy", box_xy); 
    //     ANNIWOLOG(INFO) <<  "before + grid box_xy[8][35][0] :" <<xt::view(box_xy,8,35,0,xt::all()) <<std::endl;  
         
    // } 

// python:box_xy += grid
    auto tmp2box_xy=box_xy + grid;

    // if(grid_h==38)
    // { 
    //     // xt::dump_npy("box_xy_gridwh_xtensor.npy", box_xy); 
    //     ANNIWOLOG(INFO) <<  "after + grid box_xy[8][35][0] :" <<xt::view(box_xy,8,35,0,xt::all()) <<std::endl;  
         
    // } 
    
// python:box_xy /= (grid_w, grid_h)
    std::vector<std::size_t> gridwhshape = {1, 1, 1, 2};
    //todo:注意！ auto 类型 xtensor adapt来的，reshape不能用！
    //               arrange来的，reshape也不能用!!!
    auto grid_wh= xt::adapt(vecgridwh,gridwhshape);
    // grid_wh.reshape({1, 1, 1, 2});
    
    box_xy=tmp2box_xy/grid_wh;  //box_xy shape:{19, 19,  3,  2}
    // if(grid_h==38)
    // { 
    //     // xt::dump_npy("box_xy_gridwh_xtensor.npy", box_xy); 
    //     ANNIWOLOG(INFO) <<  "after /grid_wh box_xy[8][35][0] :" <<xt::view(box_xy,8,35,0,xt::all()) <<std::endl;  
         
    // } 
    

// python:box_wh /= self.input_shape
    std::vector<std::size_t> inputimgshape = {1, 1, 1, 2};
    auto xarrinputshape = xt::adapt(input_image_shape,inputimgshape);

    auto tmp2box_wh=box_wh/xarrinputshape;
    box_wh=tmp2box_wh;
    // ANNIWOLOG(INFO) <<  "box_wh :" <<box_wh <<std::endl; 
    // if(grid_h==38)
    // { 
    //     // xt::dump_npy("box_xy_gridwh_xtensor.npy", box_xy); 
    //     ANNIWOLOG(INFO) <<  "box_wh[8][35][0] :" <<xt::view(box_wh,8,35,0,xt::all()) <<std::endl;  
         
    // }  


// python:box_xy -= (box_wh / 2.)   # 坐标格式是左上角xy加矩形宽高wh，xywh都除以图片边长归一化了。
    auto tmp3box_xy = box_xy - (box_wh / 2.);
    box_xy=tmp3box_xy;
    // ANNIWOLOG(INFO) <<  "box_xy :" <<box_xy <<std::endl; 
    // if(grid_h==38)
    // { 
    //     xt::dump_npy("box_xy_xtensor.npy", box_xy); 
    //     xt::dump_npy("box_wh_xtensor.npy", box_wh); 
         
    // } 


// python:boxes = np.concatenate((box_xy, box_wh), axis=-1) 
    boxes = xt::concatenate(xtuple(box_xy, box_wh), 3); //shape like:{19, 19,  3,  4}

    // ANNIWOLOG(INFO) <<  "process_feats:boxes shape:" <<xt::adapt(boxes.shape()) <<std::endl;  
    // ANNIWOLOG(INFO) <<  "process_feats:boxes[0][0][0]:" << xt::view(boxes,0, 0, 0,xt::all()) <<std::endl;  
    // ANNIWOLOG(INFO) <<  "process_feats:box_class_probs[0][0][0]:" << xt::view(box_class_probs,0, 0, 0,xt::all()) <<std::endl;  
    // ANNIWOLOG(INFO) <<  "process_feats:box_confidence[0][0]:" << xt::view(box_confidence,0, 0, xt::all(),xt::all()) <<std::endl;  
    

}


// todo:inline later
inline xt::xtensor<float, 2>  filter_boxes(xt::xtensor<float,4>& boxes, 
                  xt::xtensor<float,4>& box_confidences,
                  xt::xtensor<float,4>& box_class_probs,
                  /*out*/xt::xtensor<int,1>& classes,
                  /*out*/xt::xtensor<float,1>& scores,
                  float BBOX_CONF_THRESH
)
{
  
  //python: box_scores = box_confidences * box_class_probs
  xt::xtensor<float,4> box_scores = box_confidences * box_class_probs;
//   ANNIWOLOG(INFO) <<  "box_scores shape:" <<xt::adapt(box_scores.shape()) <<std::endl;  //shape:{19, 19,  3,  9}

  //python: box_classes = np.argmax(box_scores, axis=-1)
  xt::xtensor<int,3> box_classes=xt::argmax(box_scores, 3);
//   ANNIWOLOG(INFO) <<  "box_classes shape:" <<xt::adapt(box_classes.shape()) <<std::endl;  

  //python: box_class_scores = np.max(box_scores, axis=-1)
  xt::xtensor<float,3> box_class_scores = xt::amax(box_scores,3);
//   ANNIWOLOG(INFO) <<  "box_class_scores shape:" <<xt::adapt(box_class_scores.shape()) <<std::endl;  
//   ANNIWOLOG(INFO) <<  "box_class_scores:" <<box_class_scores <<std::endl;  


  //python: pos = np.where(box_class_scores >= self._t1)
  auto pos = xt::where(box_class_scores >= BBOX_CONF_THRESH);
//   ANNIWOLOG(INFO) <<  "where pos:" << xt::from_indices(pos) <<std::endl;  
  
//   if(pos.size()>0 && pos[1].size()>0 && pos[1][0] == 35)
//   {
//     ANNIWOLOG(INFO) <<  "filter_boxes in boxes :" <<boxes <<std::endl;  
//     xt::dump_npy("boxes_xtensor.npy", boxes);  
//   }




  //python: boxes = boxes[pos]
  auto boxes_result = xt::xtensor<float, 2>::from_shape({pos[0].size(),4});

  for(std::size_t i =0; i < pos[0].size(); ++i)
  {
    // ANNIWOLOG(INFO) <<  " i:"<<i ;
    auto tmparr = xt::view(boxes,pos[0][i], pos[1][i],pos[2][i],xt::all());
    // ANNIWOLOG(INFO) <<  "filter_boxes tmparr :" <<tmparr <<std::endl;  

    auto res_i = xt::view(boxes_result,i,xt::all());
    res_i=tmparr;
    // ANNIWOLOG(INFO) <<  "filter_boxes res_i :" <<res_i <<std::endl;  

    // ANNIWOLOG(INFO) <<  "filter_boxes in iter boxes_result :" <<boxes_result <<std::endl;  

  }
//   ANNIWOLOG(INFO) <<  "filter_boxes boxes_result shape :" <<xt::adapt(boxes_result.shape()) <<std::endl;  
//   ANNIWOLOG(INFO) <<  "filter_boxes boxes_result :" <<boxes_result <<std::endl;  

  /////////////////////////////

  auto indices = xt::argwhere(box_class_scores >= BBOX_CONF_THRESH);
//   ANNIWOLOG(INFO) <<  "argwhere indices :" << xt::from_indices(indices) <<std::endl;  

  //python: classes = box_classes[pos]
  // when you do a(n), you really do a(0, n), because a is a 2-D array and you only provide one index (we prepend with 0s).
  classes = xt::index_view(box_classes,indices); //box_classes[pos];
//   ANNIWOLOG(INFO) <<  "filter_boxes classes  :" <<classes <<std::endl;  

  //python: scores = box_class_scores[pos]
  scores = xt::index_view(box_class_scores,indices);//box_class_scores[pos];
//   ANNIWOLOG(INFO) <<  "filter_boxes scores shape :" <<xt::adapt(scores.shape()) <<std::endl; 
//   ANNIWOLOG(INFO) <<  "filter_boxes scores  :" <<scores <<std::endl; 

//   ANNIWOLOG(INFO) <<  "end of filter_boxes:boxes_result :" <<boxes_result <<std::endl;  
//   ANNIWOLOG(INFO) <<  "end of filter_boxes:boxes_result shape :" <<xt::adapt(boxes_result.shape()) <<std::endl;  

  return boxes_result;


}

//todo:inline
inline void decodeOne(xt::xarray<float>& outIn,
                  xt::xarray<int>& anchors,
                  /*out*/xt::xtensor<float, 2>& boxes_result,
                  /*out*/xt::xtensor<int,1>& classes,
                  /*out*/xt::xtensor<float,1>& scores,
                  float BBOX_CONF_THRESH
                  )
{
    xt::xtensor<float,4> box_confidences;
    xt::xtensor<float,4> box_class_probs;
    xt::xtensor<float,4> boxes;

    process_feats( outIn,
                   anchors,
            /*out*/box_confidences,
            /*out*/box_class_probs,
            /*out*/boxes
    );

    //python:self._filter_boxes(b, c, s)

    /*out*/boxes_result = filter_boxes(boxes, 
                box_confidences,
                box_class_probs,
                /*out*/classes,
                /*out*/scores,
                BBOX_CONF_THRESH
    );

    // ANNIWOLOG(INFO) <<  "decodeOne:boxes_result shape:" <<xt::adapt(boxes_result.shape()) <<std::endl;  

}

void decode_outputs(std::vector<float>& prob, std::vector<Object>& objects, float scale, const int img_w, 
            const int img_h,float BBOX_CONF_THRESH,int NUM_CLASSES, std::string& logtitleSTRin, bool isDebug)
{

    
    int dims =  NUM_ANCHORS* (NUM_CLASSES + 5);

    std::vector<std::size_t> shape = { 1,7581,dims };
    auto a1 = xt::adapt(prob, shape);
    // if(isDebug)
    // {
    //     xt::dump_npy("out_xtensor.npy", a1);
    // }

    // auto a1=xt::load_npy<float>("outs_py.npy");

    if(isDebug)
    {
        ANNIWOLOG(INFO) <<  "a1[0] :" <<xt::view(a1,0,0)<<std::endl; 
        ANNIWOLOG(INFO) <<  "NUM_CLASSES:" <<NUM_CLASSES<<std::endl; 
    }

    // a1.reshape(shape);




    //python: output_l = outs[:,0:19 * 19,:]
    //python: output_m = outs[:, 19 * 19:19 * 19 + 38 * 38, :]
    //python: output_s = outs[:,  19 * 19 + 38 * 38:38*38+19*19+76*76, :]
    xt::xarray<float> output_l =  xt::view(a1,xt::all(), xt::range(0, 19 * 19), xt::all());
    xt::xarray<float> output_m = xt::view(a1,xt::all(), xt::range( 19 * 19,19 * 19 + 38 * 38), xt::all());
    xt::xarray<float> output_s =xt::view(a1,xt::all(), xt::range(19 * 19 + 38 * 38, 38*38+19*19+76*76), xt::all());

    if(isDebug)
    {
        ANNIWOLOG(INFO) <<  "output_l shape:" <<xt::adapt(output_l.shape()) <<std::endl; 
        ANNIWOLOG(INFO) <<  "output_m shape:" <<xt::adapt(output_m.shape()) <<std::endl;  
        ANNIWOLOG(INFO) <<  "output_s shape:" <<xt::adapt(output_s.shape()) <<std::endl;  

    }
    
    //python: a1 = np.reshape(output_l, (1, self.input_image_shape[0]//32, self.input_image_shape[1]//32, 3, 5+self.NUM_CLASSES))
    //python: a2 = np.reshape(output_m, (1, self.input_image_shape[0]//16, self.input_image_shape[1]//16, 3, 5+self.NUM_CLASSES))
    //python: a3 = np.reshape(output_s, (1, self.input_image_shape[0]//8, self.input_image_shape[1]//8, 3, 5+self.NUM_CLASSES))
    
    output_l.reshape({1, input_image_shape[0]/32, input_image_shape[1]/32, 3, 5+NUM_CLASSES});
    output_m.reshape({1, input_image_shape[0]/16, input_image_shape[1]/16, 3, 5+NUM_CLASSES});
    output_s.reshape({1, input_image_shape[0]/8, input_image_shape[1]/8, 3, 5+NUM_CLASSES});

    if(isDebug)
    {
        ANNIWOLOG(INFO) <<  "output_l shape:" <<xt::adapt(output_l.shape()) <<std::endl; 
        ANNIWOLOG(INFO) <<  "output_m shape:" <<xt::adapt(output_m.shape()) <<std::endl;  
        ANNIWOLOG(INFO) <<  "output_s shape:" <<xt::adapt(output_s.shape()) <<std::endl;  

    }
    
    // //self._yolo_out([a1, a2, a3], orig_images_hape)
    // // masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    // xt::xarray<int> masks={{6, 7, 8}, {3, 4, 5}, {0, 1, 2}};
    // // anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55],
    // //            [72, 146], [142, 110], [192, 243], [459, 401]]
    // xt::xarray<int> anchors_all= {{12, 16}, {19, 36}, {40, 28}, {36, 75}, {76, 55},
    //            {72, 146}, {142, 110}, {192, 243}, {459, 401}};

    //python: boxes, classes, scores = [], [], []
    //python: for out, mask in zip(outs, masks):
        //python:b, c, s = self._process_feats(out, anchors, mask)
        //python: anchors = [anchors[i] for i in mask]
        // ANNIWOLOG(INFO) <<  "-----------------------handle output_l---------------------------"<<std::endl;
        xt::xarray<int> anchors_l = {{142, 110}, {192, 243}, {459, 401}};
        xt::xtensor<float, 2> boxes_result_l;
        xt::xtensor<int,1> classes_l;
        xt::xtensor<float,1> scores_l;


        decodeOne(output_l,
                  anchors_l,
                  /*out*/ boxes_result_l,
                  /*out*/classes_l,
                  /*out*/ scores_l,
                  BBOX_CONF_THRESH
                  );
        
        // ANNIWOLOG(INFO) <<  "-----------------------handle output_m---------------------------"<<std::endl;

        xt::xarray<int> anchors_m = {{36, 75}, {76, 55},{72, 146}};
        xt::xtensor<float, 2> boxes_result_m;
        xt::xtensor<int,1> classes_m;
        xt::xtensor<float,1> scores_m;


        decodeOne(output_m,
                  anchors_m,
                  /*out*/ boxes_result_m,
                  /*out*/classes_m,
                  /*out*/ scores_m,
                  BBOX_CONF_THRESH
                  );
        
        // ANNIWOLOG(INFO) <<  "-----------------------handle output_s---------------------------"<<std::endl;
        xt::xarray<int> anchors_s = {{12, 16}, {19, 36}, {40, 28}};
        xt::xtensor<float, 2> boxes_result_s;
        xt::xtensor<int,1> classes_s;
        xt::xtensor<float,1> scores_s;


        decodeOne(output_s,
                  anchors_s,
                  /*out*/ boxes_result_s,
                  /*out*/classes_s,
                  /*out*/ scores_s,
                  BBOX_CONF_THRESH
                  );

    // ANNIWOLOG(INFO) <<  "---------------------------------------------------------------------"<<std::endl;
    if(isDebug)
    {
        ANNIWOLOG(INFO) <<  "boxes_result_l shape:" <<xt::adapt(boxes_result_l.shape()) <<std::endl;  
        ANNIWOLOG(INFO) <<  "boxes_result_m shape:" <<xt::adapt(boxes_result_m.shape()) <<std::endl;  
        ANNIWOLOG(INFO) <<  "boxes_result_s shape:" <<xt::adapt(boxes_result_s.shape()) <<std::endl;  
    }
    
    // python:boxes = np.concatenate(boxes)
    xt::xtensor<float, 2> boxes = xt::concatenate(xtuple(boxes_result_l, boxes_result_m,boxes_result_s), 0);
    // xt::xarray<float> boxes = xt::concatenate(xtuple(boxes_result_l, boxes_result_m,boxes_result_s), 0);
    if(isDebug)
    {
        ANNIWOLOG(INFO) <<  "boxes shape:" <<xt::adapt(boxes.shape()) <<std::endl;  
    }
    
    // python:classes = np.concatenate(classes)
    xt::xtensor<int,1> classes = xt::concatenate(xtuple(classes_l, classes_m, classes_s), 0);
    // ANNIWOLOG(INFO) <<  "classes shape:" <<xt::adapt(classes.shape()) <<std::endl; 
    if(isDebug)
    {
        ANNIWOLOG(INFO) <<  "classes:"  << classes <<std::endl; 
    }
    

    // python:scores = np.concatenate(scores)
    xt::xtensor<float,1> scores = xt::concatenate(xtuple(scores_l, scores_m, scores_s), 0);
    // ANNIWOLOG(INFO) <<  "scores shape:" <<xt::adapt(scores.shape()) <<std::endl;  
    if(isDebug)
    {
        ANNIWOLOG(INFO) <<  "scores:"  << scores <<std::endl; 
    }

    //////////////////////////////////////////////////此处恢复正常尺寸.
    // // python:# boxes坐标格式是左上角xy加矩形宽高wh，xywh都除以图片边长归一化了。
    // // python:# Scale boxes back to original image shape.
    // // python:w, h = shape[1], shape[0]
    auto w = img_w; //原图尺寸
    auto h = img_h;
    // python:image_dims = [w, h, w, h]
    xt::xtensor<float, 1> image_dims({w, h, w, h});
    // python:boxes = boxes * image_dims     
    boxes = boxes * image_dims;



    // struct Object
    // {
    //     cv::Rect_<float> rect;
    //     int label;
    //     float prob;
    // };

///////////////////////////////////////

    if(classes.size() == 0)
    {
        return;
    }

    std::vector<Object> proposals;
    std::unordered_set<int> setclsIDs ;
    for(size_t i=0;i<classes.size();i++)
    {
        setclsIDs.insert(classes(i));
    }

    for (auto& clsID : setclsIDs)
    {
        proposals.clear();

        for(size_t i=0;i<classes.size();i++)
        {
            if( classes(i)  == clsID)
            {
                Object obj;
                obj.rect.x = boxes(i, 0);
                obj.rect.y = boxes(i, 1);
                obj.rect.width = boxes(i, 2);
                obj.rect.height = boxes(i, 3);
                obj.label = classes(i);
                obj.prob = scores(i);

                if(obj.rect.x < 0 || obj.rect.y < 0 || obj.rect.width < 0 || obj.rect.height < 0 )
                {
                    ANNIWOLOG(INFO) <<  "Ignored  score:"  << obj.prob
                            <<" label:"  << obj.label
                            <<" x:"  << obj.rect.x 
                            <<" y:"  << obj.rect.y
                            <<" width:"  << obj.rect.width
                            <<" height:"  << obj.rect.height 
                            <<std::endl; 
                }else
                {
                    proposals.push_back(obj);
                }
            }

        }


        if(isDebug)
        {
            ANNIWOLOG(INFO) << "clsID:"<<clsID<<",num of boxes before nms: " << proposals.size() ;
        }

        if(proposals.size() > 150)
        {
            ANNIWOLOG(INFO) <<  logtitleSTRin <<" decode_outputs():num of boxes before nms: " << proposals.size() <<"WRONG! Ignore!";
            return;
        }

        qsort_descent_inplace(proposals);

        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, NMS_THRESH);


        int count = picked.size();


        if(isDebug)
        {
            ANNIWOLOG(INFO) << "clsID:"<<clsID<<",num of boxes after nms: " << count;
        }

        if(count > 50)
        {
            ANNIWOLOG(INFO) <<  logtitleSTRin <<" decode_outputs():num of boxes after nms: " << count <<"WRONG! Ignore!";
            return;
        }


        for (int i = 0; i < count; i++)
        {
            objects.push_back(proposals[picked[i]]);


            // adjust offset to original unpadded
            float x0 = (objects.back().rect.x) ;
            float y0 = (objects.back().rect.y) ;
            float x1 = (objects.back().rect.x + objects.back().rect.width) ;
            float y1 = (objects.back().rect.y + objects.back().rect.height) ;


            // std::cout << "x0,y0,x1,y1:"<< x0 <<","<< y0<<"," << x1<<"," <<y1<<std::endl; 


            // clip
            x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

            objects.back().rect.x = x0;
            objects.back().rect.y = y0;
            objects.back().rect.width = x1 - x0;
            objects.back().rect.height = y1 - y0;

            // std::cout << "after clip x0,y0,x1,y1:"<< x0 <<","<< y0<<"," << x1<<"," <<y1<<std::endl; 
            
            // std::cout << "decode_outputs will return: score:"  << objects.back().prob
            // <<" label:"  << objects.back().label
            // <<" x:"  << objects.back().rect.x 
            // <<" y:"  << objects.back().rect.y
            // <<" width:"  << objects.back().rect.width
            // <<" height:"  << objects.back().rect.height 
            // <<std::endl; 

        }

    }

///////////////////////////////////////


}


void yolov4_detection_staff(std::unordered_map<int, std::vector<float> >& m_input_datas,
    int camID,int instanceID, cv::Mat img,
    nvinfer1::IRuntime* runtime,
    nvinfer1::ICudaEngine* engine,
    TrtSampleUniquePtr<nvinfer1::IExecutionContext>& context,
    int gpuNum,
    std::unique_ptr<std::mutex>&  lockptr,
    int YOLO4_OUTPUT_SIZE, int INPUT_W, int INPUT_H, 
    /*out*/std::vector<Object>& objects,
    float BBOX_CONF_THRESH,
    int NUM_CLASSES,
    std::string logstring )
{
    ANNIWOCHECK(img.data != nullptr);

    ANNIWOLOG(INFO) << logstring <<":detect entered."<<"camID:"<<camID <<"instanceID:"<<instanceID ;

    std::unordered_map<int, std::vector<float> >::iterator got_m_input_datas = m_input_datas.find(instanceID);

    if (got_m_input_datas == m_input_datas.end())
    {
        ANNIWOLOG(INFO) << logstring <<":detect "<<"instanceID:"<<instanceID<<"NOT in planned map!";
        ANNIWOCHECK(false);

        return ;
    }




    int img_w = img.cols;
    int img_h = img.rows;
    cv::Mat pr_img = static_resize(img,INPUT_W,INPUT_H);
    ANNIWOLOG(INFO)  << logstring <<": iniput image:w:"<<img_w<<",h:"<<img_h<<"camID:"<<camID <<"instanceID:"<<instanceID<<"gpuId:"<<gpuNum;

    cv::Mat rgb_img;
    cv::cvtColor(pr_img,rgb_img,cv::COLOR_BGR2RGB);
    pr_img=rgb_img;

    blobFromImageAndNormalize(pr_img,m_input_datas[instanceID]);

    float scale = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));



    ANNIWOCHECK(runtime != nullptr);
    ANNIWOCHECK(engine != nullptr); 
    ANNIWOCHECK(bool(context)); //smartpoint bool operator
    ANNIWOCHECK(img.data != nullptr);
    
    
    
    std::vector<float> out_data( YOLO4_OUTPUT_SIZE, 1.0);

    // run inference
    auto start = std::chrono::system_clock::now();
    
    TrtGPUInfer( *context, gpuNum, *lockptr, m_input_datas[instanceID].data(), out_data.data(), YOLO4_OUTPUT_SIZE);

    auto end = std::chrono::system_clock::now();

    ANNIWOLOG(INFO) << logstring <<":infer time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms,"<<"camID:"<<camID ;


    // #ifdef __aarch64__
    //     omp_set_num_threads(8); 
    // #else
    //     //x86_64 server
    //     omp_set_num_threads(48); 
    // #endif

    decode_outputs(out_data, objects, scale, img_w, img_h,BBOX_CONF_THRESH,NUM_CLASSES,logstring,false);
}




//for debugging usage.
void anniwo_debug_draw_objects(const cv::Mat bgr, const std::vector<Object>& objects, std::string output_path, bool isDrawResult)
{
    cv::Mat image = bgr.clone();

    if(isDrawResult)
    {

        static const char* class_names[] = {
            "work_clothe_blue",
            "person",
            "car",
            "work_clothe_yellow",
            "tank_truck",
            "truck",
            "motor",
            "unloader",
            "reflective_vest",
            "work_clothe_wathet",
            "rider",
            "cement_truck"
            "person",
            "person_indistinct",
            "helmet_head_indistinct", 
            "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        };


        for (size_t i = 0; i < objects.size(); i++)
        {
            const Object& obj = objects[i];

            fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                    obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

            cv::Scalar color = cv::Scalar(color_list[obj.label][0], color_list[obj.label][1], color_list[obj.label][2]);
            float c_mean = cv::mean(color)[0];
            cv::Scalar txt_color;
            if (c_mean > 0.5){
                txt_color = cv::Scalar(0, 0, 0);
            }else{
                txt_color = cv::Scalar(255, 255, 255);
            }

            cv::rectangle(image, obj.rect, color * 255, 2);

            char text[256];
            sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

            cv::Scalar txt_bk_color = color * 0.7 * 255;

            int x = obj.rect.x;
            int y = obj.rect.y + 1;
            //int y = obj.rect.y - label_size.height - baseLine;
            if (y > image.rows)
                y = image.rows;
            //if (x + label_size.width > image.cols)
                //x = image.cols - label_size.width;

            cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                        txt_bk_color, -1);

            cv::putText(image, text, cv::Point(x, y + label_size.height),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
        }

    }


    static int dbugoutpicCnt=0;

    std::stringstream buffer;  

    buffer <<"img_"<<dbugoutpicCnt++<<".jpg";  

    std::string text(buffer.str());

    output_path+=text;
    cv::imwrite(output_path, image);
    ANNIWOLOGF(INFO, "anniwo_debug_draw_objects:Visualized output saved as %s\n", output_path.c_str());

}
