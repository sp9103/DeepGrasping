#ifdef USE_CUDNN
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include <opencv2\opencv.hpp>

namespace caffe {

__global__ void sync_conv_groups() { }

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	//const int tWidth = bottom[0]->shape()[2];
	//const int tHeight = bottom[0]->shape()[3];
	//for (int i = 0; i < bottom[0]->shape()[0]; i++){
	//	cv::Mat img;
	//	img.create(tWidth, tHeight, CV_32FC1);
	//	Dtype ImgTemp[80 * 80];
	//	cudaMemcpy(ImgTemp, &bottom[0]->gpu_data()[tWidth * tHeight * i], sizeof(Dtype) * tWidth * tHeight, cudaMemcpyDeviceToHost);

	//	for (int h = 0; h < tHeight; h++){
	//		for (int w = 0; w < tWidth; w++)
	//			img.at<float>(h, w) = (float)ImgTemp[h * tWidth + w];
	//	}

	//	cv::imshow("img", img);
	//	cv::waitKey(0);
	//}

  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    const Dtype* weight = this->blobs_[0]->gpu_data();

    size_t workspace_limit_bytes = this->kernel_h_ *
                                   this->kernel_w_ *
                                   this->channels_ *
                                   sizeof(int) + 1;

    // Forward through cuDNN in parallel over groups.
    for (int g = 0; g < this->group_; g++) {
      cudnnConvolutionFwdAlgo_t algo;

      // pick the convolution algorithm
      // TODO(shelhamer) this should be done during reshape
      // TODO(shelhamer) the choice of automatic or manual algorithm picking
      // should be exposed in proto
      CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle_[g],
        bottom_descs_[i],
        filter_desc_,
        conv_descs_[i],
        top_descs_[i],
        CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
        workspace_limit_bytes,  // memoryLimitInBytes,
        &algo));

      // get minimum size of the workspace needed for the desired algorithm
      size_t workspaceSizeInBytes_temp = 0;

      CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle_[g],
        bottom_descs_[i],
        filter_desc_,
        conv_descs_[i],
        top_descs_[i],
        algo,
        &workspaceSizeInBytes_temp));

      if (workspaceSizeInBytes_temp > workspaceSizeInBytes) {
        workspaceSizeInBytes = workspaceSizeInBytes_temp;
        // free the existing workspace and allocate a new (larger) one
        cudaFree(this->workspace);
        cudaError_t err = cudaMalloc(&(this->workspace), workspaceSizeInBytes);
        if (err != cudaSuccess) {
          // force zero memory path
          algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
          workspace = NULL;
          workspaceSizeInBytes = 0;
        }
      }

      // Filters.
      CUDNN_CHECK(cudnnConvolutionForward(handle_[g],
            cudnn::dataType<Dtype>::one,
            bottom_descs_[i], bottom_data + bottom_offset_ * g,
            filter_desc_, weight + weight_offset_ * g,
            conv_descs_[i],
            algo, workspace, workspaceSizeInBytes,
            cudnn::dataType<Dtype>::zero,
            top_descs_[i], top_data + top_offset_ * g));

      // Bias.
      if (this->bias_term_) {
        const Dtype* bias_data = this->blobs_[1]->gpu_data();
        CUDNN_CHECK(cudnnAddTensor(handle_[g], CUDNN_ADD_SAME_C,
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_data + bias_offset_ * g,
              cudnn::dataType<Dtype>::one,
              top_descs_[i], top_data + top_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_conv_groups<<<1, 1>>>();
  }

  ///////////그리기
  //cv::Mat img;
  //img.create(top[0]->shape()[2], top[0]->shape()[3], CV_8UC1);
  //for (int i = 0; i < top[0]->shape()[1]; i++){
	 // int width = top[0]->shape()[2];
	 // int height = top[0]->shape()[3];
	 // Dtype map[117 * 117];
	 // char buf[32];
	 // sprintf(buf, "%d", i);
	 // cudaMemcpy(map, &top[0]->gpu_data()[top[0]->shape()[2] * top[0]->shape()[3] * i], sizeof(Dtype) * top[0]->shape()[2] * top[0]->shape()[3], cudaMemcpyDeviceToHost);

	 // Dtype max = -1;
	 // Dtype min = 9999;
	 // for (int j = 0; j < top[0]->shape()[2] * top[0]->shape()[3]; j++){
		//  if (max < map[j])		max = map[j];
		//  if (min > map[j])		min = map[j];
	 // }


	 // for (int j = 0; j < top[0]->shape()[2] * top[0]->shape()[3]; j++){
		//  img.at<uchar>(j) = (uchar)((map[j] - min)/(max-min) * 255.f);
	 // }
	 // cv::imshow(buf, img);
	 // cv::waitKey(0);
  //}
  //cv::destroyAllWindows();

  //weight 그리기
  //static int frameCount = 0;
  if (visualize_ != 0 /*&& frameCount == 0*/){
	  //printf("Draw weight..\n");
	  visualize_weight(visualize_, this->blobs_[0], 20, 20);
  }
  //frameCount = (frameCount+1)%1000;
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
  }
  Dtype* bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Backward through cuDNN in parallel over groups and gradients.
    for (int g = 0; g < this->group_; g++) {
      // Gradient w.r.t. bias.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              top_descs_[i],  top_diff + top_offset_ * g,
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_diff + bias_offset_ * g));
      }

      // Gradient w.r.t. weights.
      if (this->param_propagate_down_[0]) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(handle_[1*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              bottom_descs_[i], bottom_data + bottom_offset_ * g,
              top_descs_[i],    top_diff + top_offset_ * g,
              conv_descs_[i],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight_diff + weight_offset_ * g));
      }

      // Gradient w.r.t. bottom data.
      if (propagate_down[i]) {
        if (weight == NULL) {
          weight = this->blobs_[0]->gpu_data();
        }
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        CUDNN_CHECK(cudnnConvolutionBackwardData(handle_[2*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight + weight_offset_ * g,
              top_descs_[i], top_diff + top_offset_ * g,
              conv_descs_[i],
              cudnn::dataType<Dtype>::zero,
              bottom_descs_[i], bottom_diff + bottom_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_conv_groups<<<1, 1>>>();
  }
}

//weight 한판에 보여주기 위한 함수
template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::visualize_weight(int windowID, boost::shared_ptr<Blob<Dtype>> weight, int visChannel, int visCount){
	char buf[16];
	sprintf(buf, "weight_%d", windowID);
	
	int tchannels = weight->shape()[1];
	int twidth = weight->shape()[2];
	int theight = weight->shape()[3];
	int tcount = weight->shape()[0];
	
	int drawChannel = tchannels < visChannel ? tchannels : visChannel;
	int drawCount = tcount < visCount ? tcount : visCount;
	int scaleparam = 4;

	cv::Mat weightMap, singleWeight;
	weightMap.create(theight*drawCount, twidth*drawChannel, CV_8UC1);
	singleWeight.create(theight, twidth, CV_8UC1);

	for (int i = 0; i < drawCount; i++){
		for (int c = 0; c < drawChannel; c++){
			Dtype map[9 * 9];
			int mapstartIdx = twidth * theight * i * tchannels + c * twidth * theight;
			//singleMap.create(theight, twidth, CV_8UC1);

			cudaMemcpy(map, &weight->gpu_data()[mapstartIdx], sizeof(Dtype) * twidth * theight, cudaMemcpyDeviceToHost);

			Dtype max = -1;
			Dtype min = 9999;
			for (int j = 0; j < twidth * theight; j++){
				if (max < map[j])		max = map[j];
				if (min > map[j])		min = map[j];
			}
			
			//for (int j = 0; j < twidth * theight; j++)			singleMap.at<uchar>(j) = (uchar)((map[j] - min) / (max - min) * 255.f);
			//cv::resize(singleMap, singleMap, cv::Size(twidth*scaleparam, theight*scaleparam));
			//cv::imshow("test", singleMap);
			//cv::waitKey(0);

			int startX = c*twidth;
			int startY = i*theight;
			for (int j = 0; j < twidth * theight; j++){
				singleWeight.at<uchar>(j) = (uchar)((map[j] - min) / (max - min) * 255.f);
				weightMap.at<uchar>(startY + j / theight, startX + j%twidth) = (uchar)((map[j] - min) / (max - min) * 255.f);
			}

			char sigleBuf[256];
			sprintf(sigleBuf, "weight\\%d\\%d_%d.jpg", windowID, i, c);
			cv::imwrite(sigleBuf, singleWeight);
		}
	}

	int weightMapHeight = weightMap.size().height;
	int weightMapWidth = weightMap.size().width;
	cv::resize(weightMap, weightMap, cv::Size(weightMapWidth*scaleparam, weightMapHeight * scaleparam));
	cv::imshow(buf, weightMap);
	strcat(buf, ".jpg");
	cv::waitKey(0);

	cv::imwrite(buf, weightMap);
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNConvolutionLayer);

}  // namespace caffe
#endif
