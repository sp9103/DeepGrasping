#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include <opencv2\opencv.hpp>

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  this->height_out_ = (this->height_ + 2 * this->pad_h_ - this->kernel_h_)
      / this->stride_h_ + 1;
  this->width_out_ = (this->width_ + 2 * this->pad_w_ - this->kernel_w_)
      / this->stride_w_ + 1;
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + bottom[i]->offset(n), weight,
          top_data + top[i]->offset(n));
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + top[i]->offset(n), bias);
      }
    }
  }

  //Dtype bottomtemp[80 * 80];
  //memcpy(bottomtemp, bottom[0]->cpu_data(), sizeof(Dtype) * 80 * 80 * 3);
  //Dtype botMin = 9999;
  //Dtype botMax = -9999;
  //cv::Mat botImg;
  //botImg.create(80, 80, CV_32FC3);
  ///*for (int i = 0; i < 80 * 80; i++){
	 // if (botMin > bottomtemp[i])		botMin = bottomtemp[i];
	 // if (botMax < bottomtemp[i])		botMax = bottomtemp[i];
  //}*/
  //for (int i = 0; i < 80 * 80; i++){
	 // botImg.at<cv::Vec3f>(i)[0] = bottomtemp[i];
	 // botImg.at<cv::Vec3f>(i)[1] = bottomtemp[i + 80*80];
	 // botImg.at<cv::Vec3f>(i)[2] = bottomtemp[i + 80*80*2];
  //}
  //cv::imshow("botImg", botImg);
  //cv::waitKey(0);

  /*Dtype toptemp[];
  Dtype topmin = 9999;
  Dtype topmax = -9999;
  for (int i = 0; i < ; i++){
	  if (botMin > bottomtemp[i])		botMin = bottomtemp[i];
	  if (botMax < bottomtemp[i])		botMax = bottomtemp[i];
  }*/

  //Dtype topTemp[37 * 37];
  //Dtype topMin = 9999;
  //Dtype topMax = -9999;
  //const int channels = 32;
  //for (int i = 0; i < channels; i++){
	 // memcpy(topTemp, &top[0]->cpu_data()[37 * 37 * i], sizeof(Dtype) * 37 * 37);

	 // for (int j = 0; j < 37 * 37; j++){
		//  if (topMin > topTemp[j])	topMin = topTemp[j];
		//  if (topMax < topTemp[j])	topMax = topTemp[j];
	 // }
  //}

  //Dtype biasTemp[64];
  //memcpy(biasTemp, this->blobs_[1]->cpu_data(), sizeof(Dtype) * channels);
  //Dtype biasMin = 9999;
  //Dtype biasMax = -9999;
  //for (int i = 0; i < channels; i++){
	 // if (biasMin > biasTemp[i])		biasMin = biasTemp[i];
	 // if (biasMax < biasTemp[i])		biasMax = biasTemp[i];
  //}

  //Dtype weightTemp[7*7*1];
  //Dtype TotalMin = 9999;
  //Dtype TotalMax = -9999;
  //for (int i = 0; i < 5; i++){
	 // memcpy(weightTemp, &this->blobs_[0]->cpu_data()[7 * 7 * 1 * i], sizeof(Dtype) * 7 * 7 * 1);

	 // Dtype weightMin = 9999;
	 // Dtype weightMax = -9999;
	 // for (int j = 0; j < 7 * 7 * 1; j++){
		//  if (weightMax < weightTemp[j])	weightMax = weightTemp[j];
		//  if (weightMin > weightTemp[j])	weightMin = weightTemp[j];

		//  if (TotalMax < weightTemp[j])	TotalMax = weightTemp[j];
		//  if (TotalMin > weightTemp[j])	TotalMin = weightTemp[j];
	 // }
  //}

  //int tchannels = this->blobs_[0]->shape()[1];
  //int twidth = this->blobs_[0]->shape()[2];
  //int theight = this->blobs_[0]->shape()[3];
  //int tcount = this->blobs_[0]->shape()[0];

  //int drawChannel = tchannels < 10 ? tchannels : 10;
  //int drawCount = tcount < 10 ? tcount : 10;
  //int scaleparam = 4;

  //cv::Mat weightMap;
  //weightMap.create(theight*drawCount, twidth*drawChannel, CV_8UC1);

  //for (int i = 0; i < drawCount; i++){
	 // for (int c = 0; c < drawChannel; c++){
		//  Dtype map[7 * 7];
		//  cv::Mat singleMap;
		//  int mapstartIdx = twidth * theight * i * tchannels + c * twidth * theight;
		//  //singleMap.create(theight, twidth, CV_8UC1);

		//  memcpy(map, &this->blobs_[0]->cpu_data()[mapstartIdx], sizeof(Dtype) * twidth * theight);

		//  Dtype max = -1;
		//  Dtype min = 9999;
		//  for (int j = 0; j < twidth * theight; j++){
		//	  if (max < map[j])		max = map[j];
		//	  if (min > map[j])		min = map[j];
		//  }

		//  int startX = c*twidth;
		//  int startY = i*theight;
		//  for (int j = 0; j < twidth * theight; j++){
		//	  weightMap.at<uchar>(startY + j / theight, startX + j%twidth) = (uchar)((map[j] - min) / (max - min) * 255.f);
		//  }
	 // }
  //}

  //int weightMapHeight = weightMap.size().height;
  //int weightMapWidth = weightMap.size().width;
  //cv::resize(weightMap, weightMap, cv::Size(weightMapWidth*scaleparam, weightMapHeight * scaleparam));
  //cv::imshow("weight", weightMap);
  //cv::waitKey(0);
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + top[i]->offset(n));
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + bottom[i]->offset(n),
              top_diff + top[i]->offset(n), weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + top[i]->offset(n), weight,
              bottom_diff + bottom[i]->offset(n));
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
