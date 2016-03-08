#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include <opencv2\opencv.hpp>

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;

  ////if (!std::isnan(loss)){

  //////////////////////////////////////////////
  //cv::Mat lossLayer;
  //const int labelwidth = 40;
  //const int labelheight = 40;
  //lossLayer.create(labelheight, labelwidth * 3, CV_32FC1);

  //////////////label & output Ãâ·Â
  //int cCount = bottom[0]->num() < 10 ? bottom[0]->num() : 10;
  //for (int c = 0; c < cCount; c++){
	 // char buf[32];
	 // Dtype labelarr[1600], outputarr[1600], diffarr[1600];
	 // memcpy(labelarr, &bottom[1]->cpu_data()[1600 * c], sizeof(Dtype) * 1600);
	 // memcpy(outputarr, &bottom[0]->cpu_data()[1600 * c], sizeof(Dtype) * 1600);
	 // memcpy(diffarr, &diff_.cpu_data()[1600 * c], sizeof(Dtype) * 1600);

	 // for (int j = 0; j < labelheight * labelwidth; j++){
		//  lossLayer.at<float>(j / labelwidth, j % labelwidth) = (float)labelarr[j];
		//  lossLayer.at<float>(j / labelwidth, j % labelwidth + labelwidth) = (float)outputarr[j];
	 // }

	 // Dtype max = -9999;
	 // Dtype min = 9999;

	 // for (int j = 0; j < labelheight * labelwidth; j++){
		//  if (max < diffarr[j])		max = diffarr[j];
		//  if (min > diffarr[j])		min = diffarr[j];
	 // }

	 // for (int j = 0; j < labelheight * labelwidth; j++)
		//  lossLayer.at<float>(j / labelwidth, j % labelwidth + labelwidth * 2) = (float)((diffarr[j] - min) / (max - min));

	 // sprintf(buf, "Loss_%d", c);
	 // cv::imshow(buf, lossLayer);
	 // cv::waitKey(0);
  //}
  ////}
  //cv::destroyAllWindows();
  //////////////////////////////////////////////////////
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanLossLayer);
REGISTER_LAYER_CLASS(EuclideanLoss);

}  // namespace caffe
