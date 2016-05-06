#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanDistLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.accuracy_param().top_k();
}

template <typename Dtype>
void EuclideanDistLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
  //    << "top_k must be less than or equal to the number of classes.";
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  //CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
  //    << "Number of labels must match number of predictions; "
  //    << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
  //    << "label count (number of labels) must be N*H*W, "
  //    << "with integer values in {0, 1, ..., C-1}.";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);


  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EuclideanDistLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
	  count,
	  bottom[0]->cpu_data(),
	  bottom[1]->cpu_data(),
	  diff_.mutable_cpu_data());

  int batchsize = bottom[0]->shape()[0];
  int botLen = bottom[0]->shape()[1];
  Dtype TotalLeftDist, TotalRightDist, TotalThumbDist;
  const Dtype* diffdata = diff_.cpu_data();
  const Dtype* bot = bottom[1]->cpu_data();
  const Dtype* output = bottom[0]->cpu_data();
  TotalLeftDist = TotalRightDist = TotalThumbDist = 0;

  for (int i = 0; i < batchsize; i++){
	  float sq = 0;
	  Dtype LeftDist, RightDist, ThumbDist;
	  LeftDist = RightDist = ThumbDist = 0;
	  for (int k = 0; k < 3; k++){
		  LeftDist += pow(diffdata[9 * i + 3 * 0 + k], 2);
		  RightDist += pow(diffdata[9 * i + 3 * 1 + k], 2);
		  ThumbDist += pow(diffdata[9 * i + 3 * 2 + k], 2);
	  }
	  TotalLeftDist += LeftDist / batchsize;
	  TotalRightDist += RightDist / batchsize;
	  TotalThumbDist += ThumbDist / batchsize;
  }

  top[0]->mutable_cpu_data()[0] = (TotalLeftDist + TotalRightDist + TotalThumbDist);
   LOG(INFO) << "Left: " << (TotalLeftDist * 10);
   LOG(INFO) << "Right: " << (TotalRightDist * 10);
   LOG(INFO) << "Thumb: " << (TotalThumbDist * 10);

}

INSTANTIATE_CLASS(EuclideanDistLayer);
REGISTER_LAYER_CLASS(EuclideanDist);

}  // namespace caffe
