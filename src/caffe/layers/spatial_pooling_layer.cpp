#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SpatialPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //const int num_output = this->layer_param_.spatial_param().num_output();
	const int num_output = bottom[0]->shape()[1]*2;
  N_ = num_output;
}

template <typename Dtype>
void SpatialPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  const int axis = 1;
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);

  //max result blob allocation
  vector<int> max_dims = bottom[0]->shape();
  max_dims[2] = max_dims[3] = 1;
  index_.Reshape(max_dims);
}

template <typename Dtype>
void SpatialPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void SpatialPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

#ifdef CPU_ONLY
STUB_GPU(SpatialPoolingLayer);
#endif

INSTANTIATE_CLASS(SpatialPoolingLayer);
REGISTER_LAYER_CLASS(SpatialPooling);

}  // namespace caffe
