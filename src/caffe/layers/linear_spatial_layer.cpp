#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void LinearSpatialLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //const int num_output = this->layer_param_.spatial_param().num_output();
	const int num_output = bottom[0]->shape()[1]*2;
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
	  this->layer_param_.linear_spatial_param().axis());

  K_ = bottom[0]->count(axis);
  threshold_ = this->layer_param_.linear_spatial_param().threshold();
}

template <typename Dtype>
void LinearSpatialLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
	  this->layer_param_.spatial_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);

  //max result blob allocation
  vector<int> max_dims = bottom[0]->shape();
  max_dims[2] = max_dims[3] = 1;
  thresholdVal_.Reshape(max_dims);
  sum_.Reshape(max_dims);

  //softmax result blob allocation
  vector<int> scale_dims = bottom[0]->shape();
  linearOutput_.Reshape(scale_dims);
}

template <typename Dtype>
void LinearSpatialLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();

  //CPU Version not implemented
}

template <typename Dtype>
void LinearSpatialLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0]) {			//pooling layer¿¡¼­ µû¿È
		return;
	}

	//CPU Version not implemented
}

#ifdef CPU_ONLY
STUB_GPU(LinearSpatiallayer);
#endif

INSTANTIATE_CLASS(LinearSpatialLayer);
REGISTER_LAYER_CLASS(LinearSpatial);

}  // namespace caffe
