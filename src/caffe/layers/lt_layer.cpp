#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void LTLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const int cols = bottom[0]->shape()[1];
	CHECK_EQ(cols, 3)
		<< "bottom input & top output must 3";

	vector<int> top_shape = bottom[0]->shape();
	top[0]->Reshape(top_shape);

	N_ = 3;
	K_ = 3;
	M_ = bottom[0]->count(0, 1);

	//Linear transform matrix allocation
	vector<int> weight_shape(2);
	weight_shape[0] = N_;
	weight_shape[1] = K_;
	R.Reshape(weight_shape);
	vector<int> bias_shape(1, N_);
	T.Reshape(bias_shape);

	vector<int> bias_mul_shape(1, M_);
	bias_multiplier_.Reshape(bias_shape);
	caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());

	//Matrix setting
	Dtype RMat[9] = { -20.683149, 957.240906, 290.486237,
					963.990662, -9.238551, 198.420578,
					201.575638, 238.804276, -907.819397 }; 
	Dtype TMat[3] = { 54.133713, -222.452713, 879.145020 };

	caffe_copy(9, RMat, R.mutable_cpu_data());
	caffe_copy(3, TMat, T.mutable_cpu_data());
}

template <typename Dtype>
void LTLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void LTLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	//CPU version not implemented
}

template <typename Dtype>
void LTLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	//CPU version not implemented
}

#ifdef CPU_ONLY
STUB_GPU(CrumpleLayer);
#endif

INSTANTIATE_CLASS(LTLayer);
REGISTER_LAYER_CLASS(LT);

}  // namespace caffe
