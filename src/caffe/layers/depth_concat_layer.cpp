#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

	template <typename Dtype>
	void DepthConcatLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		//const int num_output = this->layer_param_.spatial_param().num_output();
		const int num_output = bottom[1]->shape()[1] * 3 / 2;
		N_ = num_output;
		const int axis = bottom[0]->CanonicalAxisIndex(
			this->layer_param_.spatial_param().axis());

		K_ = bottom[0]->count(axis);
	}

	template <typename Dtype>
	void DepthConcatLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
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
	}

	template <typename Dtype>
	void DepthConcatLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		//not implemented
	}

	template <typename Dtype>
	void DepthConcatLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		//not implemented
	}

#ifdef CPU_ONLY
	STUB_GPU(SpatialLayer);
#endif

	INSTANTIATE_CLASS(DepthConcatLayer);
	REGISTER_LAYER_CLASS(DepthConcat);

}  // namespace caffe
