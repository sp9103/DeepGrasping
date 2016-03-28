#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include <opencv2\opencv.hpp>

namespace caffe {

template <typename Dtype>
void LTLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	const Dtype* RMat = R.gpu_data();
	const Dtype* TMat = T.gpu_data();
	if (M_ == 1) {
		caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
			RMat, bottom_data, (Dtype)0., top_data);
		caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
			TMat, top_data);
	}
	else {
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
			bottom_data, RMat, (Dtype)0., top_data);
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
			bias_multiplier_.gpu_data(),
			TMat, (Dtype)1., top_data);
	}
}

template <typename Dtype>
void LTLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	const Dtype* top_diff = top[0]->gpu_diff();
	// Gradient with respect to bottom data
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
		top_diff, this->blobs_[0]->gpu_data(), (Dtype)0.,
		bottom[0]->mutable_gpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(LTLayer);

}  // namespace caffe
