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
__global__ void kernel_linear_transform(const int num, const int count,
	const int batchsize, const Dtype* rot, const Dtype* tran,
	const Dtype* data, Dtype* out) {
	CUDA_KERNEL_LOOP(index, num) {
		int id = index % count;
		int batchid = index / count;
		if (id == 0)
			out[index] = data[batchid * 3 + 0] * rot[0] + data[batchid * 3 + 1] * rot[1] + data[batchid * 3 + 2] * rot[2] + tran[0];
		else if (id == 1)
			out[index] = data[batchid * 3 + 0] * rot[3] + data[batchid * 3 + 1] * rot[4] + data[batchid * 3 + 2] * rot[5] + tran[1];
		else if (id == 2)
			out[index] = data[batchid * 3 + 0] * rot[6] + data[batchid * 3 + 1] * rot[7] + data[batchid * 3 + 2] * rot[8] + tran[2];
	}
}

template <typename Dtype>
void LTLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	const Dtype* RMat = R.gpu_data();
	const Dtype* TMat = T.gpu_data();
	const int bottom_count = bottom[0]->count();
	
	kernel_linear_transform<Dtype> << <CAFFE_GET_BLOCKS(bottom_count),
		CAFFE_CUDA_NUM_THREADS >> >(bottom_count, N_, M_,
		RMat, TMat, bottom_data, top_data);
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
