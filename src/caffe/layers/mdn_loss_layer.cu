#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include <opencv2\opencv.hpp>

namespace caffe {

template <typename Dtype>							//mu_ik - tk calculation
__global__ void kernel_label_subtract(const int count,
	const int param_size, const int class_size, const int data_dim,
	const Dtype* data, const Dtype* label, Dtype* diff) {
	CUDA_KERNEL_LOOP(index, count) {
		int internal_idx = index % data_dim;
		int outer_idx = index / data_dim;
		int label_idx = index / (class_size * data_dim);
		diff[index] = data[outer_idx * param_size + internal_idx + 1] - label[label_idx * data_dim + internal_idx];
	}
}

template <typename Dtype>
void MDNLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	const Dtype* bottom_data = bottom[0]->gpu_data();
	const Dtype* label = bottom[1]->gpu_data();

	//subtract (mu - t)
	kernel_label_subtract<Dtype> << <CAFFE_GET_BLOCKS(data_dim*class_size), CAFFE_CUDA_NUM_THREADS >> >(data_dim*class_size,
		data_dim + 2, class_size, data_dim, bottom_data, label, diff_.mutable_gpu_data());

	//
}

//Diff 0번지는 값있고 1번지는 없음
template <typename Dtype>
void MDNLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //for (int i = 0; i < 2; ++i) {
	 // if (propagate_down[i]) {
		//  const Dtype sign = (i == 0) ? 1 : -1;
		//  const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
		//  //const Dtype alpha = 1.0 / bottom[i]->num();
		//  Dtype tt = top[0]->cpu_diff()[0];
		//  caffe_gpu_axpby(
		//	  bottom[i]->count(),              // count
		//	  alpha,                              // alpha
		//	  diff_.gpu_data(),                   // a			//차이값이 diff_에 저장됨
		//	  Dtype(0),                           // beta
		//	  bottom[i]->mutable_gpu_diff());  // b
	 // }
  //}
}

INSTANTIATE_LAYER_GPU_FUNCS(MDNLossLayer);

}  // namespace caffe
