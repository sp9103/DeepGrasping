#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include <opencv2\opencv.hpp>

namespace caffe {

template <typename Dtype>
void MDNLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //int count = bottom[0]->count();
  //caffe_gpu_sub(
  //    count,
  //    bottom[0]->gpu_data(),
  //    bottom[1]->gpu_data(),
  //    diff_.mutable_gpu_data());
  //Dtype dot;
  //caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  //Dtype loss = dot / bottom[0]->num() / Dtype(2);
  //top[0]->mutable_cpu_data()[0] = loss;
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
