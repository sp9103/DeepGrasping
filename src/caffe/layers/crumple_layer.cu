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
void CrumpleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();

	caffe_copy(bottom[0]->count(), bottom_data, top_data);

	//제대로 복사됬나 결과 꼭 확인해보기!!
}

template <typename Dtype>
void CrumpleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	const Dtype* top_diff = top[0]->gpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

	caffe_copy(bottom[0]->count(), top_diff, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(CrumpleLayer);

}  // namespace caffe
