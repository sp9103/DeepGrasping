#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void CrumpleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	//mode -> mode 1 : shrink, mode 2 : grow
	Mode_ = this->layer_param_.crumple_param().mode();

	//Layer output size
	const int num_output = this->layer_param_.crumple_param().size();
	N_ = num_output;

	if (Mode_ == 1){
		const int cols = bottom[0]->shape()[1] % num_output;
		//EQ 체크가 맞는지 추후 확인 => 한 vector가 주어진 차원으로 나눠지는지를 확인
		CHECK_EQ(cols, 0)
			<< "Bottom size cannot divided by output";
	}
	else if (Mode_ == 2){
		//check TO-DO
		const int other = bottom[0]->count() % num_output;
		//전체 데이터 갯수를 주어진 차원으로 재정렬할수있는지 확인
		CHECK_EQ(other, 0)
			<< "Bottom size cannot divided by output";
	}
}

template <typename Dtype>
void CrumpleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
	float rows = (float)bottom[0]->count() / (float)N_;
  vector<int> top_shape = bottom[0]->shape();
  top_shape[0] = (int)rows;
  top_shape[1] = N_;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void CrumpleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	//CPU version not implemented

}

template <typename Dtype>
void CrumpleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	//CPU version not implemented
}

#ifdef CPU_ONLY
STUB_GPU(CrumpleLayer);
#endif

INSTANTIATE_CLASS(CrumpleLayer);
REGISTER_LAYER_CLASS(Crumple);

}  // namespace caffe
