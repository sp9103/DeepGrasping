#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include <opencv2\opencv.hpp>

namespace caffe {

template <typename Dtype>
void MDNLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	//MDN에서 bottom과 label의 크기가 꼭 같은 필요 없음
  /*CHECK_EQ(bottom[0]->num(), bottom[1]->num())
	  << "The data and label should have the same number.";*/
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
  
}

template <typename Dtype>
void MDNLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //CPU version not implemented
}

template <typename Dtype>
void MDNLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	//CPU version not implemented
}

#ifdef CPU_ONLY
STUB_GPU(MDNLossLayer);
#endif

INSTANTIATE_CLASS(MDNLossLayer);
REGISTER_LAYER_CLASS(MDNLoss);

}  // namespace caffe
