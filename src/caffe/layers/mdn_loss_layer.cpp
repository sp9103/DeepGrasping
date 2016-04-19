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
  
  data_dim = this->layer_param_.gmm_param().data_dim();
  class_size = this->layer_param_.gmm_param().class_size();

  vector<int> diff_shape(3);		//[batchsize, class_size, data_dim] matrix
  diff_shape[0] = bottom[0]->shape()[0];
  diff_shape[1] = class_size;
  diff_shape[2] = data_dim;
  diff_.Reshape(diff_shape);
  diff_square_.Reshape(diff_shape);

  vector<int> class_dim(2);
  class_dim[0] = bottom[0]->shape()[0];
  class_dim[1] = class_size;
  diff_norm_.Reshape(class_dim);
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
