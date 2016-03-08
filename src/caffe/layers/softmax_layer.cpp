#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  top[0]->ReshapeLike(*bottom[0]);
  vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_));
  sum_multiplier_.Reshape(mult_dims);
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  vector<int> scale_dims = bottom[0]->shape();
  scale_dims[softmax_axis_] = 1;
  scale_.Reshape(scale_dims);
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();				//109*109짜리 scale_data?
  int channels = bottom[0]->shape(softmax_axis_);				//feature map 갯수
  int dim = bottom[0]->count() / outer_num_;						//한 배치의 인덱스 갯수 (109*109*60*32 / 60)
  caffe_copy(bottom[0]->count(), bottom_data, top_data);			//일단 바텀 데이터를 탑 데이터에 복사
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  for (int i = 0; i < outer_num_; ++i) {						//배치사이즈만큼 순회 outer_num_ : 배치 사이즈
    // initialize scale_data to the first plane
    caffe_copy(inner_num_, bottom_data + i * dim, scale_data);	//bottom_data + i *dim : i번째 배치의 시작 주소, 	
    for (int j = 0; j < channels; j++) {
      for (int k = 0; k < inner_num_; k++) {					//inner_num : 109*109
        scale_data[k] = std::max(scale_data[k],
            bottom_data[i * dim + j * inner_num_ + k]);			//채널 맥시멈을 갱신함.
      }
    }
    // subtraction - sum_multiplier 데이터는 모두 1로 셋팅되어있음
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_,
        1, -1., sum_multiplier_.cpu_data(), scale_data, 1., top_data);
    // exponentiation - 모든 데이터에 exponential
    caffe_exp<Dtype>(dim, top_data, top_data);
    // sum after exp - exp summation 나눠주기 위한 인자 생성
    caffe_cpu_gemv<Dtype>(CblasTrans, channels, inner_num_, 1.,
        top_data, sum_multiplier_.cpu_data(), 0., scale_data);
    // division
    for (int j = 0; j < channels; j++) {
      caffe_div(inner_num_, top_data, scale_data, top_data);
      top_data += inner_num_;									//다음 채널을 보겠다는 의미
    }
  }
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int channels = top[0]->shape(softmax_axis_);
  int dim = top[0]->count() / outer_num_;						//outer_num : 60(batch size), dim : 109*109*32(한 배치의 사이즈)
  caffe_copy(top[0]->count(), top_diff, bottom_diff);			//먼저 top_diff를 bottom_diff로 복사 - 차원이 같으니까
  for (int i = 0; i < outer_num_; ++i) {
    // compute dot(top_diff, top_data) and subtract them from the bottom diff
    for (int k = 0; k < inner_num_; ++k) {						//inner_num : 109*109;
      scale_data[k] = caffe_cpu_strided_dot<Dtype>(channels,	//dot product의 결과를 scale_data에 저장
          bottom_diff + i * dim + k, inner_num_,				//top_diff*top_data랑 dot product 
          top_data + i * dim + k, inner_num_);					//stride : bottom_diff + i * dim + k & top_data + i * dim + k
    }
    // subtraction - sum_multipilier는 모두 1로 셋팅
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1,
        -1., sum_multiplier_.cpu_data(), scale_data, 1., bottom_diff + i * dim);			//bottom_diff에서 dot product결과를 빼냄?
  }
  // elementwise multiplication
  caffe_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);		//top data와 bottom_diff를 elementwise로 곱함
}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxLayer);
#endif

INSTANTIATE_CLASS(SoftmaxLayer);

}  // namespace caffe
