#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include <opencv2\opencv.hpp>

namespace caffe {

//alpha & sigma exponential
template <typename Dtype>
__global__ void sigmaExp(const int nthreads, const int param_size, Dtype* const topdata) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int vecIdx = index % param_size;

		if (vecIdx == 0)						//alpha
			topdata[index] = exp(topdata[index]);
		else if(vecIdx == (param_size - 1))		//sigma
			topdata[index] = exp(topdata[index]);
	}
}

template <typename Dtype>
__global__ void kernel_alpha_max(const int num, const int param_size, const int class_size, const Dtype* data, Dtype* out) {
	CUDA_KERNEL_LOOP(index, num) {
		Dtype maxval = -FLT_MAX;
		for (int i = 0; i < class_size; i++)
			maxval = max(data[(index*class_size*param_size) + i*param_size], maxval);

		out[index] = maxval;
	}
}

template <typename Dtype>
__global__ void kernel_alpha_subtract(const int count,
	const int param_size, const int class_size,
	const Dtype* max, Dtype* data) {
	CUDA_KERNEL_LOOP(index, count) {
		int n = index / class_size;
		data[index*param_size] -= max[n];
	}
}

template <typename Dtype>
__global__ void kernel_alpha_sum(const int num, const int param_size, const int class_size, const Dtype* data, Dtype* out) {
	CUDA_KERNEL_LOOP(index, num) {
		Dtype sum = 0;
		for (int i = 0; i < class_size; i++)
			sum += data[index*class_size*param_size + i*param_size];
		out[index] = sum;
	}
}

template <typename Dtype>
__global__ void kernel_alpha_div(const int count,
	const int param_size, const int class_size,
	const Dtype* sum, Dtype* data) {
	CUDA_KERNEL_LOOP(index, count) {
		int n = index / class_size;
		data[index*param_size] /= sum[n];
	}
}

template <typename Dtype>
void GMMLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const int datacount = top[0]->count();
  const int batchsize = bottom[0]->shape()[0];
  if (M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weight, bottom_data, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            this->blobs_[1]->gpu_data(), top_data);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
                          bottom_data, weight, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }

  //inner product 이후 Gaussian mixture parameter calculate
  //0: alpha, 1~x : mu, x+1 : sigma 

  //find alpha max
  kernel_alpha_max<Dtype> << <CAFFE_GET_BLOCKS(batchsize), CAFFE_CUDA_NUM_THREADS >> >(batchsize, data_dim+2, class_size, top_data, maxValue_.mutable_gpu_data());

  //sub alpha max
  kernel_alpha_subtract<Dtype> << <CAFFE_GET_BLOCKS(class_size * batchsize), CAFFE_CUDA_NUM_THREADS >> >(class_size * batchsize, data_dim + 2, class_size, maxValue_.gpu_data(), top_data);

  //exponential - sigma에 exp를 취하는 것이 문제가 될 수 있음. overflow. ( alpha는 위에서 sub max를 해줌으로 overflow 예방) ==> 문제가 있을 경우 1/sigma 를 산출 ==> 1/sigma를 리턴
  sigmaExp<Dtype> << <CAFFE_GET_BLOCKS(datacount), CAFFE_CUDA_NUM_THREADS >> >(datacount, data_dim+2, top_data);

  //sum alpha
  kernel_alpha_sum<Dtype> << <CAFFE_GET_BLOCKS(batchsize), CAFFE_CUDA_NUM_THREADS >> >(batchsize, data_dim + 2, class_size, top_data, maxValue_.mutable_gpu_data());

  //div alpha
  kernel_alpha_div<Dtype> << <CAFFE_GET_BLOCKS(class_size * batchsize), CAFFE_CUDA_NUM_THREADS >> >(class_size * batchsize, data_dim + 2, class_size, maxValue_.gpu_data(), top_data);
}

template <typename Dtype>
void GMMLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();

    // Gradient with respect to weight
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
        top_diff, bottom_data, (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, this->blobs_[0]->gpu_data(), (Dtype)0.,
        bottom[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(GMMLayer);

}  // namespace caffe
