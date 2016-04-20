#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include <opencv2\opencv.hpp>

#define MATH_PI		3.14159265

namespace caffe {

template <typename Dtype>							//mu_ik - tk calculation
__global__ void kernel_label_subtract(const int count,
	const int param_size, const int class_size, const int data_dim,
	const Dtype* data, const Dtype* label, Dtype* diff) {
	CUDA_KERNEL_LOOP(index, count) {
		int internal_idx = index % data_dim;					//mu vector���� ���° �ε���
		int outer_idx = index / data_dim;						//���° Ŭ����
		int label_idx = index / (class_size * data_dim);		//���° label
		diff[index] = data[outer_idx * param_size + internal_idx + 1] - label[label_idx * data_dim + internal_idx];
	}
}

template <typename Dtype>							// || mu-t || ^ 2
__global__ void kernel_diff_norm(const int count,
	const int class_size, const int data_dim,
	const Dtype* diff_squre, Dtype* norm) {
	CUDA_KERNEL_LOOP(index, count) {
		Dtype sum = 0;
		for (int i = 0; i < data_dim; i++)
			sum += diff_squre[index * data_dim + i];
		norm[index] = sum;
	}
}

template <typename Dtype>							// alpha * gaussian distribution ���
__global__ void kernel_normal_distribution(const int count,
	const int param_size, const int class_size, const int data_dim,
	const Dtype* norm, const Dtype* data, Dtype* alpha_distribution) {
	CUDA_KERNEL_LOOP(index, count) {
		Dtype alpha = data[index*param_size];
		Dtype sigma_inverse = data[index*param_size + 1 + data_dim];
		Dtype exp_gaussian = exp(- norm[index] * sigma_inverse * sigma_inverse / 2);
		Dtype distribution = pow(sigma_inverse, data_dim) / pow(2 * MATH_PI, data_dim / 2) * exp_gaussian;
		//alpha * gaussian_distribution;
		alpha_distribution[index] = alpha * distribution;
	}
}

template <typename Dtype>							// ��(alpha * gaussian distribution) ���
__global__ void kernel_class_summation(const int count, const int class_size,
	const Dtype* alpha_pi_, Dtype* alpha_pi_sum_) {
	CUDA_KERNEL_LOOP(index, count) {
		Dtype sum = 0;
		const int batch_idx = index / class_size;
		for (int i = 0; i < class_size; i++)
			sum += alpha_pi_[batch_idx * class_size + i];
		alpha_pi_sum_[index] = sum;
	}
}

template <typename Dtype>
void MDNLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	const Dtype* bottom_data = bottom[0]->gpu_data();
	const Dtype* label = bottom[1]->gpu_data();
	const int batch_size = bottom[0]->shape()[0];

	//subtract (mu - t)
	kernel_label_subtract<Dtype> << <CAFFE_GET_BLOCKS(data_dim*class_size), CAFFE_CUDA_NUM_THREADS >> >(data_dim*class_size,
		data_dim + 2, class_size, data_dim, bottom_data, label, diff_.mutable_gpu_data());

	//square ( mu - t )^2
	caffe_gpu_mul(diff_.count(), diff_.gpu_data(), diff_.gpu_data(), diff_square_.mutable_gpu_data());

	//norm  : || mu-t || ^ 2
	kernel_diff_norm<Dtype> << <CAFFE_GET_BLOCKS(class_size * batch_size), CAFFE_CUDA_NUM_THREADS >> >(class_size * batch_size,
		class_size, data_dim, diff_square_.gpu_data(), diff_norm_.mutable_gpu_data());

	//calculate gaussian distribution
	kernel_normal_distribution<Dtype> << <CAFFE_GET_BLOCKS(class_size * batch_size), CAFFE_CUDA_NUM_THREADS >> >(class_size * batch_size,
		data_dim + 2, class_size, data_dim,
		diff_norm_.gpu_data(), bottom_data, alpha_pi_.mutable_gpu_data());

	//sumation : ��(alpha * distribution)
	kernel_class_summation<Dtype> << <CAFFE_GET_BLOCKS(batch_size), CAFFE_CUDA_NUM_THREADS >> >(batch_size, class_size, alpha_pi_.gpu_data(), alpha_pi_sum_.mutable_gpu_data());

	//loss : ln ( sumation ) / number of batchsize
	Dtype loss;
	caffe_gpu_log(alpha_pi_sum_.count(), alpha_pi_sum_.gpu_data(), batch_loss_.mutable_gpu_data());
	caffe_gpu_dot(batch_loss_.count(), batch_loss_.gpu_data(), sum_multiplier_.gpu_data(), &loss);
	loss /= bottom[0]->num();
	top[0]->mutable_cpu_data()[0] = loss;
}

//Diff 0������ ���ְ� 1������ ����
template <typename Dtype>
void MDNLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

INSTANTIATE_LAYER_GPU_FUNCS(MDNLossLayer);

}  // namespace caffe
