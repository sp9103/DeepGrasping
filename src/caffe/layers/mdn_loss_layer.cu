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
		int internal_idx = index % data_dim;					//mu vector에서 몇번째 인덱스
		int outer_idx = index / data_dim;						//몇번째 클래스
		int label_idx = index / (class_size * data_dim);		//몇번째 label == 몇번째 batch
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

template <typename Dtype>							// alpha * gaussian distribution 계산
__global__ void kernel_normal_distribution(const int count,
	const int param_size, const int class_size, const int data_dim,
	const Dtype* norm, const Dtype* data, Dtype* alpha_distribution) {
	CUDA_KERNEL_LOOP(index, count) {
		Dtype alpha = data[index*param_size];
		Dtype sigma = data[index*param_size + 1 + data_dim];
		Dtype sigma_9 = pow(sigma, data_dim);
		Dtype exp_gaussian = exp(- norm[index] / sigma / sigma / 2);
		Dtype distribution = 0;
		if (exp_gaussian != 0)
			distribution = exp_gaussian / pow(sigma, data_dim) / pow(2 * MATH_PI, data_dim / 2);
		//alpha * gaussian_distribution;
		alpha_distribution[index] = alpha * distribution;
	}
}

template <typename Dtype>							// ∑(alpha * gaussian distribution) 계산
__global__ void kernel_class_summation(const int count, const int class_size,
	const Dtype* alpha_pi_, Dtype* alpha_pi_sum_) {
	CUDA_KERNEL_LOOP(index, count) {
		Dtype sum = 0;
		for (int i = 0; i < class_size; i++)
			sum += alpha_pi_[index * class_size + i];
		if (sum != 0)
			alpha_pi_sum_[index] = sum;
	}
}

template <typename Dtype>							// posterior calculation 
__global__ void kernel_posterior_calc(const int count,
	const int batch_size, const int class_size,
	const Dtype* alpha_pi_, const Dtype* alpha_pi_sum_, Dtype* posterior) {
	CUDA_KERNEL_LOOP(index, count) {
		const int batch_idx = index / class_size;
		posterior[index] = alpha_pi_[index] / alpha_pi_sum_[batch_idx];
	}
}

template <typename Dtype>							// backpropagation delta calculation 
__global__ void kernel_delta_calc(const int count,
	const int batch_size, const int class_size, const int param_size, const int data_dim, 
 	const Dtype* posterior, const Dtype* diff, const Dtype* diff_norm, const Dtype* bottom_data, Dtype* bottom_diff) {
	CUDA_KERNEL_LOOP(index, count) {
 		const int internal_idx = index % param_size;
		const int class_idx = index / param_size;
		const Dtype sigma = bottom_data[class_idx*param_size + param_size - 1];
		if (internal_idx == 0){							//alpha delta calculate
			bottom_diff[index] = bottom_data[index] - posterior[class_idx];
		}
		else if (internal_idx == param_size - 1){		//sigma delta calculate
			bottom_diff[index] = -posterior[class_idx] * (diff_norm[class_idx] / sigma / sigma - data_dim);
		}
		else{											//mu delta calculate
			const int data_idx = internal_idx - 1;		//[0, datadim-1]
			Dtype diff_ik = diff[data_dim * class_idx + data_idx];
			bottom_diff[index] = posterior[class_idx] * (diff_ik / sigma / sigma);
		}
	}
}

template <typename Dtype>
void MDNLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	const Dtype* bottom_data = bottom[0]->gpu_data();
	const Dtype* label = bottom[1]->gpu_data();
	const int batch_size = bottom[0]->shape()[0];

	//subtract (mu - t)
	kernel_label_subtract<Dtype> << <CAFFE_GET_BLOCKS(diff_.count()), CAFFE_CUDA_NUM_THREADS >> >(diff_.count(),
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

	//sumation : ∑(alpha * distribution)
	kernel_class_summation<Dtype> << <CAFFE_GET_BLOCKS(batch_size), CAFFE_CUDA_NUM_THREADS >> >(batch_size, class_size, alpha_pi_.gpu_data(), alpha_pi_sum_.mutable_gpu_data());

	//loss : ln ( sumation ) / number of batchsize
	Dtype loss;
	caffe_gpu_log(alpha_pi_sum_.count(), alpha_pi_sum_.gpu_data(), batch_loss_.mutable_gpu_data());
	caffe_gpu_dot(batch_loss_.count(), batch_loss_.gpu_data(), sum_multiplier_.gpu_data(), &loss);
	loss /= bottom[0]->num();
	top[0]->mutable_cpu_data()[0] = -loss;

	if (visualize_){
		//write file
		FILE *fp = fopen("loss_output.txt", "w");

		Dtype *out = new Dtype[class_size * 11];
		Dtype label_out[9];
		cudaMemcpy(out, bottom[0]->gpu_data(), sizeof(Dtype) * class_size * 11, cudaMemcpyDeviceToHost);
		cudaMemcpy(label_out, bottom[1]->gpu_data(), sizeof(Dtype) * 9, cudaMemcpyDeviceToHost);

		for (int i = 0; i < 9; i++)
			fprintf(fp, "%f ", label_out[i]);
		fprintf(fp, "\n");

		for (int i = 0; i < class_size * 11; i++)
			fprintf(fp, "%f ", out[i]);

		delete[] out;

		fclose(fp);
	}

	if (std::isnan(loss) || std::isinf(loss)){
		printf("loss invalid value.\n");

		Dtype norm_box[5];
		Dtype diff_box[60], diff_squre_box[60];
		Dtype bot_box[70], label_box[12];
		Dtype dist_box[5];
		Dtype sub;
		Dtype norm;
		Dtype alpha_pi_sum__box, alpha_pi_sum__box_temp;
		for (int i = 0; i < batch_size; i++){
			cudaMemcpy(diff_box, &diff_.gpu_data()[i * 60], sizeof(Dtype) * 60, cudaMemcpyDeviceToHost);
			cudaMemcpy(label_box, &label[i * 12], sizeof(Dtype) * 12, cudaMemcpyDeviceToHost);
			cudaMemcpy(bot_box, &bottom_data[70 * i], sizeof(Dtype) * 70, cudaMemcpyDeviceToHost);
			cudaMemcpy(diff_squre_box, &diff_square_.gpu_data()[i * 60], sizeof(Dtype) * 60, cudaMemcpyDeviceToHost);
			cudaMemcpy(norm_box, &diff_norm_.gpu_data()[i * 5], sizeof(Dtype) * 5, cudaMemcpyDeviceToHost);
			cudaMemcpy(dist_box, &alpha_pi_.gpu_data()[i * 5], sizeof(Dtype) * 5, cudaMemcpyDeviceToHost);
			cudaMemcpy(&alpha_pi_sum__box, &alpha_pi_sum_.gpu_data()[i], sizeof(Dtype), cudaMemcpyDeviceToHost);

			for (int j = 0; j < 70; j++)
				if (std::isnan(bot_box[j]) || std::isinf(bot_box[j]))
					printf("bottom data overflow.\n");
			//for (int j = 0; j < 45; j++){
			//	int tClass_idx = j / 9;
			//	int internal_idx = j % 9;
			//	sub = bot_box[tClass_idx * 11 + internal_idx + 1] - label_box[j % 9];

			//	if (diff_box[j] != sub){
			//		printf("diff miss\n");
			//	}

			//	if (diff_squre_box[j] != (sub*sub))
			//		printf("square miss\n");

			//	if (std::isnan(diff_box[j]) || std::isinf(diff_box[j]))
			//		printf("diff_box data overflow.\n");

			//}

			for (int j = 0; j < 5; j++){
				norm = 0;
				for (int k = 0; k < 12; k++)
					norm += diff_squre_box[j * 12 + k];
				if (norm != norm_box[j])
					printf("norm error\n");
																																	
			}
			for (int j = 0; j < 5; j++){
				Dtype alpha = bot_box[14 * j];
				Dtype sigma = bot_box[14 * j + 13];
				float exp_gaussian = exp(-norm_box[j] / sigma / sigma / 2);
				float sigma_9 = pow(sigma, 9);
				float pi_squre = pow(2 * MATH_PI, -9 / 2);
				float gaussian = 0;
				if(exp_gaussian !=0)
					gaussian = exp_gaussian / sigma_9 / pi_squre;
				Dtype dist_temp = alpha * gaussian;
				if (std::isnan(dist_box[j]) || std::isinf(dist_box[j]) || dist_box[j] < 0)
					printf("norm_box data overflow.\n");

			}

			alpha_pi_sum__box_temp = 0;
			for (int j = 0; j < 10; j++){
				alpha_pi_sum__box_temp += dist_box[j];

			}
			if (alpha_pi_sum__box_temp != alpha_pi_sum__box)
				printf("summation error");
		}
	}
}

//Diff 0번지는 값있고 1번지는 없음
template <typename Dtype>
void MDNLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	for (int i = 0; i < 2; ++i) {
		if (propagate_down[i]) {
			// i == 0 : bottom network i == 1 : label
			//부호 +- 다시 한번 생각해보기
			Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
			const Dtype* bottom_data = bottom[i]->gpu_data();
			const int batch_size = bottom[0]->shape()[0];
			
			//calculate posterior probability ( alpha*pi / sumation ( alpha_i * pi_i )
			kernel_posterior_calc<Dtype> << <CAFFE_GET_BLOCKS(batch_size*class_size), CAFFE_CUDA_NUM_THREADS >> >
				(batch_size*class_size, batch_size, class_size, 
				alpha_pi_.gpu_data(), alpha_pi_sum_.gpu_data(), posterior_pi_.mutable_gpu_data());

			//calculate bottom diff (alpha_diff, mu_diff, sigma_diff)
			kernel_delta_calc<Dtype> << <CAFFE_GET_BLOCKS(bottom[i]->count()), CAFFE_CUDA_NUM_THREADS >> >(bottom[i]->count(),
				batch_size, class_size, data_dim + 2, data_dim, 
				posterior_pi_.gpu_data(), diff_.gpu_data(), diff_norm_.gpu_data(), bottom_data, bottom_diff);
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(MDNLossLayer);

}  // namespace caffe
