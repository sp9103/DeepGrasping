#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include <opencv2\opencv.hpp>

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;

 // caffe_sub(count, bottom[0]->cpu_data(),
	//  bottom[1]->cpu_data(),
	//  diff_.mutable_cpu_data());
 // Dtype cpudot;
 // cpudot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
 // Dtype cpuloss = dot / bottom[0]->num() / Dtype(2);
 // Dtype out[3 * 221];
 // Dtype label[3 * 221];
 // Dtype diff[3 * 221];
 //// cudaMemcpy(out, bottom[0]->gpu_data(), sizeof(Dtype) * 3 * 100, cudaMemcpyDeviceToHost);
 //// cudaMemcpy(label, bottom[1]->gpu_data(), sizeof(Dtype) * 3 * 100, cudaMemcpyDeviceToHost);
 //// cudaMemcpy(diff, diff_.mutable_gpu_data(), sizeof(Dtype) * 3 * 100, cudaMemcpyDeviceToHost);
 // memcpy(out, bottom[0]->cpu_data(), sizeof(Dtype) * 3 * 221);
 // memcpy(label, bottom[1]->cpu_data(), sizeof(Dtype) * 3 * 221);
 // memcpy(diff, diff_.cpu_data(), sizeof(Dtype) * 3 * 221);
 //// //cudaMemcpy(diff, diff_.mutable_gpu_data(), sizeof(Dtype) * 3 * 200, cudaMemcpyDeviceToHost);
 // for (int i = 0; i < 221; i++){
	//  printf("(%f %f %f), (%f %f %f)\n diff(%f %f %f)\n", (float)out[i * 3 + 0], (float)out[i * 3 + 1], (float)out[i * 3 + 2], (float)label[i * 3 + 0], (float)label[i * 3 + 1], (float)label[i * 3 + 2]
	//	  , (float)diff[i * 3 + 0], (float)diff[i * 3 + 1], (float)diff[i * 3 + 2]);
 // }

  //if (std::isnan(loss) || std::isinf(loss)){
	 // printf("loss invalide Error!\n");

	 //  for (int c = 0; c < bottom[0]->num(); c++){
	 //   cudaMemcpy(labelarr, &bottom[1]->gpu_data()[9 * c], sizeof(Dtype) * 9, cudaMemcpyDeviceToHost);
	 //   cudaMemcpy(outputarr, &bottom[0]->gpu_data()[9 * c], sizeof(Dtype) * 9, cudaMemcpyDeviceToHost);
	 //   cudaMemcpy(diffarr, &diff_.gpu_data()[9 * c], sizeof(Dtype) * 9, cudaMemcpyDeviceToHost);
	 //  }
  //}

  //if (!std::isnan(loss)){

	  //////////////////////////////////////////
	  cv::Mat lossLayer;
	  const int labelwidth = 80;
	  const int labelheight = 80;
	  lossLayer.create(labelheight, labelwidth * 3, CV_32FC1);

	  ////////////label & output 출력
	  int cCount = bottom[0]->num() < 10 ? bottom[0]->num() : 10;
	  for (int c = 0; c < cCount; c++){
		  char buf[32];
		  Dtype labelarr[6400], outputarr[6400], diffarr[6400];
		  cudaMemcpy(labelarr, &bottom[1]->gpu_data()[6400 * c], sizeof(Dtype) * 6400, cudaMemcpyDeviceToHost);
		  cudaMemcpy(outputarr, &bottom[0]->gpu_data()[6400 * c], sizeof(Dtype) * 6400, cudaMemcpyDeviceToHost);
		  cudaMemcpy(diffarr, &diff_.gpu_data()[6400 * c], sizeof(Dtype) * 6400, cudaMemcpyDeviceToHost);

		  for (int j = 0; j < labelheight * labelwidth; j++){
			  lossLayer.at<float>(j / labelwidth, j % labelwidth) = (float)labelarr[j];
			  lossLayer.at<float>(j / labelwidth, j % labelwidth + labelwidth) = (float)outputarr[j];
		  }

		  Dtype max = -9999;
		  Dtype min = 9999;
		  
		  for (int j = 0; j < labelheight * labelwidth; j++){
			  if (max < diffarr[j])		max = diffarr[j];
			  if (min > diffarr[j])		min = diffarr[j];
		  }

		  for (int j = 0; j < labelheight * labelwidth; j++)
			  lossLayer.at<float>(j / labelwidth, j % labelwidth + labelwidth * 2) = (float)((diffarr[j] - min) / (max - min));

		  sprintf(buf, "Loss_%d.bmp", c);
		  cv::imshow(buf, lossLayer);
		  cv::waitKey(0);
	  }
  //}
  cv::destroyAllWindows();
  ////////////////////////////////////////////////////////
}

//Diff 0번지는 값있고 1번지는 없음
template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
	  if (propagate_down[i]) {
		  const Dtype sign = (i == 0) ? 1 : -1;
		  const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
		  //const Dtype alpha = 1.0 / bottom[i]->num();
		  Dtype tt = top[0]->cpu_diff()[0];
		  caffe_gpu_axpby(
			  bottom[i]->count(),              // count
			  alpha,                              // alpha
			  diff_.gpu_data(),                   // a			//차이값이 diff_에 저장됨
			  Dtype(0),                           // beta
			  bottom[i]->mutable_gpu_diff());  // b
	  }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossLayer);

}  // namespace caffe
