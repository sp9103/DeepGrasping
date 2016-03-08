#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

#include <opencv2\opencv.hpp>

namespace caffe {

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }

  ///////////±×¸®±â
  //cv::Mat img;
  //img.create(top[0]->shape()[2], top[0]->shape()[3], CV_8UC1);
  //for (int i = 0; i < top[0]->shape()[1]; i++){
	 // int width = top[0]->shape()[2];
	 // int height = top[0]->shape()[3];
	 // Dtype map[117 * 117];
	 // char buf[32];
	 // sprintf(buf, "%d", i);
	 // //cudaMemcpy(map, &top[0]->gpu_data()[top[0]->shape()[2] * top[0]->shape()[3] * i], sizeof(Dtype) * top[0]->shape()[2] * top[0]->shape()[3], cudaMemcpyDeviceToHost);
	 // memcpy(map, &top[0]->cpu_data()[top[0]->shape()[2] * top[0]->shape()[3] * i], sizeof(Dtype) * top[0]->shape()[2] * top[0]->shape()[3]);

	 // Dtype max = -1;
	 // Dtype min = 9999;
	 // for (int j = 0; j < top[0]->shape()[2] * top[0]->shape()[3]; j++){
		//  if (max < map[j])		max = map[j];
		//  if (min > map[j])		min = map[j];
	 // }


	 // for (int j = 0; j < top[0]->shape()[2] * top[0]->shape()[3]; j++){
		//  img.at<uchar>(j) = (uchar)((map[j] - min) / (max - min) * 255.f);
	 // }
	 // cv::imshow(buf, img);
	 // cv::waitKey(0);
  //}

  //cv::destroyAllWindows();
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
