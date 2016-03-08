#include <vector>

#include "caffe/data_layers.hpp"

#include <opencv2\opencv.hpp>

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data				//8640000 = 240*240*3*50(batchsize)
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
      top[0]->mutable_gpu_data());

  //////데이터 보기
  //cv::Mat tImg;
  //const int idx = 0;
  //tImg.create(240, 240, CV_8UC3);
  //for (int i = 0; i < 240; i++){
	 // for (int j = 0; j < 240; j++){
		//  int tOffset = batch->data_.offset(idx, 0, i, j);
		//  Dtype temp = batch->data_.cpu_data()[tOffset];
		//  tImg.at<cv::Vec3b>(i, j)[0] = (unsigned char)temp + 103.939;
		//  tOffset = batch->data_.offset(idx, 1, i, j);
		//  temp = batch->data_.cpu_data()[tOffset];
		//  tImg.at<cv::Vec3b>(i, j)[1] = (unsigned char)temp + 116.779;
		//  tOffset = batch->data_.offset(idx, 2, i, j);
		//  temp = batch->data_.cpu_data()[tOffset];
		//  tImg.at<cv::Vec3b>(i, j)[2] = (unsigned char)temp + 123.68;
	 // }
  //}
  //int tlabel = batch->label_.cpu_data()[idx];
  //cv::imshow("test", tImg);
  //cv::waitKey(0);

  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
        top[1]->mutable_gpu_data());
  }
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);

}  // namespace caffe
