#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include <opencv2\opencv.hpp>

namespace caffe {

template <typename Dtype>
__global__ void SpatialPoolingForward(const int nthreads,
	const Dtype* const index_data, const int num,
	const int height, const int width, Dtype* const top_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {							//CUDA KERNEL LOOP 함수는 index를 nthread까지 돌리는 for 루프 - 오프셋을 점검해야함.
		top_data[index * 2 + 0] = (int)(index_data[index] / width) / (Dtype)width;
		top_data[index * 2 + 1] = ((unsigned int)index_data[index] % width) / (Dtype)height;					//왜 모듈로 연산 안됨...?
	}
}

template <typename Dtype>
__global__ void SpatialPoolingBackward(const int nthreads,
	const Dtype* const top_diff, const int batchSize, const int nChannels,
	const int bottom_height, const int bottom_width, Dtype* const bottom_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {							//CUDA KERNEL LOOP 함수는 index를 nthread까지 돌리는 for 루프 - 오프셋을 점검해야함.
		//index = 
		const int mapidx = index / (bottom_height * bottom_width);				//몇번째 맵인지 계산
		const int inMapidx = index % (bottom_height * bottom_width);			//map 안에서 몇번째 인덱스 인지
		const int w = inMapidx % bottom_width;
		const int h = inMapidx / bottom_width;
		bottom_diff[index] = (w * top_diff[2 * mapidx + 0] / (Dtype)bottom_width)
							+ (h * top_diff[2 * mapidx + 1] / (Dtype)bottom_height);
	}
}

template <typename Dtype>
__global__ void kernel_features_maxidx(const int num, const int width,
	const int height, const Dtype* data, Dtype* out) {
	CUDA_KERNEL_LOOP(index, num) {
		Dtype maxval = -FLT_MAX;
		for (int i = 0; i < height * width; i++){
			if (maxval < data[(index*width*height) + i]){
				maxval = data[(index*width*height) + i];
				out[index] = i;
			}
		}
	}
}

template <typename Dtype>
__global__ void pooling_backward(const int num, const int width,
	const int height, const Dtype* poolIndex, const Dtype* data, Dtype* out) {
	CUDA_KERNEL_LOOP(index, num) {
		int channel = index / width / height;
		int innerIdx = index % (width*height);

		if (innerIdx == poolIndex[channel]){
			out[index] /= data[index];
		}
		else{
			out[index] = 0;	
		}
	}
}

template <typename Dtype>
void SpatialPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int tWidth = bottom[0]->shape()[2];
  int tHeight = bottom[0]->shape()[3];

  ////////////////////////////////////////maxpooling/////////////////////////////////////////////////////
  //find max
  const int feature_count = bottom[0]->count() / (tWidth*tHeight);						//맵 갯수 32*60
  Dtype* index_data = index_.mutable_gpu_data();

  //Max index finding
  kernel_features_maxidx<Dtype> << <CAFFE_GET_BLOCKS(feature_count),
	  CAFFE_CUDA_NUM_THREADS >> >(feature_count, tWidth, tHeight, bottom_data,
	  index_.mutable_gpu_data());

  //////////////////////////////////extract feature postion///////////////////////////////////////////////
  //<<앞은 블록수, 뒤는 블록당 쓰레드수>>
  SpatialPoolingForward<Dtype> << <CAFFE_GET_BLOCKS(feature_count), CAFFE_CUDA_NUM_THREADS >> >(
	  feature_count, index_.gpu_data(), index_.num(),
	  tHeight, tWidth, top_data);

  /////////////////////////////////////////////////////////////////////////////////////////////
  ///softmax 그리기
  cv::Mat img;
  int batch_size = index_.shape()[0];
  const int drawRow = 4;
  char buf[32];
  Dtype posbox[64], sumbox[50];
  int width = tWidth;
  int height = tHeight;
  int channel = feature_count / batch_size;
  img.create(width * (channel / drawRow), height * 2 * drawRow, CV_8UC1);

 // for (int b = 0; b < batch_size; b++){
 // 	sprintf(buf, "pooling_spatial");
	//cudaMemcpy(posbox, &top[0]->gpu_data()[b * channel * 2], sizeof(Dtype) * channel * 2, cudaMemcpyDeviceToHost);
	//cudaMemcpy(sumbox, &index_.gpu_data()[b * channel], sizeof(Dtype) * channel, cudaMemcpyDeviceToHost);
 // 	for (int i = 0; i < channel; i++){
 // 		Dtype map[109 * 109];
 // 		int s_row = i * 2 / (drawRow*2) * width;
 // 		int s_col = i * 2 % (drawRow*2) * width;

 // 		Dtype max = -1;
 // 		Dtype min = 9999;
 // 		Dtype sum = 0;
 //  		Dtype bmax = -1;
 // 		Dtype bmin = 9999;

 // 		for (int p = i * 2; p < (i + 1) * 2; p++){
 // 			if (std::isnan(posbox[p]))
 // 				printf("pos error!\n");
 // 		}

 // 		cudaMemcpy(map, &bottom[0]->gpu_data()[b*channel*width*height + width* height * i], sizeof(Dtype) * width * height, cudaMemcpyDeviceToHost);

 // 		for (int j = 0; j < width * height; j++){
 // 			if (bmax < map[j])		bmax = map[j];
 // 			if (bmin > map[j])		bmin = map[j];
 // 		}
	//	for (int j = 0; j < width * height; j++){
	//		img.at<uchar>(s_row + j / width, s_col + j%width) = (uchar)((map[j] - bmin) / (bmax - bmin) * 255.f);
	//		img.at<uchar>(s_row + j / width, s_col + j%width + width) = (uchar)0;
	//	}
 // 		//수직 라인
 // 		cv::line(img, cv::Point(s_col, s_row), cv::Point(s_col, s_row + height), cv::Scalar(255));
 // 		cv::line(img, cv::Point(s_col + width, s_row), cv::Point(s_col + width, s_row + height), cv::Scalar(255));
 // 		//수평 라인
 // 		cv::line(img, cv::Point(s_col, s_row), cv::Point(s_col + 2 * width, s_row), cv::Scalar(255));
	//	cv::circle(img, cv::Point((s_row + (int)(posbox[2 * i + 1] * height)), s_col + width + (int)(posbox[2 * i] * width)), 2, cv::Scalar(255), -1);
	//	//cv::imshow(buf, img);
	//	//cv::waitKey(0);
 // 	}
 // 	cv::imshow(buf, img);
	//cv::imwrite("ppp.bmp", img);
 // 	cv::waitKey(0);
 // }
 // /////////////////////////////////////////////////////////////////////////////////////////////////
}

template <typename Dtype>
void SpatialPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0]) {			//pooling layer에서 따옴
		return;
	}

	const Dtype* top_diff = top[0]->gpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	const int bottom_height = bottom[0]->shape()[2];
	const int bottom_width = bottom[0]->shape()[3];
	const int count = bottom[0]->count();
	const int batchSize = bottom[0]->shape()[0];
	const int nChannels = bottom[0]->shape()[1];
	SpatialPoolingBackward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
		count, top_diff, batchSize, nChannels, bottom_height, bottom_width, bottom_diff);

	//pooling backward
	bottom_diff = bottom[0]->mutable_gpu_diff();

	pooling_backward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
		count, bottom_width, bottom_height, index_.gpu_data(), bottom[0]->gpu_data(), bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(SpatialPoolingLayer);

}  // namespace caffe
