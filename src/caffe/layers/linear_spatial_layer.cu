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
__global__ void LSForward(const int nthreads,
	const Dtype* const bottom_data, const int num,
	const int height, const int width, Dtype* const top_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {							//CUDA KERNEL LOOP 함수는 index를 nthread까지 돌리는 for 루프 - 오프셋을 점검해야함.
		Dtype tValue = 0.0f;

		for (int h = 0; h < height; h++){									
			for (int w = 0; w < width; w++){
				if (index % 2 == 0)										//짝수일때는 x좌표 관련된 작업
					tValue += w * bottom_data[(index / 2)*width*height + h * height + w] / (Dtype)width;
				else														//홀수일때는 y좌표 관련된 작업
					tValue += h * bottom_data[(index / 2)*width*height + h * height + w] / (Dtype)height;
			}
		}

		top_data[index] = tValue;
	}
}

template <typename Dtype>
__global__ void LSBackward(const int nthreads,
	const Dtype* const top_diff,
	const int bottom_height, const int bottom_width, Dtype* const bottom_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {							//CUDA KERNEL LOOP 함수는 index를 nthread까지 돌리는 for 루프 - 오프셋을 점검해야함.
		//index = 
		const int mapidx = index / (bottom_height * bottom_width);				//몇번째 맵인지 계산
		const int inMapidx = index % (bottom_height * bottom_width);			//map 안에서 몇번째 인덱스 인지
		const int w = inMapidx % bottom_width;
		const int h = inMapidx / bottom_width;
		bottom_diff[index] = w * top_diff[2 * mapidx + 0] / (Dtype)bottom_width
							+ h * top_diff[2 * mapidx + 1] / (Dtype)bottom_height;
	}
}

template <typename Dtype>
__global__ void LS_feature_threshold(const int num, const int width,
	const int height, const float threshold, const Dtype* data, Dtype* thresholdVal) {
	CUDA_KERNEL_LOOP(index, num) {
		Dtype maxval = -FLT_MAX;
		Dtype minval = FLT_MAX;
		for (int i = 0; i < height * width; i++){
			maxval = max(data[(index*width*height) + i], maxval);
			if (data[(index*width*height) + i] != 0)
				minval = min(data[(index*width*height) + i], minval);
		}
		thresholdVal[index] = (maxval - minval) * threshold + minval;
	}
}

template <typename Dtype>
__global__ void LS_thresholding(const int num, const int width,
	const int height, const Dtype* thresholdVal,
	const Dtype* bottomdata, Dtype *output) {
	CUDA_KERNEL_LOOP(index, num) {
		int idx = index / (width * height);
		
		if (bottomdata[index] < thresholdVal[idx])	output[index] = (Dtype)0;
		else										output[index] = bottomdata[index];
	}
}

template <typename Dtype>
__global__ void LS_feature_sum(const int num, const int width,
	const int height, const Dtype* data, Dtype* out) {
	CUDA_KERNEL_LOOP(index, num) {
		const Dtype* const bottom_slice =
			data + index * height * width;						//softmax에서 한 피쳐맵을 잘라냄.(채널이 1이기 때문에 채널은 고려X)

		Dtype sum = (Dtype)0.0000001;
		for (int h = 0; h < height; h++){
			for (int w = 0; w < width; w++)
				sum += bottom_slice[h*width + w];
		}

		out[index] = sum;
	}
}

//문제있는 부분
template <typename Dtype>
__global__ void LS_features_div(const int count,
	const int width, const int height,
	const Dtype* sum, Dtype* data) {
	CUDA_KERNEL_LOOP(index, count) {
		int n = index / width / height;
		Dtype temp = data[index] / sum[n];
		if ((Dtype)0 <= temp && temp <= (Dtype)1)		data[index] = temp;
			
	}
}

template <typename Dtype>
__global__ void LS_Backthresholding(const int num, const int width,
	const int height, const Dtype* thresholdVal,
	const Dtype* bottomdata, Dtype *output) {
	CUDA_KERNEL_LOOP(index, num) {
		int idx = index / (width * height);

		if (bottomdata[index] < thresholdVal[idx])	output[index] = (Dtype)0;
	}
}

template <typename Dtype>
void LinearSpatialLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	const int bottom_width = bottom[0]->shape()[2];
	const int bottom_height = bottom[0]->shape()[3];
	const int feature_count = bottom[0]->count() / (bottom_width*bottom_height);						//맵 갯수 32*60

	//calculate threshold value
	LS_feature_threshold<Dtype> << <CAFFE_GET_BLOCKS(feature_count), CAFFE_CUDA_NUM_THREADS >> >(
		feature_count, bottom_width, bottom_height, threshold_,
		bottom_data, thresholdVal_.mutable_gpu_data());

	//calculate thresholding bottom input
	LS_thresholding<Dtype> << <CAFFE_GET_BLOCKS(linearOutput_.count()), CAFFE_CUDA_NUM_THREADS >> >(
		linearOutput_.count(), bottom_width, bottom_height, thresholdVal_.gpu_data(),
		bottom_data, linearOutput_.mutable_gpu_data());

	//calculate summation
	LS_feature_sum<Dtype> << <CAFFE_GET_BLOCKS(feature_count),
		CAFFE_CUDA_NUM_THREADS >> >(feature_count, bottom_width, bottom_height, linearOutput_.gpu_data(),
		sum_.mutable_gpu_data());

	//divide output width summation
	//div result
	LS_features_div<Dtype> << <CAFFE_GET_BLOCKS(linearOutput_.count()),
		CAFFE_CUDA_NUM_THREADS >> >(linearOutput_.count(), bottom_width, bottom_height, sum_.gpu_data(), linearOutput_.mutable_gpu_data());

	//extract x,y pos of feature
	LSForward<Dtype> << <CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS >> >(
		top[0]->count(), linearOutput_.gpu_data(), linearOutput_.num(),
		bottom_height, bottom_width, top_data);

	///////////////////////////////////////////////////////////////////////////////////////////////
	///softmax 그리기
	cv::Mat img;
	int batch_size = linearOutput_.shape()[0];
	const int drawRow = 4;
	char buf[32];
	Dtype posbox[64], sumbox[32];
	int width = linearOutput_.shape()[2];
	int height = linearOutput_.shape()[3];
	int channel = linearOutput_.shape()[1];
	img.create(linearOutput_.shape()[2] * (linearOutput_.shape()[1] / drawRow), linearOutput_.shape()[3] * 3 * drawRow, CV_8UC1);

	//for (int b = 0; b < batch_size; b++){
	//	sprintf(buf, "linear_spatial");
	//	cudaMemcpy(posbox, &top[0]->gpu_data()[b * 64], sizeof(Dtype) * 64, cudaMemcpyDeviceToHost);
	//	cudaMemcpy(sumbox, &sum_.gpu_data()[b*32], sizeof(Dtype) * 32, cudaMemcpyDeviceToHost);

	//	for (int i = 0; i < channel; i++){
	//		Dtype map[109 * 109];
	//		int s_row = i * 3 / (drawRow*3) * width;
	//		int s_col = i * 3 % (drawRow*3) * width;

	//		cudaMemcpy(map, &linearOutput_.gpu_data()[b*channel*width*height + width * height * i], sizeof(Dtype) * width * height, cudaMemcpyDeviceToHost);
	//		Dtype max = -1;
	//		Dtype min = 9999;
	//		Dtype sum = 0;
	//		Dtype bmax = -1;
	//		Dtype bmin = 9999;

	//		for (int p = i * 2; p < (i + 1) * 2; p++){
	//			if (std::isnan(posbox[p]))
	//				printf("pos error!\n");
	//		}

	//		for (int j = 0; j < width * height; j++){
	//			if (max < map[j])		max = map[j];
	//			if (min > map[j])		min = map[j];
	//			sum += map[j];
	//		}
	//		for (int j = 0; j < width * height; j++){
	//			img.at<uchar>(s_row + j / width, s_col + j%width + width) = (uchar)((map[j] - min) / (max - min) * 255.f);
	//			img.at<uchar>(s_row + j / width, s_col + j%width + width * 2) = (uchar)((map[j] - min) / (max - min) * 255.f);
	//		}
	//		cudaMemcpy(map, &bottom[0]->gpu_data()[b*channel*width*height + width* height * i], sizeof(Dtype) * width * height, cudaMemcpyDeviceToHost);

	//		for (int j = 0; j < width * height; j++){
	//			if (bmax < map[j])		bmax = map[j];
	//			if (bmin > map[j])		bmin = map[j];
	//		}
	//		for (int j = 0; j < width * height; j++)
	//			img.at<uchar>(s_row + j / width, s_col + j%width) = (uchar)((map[j] - bmin) / (bmax - bmin) * 255.f);
	//		//수직 라인
	//		cv::line(img, cv::Point(s_col, s_row), cv::Point(s_col, s_row + height), cv::Scalar(255));
	//		cv::line(img, cv::Point(s_col + width, s_row), cv::Point(s_col + width, s_row + height), cv::Scalar(255));
	//		cv::line(img, cv::Point(s_col + width * 2, s_row), cv::Point(s_col+ width * 2, s_row + height), cv::Scalar(255));
	//		//수평 라인
	//		cv::line(img, cv::Point(s_col, s_row), cv::Point(s_col + 3 * width, s_row), cv::Scalar(255));
	//		cv::circle(img, cv::Point(s_col + width * 2 + (int)(posbox[2 * i] * width), (int)(s_row + posbox[2 * i + 1] * height)), 3, cv::Scalar(255), -1);
	//	}
	//	cv::imshow(buf, img);
	//	cv::waitKey(0);
	//}
	//cv::destroyAllWindows();
	///////////////////////////////////////////////////////////////////////////////////////////////////
}

template <typename Dtype>
void LinearSpatialLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0]) {			//pooling layer에서 따옴
		return;
	}

	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	const Dtype* top_diff = top[0]->gpu_diff();

	const int bottom_width = bottom[0]->shape()[2];
	const int bottom_height = bottom[0]->shape()[3];

	LSBackward<Dtype> << <CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS >> >(
		bottom[0]->count(), top_diff, bottom_height, bottom_width, bottom_diff);

	//at forward propagation - (1. thresholding, 2. / sum)
	//div top diff with sum
	LS_features_div<Dtype> << <CAFFE_GET_BLOCKS(bottom[0]->count()),
		CAFFE_CUDA_NUM_THREADS >> >(bottom[0]->count(), bottom_width, bottom_height, sum_.gpu_data(), bottom_diff);

	//adapt thresholding
	LS_Backthresholding<Dtype> << <CAFFE_GET_BLOCKS(linearOutput_.count()), CAFFE_CUDA_NUM_THREADS >> >(
		linearOutput_.count(), bottom_width, bottom_height, thresholdVal_.gpu_data(),
		bottom[0]->gpu_data(), bottom_diff);

	//cv::Mat diffimg;
	//int width = bottom[0]->shape()[2];
	//int height = bottom[0]->shape()[3];
	//int channel = bottom[0]->shape()[1];
	//int batch_size = bottom[0]->shape()[0];
	//const int drawRow = 4;
	//diffimg.create(height * (channel / drawRow), width * 3 * drawRow, CV_8UC1);

	//for (int b = 0; b < batch_size; b++){

	//	//왼쪽이 bottom data, 중간이 featuremap, 오른쪽이 bottom_diff
	//	for (int i = 0; i < channel; i++){
	//		Dtype diffmap[109 * 109];
	//		int s_row = i * 3 / (drawRow * 3) * width;
	//		int s_col = i * 3 % (drawRow * 3) * width;
	//		cudaMemcpy(diffmap, &bottom[0]->gpu_diff()[width*height*i + b*channel*width*height], sizeof(Dtype)*width*height, cudaMemcpyDeviceToHost);

	//		Dtype max = -9999;
	//		Dtype min = 9999;

	//		for (int j = 0; j < width*height; j++){
	//			if (max < diffmap[j])		max = diffmap[j];
	//			if (min > diffmap[j])		min = diffmap[j];
	//		}

	//		for (int j = 0; j < width*height; j++)
	//			diffimg.at<uchar>(s_row + j / width, s_col + j%width + width*2) = (uchar)((diffmap[j] - min) / (max - min) * 255.f);

	//		cudaMemcpy(diffmap, &bottom[0]->gpu_data()[width*height*i + b*channel*width*height], sizeof(Dtype)*width*height, cudaMemcpyDeviceToHost);

	//		max = -9999;
	//		min = 9999;
	//		for (int j = 0; j < width*height; j++){
	//			if (max < diffmap[j])		max = diffmap[j];
	//			if (min > diffmap[j])		min = diffmap[j];
	//		}

	//		for (int j = 0; j < width*height; j++)
	//			diffimg.at<uchar>(s_row + j / width, s_col + j%width) = (uchar)((diffmap[j] - min) / (max - min) * 255.f);

	//		cudaMemcpy(diffmap, &linearOutput_.gpu_data()[width*height*i + b*channel*width*height], sizeof(Dtype)*width*height, cudaMemcpyDeviceToHost);
	//		max = -9999;
	//		min = 9999;
	//		for (int j = 0; j < width*height; j++){
	//			if (max < diffmap[j])		max = diffmap[j];
	//			if (min > diffmap[j])		min = diffmap[j];
	//		}

	//		for (int j = 0; j < width*height; j++)
	//			diffimg.at<uchar>(s_row + j / width, s_col + j%width + width) = (uchar)((diffmap[j] - min) / (max - min) * 255.f);
	//		//수직 라인
	//		cv::line(diffimg, cv::Point(s_col, s_row), cv::Point(s_col, s_row + height), cv::Scalar(255));
	//		cv::line(diffimg, cv::Point(s_col + width, s_row), cv::Point(s_col + width, s_row + height), cv::Scalar(255));
	//		cv::line(diffimg, cv::Point(s_col + width * 2, s_row), cv::Point(s_col + width * 2, s_row + height), cv::Scalar(255));
	//		//수평 라인
	//		cv::line(diffimg, cv::Point(s_col, s_row), cv::Point(s_col + 3 * width, s_row), cv::Scalar(255));
	//	}

	//	cv::imshow("diffimg", diffimg);
	//	cv::waitKey(0);
	//}
}

INSTANTIATE_LAYER_GPU_FUNCS(LinearSpatialLayer);

}  // namespace caffe
