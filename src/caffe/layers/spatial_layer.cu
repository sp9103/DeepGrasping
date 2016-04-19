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
__global__ void kernel_feature_max(const int num, const int width,
	const int height, const Dtype* data, Dtype* out) {
	CUDA_KERNEL_LOOP(index, num) {
		Dtype maxval = -FLT_MAX;
		for (int i = 0; i < height * width; i++)
			maxval = max(data[(index*width*height) + i], maxval);

		out[index] = maxval + 0.0000001;
	}
}

template <typename Dtype>
__global__ void kernel_feature_subtract(const int count,
	const int width, const int height,
	const Dtype* feature_max, Dtype* data) {
	CUDA_KERNEL_LOOP(index, count) {
		int n = index / width / height;
		data[index] -= feature_max[n];
	}
}

template <typename Dtype>
__global__ void kernel_exp(const int count, const Dtype* data, Dtype* out) {
	CUDA_KERNEL_LOOP(index, count) {
		out[index] = exp(data[index]);
	}
}

template <typename Dtype>
__global__ void kernel_features_scale(const int count,
	const int width, const int height,
	const Dtype* max, Dtype scaleparam, Dtype* data) {
	CUDA_KERNEL_LOOP(index, count) {
		int n = index / width / height;
		data[index] = data[index] * scaleparam /*/ max[n]*/;
		//if (max[n] != 0)
		//	data[index] = data[index] * scaleparam /*/ max[n]*/;
	}
}

template <typename Dtype>
__global__ void kernel_features_div(const int count,
	const int width, const int height,
	const Dtype* sum, Dtype* data) {
	CUDA_KERNEL_LOOP(index, count) {
		int n = index / width / height;
		data[index] /= sum[n];
	}
}

template <typename Dtype>
__global__ void kernel_feature_sum(const int num, const int width,
	const int height, const Dtype* data, Dtype* out) {
	CUDA_KERNEL_LOOP(index, num) {
		const Dtype* const bottom_slice =
			data + index * height * width;						//softmax에서 한 피쳐맵을 잘라냄.(채널이 1이기 때문에 채널은 고려X)

		Dtype sum = 0;
		for (int h = 0; h < height; h++){
			for (int w = 0; w < width; w++)
				sum += bottom_slice[h*width + w];
		}

		out[index] = sum;
	}
}

template <typename Dtype>
__global__ void SpatialForward(const int nthreads,
	const Dtype* const bottom_data, const int num,
	const int height, const int width, Dtype* const top_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {							//CUDA KERNEL LOOP 함수는 index를 nthread까지 돌리는 for 루프 - 오프셋을 점검해야함.
		
		//직접 구현
		Dtype tValue = 0.0f;

		for (int h = 0; h < height; h++){									
			for (int w = 0; w < width; w++){
				const int element_idx = h * width + w;						//곱셈을 해야할 element
				if (index % 2 == 0)										//짝수일때는 x좌표 관련된 작업
					tValue += w * bottom_data[(index / 2)*width*height + element_idx] / (Dtype)width;
				else														//홀수일때는 y좌표 관련된 작업
					tValue += h * bottom_data[(index / 2)*width*height + element_idx] / (Dtype)height;
			}
		}

		top_data[index] = tValue;
	}
}

template <typename Dtype>
__global__ void SpatialBackward(const int nthreads,
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
__global__ void kernel_features_dot(const int num, const int width,
	const int height, const Dtype* data_1, const Dtype* data_2,
	Dtype* out) {
	CUDA_KERNEL_LOOP(index, num) {
		Dtype result = 0;
		int mapIdx = index*width*height;

		for (int i = 0; i < width*height; i++)
			result += data_1[mapIdx + i] * data_2[mapIdx + i];

		out[index] = result;
	}
}

template <typename Dtype>
void SpatialLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
	//Dtype* bottom_data = bottom[0]->mutable_gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int tWidth = bottom[0]->shape()[2];
  int tHeight = bottom[0]->shape()[3];

  /*int sTime = clock();*/

  //////////////////////////////////softmax operation///////////////////////////////////////////////////// check
  //find max
  const int feature_count = bottom[0]->count() / (tWidth*tHeight);						//맵 갯수 32*60
  Dtype* max_data = maxValue_.mutable_gpu_data();
  Dtype* softmax_result = softmaxResult_.mutable_gpu_data();
  const int softmaxCount = softmaxResult_.count();
  caffe_copy(bottom[0]->count(), bottom_data, softmax_result);

  if (alpha_ != 1.0f){
	  kernel_features_scale<Dtype> << <CAFFE_GET_BLOCKS(softmaxCount),
		   CAFFE_CUDA_NUM_THREADS >> >(softmaxCount, tWidth, tHeight,
		   max_data, (Dtype)alpha_, softmax_result);
  }

  kernel_feature_max<Dtype> << <CAFFE_GET_BLOCKS(feature_count),
	  CAFFE_CUDA_NUM_THREADS >> >(feature_count, tWidth, tHeight, softmax_result,
	  max_data);

  ////sub max
  kernel_feature_subtract<Dtype> << <CAFFE_GET_BLOCKS(softmaxCount),
	  CAFFE_CUDA_NUM_THREADS >> >(softmaxCount, tWidth, tHeight,
	  max_data, softmax_result);

  //exponential
  kernel_exp<Dtype> << <CAFFE_GET_BLOCKS(softmaxCount), CAFFE_CUDA_NUM_THREADS >> >(
	  softmaxCount, softmax_result, softmax_result);

  //calculate summation
  kernel_feature_sum<Dtype> << <CAFFE_GET_BLOCKS(feature_count),
	  CAFFE_CUDA_NUM_THREADS >> >(feature_count, tWidth, tHeight, softmax_result,
	  backwardTemp_.mutable_gpu_data());

  //div result
  kernel_features_div<Dtype> << <CAFFE_GET_BLOCKS(softmaxCount),
	  CAFFE_CUDA_NUM_THREADS >> >(softmaxCount, tWidth, tHeight, backwardTemp_.gpu_data(), softmax_result);

  //////////////////////////////////extract feature postion///////////////////////////////////////////////
  //<<앞은 블록수, 뒤는 블록당 쓰레드수>>
  const Dtype* softmax = softmaxResult_.gpu_data();
  int count = top[0]->count();
  SpatialForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
	  count, softmax, softmaxResult_.num(),
	  tHeight, tWidth, top_data);

  ////Feature 뿌리기
  ////////if (bottom->size() == 2){
	 // cv::Mat FeaturePlot;
	 // const int drawRow = 10;
	 // const int datatWidth = bottom[1]->shape()[2];
	 // const int dataHeight = bottom[1]->shape()[3];
	 // const int dataCount = bottom[1]->shape()[0];
	 // const int softChannel = bottom[0]->shape()[1];
	 // const int topCount = top[0]->shape()[1];
	 // FeaturePlot.create(datatWidth * (dataCount / drawRow + 1), dataHeight * 2 * drawRow, CV_8UC3);
	 // for (int i = 0; i < dataCount; i++){
		//  int s_row = i * 2 / (drawRow * 2) * datatWidth;
		//  int s_col = i * 2 % (drawRow * 2) * datatWidth;

		//  Dtype pos[128];
		//  Dtype *Map = new Dtype[160*160*3];
		//  cv::Point pointList[64];

		//  cudaMemcpy(pos, &top[0]->gpu_data()[i * topCount], sizeof(Dtype) * topCount, cudaMemcpyDeviceToHost);
		//  cudaMemcpy(Map, &bottom[1]->gpu_data()[i * datatWidth * dataHeight * 3], sizeof(Dtype) * datatWidth * dataHeight * 3, cudaMemcpyDeviceToHost);

		//  for (int i = 0; i < topCount / 2; i++)
		//	  pointList[i] = cv::Point(pos[2 * i] * datatWidth, pos[2 * i + 1] * dataHeight);

		//  for (int h = 0; h < dataHeight; h++){
		//	  for (int w = 0; w < datatWidth; w++){
		//		  for (int c = 0; c < 3; c++){
		//			  FeaturePlot.at<cv::Vec3b>(s_row + h, s_col + w)[c] = uchar(Map[c*dataHeight*datatWidth + datatWidth*h + w] * 255.f);
		//			  FeaturePlot.at<cv::Vec3b>(s_row + h, s_col + w + datatWidth)[c] = uchar(Map[c*dataHeight*datatWidth + datatWidth*h + w] * 255.f);
		//		  }
		//	  }
		//  }

		//  Dtype softThreshold = 0.0;
		//  const int softwidth = softmaxResult_.shape()[2];
		//  const int softheight = softmaxResult_.shape()[3];
		//  Dtype *softmap = new Dtype[softwidth * softheight];
		//  for (int j = 0; j < topCount/2/2; j++){
		//	  uchar R = (j * 7)% 255;
		//	  uchar G = (j * 37) % 255;
		//	  uchar B = (j * 103) % 255;

		//	  cudaMemcpy(softmap, &softmaxResult_.gpu_data()[i*softChannel*softwidth*softheight + softwidth * softheight * j], sizeof(Dtype) * softwidth * softheight, cudaMemcpyDeviceToHost);
		//	  Dtype Max = -99999;
		//	  for (int s = 0; s < softwidth*softheight; s++)
		//		  if (Max < softmap[s])
		//			  Max = softmap[s];

		//	  if (Max > softThreshold)
		//		  cv::circle(FeaturePlot, cv::Point(s_col + pointList[j].x, s_row + pointList[j].y), 3, cv::Scalar(B, G, R), -1);
		//  }
		//  for (int j = topCount/2/2; j < topCount; j++){
		//	  uchar R = (j * 7) % 255;
		//	  uchar G = (j * 37) % 255;
		//	  uchar B = (j * 103) % 255;

		//	  cudaMemcpy(softmap, &softmaxResult_.gpu_data()[i * softChannel * softwidth*softheight + softwidth * softheight * j], sizeof(Dtype) * softwidth * softheight, cudaMemcpyDeviceToHost);
		//	  Dtype Max = -99999;
		//	  for (int s = 0; s < softwidth*softheight; s++)
		//		  if (Max < softmap[s])
		//			  Max = softmap[s];

		//	  if (Max > softThreshold)
		//		  cv::circle(FeaturePlot, cv::Point(s_col + datatWidth + pointList[j].x, s_row + pointList[j].y), 3, cv::Scalar(B, G, R), -1);
		//  }

		//  delete[] softmap;
		//  delete[] Map;
	 // }

	 // cv::imwrite("FeaturePlot.bmp", FeaturePlot);
	 // cv::imshow("Map", FeaturePlot);
	 // cv::waitKey(0);
  //////////}
}

template <typename Dtype>
void SpatialLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
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
	SpatialBackward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
		count, top_diff, batchSize, nChannels, bottom_height, bottom_width, bottom_diff);

	//softmax backward
	bottom_diff = bottom[0]->mutable_gpu_diff();

	//top_data(softmaxResult_) * top_diff = max_data
	const int feature_count = count / (bottom_height * bottom_width);
	Dtype* backwardTemp = backwardTemp_.mutable_gpu_data();
	const Dtype* softmax_result = softmaxResult_.gpu_data();
	kernel_features_dot<Dtype> << <CAFFE_GET_BLOCKS(feature_count),
		CAFFE_CUDA_NUM_THREADS >> >(feature_count, bottom_height, bottom_width,
		bottom_diff, softmax_result, backwardTemp);

	//bottom_diff - max_data = bottom_diff
	kernel_feature_subtract<Dtype> << <CAFFE_GET_BLOCKS(count),
		CAFFE_CUDA_NUM_THREADS >> >(count, bottom_width, bottom_height,
		backwardTemp, bottom_diff);

	//bottom_diff * top_data = bottom_diff (elementwise product)
	caffe_gpu_mul<Dtype>(softmaxResult_.count(), bottom_diff, softmax_result, bottom_diff);

	if (alpha_ != 1.0f){
		kernel_features_scale<Dtype> << <CAFFE_GET_BLOCKS(softmaxResult_.count()),
			CAFFE_CUDA_NUM_THREADS >> >(softmaxResult_.count(), bottom_width, bottom_height,
			maxValue_.gpu_data(), (Dtype)alpha_, bottom_diff);
	}

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
	//			diffimg.at<uchar>(s_row + j / width, s_col + j%width + width * 2) = (uchar)((diffmap[j] - min) / (max - min) * 255.f);

	//		cudaMemcpy(diffmap, &bottom[0]->gpu_data()[width*height*i + b*channel*width*height], sizeof(Dtype)*width*height, cudaMemcpyDeviceToHost);

	//		max = -9999;
	//		min = 9999;
	//		for (int j = 0; j < width*height; j++){
	//			if (max < diffmap[j])		max = diffmap[j];
	//			if (min > diffmap[j])		min = diffmap[j];
	//		}

	//		for (int j = 0; j < width*height; j++)
	//			diffimg.at<uchar>(s_row + j / width, s_col + j%width) = (uchar)((diffmap[j] - min) / (max - min) * 255.f);

	//		cudaMemcpy(diffmap, &softmaxResult_.gpu_data()[width*height*i + b*channel*width*height], sizeof(Dtype)*width*height, cudaMemcpyDeviceToHost);
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

INSTANTIATE_LAYER_GPU_FUNCS(SpatialLayer);

}  // namespace caffe
