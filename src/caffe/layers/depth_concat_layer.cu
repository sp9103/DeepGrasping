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
	__global__ void spatial_depth_concat(const int count, const int width, const int height,
		const Dtype* spatial_pos, const Dtype* depthval, Dtype* topdata) {
		CUDA_KERNEL_LOOP(index, count) {
			const int id = index % 3;
			const int batchid = index / 3;

			topdata[index] = (Dtype)0;

			//x pos
			if (id == 0)
				topdata[index] = spatial_pos[2 * batchid + 0];
			//y pos
			else if (id == 1)
				topdata[index] = spatial_pos[2 * batchid + 1];
			//depth val
			else if (id == 2)
				topdata[index] = depthval[batchid];
		}
	}

	template <typename Dtype>
	__global__ void kernel_depth_sub(const int count, const int width, const int height, Dtype threshold,
		const Dtype* depth, const Dtype* background, Dtype* bin) {
		CUDA_KERNEL_LOOP(index, count) {
			/////////////////////////TO-DO
			int backIndex = index % (width * height);

			if (depth[index] < 0.6){
				bin[index] = 0;
				continue;
			}

			Dtype subVal = background[backIndex] - depth[index];
			bin[index] = subVal > threshold ? 1 : 0;
		}
	}

	template <typename Dtype>
	__global__ void kernel_depth_val(const int count, const int width, const int height, const int featureCount,
		Dtype alpha, const Dtype* depth, const Dtype* spatial_pos, const Dtype* bin, Dtype* data) {
		CUDA_KERNEL_LOOP(index, count) {
			const int x_pos = width * spatial_pos[2 * index + 0];
			const int y_pos = height * spatial_pos[2 * index + 1];
			const int mapIdx = index / count;

			//data[index] = depth[x_pos + y_pos * width + mapIdx*width*height];

			//TO_DO
			const int x_min = (x_pos - alpha) > 0 ? (x_pos - alpha) : 0;
			const int x_max = (x_pos + alpha) < width ? (x_pos + alpha) : width - 1;
			const int y_min = (y_pos - alpha) > 0 ? (y_pos - alpha) : 0;
			const int y_max = (y_pos + alpha) < height ? (y_pos + alpha) : height - 1;

			int whitePixel = 0;
			Dtype DepthObjVal = 0;
			Dtype DepthTotalVal = 0;
			int kernelCount = 0;
			for (int x = x_min; x <= x_max; x++){
				for (int y = y_min; y <= y_max; y++){
					if (bin[x + y*width + mapIdx*width*height] > 0){
						whitePixel++;
						DepthObjVal += depth[x + y*width + mapIdx*width*height];
					}
					if (depth[x + y*width + mapIdx*width*height] > 0.6){
						DepthTotalVal += depth[x + y*width + mapIdx*width*height];
						kernelCount++;
					}
				}
			}

			//값 입력단
			if (whitePixel > 0)		data[index] = DepthObjVal / (Dtype)whitePixel;
			else if (kernelCount == 0)	data[index] = -1;
			else					data[index] = DepthTotalVal / (Dtype)kernelCount;
		}
	}

	template <typename Dtype>
	__global__ void concat_spatial_backward(const int count,
		const Dtype* topdiff, Dtype* spatial) {
		CUDA_KERNEL_LOOP(index, count) {
			const int featureIdx = index / 2;
			const int id = index % 2;				//0 : xpos, 1 : ypos 2 : depthvalue

			const int topid = featureIdx * 3 + id;

			spatial[index] = topdiff[topid];
		}
	}

	template <typename Dtype>
	void DepthConcatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const int topcount = top[0]->count();
		const int tWidth = bottom[0]->shape()[2];
		const int tHeight = bottom[0]->shape()[3];
		const int count = bottom[0]->count();
		const int botPosCount = bottom[1]->count() / 2;
		const int batchsize = bottom[0]->shape()[0];

		const Dtype* spatialPos = bottom[1]->gpu_data();
		const Dtype* depthImg = bottom[0]->gpu_data();

		//배경 제거
		kernel_depth_sub<Dtype> << <CAFFE_GET_BLOCKS(count),
			CAFFE_CUDA_NUM_THREADS >> >(count, tWidth, tHeight, (Dtype)0.02,
			depthImg, BackGround_.gpu_data(), sub_img_.mutable_gpu_data());

		/*cv::Mat bottomImg, subImg;
		bottomImg.create(80, 80, CV_32FC1);
		subImg.create(80, 80, CV_32FC1);
		Dtype botBox[80 * 80], subBox[80 * 80];
		cudaMemcpy(subBox, BackGround_.gpu_data(), sizeof(Dtype)*tWidth*tHeight, cudaMemcpyDeviceToHost);
		for (int j = 0; j < 80 * 80; j++){
			subImg.at<float>(j) = (float)subBox[j];
		}
		cv::imshow("background", subImg);

		for (int i = 0; i < 50; i++){
			cudaMemcpy(botBox, &bottom[0]->gpu_data()[tWidth * tHeight * i], sizeof(Dtype)*tWidth*tHeight, cudaMemcpyDeviceToHost);
			cudaMemcpy(subBox, &sub_img_.gpu_data()[tWidth * tHeight * i], sizeof(Dtype)*tWidth*tHeight, cudaMemcpyDeviceToHost);

			for (int j = 0; j < 80 * 80; j++){
				bottomImg.at<float>(j) = (float)botBox[j];
				subImg.at<float>(j) = (float)subBox[j];
			}

			cv::imshow("bottom", bottomImg);
			cv::imshow("subImg", subImg);
			cv::waitKey(0);
		}*/

		//원본 이미지에서 배경아닌 바이너리 넣어주기
		//바이너리 이미지에서 없으면 그냥 배경뎁스 넣어주기
		kernel_depth_val<Dtype> << <CAFFE_GET_BLOCKS(botPosCount),
			CAFFE_CUDA_NUM_THREADS >> >(botPosCount, tWidth, tHeight, batchsize,
			(Dtype)alpha_, depthImg, spatialPos, sub_img_.gpu_data(), depth_info_.mutable_gpu_data());

		//concatenation
		spatial_depth_concat<Dtype> << <CAFFE_GET_BLOCKS(topcount),
			CAFFE_CUDA_NUM_THREADS >> >(topcount, tWidth, tHeight,
			spatialPos, depth_info_.gpu_data(), top[0]->mutable_gpu_data());
	}

	template <typename Dtype>
	void DepthConcatLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		const Dtype* topdiff = top[0]->gpu_diff();
		Dtype* spatialDiff = bottom[1]->mutable_gpu_diff();

		const int spatialcount = bottom[1]->count();

		//sptial positon Layer로만 diff를 생성해줘야함
		concat_spatial_backward<Dtype> << <CAFFE_GET_BLOCKS(spatialcount),
			CAFFE_CUDA_NUM_THREADS >> >(spatialcount, topdiff, spatialDiff);
	}

	INSTANTIATE_LAYER_GPU_FUNCS(DepthConcatLayer);

}  // namespace caffe
