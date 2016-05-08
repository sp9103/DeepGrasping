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
	__global__ void slice_depth_concat(const int count, const int top_count, const int bot_count,
		const int bottom_channel, const int bottom_width, const int bottom_height,
		const int depth_width, const int depth_height,
		const Dtype* bottom_data, const Dtype* depthval, Dtype* topdata) {
		CUDA_KERNEL_LOOP(index, count) {
			int inner_idx = index % top_count;

			if (inner_idx < bot_count)
				topdata[index] = bottom_data[index];
			else{
				int depth_idx = inner_idx - bot_count;
				int depth_val_idx = depth_idx / (bottom_height * bottom_width) * depth_width * depth_height;
				topdata[index] = depthval[depth_val_idx];
			}
		}
	}

	template <typename Dtype>
	__global__ void concat_slice_backward(const int count, const int top_count, const int bot_count,
		const Dtype* topdiff, Dtype* spatial) {
		CUDA_KERNEL_LOOP(index, count) {
			const int featureIdx = index / bot_count;
			const int inneridx = index % bot_count;				//0 : xpos, 1 : ypos 2 : depthvalue

			const int topid = featureIdx * top_count + inneridx;

			spatial[index] = topdiff[topid];
		}
	}

	template <typename Dtype>
	void DepthSliceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const int topcount = top[0]->count();
		const int tWidth = bottom[1]->shape()[1];
		const int tHeight = bottom[1]->shape()[2];
		const int bottom_width = bottom[0]->shape()[2];
		const int bottom_height = bottom[0]->shape()[3];
		const int bottom_channel = bottom[0]->shape()[1];
		int bot_num = bottom_width * bottom_height * bottom_channel;

		const Dtype* bottomImg = bottom[0]->gpu_data();						//spatial feature
		const Dtype* depthImg = bottom[1]->gpu_data();							//Depth image

		//concatenation
		slice_depth_concat<Dtype> << <CAFFE_GET_BLOCKS(topcount),
			CAFFE_CUDA_NUM_THREADS >> >(topcount, N_, bot_num,
			bottom_channel, bottom_width, bottom_height,
			tWidth, tHeight,
			bottomImg, depthImg, top[0]->mutable_gpu_data());
	}

	template <typename Dtype>
	void DepthSliceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		const Dtype* topdiff = top[0]->gpu_diff();
		Dtype* spatialDiff = bottom[0]->mutable_gpu_diff();

		const int bottom_count = bottom[0]->count();
		const int top_num = top[0]->shape()[1];
		int bot_num = bottom[0]->shape()[1] * bottom[0]->shape()[2] * bottom[0]->shape()[3];

		//sptial positon Layer로만 diff를 생성해줘야함
		concat_slice_backward<Dtype> << <CAFFE_GET_BLOCKS(bottom_count),
			CAFFE_CUDA_NUM_THREADS >> >(bottom_count, top_num, bot_num, topdiff, spatialDiff);
	}

	INSTANTIATE_LAYER_GPU_FUNCS(DepthSliceLayer);

}  // namespace caffe
