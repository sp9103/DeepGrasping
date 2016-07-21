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
			const int Internal_idx = index % 3;
			const int Feature_idx = index / 3;

			//x pos
			if (Internal_idx == 0)
				topdata[index] = spatial_pos[2 * Feature_idx + 0];
			//y pos
			else if (Internal_idx == 1)
				topdata[index] = spatial_pos[2 * Feature_idx + 1];
			//depth val
			else if (Internal_idx == 2){
				const int d_x = width * spatial_pos[2 * Feature_idx + 0];
				const int d_y = height * spatial_pos[2 * Feature_idx + 1];
				topdata[index] = depthval[d_x + d_y * width] / 1000.f;
			}
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
		const int tWidth = bottom[1]->shape()[1];
		const int tHeight = bottom[1]->shape()[2];

		const Dtype* spatialPos = bottom[0]->gpu_data();						//spatial feature
		const Dtype* depthImg = bottom[1]->gpu_data();							//Depth image

		//concatenation
		spatial_depth_concat<Dtype> << <CAFFE_GET_BLOCKS(topcount),
			CAFFE_CUDA_NUM_THREADS >> >(topcount, tWidth, tHeight,
			spatialPos, depthImg, top[0]->mutable_gpu_data());
	}

	template <typename Dtype>
	void DepthConcatLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		const Dtype* topdiff = top[0]->gpu_diff();
		Dtype* spatialDiff = bottom[0]->mutable_gpu_diff();

		const int spatialcount = bottom[0]->count();

		//sptial positon Layer로만 diff를 생성해줘야함
		concat_spatial_backward<Dtype> << <CAFFE_GET_BLOCKS(spatialcount),
			CAFFE_CUDA_NUM_THREADS >> >(spatialcount, topdiff, spatialDiff);
	}

	INSTANTIATE_LAYER_GPU_FUNCS(DepthConcatLayer);

}  // namespace caffe
