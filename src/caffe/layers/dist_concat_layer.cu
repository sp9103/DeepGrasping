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
	__global__ void spatial_dist_concat(const int count, const int concat_idx, const int bot_count,
		const Dtype* spatial_pos, const Dtype* distVal, Dtype* topdata) {
		CUDA_KERNEL_LOOP(index, count) {
			const int interIdx = index % concat_idx;
			const int outerIdx = index / concat_idx;

			if (interIdx == concat_idx - 1){
				topdata[index] = distVal[outerIdx];
			}
			else{
				topdata[index] = spatial_pos[outerIdx * bot_count + interIdx];
			}
		}
	}

	template <typename Dtype>
	__global__ void concat_dist_backward(const int count,
		const Dtype* topdiff, Dtype* spatial) {
		CUDA_KERNEL_LOOP(index, count) {
			const int featureIdx = index / 2;
			const int id = index % 2;				//0 : xpos, 1 : ypos 2 : depthvalue

			const int topid = featureIdx * 3 + id;

			spatial[index] = topdiff[topid];
		}
	}

	template <typename Dtype>
	void DistConcatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const int topcount = top[0]->count();
		const int concat_idx = bottom[0]->shape()[1] + 1;
		const int bot_count = bottom[0]->shape()[1];

		const Dtype* spatialPos = bottom[0]->gpu_data();						//spatial feature
		const Dtype* dist = bottom[1]->gpu_data();							//Depth image

		//concatenation
		spatial_dist_concat<Dtype> << <CAFFE_GET_BLOCKS(topcount),
			CAFFE_CUDA_NUM_THREADS >> >(topcount, concat_idx, bot_count,
			spatialPos, dist, top[0]->mutable_gpu_data());
	}

	template <typename Dtype>
	void DistConcatLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		const Dtype* topdiff = top[0]->gpu_diff();
		Dtype* spatialDiff = bottom[0]->mutable_gpu_diff();

		const int spatialcount = bottom[0]->count();

		//sptial positon Layer로만 diff를 생성해줘야함
		concat_dist_backward<Dtype> << <CAFFE_GET_BLOCKS(spatialcount),
			CAFFE_CUDA_NUM_THREADS >> >(spatialcount, topdiff, spatialDiff);
	}

	INSTANTIATE_LAYER_GPU_FUNCS(DistConcatLayer);

}  // namespace caffe
