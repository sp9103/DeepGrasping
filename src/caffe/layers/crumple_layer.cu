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
void CrumpleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();

	caffe_copy(bottom[0]->count(), bottom_data, top_data);

	//if (Mode_ == 2){
	//	FILE *fp = fopen("feature.txt", "w");
	//	Dtype featurebox[192];
	//	cudaMemcpy(featurebox, top[0]->gpu_data(), sizeof(Dtype) * 192, cudaMemcpyDeviceToHost);

	//	for (int i = 0; i < 192 / 3; i++){
	//		fprintf(fp, "%f %f %f\n", featurebox[i * 3 + 0], featurebox[i * 3 + 1], featurebox[i * 3 + 2]);
	//	}

	//	fclose(fp);
	//}
}

template <typename Dtype>
void CrumpleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	const Dtype* top_diff = top[0]->gpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

	caffe_copy(bottom[0]->count(), top_diff, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(CrumpleLayer);

}  // namespace caffe
