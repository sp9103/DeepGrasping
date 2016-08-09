#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MDNDistLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	data_dim = this->layer_param_.gmm_param().data_dim();
	class_size = this->layer_param_.gmm_param().class_size();
}

template <typename Dtype>
void MDNDistLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
  //    << "top_k must be less than or equal to the number of classes.";
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  //CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
  //    << "Number of labels must match number of predictions; "
  //    << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
  //    << "label count (number of labels) must be N*H*W, "
  //    << "with integer values in {0, 1, ..., C-1}.";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);


}

template <typename Dtype>
void MDNDistLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int batch_size = bottom[0]->shape()[0];
  const Dtype *bottom_data = bottom[0]->cpu_data();
  const Dtype *label_data = bottom[1]->cpu_data();
  int min_idx = -1;
  Dtype MeanDist, presentDist, TotalDist;
  Dtype alpha, alphasum;
  Dtype *diff = new Dtype[data_dim];
  Dtype ang_diff[9], ang_total[9];
  //Dtype diff[90];
  Dtype bot_box[70];
  Dtype label_box[9];

  for (int i = 0; i < data_dim; i++)	ang_total[i] = 0;
  for (int i = 0; i < batch_size; i++){
	  MeanDist = FLT_MAX;
	  //alphasum = 0;
	  memcpy(bot_box, &bottom_data[55 * i], sizeof(Dtype) * 55);
	  memcpy(label_box, &label_data[9 * i], sizeof(Dtype) * 9);
	  for (int j = 0; j < class_size; j++){
		  presentDist = 0;
		  alpha = bottom_data[i*class_size*(data_dim + 2) + j*(data_dim + 2)];
		  //alphasum += alpha;
		  for (int k = 0; k < data_dim; k++){
			  diff[k] = bottom_data[i*class_size*(data_dim + 2) + j*(data_dim + 2) + k + 1] - label_box[k];
			  presentDist += pow(diff[k], 2);
		  }

		  if (MeanDist > presentDist){
			  MeanDist = presentDist;
			  for (int k = 0; k < data_dim; k++)
				  ang_diff[k] = abs(diff[k]);
		  }
	  }

	  TotalDist += sqrt(MeanDist) / batch_size;
	  for (int j = 0; j < data_dim; j++)
		  ang_total[j] += ang_diff[j] / batch_size;
  }

   top[0]->mutable_cpu_data()[0] = TotalDist;
   for (int i = 0; i < 9; i++)
	   printf("%.3f ", ang_total[i] * 180.f / M_PI);
   printf("\n");
   delete[] diff;
}

INSTANTIATE_CLASS(MDNDistLayer);
REGISTER_LAYER_CLASS(MDNDist);

}  // namespace caffe
