#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanDistLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.accuracy_param().top_k();
}

template <typename Dtype>
void EuclideanDistLayer<Dtype>::Reshape(
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


  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EuclideanDistLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
	  count,
	  bottom[0]->cpu_data(),
	  bottom[1]->cpu_data(),
	  diff_.mutable_cpu_data());

  int batchsize = bottom[0]->shape()[0];
  int botLen = bottom[0]->shape()[1];
  Dtype dist = 0;
  const Dtype* diffdata = diff_.cpu_data();
  const Dtype* bot = bottom[1]->cpu_data();
  const Dtype* output = bottom[0]->cpu_data();

  for (int i = 0; i < batchsize; i++){
	  float sq = 0;
	  for (int k = 0; k < 3; k++){
		  sq += diffdata[botLen*i + k] * diffdata[botLen*i + k];
	  }
	  Dtype tempDist = sqrt(sq);
	  dist += tempDist / batchsize;

	  //int step = diff_.offset(1);
	  //diffdata += step;
	  //bot += step;
  }

   top[0]->mutable_cpu_data()[0] = dist;
   LOG(INFO) << "Distance: " << dist;

   //const Dtype* label = bottom[2]->cpu_data();
   //cv::Mat img(160, 160, CV_32FC3);
   //for (int i = 0; i < 20; i++){
	  // char buf[256];

	  // for (int h = 0; h < 160; h++){
		 //  for (int w = 0; w < 160; w++){
			//   for (int c = 0; c < 3; c++){
			//	   //tempdataMat.at<float>(c*height_*width_ + width_*h + w) = (float)dataimage.at<cv::Vec3b>(h, w)[c]
			//	   img.at<cv::Vec3f>(h, w)[c] = (float)bot[i * 160 * 160 * 3 + c * 160 * 160 + 160 * h + w];
			//   }
		 //  }
	  // }

	  // cv::Point pos;
	  // pos.x = output[2 * i + 0];
	  // pos.y = output[2 * i + 1];
	  // cv::circle(img, pos, 5, cv::Scalar(0, 0, 255), -1);
	  // pos.x = label[2 * i + 0];
	  // pos.y = label[2 * i + 1];
	  // cv::circle(img, pos, 5, cv::Scalar(255, 0, 0), -1);

	  // sprintf(buf, "%d", i);
	  // cv::imshow(buf, img);
	  // cv::waitKey(0);
   //}

}

INSTANTIATE_CLASS(EuclideanDistLayer);
REGISTER_LAYER_CLASS(EuclideanDist);

}  // namespace caffe
