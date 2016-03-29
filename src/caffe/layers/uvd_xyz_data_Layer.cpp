#include <opencv2/core/core.hpp>

#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"

#include <opencv2\opencv.hpp>
#include <conio.h>
#include <strsafe.h>
#include <Windows.h>
#include <time.h>

void UVDXYZmakeRandbox(int *arr, int size){
	for (int i = 0; i < size; i++)
		arr[i] = i;
	for (int i = 0; i < size; i++){
		int tidx = rand() % size;
		int t = arr[i];
		arr[i] = arr[tidx];
		arr[tidx] = t;
	}
}

namespace caffe {

template <typename Dtype>
void UVDXYZDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
	batch_size_ = this->layer_param_.sp_unsupervised_data_param().batch_size();

	data_path_ = this->layer_param_.sp_unsupervised_data_param().data_path();

  size_ = 3;
  CHECK_GT(batch_size_ * size_, 0) <<
      "batch_size, channels, height, and width must be specified and"
      " positive in memory_data_param";
  //vector<int> label_shape(1, batch_size_);

  vector<int> top_shape;
  top_shape.resize(1 + 1);
  top_shape[0] = batch_size_;
  top_shape[1] = 3;

  top[0]->Reshape(top_shape);

  top_shape[1] = 3;								//label shape
  top[1]->Reshape(top_shape);

  //top[0]->Reshape(batch_size_, 1, 1, 3);
  //top[1]->Reshape(batch_size_, 1, 1, 3);

  //전체 로드
  readData(data_path_.c_str());
  CHECK_EQ(data.size(), label.size()) << "data size != label size";
  CHECK_GT(data.size(), 0) << "data is empty";
  //CHECK_EQ(data_blob.size(), label_blob.size()) << "data size != label size";

  //랜덤 박스 생성
  srand(time(NULL));
  randbox = (int*)malloc(sizeof(int)*data.size());
  UVDXYZmakeRandbox(randbox, data.size());
  dataidx = 0;
}

template <typename Dtype>
void UVDXYZDataLayer<Dtype>::Reset(Dtype* data, Dtype* labels, int n) {
  CHECK(data);
  CHECK(labels);
  CHECK_EQ(n % batch_size_, 0) << "n must be a multiple of batch size";
  // Warn with transformation parameters since a memory array is meant to
  // be generic and no transformations are done with Reset().
  if (this->layer_param_.has_transform_param()) {
    LOG(WARNING) << this->type() << " does not transform array data on Reset()";
  }
  n_ = n;
}

template <typename Dtype>
void UVDXYZDataLayer<Dtype>::set_batch_size(int new_size) {
  /*CHECK(!has_new_data_) <<
      "Can't change batch_size until current data has been consumed.";*/
  batch_size_ = new_size;
}

template <typename Dtype>
void UVDXYZDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	////int sTime = clock();
	Dtype* uvd = top[0]->mutable_cpu_data();
	Dtype* xyz = top[1]->mutable_cpu_data();

	for (int i = 0; i < batch_size_; i++){
		cv::Point3f data = this->data.at(randbox[dataidx]);
		cv::Point3f label = this->label.at(randbox[dataidx]);

		/*if (data.z <= 0 || label.z <= 0){
			printf("Data input error\n");
			continue;
		}*/

		/*data.x /= 160;
		data.y /= 160;
		data.z /= 1000.f;*/
		uvd[0] = (Dtype)data.x;
		uvd[1] = (Dtype)data.y;
		uvd[2] = (Dtype)data.z;

		xyz[0] = (Dtype)label.x;
		xyz[1] = (Dtype)label.y;
		xyz[2] = (Dtype)label.z;

		if (dataidx + 1 >= this->data.size()){
			UVDXYZmakeRandbox(randbox, this->data.size());
			dataidx = 0;
		}
		else
			dataidx++;

		uvd += top[0]->offset(1);
		xyz += top[1]->offset(1);
	}
}

template <typename Dtype>
void UVDXYZDataLayer<Dtype>::readData(const char *path){
	//Open File
	FILE *Datafp = fopen(path, "rb");
	if (Datafp == NULL){
		printf("Can not open file\n");
	}

	//Read File
	cv::Point3f uvd, xyz;
	int i = 0;

	float xMax, yMax, zMax;
	float xMin, yMin, zMin;
	xMax = yMax = zMax = -99999.f;
	xMin = yMin = zMin = 99999.f;

	while (!feof(Datafp)){
		fread(&uvd, sizeof(cv::Point3f), 1, Datafp);
		fread(&xyz, sizeof(cv::Point3f), 1, Datafp);

		data.push_back(uvd);
		label.push_back(xyz);
	}
	printf("Data Load Complete!\n");
}

INSTANTIATE_CLASS(UVDXYZDataLayer);
REGISTER_LAYER_CLASS(UVDXYZData);

}  // namespace caffe
