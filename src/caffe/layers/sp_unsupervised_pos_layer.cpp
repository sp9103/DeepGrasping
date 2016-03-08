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

bool fileTypeCheckPos(TCHAR *fileName){
	size_t fileLen;
	StringCchLength(fileName, MAX_PATH, &fileLen);

	if (fileLen < 5)
		return false;

	if (fileName[fileLen - 1] != 'g')
		return false;
	if (fileName[fileLen - 2] != 'p')
		return false;
	if (fileName[fileLen - 3] != 'j')
		return false;

	return true;
}

void makeRandboxPos(int *arr, int size){
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
void SPUnsupervisedPosLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
	batch_size_ = this->layer_param_.sp_unsupervised_pos_param().batch_size();
	channels_ = this->layer_param_.sp_unsupervised_pos_param().channels();
	height_ = this->layer_param_.sp_unsupervised_pos_param().height();
	width_ = this->layer_param_.sp_unsupervised_pos_param().width();

	data_path_ = this->layer_param_.sp_unsupervised_pos_param().data_path();
	pos_path_ = this->layer_param_.sp_unsupervised_pos_param().pos_path();
	data_limit_ = this->layer_param_.sp_unsupervised_pos_param().data_limit();

  size_ = channels_ * height_ * width_;
  CHECK_GT(batch_size_ * size_, 0) <<
      "batch_size, channels, height, and width must be specified and"
      " positive in memory_data_param";
  //파일 읽기 부분. 읽은 후 파일의 헤더를 통해서 메모리를 할당.
  //파일 구조 : Datacount(47000), SampleCount(32), width(240), height(240), data...
  PoslaodAll(pos_path_.c_str());

  top[0]->Reshape(batch_size_, channels_, height_, width_);
  vector<int> label_shape = top[0]->shape();
  label_shape.resize(1 + 1);
  label_shape[1] = 64;
  top[1]->Reshape(label_shape);

  //전체 로드
  ImageloadAll(data_path_.c_str());
  CHECK_EQ(data.size(), label.size()) << "data size != label size";
  CHECK_GT(data.size(), 0) << "data is empty";
  //CHECK_EQ(data_blob.size(), label_blob.size()) << "data size != label size";

  //랜덤 박스 생성
  srand(time(NULL));
  randbox = (int*)malloc(sizeof(int)*data.size());
  makeRandboxPos(randbox, data.size());
  /*randbox = (int*)malloc(sizeof(int)*data_blob.size());
  makeRandbox(randbox, data_blob.size());*/
  dataidx = 0;
}

template <typename Dtype>
void SPUnsupervisedPosLayer<Dtype>::Reset(Dtype* data, Dtype* labels, int n) {
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
void SPUnsupervisedPosLayer<Dtype>::set_batch_size(int new_size) {
  /*CHECK(!has_new_data_) <<
      "Can't change batch_size until current data has been consumed.";*/
  batch_size_ = new_size;
}

template <typename Dtype>
void SPUnsupervisedPosLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	//int sTime = clock();
	const float mean_val[3] = { 103.939, 116.779, 123.68 }; // bgr mean
	Dtype* label = top[1]->mutable_cpu_data();
	Dtype* data = top[0]->mutable_cpu_data();

	//Dtype* dataarr = new Dtype[height_ * width_ * channels_];
	//Dtype* labelarr = new Dtype[height_ / 4 * width_ / 4];

	for (int i = 0; i < batch_size_; i++){
		//RGB 로드
		cv::Mat src = this->data.at(randbox[dataidx]);
		struct struct_pos dst = this->label.at(randbox[dataidx]);

		for (int h = 0; h < height_; h++){
			for (int w = 0; w < width_; w++){
				for (int c = 0; c < 3; c++){
					data[c*height_*width_ + width_*h + w] = (Dtype)src.at<cv::Vec3b>(h, w)[c] - (Dtype)mean_val[c];
				}
			}
		}

		for (int j = 0; j < 64; j++)
			label[j] = (Dtype)dst.pos[j];

		label += top[1]->offset(1);
		data += top[0]->offset(1);
		if (dataidx + 1 >= this->data.size()){
			makeRandboxPos(randbox, this->data.size());
			dataidx = 0;
		}
		else
			dataidx++;
	}
}

template <typename Dtype>
void SPUnsupervisedPosLayer<Dtype>::ImageloadAll(const char* datapath){
	WIN32_FIND_DATA ffd;
	HANDLE hFind = INVALID_HANDLE_VALUE;
	TCHAR szDir[MAX_PATH] = { 0, };

	MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, datapath, strlen(datapath), szDir, MAX_PATH);
	StringCchCat(szDir, MAX_PATH, TEXT("\\*"));

	hFind = FindFirstFile(szDir, &ffd);
	while (FindNextFile(hFind, &ffd) != 0){

		TCHAR subDir[MAX_PATH] = { 0, };
		memcpy(subDir, szDir, sizeof(TCHAR)*MAX_PATH);
		size_t len;
		StringCchLength(subDir, MAX_PATH, &len);
		subDir[len - 1] = '\0';
		StringCchCat(subDir, MAX_PATH, ffd.cFileName);
		char tBuf[MAX_PATH];
		WideCharToMultiByte(CP_ACP, 0, subDir, MAX_PATH, tBuf, MAX_PATH, NULL, NULL);

		if (fileTypeCheckPos(ffd.cFileName)){
			cv::Mat dataimage;

			dataimage = cv::imread(tBuf);

			data.push_back(dataimage);
		}

		if ((data_limit_ != 0) && data_limit_ <= data.size())
			break;

		if (ffd.dwFileAttributes == 16 && ffd.cFileName[0] != '.'){
			printf("%s\n", tBuf);
			ImageloadAll(tBuf);
		}
	}
}

template <typename Dtype>
void SPUnsupervisedPosLayer<Dtype>::PoslaodAll(const char* pospath){
	FILE *fp = fopen(pospath, "rb");
	int DataCount;
	int SampleCount;
	int Imgwidth, Imgheight;

	fread(&DataCount, sizeof(int), 1, fp);
	fread(&SampleCount, sizeof(int), 1, fp);
	fread(&Imgheight, sizeof(int), 1, fp);
	fread(&Imgwidth, sizeof(int), 1, fp);

	int loadCount;
	if (data_limit_ == 0)		loadCount = DataCount;
	else loadCount = data_limit_;

	for (int i = 0; i < loadCount; i++){
		struct struct_pos temp;
		for (int j = 0; j < SampleCount; j++){
			int X, Y;
			fread(&X, sizeof(int), 1, fp);
			fread(&Y, sizeof(int), 1, fp);

			temp.pos[j * 2 + 0] = (float)X / Imgwidth;
			temp.pos[j * 2 + 1] = (float)Y / Imgheight;
		}

		label.push_back(temp);
	}

	fclose(fp);
}

INSTANTIATE_CLASS(SPUnsupervisedPosLayer);
REGISTER_LAYER_CLASS(SPUnsupervisedPos);

}  // namespace caffe
