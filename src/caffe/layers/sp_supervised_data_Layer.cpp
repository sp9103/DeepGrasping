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


bool BinfileTypeCheck(TCHAR *fileName){
	size_t fileLen;
	StringCchLength(fileName, MAX_PATH, &fileLen);

	if (fileLen < 5)
		return false;

	if (fileName[fileLen - 1] != 'n')
		return false;
	if (fileName[fileLen - 2] != 'i')
		return false;
	if (fileName[fileLen - 3] != 'b')
		return false;

	return true;
}

void BinmakeRandbox(int *arr, int size){
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
void SPsupervisedDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
	batch_size_ = this->layer_param_.sp_supervised_data_param().batch_size();
	height_ = this->layer_param_.sp_supervised_data_param().height();
	width_ = this->layer_param_.sp_supervised_data_param().width();

	data_path_ = this->layer_param_.sp_supervised_data_param().data_path();
	data_limit_ = this->layer_param_.sp_supervised_data_param().data_limit();

  size_ = height_ * width_;
  CHECK_GT(batch_size_ * size_, 0) <<
      "batch_size, channels, height, and width must be specified and"
      " positive in memory_data_param";
  //vector<int> label_shape(1, batch_size_);
  top[0]->Reshape(batch_size_, 3, height_, width_);
  top[1]->Reshape(batch_size_, 1, height_, width_);
  
  vector<int> label_shape = top[0]->shape();
  label_shape.resize(1 + 1);
  label_shape[1] = 9;
  top[2]->Reshape(label_shape);
  //top[2]->Reshape(batch_size_, 3*3);

  //전체 로드
  BinFileloadAll(data_path_.c_str());
  CHECK_EQ(data_blob.size(), label_blob.size()) << "data size != label size";
  CHECK_GT(data_blob.size(), 0) << "data is empty";
  //CHECK_EQ(data_blob.size(), label_blob.size()) << "data size != label size";

  //랜덤 박스 생성
  srand(time(NULL));
  randbox = (int*)malloc(sizeof(int)*data_blob.size());
  BinmakeRandbox(randbox, data_blob.size());
  dataidx = 0;
}

template <typename Dtype>
void SPsupervisedDataLayer<Dtype>::Reset(Dtype* data, Dtype* labels, int n) {
  CHECK(data);
  CHECK(labels);
  CHECK_EQ(n % batch_size_, 0) << "n must be a multiple of batch size";
  // Warn with transformation parameters since a memory array is meant to
  // be generic and no transformations are done with Reset().
  if (this->layer_param_.has_transform_param()) {
    LOG(WARNING) << this->type() << " does not transform array data on Reset()";
  }
  n_ = n;
  pos_ = 0;
}

template <typename Dtype>
void SPsupervisedDataLayer<Dtype>::set_batch_size(int new_size) {
  /*CHECK(!has_new_data_) <<
      "Can't change batch_size until current data has been consumed.";*/
  batch_size_ = new_size;
}

template <typename Dtype>
void SPsupervisedDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	////int sTime = clock();
	Dtype* depth = top[1]->mutable_cpu_data();
	Dtype* rgb = top[0]->mutable_cpu_data();
	Dtype* label = top[2]->mutable_cpu_data();

	for (int i = 0; i < batch_size_; i++){
		RGBImgData srcImg = this->data_blob.at(randbox[dataidx]);
		DEPTHImgData depthImg = this->Depth_blob.at(randbox[dataidx]);
		LabelPosData labelpos = this->label_blob.at(randbox[dataidx]);

		caffe_copy(3 * height_ * width_, srcImg.data, rgb);
		caffe_copy(1 * height_ * width_, depthImg.data, depth);
		caffe_copy(3*3, labelpos.pos, label);

		//cv::Mat rgbDtype, depthDtype;
		//Dtype pos[9];
		//rgbDtype.create(80, 80, CV_32FC3);
		//depthDtype.create(80, 80, CV_32FC1);

		//for (int i = 0; i < 80 * 80; i++){
		//	for (int c = 0; c < 3; c++){
		//		//tempRGBData.data[c*height_*width_ + width_*h + w] = (float)RGBimg.at<cv::Vec3b>(h, w)[c] / 255.0f;
		//		rgbDtype.at<cv::Vec3f>(i / width_, i % width_)[c] = rgb[c*height_*width_ + i];
		//	}
		//	depthDtype.at<float>(i / width_, i % width_) = depthImg.data[i];
		//}

		//memcpy(pos, label, sizeof(Dtype) * 9);

		label += top[2]->offset(1);
		rgb += top[0]->offset(1);
		depth += top[1]->offset(1);

		if (dataidx + 1 >= this->data_blob.size()){
			BinmakeRandbox(randbox, this->data_blob.size());
			dataidx = 0;
		}
		else
			dataidx++;
	}
}

template <typename Dtype>
void SPsupervisedDataLayer<Dtype>::BinFileloadAll(const char* datapath){
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

		if (BinfileTypeCheck(ffd.cFileName)){
			cv::Mat dataimage, depthImage;
			cv::Point3f label[3];
			FILE *fp_ = fopen(tBuf, "rb");;

			//하나의 바이너리마다 하나의 EndEffector
			int RGBPathLen, DEPTHPathLen;
			char RGBPath[256], DepthPath[256];
			cv::Point3f UpperLeft, UpperRight, Thumb;
			int UpperLeftAngle, UpperRightAngle, ThumbAngle;

			fread(&RGBPathLen, sizeof(int), 1, fp_);
			fread(RGBPath, sizeof(char), RGBPathLen, fp_);
			RGBPath[RGBPathLen] = '\0';
			fread(&DEPTHPathLen, sizeof(int), 1, fp_);
			fread(DepthPath, sizeof(char), DEPTHPathLen, fp_);
			DepthPath[DEPTHPathLen] = '\0';

			char rootDir[MAX_PATH], rootAddedRGB[MAX_PATH], rootAddedDEPTH[MAX_PATH];
			WideCharToMultiByte(CP_ACP, 0, szDir, MAX_PATH, rootDir, MAX_PATH, NULL, NULL);

			int len;
			strcpy(rootAddedRGB, rootDir);
			len = strlen(rootAddedRGB);
			rootAddedRGB[len-1] = '\0';
			strcat(rootAddedRGB, RGBPath);

			strcpy(rootAddedDEPTH, rootDir);
			len = strlen(rootAddedDEPTH);
			rootAddedDEPTH[len-1] = '\0';
			strcat(rootAddedDEPTH, DepthPath);

			cv::Mat RGBimg = cv::imread(rootAddedRGB);
			cv::Mat DepthImg = cv::imread(rootAddedDEPTH);

			fread(&UpperLeftAngle, sizeof(int), 1, fp_);
			fread(&UpperRightAngle, sizeof(int), 1, fp_);
			fread(&ThumbAngle, sizeof(int), 1, fp_);

			if (UpperLeftAngle < 0 || UpperLeftAngle > 3000)
				continue;
			if (UpperRightAngle < 0 || UpperRightAngle > 3000)
				continue;
			if (ThumbAngle < 0 || ThumbAngle > 3000)
				continue;

			float x, y, z;
			fread(&x, sizeof(float), 1, fp_);
			fread(&y, sizeof(float), 1, fp_);
			fread(&z, sizeof(float), 1, fp_);
			UpperLeft = cv::Point3f(x, y, z);

			fread(&x, sizeof(float), 1, fp_);
			fread(&y, sizeof(float), 1, fp_);
			fread(&z, sizeof(float), 1, fp_);
			UpperRight = cv::Point3f(x, y, z);

			fread(&x, sizeof(float), 1, fp_);
			fread(&y, sizeof(float), 1, fp_);
			fread(&z, sizeof(float), 1, fp_);
			Thumb = cv::Point3f(x, y, z);

			fclose(fp_);

			if (Thumb.z > 100 || UpperRight.z > 100 && UpperLeft.z > 100)
				continue;

			cv::resize(RGBimg, RGBimg, cv::Size(height_, width_));
			cv::resize(DepthImg, DepthImg, cv::Size(height_, width_));
			cv::cvtColor(DepthImg, DepthImg, CV_BGR2GRAY);

			RGBImgData tempRGBData;
			DEPTHImgData tempDepthData;
			LabelPosData tempLabel;
			for (int h = 0; h < height_; h++){
				for (int w = 0; w < width_; w++){
					for (int c = 0; c < 3; c++)
						tempRGBData.data[c*height_*width_ + width_*h + w] = (float)RGBimg.at<cv::Vec3b>(h, w)[c] / 255.0f;
					tempDepthData.data[width_*h + w] = (float)DepthImg.at<uchar>(h, w) / 255.0f;
				}
			}

			tempLabel.pos[0] = UpperLeft.x;
			tempLabel.pos[1] = UpperLeft.y;
			tempLabel.pos[2] = UpperLeft.z;

			tempLabel.pos[3] = UpperRight.x;
			tempLabel.pos[4] = UpperRight.y;
			tempLabel.pos[5] = UpperRight.z;

			tempLabel.pos[6] = Thumb.x;
			tempLabel.pos[7] = Thumb.y;
			tempLabel.pos[8] = Thumb.z;

			data_blob.push_back(tempRGBData);
			Depth_blob.push_back(tempDepthData);
			label_blob.push_back(tempLabel);
		}

		if ((data_limit_ != 0) && data_limit_ <= data_blob.size())
			break;

		if (ffd.dwFileAttributes == 16 && ffd.cFileName[0] != '.'){
			printf("%s\n", tBuf);
			BinFileloadAll(tBuf);
		}
	}
}

INSTANTIATE_CLASS(SPsupervisedDataLayer);
REGISTER_LAYER_CLASS(SPsupervisedData);

}  // namespace caffe
