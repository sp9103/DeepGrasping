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

#define SWAP(a,b,t) (t)=(a), (a)=(b), (b)=(t)

namespace caffe {

template <typename Dtype>
void ApproachDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
	batch_size_ = this->layer_param_.sp_unsupervised_data_param().batch_size();
	channels_ = this->layer_param_.sp_unsupervised_data_param().channels();
	height_ = this->layer_param_.sp_unsupervised_data_param().height();
	width_ = this->layer_param_.sp_unsupervised_data_param().width();

	data_path_ = this->layer_param_.sp_unsupervised_data_param().data_path();
	data_limit_ = this->layer_param_.sp_unsupervised_data_param().data_limit();

  size_ = channels_ * height_ * width_;
  CHECK_GT(batch_size_ * size_, 0) <<
      "batch_size, channels, height, and width must be specified and"
      " positive in memory_data_param";

  top[0]->Reshape(batch_size_, channels_, height_, width_);					//[0] RGB
  std::vector<int> depth_dim(3);
  depth_dim[0] = batch_size_;
  depth_dim[1] = height_;
  depth_dim[2] = width_;
  top[1]->Reshape(depth_dim);												//[1] Depth

  std::vector<int> pos_dim(2);
  pos_dim[0] = batch_size_;
  pos_dim[1] = 9;
  top[2]->Reshape(pos_dim);													//[2] Pregrasping postion (label)

  //전체 로드
  Approach_DataLoadAll(data_path_.c_str());
  CHECK_GT(approach_blob.size(), 0) << "data is empty";

  //랜덤 박스 생성
  srand(time(NULL));
  randbox = (int*)malloc(sizeof(int)*approach_blob.size());
  makeRandbox(randbox, approach_blob.size());
  dataidx = 0;
}

template <typename Dtype>
void ApproachDataLayer<Dtype>::Reset(Dtype* data, Dtype* labels, int n) {
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
void ApproachDataLayer<Dtype>::set_batch_size(int new_size) {
  /*CHECK(!has_new_data_) <<
      "Can't change batch_size until current data has been consumed.";*/
  batch_size_ = new_size;
}

template <typename Dtype>
void ApproachDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	Dtype* rgb_data = top[0]->mutable_cpu_data();					//[0] RGB
	Dtype* depth_data = top[1]->mutable_cpu_data();					//[1] Depth
	Dtype* motion_data = top[2]->mutable_cpu_data();					//[2] MotionAngle

	for (int i = 0; i < batch_size_; i++){
		//RGB 로드
		appraoch_label tempLabel = this->approach_blob.at(randbox[dataidx]);
		int idx = tempLabel.imgIdx;
		cv::Mat	rgbImg = this->image_blob.at(idx);
		cv::Mat depthImg = this->depth_blob.at(idx).clone();
		float *label_motion = tempLabel.angle;

		caffe_copy(channels_ * height_ * width_, rgbImg.ptr<Dtype>(0), rgb_data);
		caffe_copy(height_ * width_, depthImg.ptr<Dtype>(0), depth_data);
		caffe_copy(9, (Dtype*)label_motion, motion_data);

		//motion data 확인해보기

		rgb_data += top[0]->offset(1);
		depth_data += top[1]->offset(1);
		motion_data += top[2]->offset(1);

		if (dataidx + 1 >= this->approach_blob.size()){
			makeRandbox(randbox, this->approach_blob.size());
			dataidx = 0;
		}
		else
			dataidx++;
	}
}

template <typename Dtype>
void ApproachDataLayer<Dtype>::Approach_DataLoadAll(const char* datapath){
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

		//Tchar to char
		char ccFileName[256];
		WideCharToMultiByte(CP_ACP, 0, ffd.cFileName, len, ccFileName, 256, NULL, NULL);
		printf("Object : %s load.\n", ccFileName);

		if (ccFileName[0] != '.'){
			WIN32_FIND_DATA class_ffd;
			TCHAR szObjDir[MAX_PATH] = { 0, };
			HANDLE hObjFind = INVALID_HANDLE_VALUE;
			char ObjDir[256];
			strcpy(ObjDir, tBuf);
			strcat(ObjDir, "\\RGB\\*");
			MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, ObjDir, strlen(ObjDir), szObjDir, MAX_PATH);
			hObjFind = FindFirstFile(szObjDir, &class_ffd);

			while (FindNextFile(hObjFind, &class_ffd) != 0){
				char ObjFileName[256];
				size_t Grasplen;
				StringCchLength(class_ffd.cFileName, MAX_PATH, &Grasplen);
				WideCharToMultiByte(CP_ACP, 0, class_ffd.cFileName, 256, ObjFileName, 256, NULL, NULL);

				if (ObjFileName[0] == '.')
					continue;

				char MotionDataFile[256], ObjImageFile[256], ObjDepthFile[256], GraspCOMFile[256];
				int filePathLen;
				FILE *fp;
				//1. RGB 읽어오기
				sprintf(ObjImageFile, "%s\\RGB\\%s", tBuf, ObjFileName);
				cv::Mat rgb = cv::imread(ObjImageFile);
				cv::Mat tempdataMat(height_, width_, CV_32FC3);
				for (int h = 0; h < rgb.rows; h++){
					for (int w = 0; w < rgb.cols; w++){
						for (int c = 0; c < rgb.channels(); c++){
							tempdataMat.at<float>(c*height_*width_ + width_*h + w) = (float)rgb.at<cv::Vec3b>(h, w)[c] / 255.0f;
						}
					}
				}

				//2.Depth 읽어오기
				sprintf(ObjDepthFile, "%s\\DEPTH\\%s", tBuf, ObjFileName);
				int depthwidth, depthheight, depthType;
				filePathLen = strlen(ObjDepthFile);
				ObjDepthFile[filePathLen - 1] = 'n';
				ObjDepthFile[filePathLen - 2] = 'i';
				ObjDepthFile[filePathLen - 3] = 'b';
				strcat(ObjDepthFile, ".bin");
				fp = fopen(ObjDepthFile, "rb");
				if (fp == NULL)	continue;
				fread(&depthwidth, sizeof(int), 1, fp);
				fread(&depthheight, sizeof(int), 1, fp);
				fread(&depthType, sizeof(int), 1, fp);
				cv::Mat depthMap(depthheight, depthwidth, depthType);
				for (int i = 0; i < depthMap.rows * depthMap.cols; i++)		fread(&depthMap.at<float>(i), sizeof(float), 1, fp);
				fclose(fp);

				int imgCount = image_blob.size();
				//APPROACH motion 
				TCHAR szMotionDir[MAX_PATH] = { 0, };
				sprintf(MotionDataFile, "%s\\APPRAOCH\\%s", tBuf, ObjFileName);
				filePathLen = strlen(MotionDataFile);
				MotionDataFile[filePathLen - 4] = '\0';
				strcat(MotionDataFile, "\\");
				strcat(MotionDataFile, "MOTION\\*");
				MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, MotionDataFile, strlen(MotionDataFile), szMotionDir, MAX_PATH);
				HANDLE hMotionFind = FindFirstFile(szMotionDir, &class_ffd);
				while (FindNextFile() ! = ){
				}
				//fp = fopen(MotionDataFile, "r");
				//while (!feof(fp)){
				//	//upper Left, Upper Right, Thumb
				//	cv::Mat posMat(9, 1, CV_32FC1);
				//	float tempFloat;
				//	cv::Point3f UpperLeft, UpperRight, Thumb, swapPoint;
				//	fscanf(fp, "%f %f %f ", &UpperLeft.x, &UpperLeft.y, &UpperLeft.z);
				//	fscanf(fp, "%f %f %f ", &UpperRight.x, &UpperRight.y, &UpperRight.z);
				//	fscanf(fp, "%f %f %f\n", &Thumb.x, &Thumb.y, &Thumb.z);

				//	//mm -> M 단위로 변경
				//	posMat.at<float>(0) = UpperLeft.x / 100.f;
				//	posMat.at<float>(1) = UpperLeft.y / 100.f;
				//	posMat.at<float>(2) = UpperLeft.z / 100.f;
				//	posMat.at<float>(3) = UpperRight.x / 100.f;
				//	posMat.at<float>(4) = UpperRight.y / 100.f;
				//	posMat.at<float>(5) = UpperRight.z / 100.f;
				//	posMat.at<float>(6) = Thumb.x / 100.f;
				//	posMat.at<float>(7) = Thumb.y / 100.f;
				//	posMat.at<float>(8) = Thumb.z / 100.f;

				//	std::pair<int, cv::Mat> tempPair;
				//	tempPair.first = imgCount;
				//	tempPair.second = posMat.clone();
				//	pos_blob.push_back(tempPair);

				//	if ((data_limit_ != 0) && data_limit_ <= pos_blob.size())
				//		break;
				//}
				//fclose(fp);
				//motion file search

				//save
				image_blob.push_back(tempdataMat.clone());
				depth_blob.push_back(depthMap.clone());
				//approach_blob.push_back();

				if ((data_limit_ != 0) && data_limit_ <= approach_blob.size())
					break;
			}

		}
	}
}

template <typename Dtype>
bool ApproachDataLayer<Dtype>::fileTypeCheck(char *fileName){
	size_t fileLen;
	fileLen = strlen(fileName);

	if (fileLen < 5)
		return false;

	if (fileName[fileLen - 1] != 'g' && fileName[fileLen - 1] != 'p')
		return false;
	if (fileName[fileLen - 2] != 'p' && fileName[fileLen - 2] != 'm')
		return false;
	if (fileName[fileLen - 3] != 'j' && fileName[fileLen - 3] != 'b')
		return false;

	return true;
}

template <typename Dtype>
void ApproachDataLayer<Dtype>::makeRandbox(int *arr, int size){
	for (int i = 0; i < size; i++)
		arr[i] = i;
	for (int i = 0; i < size; i++){
		int tidx = rand() % size;
		int t = arr[i];
		arr[i] = arr[tidx];
		arr[tidx] = t;
	}
}

INSTANTIATE_CLASS(ApproachDataLayer);
REGISTER_LAYER_CLASS(ApproachData);

}  // namespace caffe
