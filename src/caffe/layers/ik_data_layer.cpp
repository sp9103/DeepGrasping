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
void IKDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
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

  std::vector<int> ang_dim(2);
  ang_dim[0] = batch_size_;
  ang_dim[1] = 9;
  top[2]->Reshape(ang_dim);													//[2] Angle (label)

  //전체 로드
  IK_DataLoadAll(data_path_.c_str());
  CHECK_GT(ang_blob.size(), 0) << "data is empty";

  //랜덤 박스 생성
  srand(time(NULL));
  randbox = (int*)malloc(sizeof(int)*ang_blob.size());
  makeRandbox(randbox, ang_blob.size());
  dataidx = 0;
}

template <typename Dtype>
void IKDataLayer<Dtype>::Reset(Dtype* data, Dtype* labels, int n) {
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
void IKDataLayer<Dtype>::set_batch_size(int new_size) {
  /*CHECK(!has_new_data_) <<
      "Can't change batch_size until current data has been consumed.";*/
  batch_size_ = new_size;
}

template <typename Dtype>
void IKDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	Dtype* rgb_data = top[0]->mutable_cpu_data();					//[0] RGB
	Dtype* depth_data = top[1]->mutable_cpu_data();					//[1] Depth

	Dtype* ang_data = top[2]->mutable_cpu_data();					//[2] ang postion (label)

	for (int i = 0; i < batch_size_; i++){
		//RGB 로드
		std::pair<int, cv::Mat> angPair = this->ang_blob.at(randbox[dataidx]);
		int idx = angPair.first;
		cv::Mat	rgbImg = this->image_blob.at(idx);
		cv::Mat depthImg = this->depth_blob.at(idx).clone();
		cv::Mat ang = angPair.second.clone();

		caffe_copy(channels_ * height_ * width_, rgbImg.ptr<Dtype>(0), rgb_data);
		caffe_copy(height_ * width_, depthImg.ptr<Dtype>(0), depth_data);
		caffe_copy(9, ang.ptr<Dtype>(0), ang_data);

		rgb_data += top[0]->offset(1);
		depth_data += top[1]->offset(1);

		ang_data += top[2]->offset(1);
		if (dataidx + 1 >= this->ang_blob.size()){
			makeRandbox(randbox, this->ang_blob.size());
			dataidx = 0;
		}
		else
			dataidx++;
	}
}

template <typename Dtype>
void IKDataLayer<Dtype>::IK_DataLoadAll(const char* datapath){
	WIN32_FIND_DATA ffd;
	HANDLE hFind = INVALID_HANDLE_VALUE;
	TCHAR szDir[MAX_PATH] = { 0, };

	MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, datapath, strlen(datapath), szDir, MAX_PATH);
	StringCchCat(szDir, MAX_PATH, TEXT("\\*"));

	hFind = FindFirstFile(szDir, &ffd);
	while (FindNextFile(hFind, &ffd) != 0){
		//TCHAR subDir[MAX_PATH] = { 0, };
		//memcpy(subDir, szDir, sizeof(TCHAR)*MAX_PATH);
		//size_t len;
		//StringCchLength(subDir, MAX_PATH, &len);
		//subDir[len - 1] = '\0';
		//StringCchCat(subDir, MAX_PATH, ffd.cFileName);
		//char tBuf[MAX_PATH];
		//WideCharToMultiByte(CP_ACP, 0, subDir, MAX_PATH, tBuf, MAX_PATH, NULL, NULL);

		////Tchar to char
		//char ccFileName[256];
		//WideCharToMultiByte(CP_ACP, 0, ffd.cFileName, len, ccFileName, 256, NULL, NULL);
		//printf("Object : %s load.\n", ccFileName);

		//if (ccFileName[0] != '.' && strcmp("background", ccFileName)){
		//	WIN32_FIND_DATA class_ffd;
		//	TCHAR szGraspDir[MAX_PATH] = { 0, };
		//	HANDLE hGraspFind = INVALID_HANDLE_VALUE;
		//	char GraspPosDir[256];
		//	strcpy(GraspPosDir, tBuf);
		//	strcat(GraspPosDir, "\\GraspingPos\\*");
		//	MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, GraspPosDir, strlen(GraspPosDir), szGraspDir, MAX_PATH);
		//	hGraspFind = FindFirstFile(szGraspDir, &class_ffd);

		//	while (FindNextFile(hGraspFind, &class_ffd) != 0){
		//		char GraspFileName[256];
		//		size_t Grasplen;
		//		StringCchLength(class_ffd.cFileName, MAX_PATH, &Grasplen);
		//		WideCharToMultiByte(CP_ACP, 0, class_ffd.cFileName, 256, GraspFileName, 256, NULL, NULL);

		//		if (GraspFileName[0] == '.')
		//			continue;

		//		char GraspDataFile[256], GraspImageFile[256], GraspDepthFile[256], GraspCOMFile[256];
		//		int imgCount = image_blob.size();
		//		FILE *fp;
		//		int filePathLen;
		//		//Grasping pos 읽어오기
		//		strcpy(GraspDataFile, GraspPosDir);
		//		filePathLen = strlen(GraspDataFile);
		//		GraspDataFile[filePathLen - 1] = '\0';
		//		strcat(GraspDataFile, GraspFileName);
		//		fp = fopen(GraspDataFile, "r");
		//		while (!feof(fp)){
		//			//upper Left, Upper Right, Thumb
		//			cv::Mat posMat(9, 1, CV_32FC1);
		//			float tempFloat;
		//			cv::Point3f UpperLeft, UpperRight, Thumb, swapPoint;
		//			fscanf(fp, "%f %f %f ", &UpperLeft.x, &UpperLeft.y, &UpperLeft.z);
		//			fscanf(fp, "%f %f %f ", &UpperRight.x, &UpperRight.y, &UpperRight.z);
		//			fscanf(fp, "%f %f %f\n", &Thumb.x, &Thumb.y, &Thumb.z);

		//			//mm -> M 단위로 변경
		//			posMat.at<float>(0) = UpperLeft.x / 100.f;
		//			posMat.at<float>(1) = UpperLeft.y / 100.f;
		//			posMat.at<float>(2) = UpperLeft.z / 100.f;
		//			posMat.at<float>(3) = UpperRight.x / 100.f;
		//			posMat.at<float>(4) = UpperRight.y / 100.f;
		//			posMat.at<float>(5) = UpperRight.z / 100.f;
		//			posMat.at<float>(6) = Thumb.x / 100.f;
		//			posMat.at<float>(7) = Thumb.y / 100.f;
		//			posMat.at<float>(8) = Thumb.z / 100.f;

		//			std::pair<int, cv::Mat> tempPair;
		//			tempPair.first = imgCount;
		//			tempPair.second = posMat.clone();
		//			ang_blob.push_back(tempPair);

		//			if ((data_limit_ != 0) && data_limit_ <= pos_blob.size())
		//				break;
		//		}
		//		fclose(fp);

		//		//RGB 읽어오기
		//		sprintf(GraspImageFile, "%s\\RGB\\%s", tBuf, GraspFileName);
		//		filePathLen = strlen(GraspImageFile);
		//		GraspImageFile[filePathLen - 1] = 'p';
		//		GraspImageFile[filePathLen - 2] = 'm';
		//		GraspImageFile[filePathLen - 3] = 'b';
		//		cv::Mat rgb = cv::imread(GraspImageFile);
		//		cv::Mat tempdataMat(height_, width_, CV_32FC3);
		//		for (int h = 0; h < rgb.rows; h++){
		//			for (int w = 0; w < rgb.cols; w++){
		//				for (int c = 0; c < rgb.channels(); c++){
		//					tempdataMat.at<float>(c*height_*width_ + width_*h + w) = (float)rgb.at<cv::Vec3b>(h, w)[c] / 255.0f;
		//				}
		//			}
		//		}
		//		image_blob.push_back(tempdataMat.clone());

		//		//Depth 읽어오기
		//		sprintf(GraspDepthFile, "%s\\DEPTH\\%s", tBuf, GraspFileName);
		//		int depthwidth, depthheight, depthType;
		//		filePathLen = strlen(GraspDepthFile);
		//		GraspDepthFile[filePathLen - 1] = 'n';
		//		GraspDepthFile[filePathLen - 2] = 'i';
		//		GraspDepthFile[filePathLen - 3] = 'b';
		//		fp = fopen(GraspDepthFile, "rb");
		//		fread(&depthwidth, sizeof(int), 1, fp);
		//		fread(&depthheight, sizeof(int), 1, fp);
		//		fread(&depthType, sizeof(int), 1, fp);
		//		cv::Mat depthMap(depthheight, depthwidth, depthType);
		//		for (int i = 0; i < depthMap.rows * depthMap.cols; i++)		fread(&depthMap.at<float>(i), sizeof(float), 1, fp);
		//		depth_blob.push_back(depthMap.clone());
		//		fclose(fp);

		//		if ((data_limit_ != 0) && data_limit_ <= pos_blob.size())
		//			break;
		//	}

		//}
	}
}

template <typename Dtype>
bool IKDataLayer<Dtype>::fileTypeCheck(char *fileName){
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
void IKDataLayer<Dtype>::makeRandbox(int *arr, int size){
	for (int i = 0; i < size; i++)
		arr[i] = i;
	for (int i = 0; i < size; i++){
		int tidx = rand() % size;
		int t = arr[i];
		arr[i] = arr[tidx];
		arr[tidx] = t;
	}
}

INSTANTIATE_CLASS(IKDataLayer);
REGISTER_LAYER_CLASS(IKData);

}  // namespace caffe
