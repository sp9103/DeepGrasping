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

namespace caffe {

template <typename Dtype>
void PreGraspDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
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

  int tSize = 2;
  top[0]->Reshape(batch_size_, channels_, height_, width_);					//[0] RGB
  std::vector<int> depth_dim(3);
  depth_dim[0] = batch_size_;
  depth_dim[1] = height_;
  depth_dim[2] = width_;
  top[1]->Reshape(depth_dim);												//[1] Depth
  std::vector<int> pos_dim(2);
  pos_dim[0] = batch_size_;
  pos_dim[1] = 3;
  top[2]->Reshape(pos_dim);													//[2] COM
  pos_dim[1] = 9;
  top[3]->Reshape(pos_dim);													//[3] Pregrasping postion (label)

  //전체 로드
  /*RGBDloadAll_calcCom(data_path_.c_str(), data_path_.c_str());
  CHECK_EQ(data_blob.size(), label_blob.size()) << "data size != label size";
  CHECK_GT(data_blob.size(), 0) << "data is empty";*/

  //랜덤 박스 생성
  srand(time(NULL));
  randbox = (int*)malloc(sizeof(int)*pos_blob.size());
  makeRandbox(randbox, pos_blob.size());
  dataidx = 0;
}

template <typename Dtype>
void PreGraspDataLayer<Dtype>::Reset(Dtype* data, Dtype* labels, int n) {
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
void PreGraspDataLayer<Dtype>::set_batch_size(int new_size) {
  /*CHECK(!has_new_data_) <<
      "Can't change batch_size until current data has been consumed.";*/
  batch_size_ = new_size;
}

template <typename Dtype>
void PreGraspDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	Dtype* rgb_data = top[0]->mutable_cpu_data();					//[0] RGB
	Dtype* depth_data = top[1]->mutable_cpu_data();					//[1] Depth
	Dtype* com_data = top[2]->mutable_cpu_data();					//[2] COM
	Dtype* pos_data = top[3]->mutable_cpu_data();					//[3] Pregrasping postion (label)

	for (int i = 0; i < batch_size_; i++){
		//RGB 로드
		int idx = this->pos_blob.at(randbox[dataidx]).first;
		cv::Mat	rgbImg = this->image_blob.at(idx);
		cv::Mat depthImg = this->depth_blob.at(idx);
		cv::Mat com = this->com_blob.at(idx);
		cv::Mat pos = this->pos_blob.at(randbox[dataidx]).second;

		caffe_copy(channels_ * height_ * width_, rgbImg.ptr<Dtype>(0), rgb_data);
		caffe_copy(height_ * width_, depthImg.ptr<Dtype>(0), depth_data);
		caffe_copy(3, com.ptr<Dtype>(0), com_data);
		caffe_copy(9, pos.ptr<Dtype>(0), pos_data);

		rgb_data += top[0]->offset(1);
		depth_data += top[1]->offset(1);
		com_data += top[1]->offset(1);
		pos_data += top[1]->offset(1);
		if (dataidx + 1 >= this->pos_blob.size()){
			makeRandbox(randbox, this->pos_blob.size());
			dataidx = 0;
		}
		else
			dataidx++;
	}
}

template <typename Dtype>
void PreGraspDataLayer<Dtype>::RGBDloadAll_calcCom(const char* datapath, const char* depthpath){
	//WIN32_FIND_DATA ffd;
	//HANDLE hFind = INVALID_HANDLE_VALUE;
	//TCHAR szDir[MAX_PATH] = { 0, }, szDepthDir[MAX_PATH] = { 0, };

	//MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, datapath, strlen(datapath), szDir, MAX_PATH);
	//StringCchCat(szDir, MAX_PATH, TEXT("\\*"));
	//MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, depthpath, strlen(depthpath), szDepthDir, MAX_PATH);
	//StringCchCat(szDepthDir, MAX_PATH, TEXT("\\*"));

	//hFind = FindFirstFile(szDir, &ffd);
	//while (FindNextFile(hFind, &ffd) != 0){

	//	TCHAR subDir[MAX_PATH] = { 0, }, subDepthDir[MAX_PATH] = { 0, };
	//	memcpy(subDir, szDir, sizeof(TCHAR)*MAX_PATH);
	//	size_t len;
	//	StringCchLength(subDir, MAX_PATH, &len);
	//	subDir[len - 1] = '\0';
	//	StringCchCat(subDir, MAX_PATH, ffd.cFileName);
	//	char tBuf[MAX_PATH], tdepthBuf[MAX_PATH];
	//	WideCharToMultiByte(CP_ACP, 0, subDir, MAX_PATH, tBuf, MAX_PATH, NULL, NULL);

	//	memcpy(subDepthDir, szDepthDir, sizeof(TCHAR)*MAX_PATH);
	//	StringCchLength(subDepthDir, MAX_PATH, &len);
	//	subDepthDir[len - 1] = '\0';

	//	//Tchar to char
	//	char ccFileName[256];
	//	WideCharToMultiByte(CP_ACP, 0, ffd.cFileName, len, ccFileName, 256, NULL, NULL);

	//	if (!strcmp(ccFileName, "RGB")){
	//		WideCharToMultiByte(CP_ACP, 0, subDepthDir, MAX_PATH, tdepthBuf, MAX_PATH, NULL, NULL);
	//		strcat(tdepthBuf, "DEPTHMAP");
	//	}
	//	else{
	//		StringCchCat(subDepthDir, MAX_PATH, ffd.cFileName);
	//		WideCharToMultiByte(CP_ACP, 0, subDepthDir, MAX_PATH, tdepthBuf, MAX_PATH, NULL, NULL);
	//	}

	//	if (!strcmp(ccFileName, "DEPTHMAP") || !strcmp(ccFileName, "XYZMAP"))		continue;

	//	if (fileTypeCheck(ccFileName)){
	//		cv::Mat dataimage;
	//		cv::Mat labelimage;
	//		cv::Mat depthMap;

	//		cv::Mat tempdataMat;
	//		cv::Mat templabelMat;
	//		
	//		//tempdataMat.create(height_, width_, CV_32FC4);
	//		tempdataMat.create(height_, width_, CV_32FC3);
	//		templabelMat.create(2, 1, CV_32FC1);

	//		dataimage = cv::imread(tBuf);

	//		//Depth 열고 넣어주기
	//		int depthpathlen = strlen(tdepthBuf);
	//		int depthwidth, depthheight, depthType;
	//		tdepthBuf[depthpathlen - 3] = 'b';
	//		tdepthBuf[depthpathlen - 2] = 'i';
	//		tdepthBuf[depthpathlen - 1] = 'n';
	//		FILE *fp = fopen(tdepthBuf, "rb");
	//		fread(&depthwidth, sizeof(int), 1, fp);
	//		fread(&depthheight, sizeof(int), 1, fp);
	//		fread(&depthType, sizeof(int), 1, fp);
	//		depthMap.create(depthheight, depthwidth, depthType);
	//		for (int i = 0; i < depthMap.rows * depthMap.cols; i++)		fread(&depthMap.at<float>(i), sizeof(float), 1, fp);
	//		fclose(fp);

	//		if (height_ != dataimage.rows || width_ != dataimage.cols)
	//			cv::resize(dataimage, dataimage, cv::Size(height_, width_));
	//		labelimage = subBackground(dataimage, depthMap);

	//		for (int r = 0; r < 4; r++){
	//			cv::transpose(dataimage, dataimage);
	//			cv::flip(dataimage, dataimage, 1);
	//			cv::transpose(labelimage, labelimage);
	//			cv::flip(labelimage, labelimage, 1);

	//			if (dataimage.rows == height_ && dataimage.cols == width_){
	//				for (int h = 0; h < dataimage.rows; h++){
	//					for (int w = 0; w < dataimage.cols; w++){
	//						for (int c = 0; c < dataimage.channels(); c++){
	//							tempdataMat.at<float>(c*height_*width_ + width_*h + w) = (float)dataimage.at<cv::Vec3b>(h, w)[c] / 255.0f;
	//						}
	//					}
	//				}
	//				cv::Point2f objPos = cv::Point2f(0, 0);
	//				int objpixelcount = 0;
	//				for (int h = 0; h < labelimage.rows; h++){
	//					for (int w = 0; w < labelimage.cols; w++){
	//						uchar val = labelimage.at<uchar>(h, w);
	//						if (val > 0){
	//							objPos.x += w;
	//							objPos.y += h;
	//							objpixelcount++;
	//						}
	//					}
	//				}
	//				objPos.x /= objpixelcount;
	//				objPos.y /= objpixelcount;

	//				templabelMat.at<float>(0) = objPos.x;
	//				templabelMat.at<float>(1) = objPos.y;

	//				data_blob.push_back(tempdataMat.clone());
	//				label_blob.push_back(templabelMat.clone());
	//			}
	//		}
	//	}

	//	if ((data_limit_ != 0) && data_limit_ <= data_blob.size())
	//		break;

	//	if (ffd.dwFileAttributes == 16 && ffd.cFileName[0] != '.'){
	//		printf("%s\n", tBuf);
	//		RGBDloadAll_calcCom(tBuf, tdepthBuf);
	//	}
	//}
}

template <typename Dtype>
bool PreGraspDataLayer<Dtype>::fileTypeCheck(char *fileName){
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
void PreGraspDataLayer<Dtype>::makeRandbox(int *arr, int size){
	for (int i = 0; i < size; i++)
		arr[i] = i;
	for (int i = 0; i < size; i++){
		int tidx = rand() % size;
		int t = arr[i];
		arr[i] = arr[tidx];
		arr[tidx] = t;
	}
}

INSTANTIATE_CLASS(PreGraspDataLayer);
REGISTER_LAYER_CLASS(PreGraspData);

}  // namespace caffe
