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

  //top[0]->Reshape(batch_size_, channels_, height_, width_);					//[0] RGB
  //std::vector<int> depth_dim(3);
  //depth_dim[0] = batch_size_;
  //depth_dim[1] = height_;
  //depth_dim[2] = width_;
  //top[1]->Reshape(depth_dim);												//[1] Depth
  //std::vector<int> pos_dim(2);
  //pos_dim[0] = batch_size_;
  //pos_dim[1] = 3;
  //top[2]->Reshape(pos_dim);													//[2] COM
  //pos_dim[1] = 9;
  //top[3]->Reshape(pos_dim);													//[3] Pregrasping postion (label)

  //std::vector<int> pos_dim(2);
  //pos_dim[0] = batch_size_;
  //pos_dim[1] = 9;
  //top[2]->Reshape(pos_dim);													//[2] Pregrasping postion (label)

  std::vector<int> pos_dim(2);
  pos_dim[0] = batch_size_;
  pos_dim[1] = 9;
  top[0]->Reshape(pos_dim);		

  //전체 로드
  PreGrasp_DataLoadAll(data_path_.c_str());
  CHECK_GT(pos_blob.size(), 0) << "data is empty";

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
	//Dtype* rgb_data = top[0]->mutable_cpu_data();					//[0] RGB
	//Dtype* depth_data = top[1]->mutable_cpu_data();					//[1] Depth
	//Dtype* com_data = top[2]->mutable_cpu_data();					//[2] COM
	//Dtype* pos_data = top[3]->mutable_cpu_data();					//[3] Pregrasping postion (label)

	//Dtype* pos_data = top[2]->mutable_cpu_data();					//[3] Pregrasping postion (label)
	Dtype* pos_data = top[0]->mutable_cpu_data();					//[3] Pregrasping postion (label)

	for (int i = 0; i < batch_size_; i++){
		//RGB 로드
		std::pair<int, cv::Mat> posPair = this->pos_blob.at(randbox[dataidx]);
		int idx = posPair.first;
		//cv::Mat	rgbImg = this->image_blob.at(idx);
		//cv::Mat depthImg = this->depth_blob.at(idx).clone();
		//cv::Mat com = this->com_blob.at(idx);
		cv::Mat pos = posPair.second.clone();

		/*caffe_copy(channels_ * height_ * width_, rgbImg.ptr<Dtype>(0), rgb_data);
		caffe_copy(height_ * width_, depthImg.ptr<Dtype>(0), depth_data);*/
		//caffe_copy(3, com.ptr<Dtype>(0), com_data);
		caffe_copy(9, pos.ptr<Dtype>(0), pos_data);

		///////////////////////////
		//cv::Mat tempMat;
		//tempMat.create(height_, width_, CV_32FC3);
		//int tidx = 0;
		//for (int c = 0; c < 3; c++){
		//	for (int h = 0; h < height_; h++){
		//		for (int w = 0; w < width_; w++){
		//			tempMat.at<cv::Vec3f>(h, w)[c] = (float)rgb_data[tidx++];
		//		}
		//	}
		//}
		//cv::Mat tempDepth(height_, width_, CV_32FC1);
		//for (int i = 0; i < height_*width_; i++)	tempDepth.at<float>(i) = depth_data[i] / (8000 / 256) / 255.f;
		//cv::imshow("rgb", tempMat);
		//cv::imshow("depth", tempDepth);
		//cv::waitKey(0);

		/*rgb_data += top[0]->offset(1);
		depth_data += top[1]->offset(1);*/
		//com_data += top[2]->offset(1);
		/*pos_data += top[3]->offset(1);*/

		/*pos_data += top[2]->offset(1);*/
		pos_data += top[0]->offset(1);
		if (dataidx + 1 >= this->pos_blob.size()){
			makeRandbox(randbox, this->pos_blob.size());
			dataidx = 0;
		}
		else
			dataidx++;
	}
}

template <typename Dtype>
void PreGraspDataLayer<Dtype>::PreGrasp_DataLoadAll(const char* datapath){
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

		if (ccFileName[0] != '.' && strcmp("background", ccFileName)){
			WIN32_FIND_DATA class_ffd;
			TCHAR szGraspDir[MAX_PATH] = { 0, };
			HANDLE hGraspFind = INVALID_HANDLE_VALUE;
			char GraspPosDir[256];
			strcpy(GraspPosDir, tBuf);
			strcat(GraspPosDir, "\\GraspingPos\\*");
			MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, GraspPosDir, strlen(GraspPosDir), szGraspDir, MAX_PATH);
			hGraspFind = FindFirstFile(szGraspDir, &class_ffd);

			while (FindNextFile(hGraspFind, &class_ffd) != 0){
				char GraspFileName[256];
				size_t Grasplen;
				StringCchLength(class_ffd.cFileName, MAX_PATH, &Grasplen);
				WideCharToMultiByte(CP_ACP, 0, class_ffd.cFileName, 256, GraspFileName, 256, NULL, NULL);

				if (GraspFileName[0] == '.')
					continue;

				char GraspDataFile[256], GraspImageFile[256], GraspDepthFile[256], GraspCOMFile[256];
				int imgCount = image_blob.size();
				FILE *fp;
				int filePathLen;
				//Grasping pos 읽어오기
				strcpy(GraspDataFile, GraspPosDir);
				filePathLen = strlen(GraspDataFile);
				GraspDataFile[filePathLen - 1] = '\0';
				strcat(GraspDataFile, GraspFileName);
				fp = fopen(GraspDataFile, "r");
				while (!feof(fp)){
					//upper Left, Upper Right, Thumb
					cv::Mat posMat(9, 1, CV_32FC1);
					float tempFloat;
					cv::Point3f UpperLeft, UpperRight, Thumb, swapPoint;
					fscanf(fp, "%f %f %f ", &UpperLeft.x, &UpperLeft.y, &UpperLeft.z);
					fscanf(fp, "%f %f %f ", &UpperRight.x, &UpperRight.y, &UpperRight.z);
					fscanf(fp, "%f %f %f\n", &Thumb.x, &Thumb.y, &Thumb.z);

					//손가락 재정렬
					calcFingerSort(&UpperLeft, &UpperRight, &Thumb);

					//mm -> M 단위로 변경
					posMat.at<float>(0) = UpperLeft.x / 100.f;
					posMat.at<float>(1) = UpperLeft.y / 100.f;
					posMat.at<float>(2) = UpperLeft.z / 100.f;
					posMat.at<float>(3) = UpperRight.x / 100.f;
					posMat.at<float>(4) = UpperRight.y / 100.f;
					posMat.at<float>(5) = UpperRight.z / 100.f;
					posMat.at<float>(6) = Thumb.x / 100.f;
					posMat.at<float>(7) = Thumb.y / 100.f;
					posMat.at<float>(8) = Thumb.z / 100.f;

					std::pair<int, cv::Mat> tempPair;
					tempPair.first = imgCount;
					tempPair.second = posMat.clone();
					pos_blob.push_back(tempPair);

					if ((data_limit_ != 0) && data_limit_ <= pos_blob.size())
						break;
				}
				fclose(fp);

				//RGB 읽어오기
				/*sprintf(GraspImageFile, "%s\\RGB\\%s", tBuf, GraspFileName);
				filePathLen = strlen(GraspImageFile);
				GraspImageFile[filePathLen - 1] = 'p';
				GraspImageFile[filePathLen - 2] = 'm';
				GraspImageFile[filePathLen - 3] = 'b';
				cv::Mat rgb = cv::imread(GraspImageFile);
				cv::Mat tempdataMat(height_, width_, CV_32FC3);
				for (int h = 0; h < rgb.rows; h++){
					for (int w = 0; w < rgb.cols; w++){
						for (int c = 0; c < rgb.channels(); c++){
							tempdataMat.at<float>(c*height_*width_ + width_*h + w) = (float)rgb.at<cv::Vec3b>(h, w)[c] / 255.0f;
						}
					}
				}
				image_blob.push_back(tempdataMat.clone());*/

				//Depth 읽어오기
				/*sprintf(GraspDepthFile, "%s\\DEPTH\\%s", tBuf, GraspFileName);
				int depthwidth, depthheight, depthType;
				filePathLen = strlen(GraspDepthFile);
				GraspDepthFile[filePathLen - 1] = 'n';
				GraspDepthFile[filePathLen - 2] = 'i';
				GraspDepthFile[filePathLen - 3] = 'b';
				fp = fopen(GraspDepthFile, "rb");
				fread(&depthwidth, sizeof(int), 1, fp);
				fread(&depthheight, sizeof(int), 1, fp);
				fread(&depthType, sizeof(int), 1, fp);
				cv::Mat depthMap(depthheight, depthwidth, depthType);
				for (int i = 0; i < depthMap.rows * depthMap.cols; i++)		fread(&depthMap.at<float>(i), sizeof(float), 1, fp);
				depth_blob.push_back(depthMap.clone());
				fclose(fp);*/

				////COM 읽어오기
				//sprintf(GraspCOMFile, "%s\\COM\\%s", tBuf, GraspFileName);
				//filePathLen = strlen(GraspCOMFile);
				//GraspCOMFile[filePathLen - 1] = 't';
				//GraspCOMFile[filePathLen - 2] = 'x';
				//GraspCOMFile[filePathLen - 3] = 't';
				//fp = fopen(GraspCOMFile, "r");
				//cv::Mat comMat(3, 1, CV_32FC1);
				//fscanf(fp, "%f %f %f", &comMat.at<float>(0), &comMat.at<float>(1), &comMat.at<float>(2));
				//com_blob.push_back(comMat.clone());
				//fclose(fp);

				if ((data_limit_ != 0) && data_limit_ <= pos_blob.size())
					break;
			}

		}
	}
}
template <typename Dtype>
void PreGraspDataLayer<Dtype>::calcFingerSort(cv::Point3f *upperLeft, cv::Point3f *upperRight, cv::Point3f *thumb){
	cv::Point3f tLeft = *upperLeft;
	cv::Point3f tRight = *upperRight;
	cv::Point3f tThumb = *thumb;
	cv::Point3f swapPoint;

	float leftRightDist = calcDist3D(tLeft, tRight);
	float thumbRightDist = calcDist3D(tThumb, tRight);
	float thumbleftDist = calcDist3D(tThumb, tLeft);

	//엄지 교환
	if (leftRightDist < thumbRightDist && leftRightDist < thumbleftDist){
		*thumb = tThumb;
	}
	else if (thumbRightDist < leftRightDist && thumbRightDist < thumbleftDist){
		*thumb = tLeft;
		SWAP(tThumb, tLeft, swapPoint);
	}
	else if (thumbleftDist < leftRightDist && thumbleftDist < thumbRightDist){
		*thumb = tRight;
		SWAP(tThumb, tRight, swapPoint);
	}

	//Left Right 구분
	thumbRightDist = calcDist3D(*thumb, tRight);
	thumbleftDist = calcDist3D(*thumb, tLeft);

	if (thumbleftDist < thumbRightDist)
		SWAP(tRight, tLeft, swapPoint);
	
	*upperLeft = tLeft;
	*upperRight = tRight;

	if (*upperLeft == *upperRight || *upperRight == *thumb)
		printf("Finger Data Error.\n");
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
	//for (int i = 0; i < size; i++){
	//	int tidx = rand() % size;
	//	int t = arr[i];
	//	arr[i] = arr[tidx];
	//	arr[tidx] = t;
	//}
}

template <typename Dtype>
float PreGraspDataLayer<Dtype>::calcDist3D(cv::Point3f A, cv::Point3f B){
	return sqrt(pow(A.x - B.x, 2) + pow(A.y - B.y, 2) + pow(A.z - B.z, 2));
}

INSTANTIATE_CLASS(PreGraspDataLayer);
REGISTER_LAYER_CLASS(PreGraspData);

}  // namespace caffe
