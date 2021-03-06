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
void SPRGBDCOMDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
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
  top[0]->Reshape(batch_size_, channels_, height_, width_);
  vector<int> label_shape = top[0]->shape();
  label_shape.resize(1 + 1);
  label_shape[1] = tSize;
  top[1]->Reshape(label_shape);

  //Background load - 배경제거를 위해서
  BackgroudLoad("D:\\RGBDData\\background", "0_7");

  //전체 로드
  RGBDloadAll_calcCom(data_path_.c_str(), data_path_.c_str());
  CHECK_EQ(data_blob.size(), label_blob.size()) << "data size != label size";
  CHECK_GT(data_blob.size(), 0) << "data is empty";
  //CHECK_EQ(data_blob.size(), label_blob.size()) << "data size != label size";

  //랜덤 박스 생성
  srand(time(NULL));
  randbox = (int*)malloc(sizeof(int)*data_blob.size());
  makeRandbox(randbox, data_blob.size());
  dataidx = 0;
}

template <typename Dtype>
void SPRGBDCOMDataLayer<Dtype>::Reset(Dtype* data, Dtype* labels, int n) {
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
void SPRGBDCOMDataLayer<Dtype>::set_batch_size(int new_size) {
  /*CHECK(!has_new_data_) <<
      "Can't change batch_size until current data has been consumed.";*/
  batch_size_ = new_size;
}

template <typename Dtype>
void SPRGBDCOMDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	Dtype* label = top[1]->mutable_cpu_data();
	Dtype* data = top[0]->mutable_cpu_data();

	for (int i = 0; i < batch_size_; i++){
		//RGB 로드
		cv::Mat srcImg = this->data_blob.at(randbox[dataidx]);
		cv::Mat labelImg = this->label_blob.at(randbox[dataidx]);

		caffe_copy(channels_ * height_ * width_, srcImg.ptr<Dtype>(0), data);
		caffe_copy(2, labelImg.ptr<Dtype>(0), label);
		
		///////////////////////////
		/*cv::Mat tempMat;
		tempMat.create(height_, width_, CV_32FC3);
		int idx = 0;
		for (int c = 0; c < 3; c++){
			for (int h = 0; h < height_; h++){
				for (int w = 0; w < width_; w++){
					tempMat.at<cv::Vec3f>(h, w)[c] = (float)data[idx++];
				}
			}
		}*/

		label += top[1]->offset(1);
		data += top[0]->offset(1);
		if (dataidx + 1 >= this->data_blob.size()){
			makeRandbox(randbox, this->data_blob.size());
			dataidx = 0;
		}
		else
			dataidx++;
	}
}

template <typename Dtype>
void SPRGBDCOMDataLayer<Dtype>::RGBDloadAll_calcCom(const char* datapath, const char* depthpath){
	WIN32_FIND_DATA ffd;
	HANDLE hFind = INVALID_HANDLE_VALUE;
	TCHAR szDir[MAX_PATH] = { 0, }, szDepthDir[MAX_PATH] = { 0, };

	MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, datapath, strlen(datapath), szDir, MAX_PATH);
	StringCchCat(szDir, MAX_PATH, TEXT("\\*"));
	MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, depthpath, strlen(depthpath), szDepthDir, MAX_PATH);
	StringCchCat(szDepthDir, MAX_PATH, TEXT("\\*"));

	hFind = FindFirstFile(szDir, &ffd);
	while (FindNextFile(hFind, &ffd) != 0){

		TCHAR subDir[MAX_PATH] = { 0, }, subDepthDir[MAX_PATH] = { 0, };
		memcpy(subDir, szDir, sizeof(TCHAR)*MAX_PATH);
		size_t len;
		StringCchLength(subDir, MAX_PATH, &len);
		subDir[len - 1] = '\0';
		StringCchCat(subDir, MAX_PATH, ffd.cFileName);
		char tBuf[MAX_PATH], tdepthBuf[MAX_PATH];
		WideCharToMultiByte(CP_ACP, 0, subDir, MAX_PATH, tBuf, MAX_PATH, NULL, NULL);

		memcpy(subDepthDir, szDepthDir, sizeof(TCHAR)*MAX_PATH);
		StringCchLength(subDepthDir, MAX_PATH, &len);
		subDepthDir[len - 1] = '\0';

		//Tchar to char
		char ccFileName[256];
		WideCharToMultiByte(CP_ACP, 0, ffd.cFileName, len, ccFileName, 256, NULL, NULL);

		if (!strcmp(ccFileName, "RGB")){
			WideCharToMultiByte(CP_ACP, 0, subDepthDir, MAX_PATH, tdepthBuf, MAX_PATH, NULL, NULL);
			strcat(tdepthBuf, "DEPTHMAP");
		}
		else{
			StringCchCat(subDepthDir, MAX_PATH, ffd.cFileName);
			WideCharToMultiByte(CP_ACP, 0, subDepthDir, MAX_PATH, tdepthBuf, MAX_PATH, NULL, NULL);
		}

		if (!strcmp(ccFileName, "DEPTHMAP") || !strcmp(ccFileName, "XYZMAP"))		continue;

		if (fileTypeCheck(ccFileName)){
			cv::Mat dataimage;
			cv::Mat labelimage;
			cv::Mat depthMap;

			cv::Mat tempdataMat;
			cv::Mat templabelMat;
			
			//tempdataMat.create(height_, width_, CV_32FC4);
			tempdataMat.create(height_, width_, CV_32FC3);
			templabelMat.create(2, 1, CV_32FC1);

			dataimage = cv::imread(tBuf);

			//Depth 열고 넣어주기
			int depthpathlen = strlen(tdepthBuf);
			int depthwidth, depthheight, depthType;
			tdepthBuf[depthpathlen - 3] = 'b';
			tdepthBuf[depthpathlen - 2] = 'i';
			tdepthBuf[depthpathlen - 1] = 'n';
			FILE *fp = fopen(tdepthBuf, "rb");
			fread(&depthwidth, sizeof(int), 1, fp);
			fread(&depthheight, sizeof(int), 1, fp);
			fread(&depthType, sizeof(int), 1, fp);
			depthMap.create(depthheight, depthwidth, depthType);
			for (int i = 0; i < depthMap.rows * depthMap.cols; i++)		fread(&depthMap.at<float>(i), sizeof(float), 1, fp);
			fclose(fp);

			if (height_ != dataimage.rows || width_ != dataimage.cols)
				cv::resize(dataimage, dataimage, cv::Size(height_, width_));
			labelimage = subBackground(dataimage, depthMap);

			for (int r = 0; r < 4; r++){
				cv::transpose(dataimage, dataimage);
				cv::flip(dataimage, dataimage, 1);
				cv::transpose(labelimage, labelimage);
				cv::flip(labelimage, labelimage, 1);

				if (dataimage.rows == height_ && dataimage.cols == width_){
					for (int h = 0; h < dataimage.rows; h++){
						for (int w = 0; w < dataimage.cols; w++){
							for (int c = 0; c < dataimage.channels(); c++){
								tempdataMat.at<float>(c*height_*width_ + width_*h + w) = (float)dataimage.at<cv::Vec3b>(h, w)[c] / 255.0f;
							}
						}
					}
					cv::Point2f objPos = cv::Point2f(0, 0);
					int objpixelcount = 0;
					for (int h = 0; h < labelimage.rows; h++){
						for (int w = 0; w < labelimage.cols; w++){
							uchar val = labelimage.at<uchar>(h, w);
							if (val > 0){
								objPos.x += w;
								objPos.y += h;
								objpixelcount++;
							}
						}
					}
					objPos.x /= objpixelcount;
					objPos.y /= objpixelcount;

					templabelMat.at<float>(0) = objPos.x;
					templabelMat.at<float>(1) = objPos.y;

					data_blob.push_back(tempdataMat.clone());
					label_blob.push_back(templabelMat.clone());
				}
			}
		}

		if ((data_limit_ != 0) && data_limit_ <= data_blob.size())
			break;

		if (ffd.dwFileAttributes == 16 && ffd.cFileName[0] != '.'){
			printf("%s\n", tBuf);
			RGBDloadAll_calcCom(tBuf, tdepthBuf);
		}
	}
}

template <typename Dtype>
bool SPRGBDCOMDataLayer<Dtype>::fileTypeCheck(char *fileName){
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
void SPRGBDCOMDataLayer<Dtype>::makeRandbox(int *arr, int size){
	for (int i = 0; i < size; i++)
		arr[i] = i;
	for (int i = 0; i < size; i++){
		int tidx = rand() % size;
		int t = arr[i];
		arr[i] = arr[tidx];
		arr[tidx] = t;
	}
}

template <typename Dtype>
void SPRGBDCOMDataLayer<Dtype>::BackgroudLoad(const char *path, const char *fileName){
	char RGBbuf[256], DepthBuf[256];
	sprintf(RGBbuf, "%s\\RGB\\%s.bmp", path, fileName);
	sprintf(DepthBuf, "%s\\DEPTHMAP\\%s.bin", path, fileName);
	backRGB = cv::imread(RGBbuf);

	int depthwidth, depthheight, depthType;
	FILE *backfp = fopen(DepthBuf, "rb");
	fread(&depthwidth, sizeof(int), 1, backfp);
	fread(&depthheight, sizeof(int), 1, backfp);
	fread(&depthType, sizeof(int), 1, backfp);
	backDepth.create(depthheight, depthwidth, depthType);
	for (int i = 0; i < backDepth.rows * backDepth.cols; i++)		fread(&backDepth.at<float>(i), sizeof(float), 1, backfp);
	fclose(backfp);
}

template <typename Dtype>
cv::Mat SPRGBDCOMDataLayer<Dtype>::subBackground(cv::Mat rgb, cv::Mat depth){
	cv::Mat label;
	label.create(rgb.rows, rgb.cols, rgb.type());

	//배경이 입력되지 않으면 그냥 label 생성
	if (backRGB.cols == 0){
		cv::resize(rgb, label, cv::Size(labelHeight_, labelWidth_));
		if (rgb.channels() != 1){
			cv::cvtColor(label, label, CV_BGR2GRAY);
		}
	}
	else{
		const float threshold = 5;
		cv::Mat tMat(rgb.rows, rgb.cols, CV_8UC1);
		for (int i = 0; i < rgb.rows * rgb.cols; i++){
			float bVal = backDepth.at<float>(i);
			float oVal = depth.at<float>(i);

			if (bVal == 0 || oVal == 0)		continue;
			if (abs(oVal - bVal) > threshold){
				tMat.at<uchar>(i) = 255;
			}
			else{
				tMat.at<uchar>(i) = 0;
			}
		}
		cv::erode(tMat, tMat, cv::Mat(), cv::Point(-1, -1), 2);
		cv::dilate(tMat, tMat, cv::Mat(), cv::Point(-1, -1), 8);

		//ColorBase sub image
		for (int i = 0; i < rgb.rows * rgb.cols; i++){
			uchar binVal = tMat.at<uchar>(i);
			if (binVal > 0){
				cv::Vec3b bVal = backRGB.at<cv::Vec3b>(i);
				cv::Vec3b oVal = rgb.at<cv::Vec3b>(i);
				cv::Vec3b subVal;
				const int cThreshold = 30;
				for (int j = 0; j < 3; j++)		subVal[j] = abs((int)bVal[j] - (int)oVal[j]);
				if (subVal[0] < cThreshold && subVal[1] < cThreshold && subVal[2] < cThreshold)
					tMat.at<uchar>(i) = 0;
			}
		}

		//mask
		for (int i = 0; i < rgb.rows * rgb.cols; i++){
			uchar val = tMat.at<uchar>(i);
			if (val > 0)	label.at<cv::Vec3b>(i) = rgb.at<cv::Vec3b>(i);
			else			label.at<cv::Vec3b>(i) = cv::Vec3f(0, 0, 0);
		}

		if (rgb.channels() != 1){
			cv::cvtColor(label, label, CV_BGR2GRAY);
		}
	}

	return label.clone();
}

INSTANTIATE_CLASS(SPRGBDCOMDataLayer);
REGISTER_LAYER_CLASS(SPRGBDCOMData);

}  // namespace caffe
