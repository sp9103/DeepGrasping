#include <opencv2/core/core.hpp>

#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"

#include <opencv2\opencv.hpp>
#include <conio.h>
#include <time.h>
#include <iostream>
#include <fstream>

void makeRandboxMNIST(int *arr, int size){
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
void MNISTDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
	batch_size_ = this->layer_param_.mnist_data_param().batch_size();

	data_path_ = this->layer_param_.mnist_data_param().data_path();
	label_path_ = this->layer_param_.mnist_data_param().label_path();

	//전체 로드
	ImageloadAll(data_path_.c_str(), label_path_.c_str());
	CHECK_EQ(data.size(), label.size()) << "data size != label size";
	CHECK_GT(data.size(), 0) << "data is empty";
	//CHECK_EQ(data_blob.size(), label_blob.size()) << "data size != label size";

	size_ = n_rows * n_cols;
  CHECK_GT(batch_size_ * size_, 0) <<
      "batch_size, channels, height, and width must be specified and"
      " positive in memory_data_param";
  //int tSize = height_ * width_ / 4 / 4;
  int tSize = 1;
  //vector<int> label_shape(1, batch_size_);
  top[0]->Reshape(batch_size_, 1, n_rows, n_cols);
  vector<int> label_shape = top[0]->shape();
  label_shape.resize(1 + 1);
  label_shape[1] = tSize;
  top[1]->Reshape(label_shape);

  //랜덤 박스 생성
  srand(time(NULL));
  randbox = (int*)malloc(sizeof(int)*data.size());
  makeRandboxMNIST(randbox, data.size());
  dataidx = 0;
}

template <typename Dtype>
void MNISTDataLayer<Dtype>::Reset(Dtype* data, Dtype* labels, int n) {
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
void MNISTDataLayer<Dtype>::set_batch_size(int new_size) {
  /*CHECK(!has_new_data_) <<
      "Can't change batch_size until current data has been consumed.";*/
  batch_size_ = new_size;
}

template <typename Dtype>
void MNISTDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	Dtype* label = top[1]->mutable_cpu_data();
	Dtype* data = top[0]->mutable_cpu_data();

	for (int i = 0; i < batch_size_; i++){
		//RGB 로드
		cv::Mat src = this->data.at(randbox[dataidx]);
		cv::Mat labelimg = this->label.at(randbox[dataidx]);
		
		int imgHeight = src.rows;
		int imgWidth = src.cols;

		for (int h = 0; h < imgHeight; h++){
			for (int w = 0; w < imgWidth; w++){
				data[imgWidth*h + w] = (Dtype)src.at<float>(h, w);
			}
		}

		for (int j = 0; j < 10; j++){
			//label[j] = (Dtype)labelimg.at</*uchar*/float>(j);
			if ((Dtype)labelimg.at<float>(j) != 0)
				label[0] = (Dtype)j;
		}

		label += top[1]->offset(1);
		data += top[0]->offset(1);
		if (dataidx + 1 >= this->data.size()){
			makeRandboxMNIST(randbox, this->data.size());
			dataidx = 0;
		}
		else
			dataidx++;
	}
}

template <typename Dtype>
void MNISTDataLayer<Dtype>::ImageloadAll(const char* datapath, const char* labelpath){
	FileOpenData(datapath);

	FileOpenLabel(labelpath);
}

template <typename Dtype>
int MNISTDataLayer<Dtype>::reverseInt(int i) {
	unsigned char c1, c2, c3, c4;
	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;
	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

template <typename Dtype>
void MNISTDataLayer<Dtype>::FileOpenData(const char *fileName){
	ifstream m_file;
	m_file.open(fileName, ios::binary);

	//File Header read
	m_file.read((char*)&magic_number, sizeof(magic_number));
	magic_number = reverseInt(magic_number);

	m_file.read((char*)&number_of_images, sizeof(number_of_images));
	number_of_images = reverseInt(number_of_images);

	m_file.read((char*)&n_rows, sizeof(n_rows));
	n_rows = reverseInt(n_rows);
	m_file.read((char*)&n_cols, sizeof(n_cols));
	n_cols = reverseInt(n_cols);

	//모두 로드
	for (int i = 0; i < number_of_images; i++){
		cv::Size size;
		unsigned char temp = 0;
		cv::Mat dataImg;

		dataImg.create(n_rows, n_cols, CV_32FC1);		//28*28

		//Image Read
		unsigned char arr[28][28];

		for(int r = 0; r<n_rows; ++r)
		{
			for (int c = 0; c<n_cols; ++c)
			{
				m_file.read((char*)&temp, sizeof(temp));
				arr[r][c] = temp;
			}
		}
		size.height = n_rows;
		size.width = n_cols;
		create_image(&dataImg, size, 1, arr, i);

		data.push_back(dataImg);
	}

	m_file.close();
}

template <typename Dtype>
void MNISTDataLayer<Dtype>::FileOpenLabel(const char *fileName){
	ifstream m_file;
	m_file.open(fileName, ios::binary);

	//File Header read
	m_file.read((char*)&magic_number, sizeof(magic_number));
	magic_number = reverseInt(magic_number);

	m_file.read((char*)&number_of_images, sizeof(number_of_images));
	number_of_images = reverseInt(number_of_images);

	//모두 로드
	for (int i = 0; i < number_of_images; i++){
		unsigned char temp = 0;

		cv::Mat labelImg;
		labelImg.create(10, 1, CV_32FC1);

		//Label Read
		
		m_file.read((char*)&temp, sizeof(temp));

		for (int j = 0; j < 10; j++)
			if (j == (int)temp)
				labelImg.at<float>(j) = 1.0f;
			else
				labelImg.at<float>(j) = 0.0f;

		label.push_back(labelImg);
	}

	m_file.close();
}

template <typename Dtype>
int MNISTDataLayer<Dtype>::getDataCount(){
	return number_of_images;
}

template <typename Dtype>
void MNISTDataLayer<Dtype>::create_image(cv::Mat *dst, cv::Size size, int channels, unsigned char data[28][28], int imagenumber) {
	string imgname; ostringstream imgstrm; string fullpath;
	imgstrm << imagenumber;
	imgname = imgstrm.str();

	/*if(*dst != NULL)
	cvReleaseImage(&(*dst));*/
	IplImage* tempImg = NULL;
	tempImg = cvCreateImageHeader(size, IPL_DEPTH_8U, channels);
	cvSetData(tempImg, data, size.width);
	for (int i = 0; i < tempImg->height * tempImg->width; i++){
		dst->at<float>(i) = (float)(uchar)tempImg->imageData[i] / 255.0f;
	}

	//cvReleaseImage(tempImg);
}

INSTANTIATE_CLASS(MNISTDataLayer);
REGISTER_LAYER_CLASS(MNISTData);

}  // namespace caffe
