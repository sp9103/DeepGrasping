#ifndef CAFFE_DATA_LAYERS_HPP_
#define CAFFE_DATA_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "hdf5/hdf5.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/filler.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/db.hpp"

#include <opencv2\opencv.hpp>

namespace caffe {

/**
 * @brief Provides base for data layers that feed blobs to the Net.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class BaseDataLayer : public Layer<Dtype> {
 public:
  explicit BaseDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden except by the BasePrefetchingDataLayer.
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Data layers should be shared by multiple solvers in parallel
  virtual inline bool ShareInParallel() const { return true; }
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

 protected:
  TransformationParameter transform_param_;
  shared_ptr<DataTransformer<Dtype> > data_transformer_;
  bool output_labels_;
};

template <typename Dtype>
class Batch {
 public:
  Blob<Dtype> data_, label_;
};

template <typename Dtype>
class BasePrefetchingDataLayer :
    public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit BasePrefetchingDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  // Prefetches batches (asynchronously if to GPU memory)
  static const int PREFETCH_COUNT = 3;

 protected:
  virtual void InternalThreadEntry();
  virtual void load_batch(Batch<Dtype>* batch) = 0;

  Batch<Dtype> prefetch_[PREFETCH_COUNT];
  BlockingQueue<Batch<Dtype>*> prefetch_free_;
  BlockingQueue<Batch<Dtype>*> prefetch_full_;

  Blob<Dtype> transformed_data_;
};

template <typename Dtype>
class DataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit DataLayer(const LayerParameter& param);
  virtual ~DataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "Data"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);

  DataReader reader_;
};

/**
 * @brief Provides data to the Net generated by a Filler.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class DummyDataLayer : public Layer<Dtype> {
 public:
  explicit DummyDataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Data layers should be shared by multiple solvers in parallel
  virtual inline bool ShareInParallel() const { return true; }
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual inline const char* type() const { return "DummyData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

  vector<shared_ptr<Filler<Dtype> > > fillers_;
  vector<bool> refill_;
};

/**
 * @brief Provides data to the Net from HDF5 files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class HDF5DataLayer : public Layer<Dtype> {
 public:
  explicit HDF5DataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~HDF5DataLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Data layers should be shared by multiple solvers in parallel
  virtual inline bool ShareInParallel() const { return true; }
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual inline const char* type() const { return "HDF5Data"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void LoadHDF5FileData(const char* filename);

  std::vector<std::string> hdf_filenames_;
  unsigned int num_files_;
  unsigned int current_file_;
  hsize_t current_row_;
  std::vector<shared_ptr<Blob<Dtype> > > hdf_blobs_;
  std::vector<unsigned int> data_permutation_;
  std::vector<unsigned int> file_permutation_;
};

/**
 * @brief Write blobs to disk as HDF5 files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class HDF5OutputLayer : public Layer<Dtype> {
 public:
  explicit HDF5OutputLayer(const LayerParameter& param)
      : Layer<Dtype>(param), file_opened_(false) {}
  virtual ~HDF5OutputLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Data layers should be shared by multiple solvers in parallel
  virtual inline bool ShareInParallel() const { return true; }
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual inline const char* type() const { return "HDF5Output"; }
  // TODO: no limit on the number of blobs
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 0; }

  inline std::string file_name() const { return file_name_; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void SaveBlobs();

  bool file_opened_;
  std::string file_name_;
  hid_t file_id_;
  Blob<Dtype> data_blob_;
  Blob<Dtype> label_blob_;
};

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class ImageDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ImageDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~ImageDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

  vector<std::pair<std::string, int> > lines_;
  int lines_id_;
};

/**
 * @brief Provides data to the Net from memory.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class MemoryDataLayer : public BaseDataLayer<Dtype> {
 public:
  explicit MemoryDataLayer(const LayerParameter& param)
      : BaseDataLayer<Dtype>(param), has_new_data_(false) {}
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MemoryData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

  virtual void AddDatumVector(const vector<Datum>& datum_vector);
  virtual void AddMatVector(const vector<cv::Mat>& mat_vector,
      const vector<int>& labels);

  // Reset should accept const pointers, but can't, because the memory
  //  will be given to Blob, which is mutable
  void Reset(Dtype* data, Dtype* label, int n);
  void set_batch_size(int new_size);

  int batch_size() { return batch_size_; }
  int channels() { return channels_; }
  int height() { return height_; }
  int width() { return width_; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  int batch_size_, channels_, height_, width_, size_;
  Dtype* data_;
  Dtype* labels_;
  int n_;
  size_t pos_;
  Blob<Dtype> added_data_;
  Blob<Dtype> added_label_;
  bool has_new_data_;
};

/**
 * @brief Provides data to the Net from windows of images files, specified
 *        by a window data file.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class WindowDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit WindowDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~WindowDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "WindowData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  virtual unsigned int PrefetchRand();
  virtual void load_batch(Batch<Dtype>* batch);

  shared_ptr<Caffe::RNG> prefetch_rng_;
  vector<std::pair<std::string, vector<int> > > image_database_;
  enum WindowField { IMAGE_INDEX, LABEL, OVERLAP, X1, Y1, X2, Y2, NUM };
  vector<vector<float> > fg_windows_;
  vector<vector<float> > bg_windows_;
  Blob<Dtype> data_mean_;
  vector<Dtype> mean_values_;
  bool has_mean_file_;
  bool has_mean_values_;
  bool cache_images_;
  vector<std::pair<std::string, Datum > > image_database_cache_;
};

/**
* @brief Provides data to the Net from memory.
*
* TODO(dox): thorough documentation for Forward and proto params.
*/
template <typename Dtype>
class SPUnsupervisedDataLayer : public BaseDataLayer<Dtype> {
public:
	explicit SPUnsupervisedDataLayer(const LayerParameter& param)
		: BaseDataLayer<Dtype>(param) {}
	virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "SPUnsupervisedData"; }
	virtual inline int ExactNumBottomBlobs() const { return 0; }
	virtual inline int ExactNumTopBlobs() const { return 2; }

	// Reset should accept const pointers, but can't, because the memory
	//  will be given to Blob, which is mutable
	void Reset(Dtype* data, Dtype* label, int n);
	void set_batch_size(int new_size);

	int batch_size() { return batch_size_; }
	int channels() { return channels_; }
	int height() { return height_; }
	int width() { return width_; }
	int data_limit() { return data_limit_; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	void UnsupervisedImageloadAll(const char* datapath);
	void makeRandbox(int *arr, int size);
	bool fileTypeCheck(char *fileName);

	int batch_size_, channels_, height_, width_, size_;
	int labelHeight_, labelWidth_;
	int n_;
	int data_limit_;
	size_t pos_;

	std::string data_path_;
	std::string label_path_;

	std::vector<cv::Mat> data_blob;
	std::vector<cv::Mat> label_blob;

	int *randbox;
	int dataidx;

	cv::Mat background;
};

/**
* @brief Provides data to the Net from memory.
*
* TODO(dox): thorough documentation for Forward and proto params.
*/
template <typename Dtype>
class SPRGBDUnsupervisedDataLayer : public BaseDataLayer<Dtype> {
public:
	explicit SPRGBDUnsupervisedDataLayer(const LayerParameter& param)
		: BaseDataLayer<Dtype>(param) {}
	virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "SPRGBDUnsupervisedData"; }
	virtual inline int ExactNumBottomBlobs() const { return 0; }
	virtual inline int ExactNumTopBlobs() const { return 2; }

	// Reset should accept const pointers, but can't, because the memory
	//  will be given to Blob, which is mutable
	void Reset(Dtype* data, Dtype* label, int n);
	void set_batch_size(int new_size);

	int batch_size() { return batch_size_; }
	int channels() { return channels_; }
	int height() { return height_; }
	int width() { return width_; }
	int data_limit() { return data_limit_; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	void RGBDImageloadAll(const char* datapath, const char* depthpath);
	bool fileTypeCheck(char *fileName);
	void makeRandbox(int *arr, int size);
	void BackgroudLoad(const char *path, const char *fileName);
	cv::Mat subBackground(cv::Mat rgb, cv::Mat depth);

	int batch_size_, channels_, height_, width_, size_;
	int labelHeight_, labelWidth_;
	int n_;
	int data_limit_;
	size_t pos_;

	std::string data_path_;
	std::string label_path_;

	std::vector<cv::Mat> data_blob;
	std::vector<cv::Mat> label_blob;
	
	cv::Mat backRGB;
	cv::Mat backDepth;

	int *randbox;
	int dataidx;
};

/**
* @brief Provides data to the Net from memory.
*
* TODO(dox): thorough documentation for Forward and proto params.
*/
template <typename Dtype>
class SPRGBDCOMDataLayer : public BaseDataLayer<Dtype> {
public:
	explicit SPRGBDCOMDataLayer(const LayerParameter& param)
		: BaseDataLayer<Dtype>(param) {}
	virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "SPRGBDCOMData"; }
	virtual inline int ExactNumBottomBlobs() const { return 0; }
	virtual inline int ExactNumTopBlobs() const { return 2; }

	// Reset should accept const pointers, but can't, because the memory
	//  will be given to Blob, which is mutable
	void Reset(Dtype* data, Dtype* label, int n);
	void set_batch_size(int new_size);

	int batch_size() { return batch_size_; }
	int channels() { return channels_; }
	int height() { return height_; }
	int width() { return width_; }
	int data_limit() { return data_limit_; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	void RGBDloadAll_calcCom(const char* datapath, const char* depthpath);
	bool fileTypeCheck(char *fileName);
	void makeRandbox(int *arr, int size);
	void BackgroudLoad(const char *path, const char *fileName);
	cv::Mat subBackground(cv::Mat rgb, cv::Mat depth);

	int batch_size_, channels_, height_, width_, size_;
	int labelHeight_, labelWidth_;
	int n_;
	int data_limit_;
	size_t pos_;

	std::string data_path_;
	std::string label_path_;

	std::vector<cv::Mat> data_blob;
	std::vector<cv::Mat> label_blob;

	cv::Mat backRGB;
	cv::Mat backDepth;

	int *randbox;
	int dataidx;
};

/**
* @brief Provides data to the Net from memory.
*
* TODO(dox): thorough documentation for Forward and proto params.
*/
//image, depth, pregrasping pos, COM 
template <typename Dtype>
class PreGraspDataLayer : public BaseDataLayer<Dtype> {
public:
	explicit PreGraspDataLayer(const LayerParameter& param)
		: BaseDataLayer<Dtype>(param) {}
	virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "PreGraspData"; }
	virtual inline int ExactNumBottomBlobs() const { return 0; }

	// Reset should accept const pointers, but can't, because the memory
	//  will be given to Blob, which is mutable
	void Reset(Dtype* data, Dtype* label, int n);
	void set_batch_size(int new_size);

	int batch_size() { return batch_size_; }
	int channels() { return channels_; }
	int height() { return height_; }
	int width() { return width_; }
	int data_limit() { return data_limit_; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	void PreGrasp_DataLoadAll(const char* datapath);
	bool fileTypeCheck(char *fileName);
	void makeRandbox(int *arr, int size);
	float calcDist3D(cv::Point3f A, cv::Point3f B);
	bool calcFingerSort(cv::Point3f *upperLeft, cv::Point3f *upperRight, cv::Point3f *thumb);

	int batch_size_, channels_, height_, width_, size_;
	int n_;
	int data_limit_;

	std::string data_path_;

	std::vector<cv::Mat> image_blob;						//rgb image
	std::vector<cv::Mat> depth_blob;						//depth image
	std::vector<cv::Mat> com_blob;							//center of mass blob
	std::vector<std::pair<int, cv::Mat>> pos_blob;			//pregrasping pos (image idx, pos)

	int *randbox;
	int dataidx;
};

/**
* @brief Provides data to the Net from memory.
*
* TODO(dox): thorough documentation for Forward and proto params.
*/
//image, depth, pregrasping pos, COM 
template <typename Dtype>
class IKDataLayer : public BaseDataLayer<Dtype> {
public:
	explicit IKDataLayer(const LayerParameter& param)
		: BaseDataLayer<Dtype>(param) {}
	virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "IKData"; }
	virtual inline int ExactNumBottomBlobs() const { return 0; }

	// Reset should accept const pointers, but can't, because the memory
	//  will be given to Blob, which is mutable
	void Reset(Dtype* data, Dtype* label, int n);
	void set_batch_size(int new_size);

	int batch_size() { return batch_size_; }
	int channels() { return channels_; }
	int height() { return height_; }
	int width() { return width_; }
	int data_limit() { return data_limit_; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	void IK_DataLoadAll(const char* datapath);
	bool fileTypeCheck(char *fileName);
	void makeRandbox(int *arr, int size);

	int batch_size_, channels_, height_, width_, size_;
	int n_;
	int data_limit_;

	std::string data_path_;

	std::vector<cv::Mat> image_blob;						//rgb image
	std::vector<cv::Mat> depth_blob;						//depth image
	std::vector<cv::Mat> ang_blob;			//pregrasping pos (image idx, pos)

	int *randbox;
	int dataidx;
};

//MNIST LOADER
template <typename Dtype>
class MNISTDataLayer : public BaseDataLayer<Dtype> {
public:
	explicit MNISTDataLayer(const LayerParameter& param)
		: BaseDataLayer<Dtype>(param) {}
	virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "MNISTData"; }
	virtual inline int ExactNumBottomBlobs() const { return 0; }
	virtual inline int ExactNumTopBlobs() const { return 2; }

	// Reset should accept const pointers, but can't, because the memory
	//  will be given to Blob, which is mutable
	void Reset(Dtype* data, Dtype* label, int n);
	void set_batch_size(int new_size);

	int batch_size() { return batch_size_; }
	int channels() { return channels_; }
	int data_limit() { return data_limit_; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	void ImageloadAll(const char* datapath, const char* labelpath);
	int reverseInt(int i);
	void FileOpenData(const char *fileName);
	void FileOpenLabel(const char *fileName);
	int getDataCount();
	void create_image(cv::Mat *dst, cv::Size size, int channels, unsigned char data[28][28], int imagenumber);

	int batch_size_, channels_, size_;
	int n_;
	int data_limit_;
	int magic_number;
	int number_of_images;
	int n_rows, n_cols;
	size_t pos_;

	std::string data_path_;
	std::string label_path_;

	std::vector<cv::Mat> data;
	std::vector<cv::Mat> label;

	int *randbox;
	int dataidx;
};

//uvd_xyz_ Data Loader
template <typename Dtype>
class UVDXYZDataLayer : public BaseDataLayer<Dtype> {
public:
	explicit UVDXYZDataLayer(const LayerParameter& param)
		: BaseDataLayer<Dtype>(param) {}
	virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "UVDXYZData"; }
	virtual inline int ExactNumBottomBlobs() const { return 0; }
	virtual inline int ExactNumTopBlobs() const { return 2; }

	// Reset should accept const pointers, but can't, because the memory
	//  will be given to Blob, which is mutable
	void Reset(Dtype* data, Dtype* label, int n);
	void set_batch_size(int new_size);

	int batch_size() { return batch_size_; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	
	void readData(const char *path);

	int batch_size_;
	int n_;
	int size_;

	std::string data_path_;

	std::vector<cv::Point3f> data;
	std::vector<cv::Point3f> label;

	int *randbox;
	int dataidx;
};

/**
* @brief Provides data to the Net from memory.
*
* TODO(dox): thorough documentation for Forward and proto params.
*/
template <typename Dtype>
class SPsupervisedDataLayer : public BaseDataLayer<Dtype> {
public:
	explicit SPsupervisedDataLayer(const LayerParameter& param)
		: BaseDataLayer<Dtype>(param) {}
	virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "SPsupervisedData"; }
	virtual inline int ExactNumBottomBlobs() const { return 0; }
	virtual inline int ExactNumTopBlobs() const { return 3; }

	// Reset should accept const pointers, but can't, because the memory
	//  will be given to Blob, which is mutable
	void Reset(Dtype* data, Dtype* label, int n);
	void set_batch_size(int new_size);

	int batch_size() { return batch_size_; }
	int height() { return height_; }
	int width() { return width_; }
	int data_limit() { return data_limit_; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	void BinFileloadAll(const char* datapath);

	int batch_size_, height_, width_, size_;
	int n_;
	int data_limit_;
	size_t pos_;

	typedef struct RGBimgData_{
		Dtype data[80 * 80 * 3];
	}RGBImgData;

	typedef struct DEPTHimgData_{
		Dtype data[80 * 80];
	}DEPTHImgData;

	typedef struct LabelPosData_{
		Dtype pos[3*3];
	}LabelPosData;

	std::string data_path_;
	std::string label_path_;

	std::vector<RGBImgData> data_blob;
	std::vector<DEPTHImgData> Depth_blob;
	std::vector<LabelPosData> label_blob;

	int *randbox;
	int dataidx;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
