#include "windows.h"
#include <iostream>
#include <chrono>
#include <time.h>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"
#include <opencv2/opencv.hpp>
#include <regex>
#include <algorithm>
#include <chrono>
#include <highgui/highgui_c.h>
#include "Net.h"
#include <ctime>
#include <device_launch_parameters.h>
#include <io.h>
#include <direct.h>

#define DEVICE0 0
#define DEVICE1 1
#define OK 0;
#define ERR -1;
#define KEY 0x9b;
#define BATCH_SIZE 1
#define BYTE unsigned char

const string DLL_VERSION = "1.0.1"; //在上一版本之上，修改了UNet的预处理部分，解决了边缘填充时，像素值不能填充成一个正方形的问题。
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
const int model_select = 0;
int tcount = 0;

cv::Mat uchar2Mat[15][12];
bool Flags = false;
bool Handle_Flags = false;

static Logger gLogger;
using namespace std;
using namespace cv;
using namespace nvinfer1;


struct EngineParameter_
{
	float* tdata;
	float* prob;
	IRuntime* runtime;
	ICudaEngine* engine;
	IExecutionContext* context;
	cudaStream_t stream;
	void* buffer[2];
	
	int input_index;
	int output_index;
};

EngineParameter_ struct_cam[15][12] = {0};

template<class T>
int length(T& arr)
{
	return sizeof(arr) / sizeof(arr[0]);
}

struct axi 
{ 
	float x, y; 
}Num[4] = {0};

struct Scaling_ratio
{
	int top;
	int bottom;
	int left;
	int right;
	double ratio;
}get_ratio = {0};

struct return_data
{
	ICudaEngine* eng = nullptr;
	int img_h;
	int img_w;
	int cls;
} sEng[15][12] = {0};

list<return_data> aa ;

/// <summary>
/// 坐标排序的规则
/// </summary>
/// <param name="ax_A"> 第一个坐标的值</param>
/// <param name="ax_B"> 第二个坐标的值</param>
/// <returns>由小到大返回坐标的值</returns>
float cmp(axi ax_A, axi ax_B)
{
	if (ax_A.x == ax_B.x) 
	return ax_A.y < ax_B.y;
	return ax_A.x < ax_B.x;
};


/// <summary>
/// 图像预处理
/// </summary>
/// <param name="img"> 需要处理的图像</param>
/// <param name="data"> 创建的数组指针，以数组的形式存放resize后图像的值</param>
/// <param name="in_H"> resize后图像的高</param>
/// <param name="in_W"> resize后图像的宽</param>
void proccess_img_ResNet(cv::Mat img, float* data, int in_H, int in_W)
{
	cv::Mat imgDst;
	cv::resize(img, imgDst, cv::Size(in_W, in_H), cv::INTER_CUBIC);
	
	//先二值化，再归一化 ---> 减均值，除以标准差
	int i = 0;
	for (int row = 0; row < in_H; ++row)
	{
		uchar* uc_pixel = imgDst.data + row * imgDst.step;
		for (int col = 0; col < in_W; ++col) {
			data[i] = ((float)uc_pixel[2]/255 - 0.485) / 0.229;
			data[i + in_H * in_W] = ((float)uc_pixel[1]/255 - 0.456) / 0.224;
			data[i + 2 * in_H * in_W] = ((float)uc_pixel[0]/255 - 0.406) / 0.225;
			uc_pixel += 3;
			++i;
		}
	}
}


/// <summary>
/// UNet网络的图像预处理部分
/// </summary>
/// <param name="img"> 需要处理的图像</param>
/// <param name="data"> 创建的数组指针，以数组的形式存放resize后图像的值</param>
/// <param name="in_H"> resize后图像的高</param>
/// <param name="in_W"> resize后图像的宽</param>
/// <returns> 以结构体数组的形式返回resize后，上下左右图像分别较少的宽度，还返回图像的缩放比例</returns>
Scaling_ratio proccess_img_UNet(cv::Mat img, float* data, int in_H, int in_W)
{
	double ih = img.rows;
	double iw = img.cols;
	double scale = min(in_H / ih, in_W / iw);
	int nw = int(iw * scale);
	if (nw < in_W) nw += 1;
	int nh = int(ih * scale);


	int topborder = (in_H - nh) / 2;
	int bootomborder = (in_H - nh) / 2;
	int leftborder = (in_W - nw) / 2;
	int rightborder = (in_W - nw) / 2;

	if ((in_H - nh) % 2 != 0)
	{
		bootomborder += 1;
	}
	if ((in_W - nw) % 2 != 0)
	{
		rightborder += 1;
	}

	//cv::Mat new_img;
	cv::resize(img, img, Size(nw, nh));
	Mat U_img = Mat::zeros(in_H, in_W, CV_8UC3);
	cv::copyMakeBorder(img, U_img, topborder, bootomborder, leftborder, rightborder, cv::BORDER_CONSTANT, cv::Scalar::all(0));

	int i = 0;
	for (int row = 0; row < in_H; ++row)
	{
		uchar* uc_pixel = U_img.data + row * U_img.step;
		for (int col = 0; col < in_W; ++col) {
			data[i] = (float)uc_pixel[2] / 255.0;
			data[i + in_H * in_W] = (float)uc_pixel[1] / 255.0;
			data[i + 2 * in_H * in_W] = (float)uc_pixel[0] / 255.0;
			uc_pixel += 3;
			++i;
		}
	}

	Scaling_ratio total;
	total.top = (in_H - nh) / 2;
	total.bottom = (in_H - nh) / 2;
	total.left = (in_W - nw) / 2;
	total.right = (in_W - nw) / 2;
	total.ratio = scale;
	return  total;
}


/// <summary>
/// 对推理结果进行后处理部分，将所有类别的推理数值映射到（0,1）区间内，即每类概率。
/// </summary>
/// <param name="in_H"></param>
/// <param name="in_W"></param>
/// <param name="tcls_num"></param>
/// <param name="tprob"></param>
/// <param name="tout"></param>
void softmax_UNet(int in_H, int in_W, int tcls_num, float* tprob, double* tout)
{
	for (int t = 0; t < in_H * in_W; t++)
	{
		float sum1 = 0.;
		for (int p = 0; p < tcls_num; p++)
		{
			sum1 += exp(tprob[t + in_H * in_W * p]);
		}

		for (int y = 0; y < tcls_num; y++)
		{
			tout[t + in_H * in_W * y] = tprob[t + in_H * in_W * y] / sum1;
		}
		
	}
}

/// <summary>
/// 找出多通道图像中的每个像素值最高的哪一个通道，并将该通道像素变成255
/// </summary>
/// <param name="in_H"> 推理图像的高</param>
/// <param name="in_W"> 推理图像的宽</param>
/// <param name="tcls_num"> 训练的图像类别数</param>
/// <param name="threshold"> 像素阈值</param>
/// <param name="out"> 返回所有通道的像素数组指针</param>
void argmax_UNet(int in_H, int in_W, int tcls_num, float threshold, float* out)
{
	for (int t = 0; t < in_H * in_W; t++)
	{
		double max_value = -INFINITY;
		int count_c = 0;
		for (int k = 0; k < tcls_num; k++)
		{
			if (max_value < out[t + in_H * in_W * k] && out[t + in_H * in_W * k] > threshold)
			{
				max_value = out[t + in_H * in_W * k];
				count_c = k;
			}
		}
		out[t] = count_c;

		for (int v = 0; v < tcls_num; v++)
		{
			if (v == count_c && v != 0)
			{
				out[t + in_H * in_W * v] = 255;
			}
			else {
				out[t + in_H * in_W * v] = 0;
			}
		}
	}
}


static void HandleError(cudaError_t err, const char* file, int line, int cam, int thrd)
{
	if (err != cudaSuccess) {
		Handle_Flags = true;
		ofstream receiveData;
		receiveData.open("D:\\NewVision\\Log\\trt_Log\\cudaError.txt",ios::app);
		time_t now = time(0);
		receiveData << ctime(&now) <<"  Error: " << int(err) << "---" << cudaGetErrorString(err) << "in " << file << "  at line: " << line << endl;
		receiveData.close();

		cv::imwrite("D:\\NewVision\\Log\\trt_Log\\Failure.bmp", uchar2Mat[cam][thrd]);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err, cam, thrd ) (HandleError( err, __FILE__, __LINE__, cam, thrd))


/// <summary>
/// UNet的主要推理函数
/// </summary>
/// <param name="context"> cuda中构建的上下文</param>
/// <param name="engine"> cuda中构建的引擎</param>
/// <param name="stream"> cuda流</param>
/// <param name="buffers"> cuda的缓存空间</param>
/// <param name="input"> 输入的要推理的图像</param>
/// <param name="output"> 输出的推理结果</param>
/// <param name="in_H"> 推理图像的高</param>
/// <param name="in_W"> 推理图像的宽</param>
/// <param name="select_gpu"> 指定推理要用的gpu，这个要与LoadFile时的gpu对应上，否则报错</param>
void doInference_ResNet(IExecutionContext& context, ICudaEngine& engine, cudaStream_t stream, void* buffers[2],
	float* input, float* output, int in_H, int in_W, int select_gpu, int inputIndex, int outputIndex, int cam, int thrd) {
	int CudaCount;
	cudaGetDeviceCount(&CudaCount);

	if (CudaCount >= select_gpu)
	{
		if (select_gpu == 1)
		{
			cudaSetDevice(DEVICE0);
		}

		else if (select_gpu == 2)
		{
			cudaSetDevice(DEVICE1);
		}
	}
	else
	{
		cudaSetDevice(DEVICE0);
	}
	
	//HANDLE_ERROR(cudaMemcpy(buffers[inputIndex], input, 1 * 3 * in_H * in_W * sizeof(float), cudaMemcpyHostToDevice),cam, thrd);
	////HANDLE_ERROR(cudaStreamSynchronize(stream), cam, thrd);
	//context.enqueue(1, buffers, stream, nullptr);
	//HANDLE_ERROR(cudaMemcpy(output, buffers[outputIndex], 1 * 2 * sizeof(float), cudaMemcpyDeviceToHost), cam, thrd);
	////HANDLE_ERROR(cudaStreamSynchronize(stream), cam, thrd);
	
	HANDLE_ERROR(cudaMemcpyAsync(buffers[inputIndex], input, 1 * 3 * in_H * in_W * sizeof(float), cudaMemcpyHostToDevice, stream),cam,thrd);
	//HANDLE_ERROR(cudaStreamSynchronize(stream), cam, thrd);
	context.enqueue(1, buffers, stream, nullptr);	
	HANDLE_ERROR(cudaMemcpyAsync(output, buffers[outputIndex], 1 * 2 * sizeof(float), cudaMemcpyDeviceToHost, stream), cam, thrd);
	//HANDLE_ERROR(cudaStreamSynchronize(stream), cam, thrd);
	
	// Release stream and buffers
	/*cudaStreamDestroy(stream);
	HANDLE_ERROR(cudaFree(buffers[inputIndex]));
	HANDLE_ERROR(cudaFree(buffers[outputIndex]));*/
}

/// <summary>
/// ResNet的主要推理函数
/// </summary>
/// <param name="context"> cuda中构建的上下文</param>
/// <param name="engine"> cuda中构建的引擎</param>
/// <param name="input"> 输入的要推理的图像</param>
/// <param name="output"> 输出的推理结果</param>
/// <param name="batchSize"> 一次推理几张</param>
/// <param name="in_H"> 推理图像的高</param>
/// <param name="in_W"> 推理图像的宽</param>
/// <param name="out_size"> 保存推理结果的数组</param>
//void doInference_UNet(IExecutionContext& context, ICudaEngine& engine, cudaStream_t stream, void* buffers[2], float* input, float* output, int batchSize, int in_H, int in_W, int out_size, int select_gpu) {
//	
//	int CudaCount;
//	cudaGetDeviceCount(&CudaCount);
//
//	if (CudaCount >= select_gpu)
//	{
//		if (select_gpu == 1)
//		{
//			cudaSetDevice(DEVICE0);
//		}
//
//		else if (select_gpu == 2)
//		{
//			cudaSetDevice(DEVICE1);
//		}
//	}
//	else
//	{
//		cudaSetDevice(DEVICE0);
//	}
//
//	
//	
//	//IExecutionContext& context = contextX;
//	//ICudaEngine& engine = engineX;
//	engine = context.getEngine();
//
//	// Pointers to input and output device buffers to pass to engine.
//	// Engine requires exactly IEngine::getNbBindings() number of buffers.
//	assert(engine.getNbBindings() == 2);
//	
//
//	// In order to bind the buffers, we need to know the names of the input and output tensors.
//	// Note that indices are guaranteed to be less than IEngine::getNbBindings()
//	const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
//	const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
//
//	// Create GPU buffers on device
//	HANDLE_ERROR(cudaMalloc(&buffers[inputIndex], batchSize * 3. * in_H * in_W * sizeof(float)));
//	HANDLE_ERROR(cudaMalloc(&buffers[outputIndex], 1.* batchSize * out_size * sizeof(float)));
//
//	// Create stream
//	HANDLE_ERROR(cudaStreamCreate(&stream));
//
//	// DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
//	HANDLE_ERROR(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3. * in_H * in_W * sizeof(float), cudaMemcpyHostToDevice, stream));
//	context.enqueue(batchSize, buffers, stream, nullptr);
//	HANDLE_ERROR(cudaMemcpyAsync(output, buffers[outputIndex], 1. * batchSize * out_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
//	//流同步：通过cudaStreamSynchronize()来协调。
//	cudaStreamSynchronize(stream);
//
//	// Release stream and buffers
//	cudaStreamDestroy(stream);
//	HANDLE_ERROR(cudaFree(buffers[inputIndex]));
//	HANDLE_ERROR(cudaFree(buffers[outputIndex]));
//}

void Match_device(int deviceCount, int select_gpu)
{
	if (deviceCount >= select_gpu)
	{
		if (select_gpu == 1)
		{
			cudaSetDevice(DEVICE0);
		}
		else if (select_gpu == 2)
		{
			cudaSetDevice(DEVICE1);
		}
	}
	else
	{
		cudaSetDevice(DEVICE0);
	}
}


extern"C"
{
	/*******************************************************
		-函数名称：Loadfile

		-功能描述：用于开启模型推理的引擎；

		-输入参数：
				   file_path: 引擎文件的路径；
				     cam_cls: 开启的第几个相机；
				  cam_thread: 这个相机要开几个线程(从0开始计数)；
				model_select:
							  当设置为0时，使用UNet;
						      当设置为1时，使用ResNet;

		-输出参数：无，此函数功能是开启引擎，无参数输出;
	*******************************************************/
	_declspec (dllexport)void Loadfile(const char* file_path, 
									   int cam_cls, 
									   int cam_thread, 
									   int select_gpu, 
									   int model_select)
	
	{
		int cudaSetPrecision=32;
		int dev;
		//判断显卡数量
		Flags = true;
		char* trtModelStream{ nullptr };
		int deviceCount;
		cudaGetDeviceCount(&deviceCount);
		int INPUT_h = 0;
		int INPUT_w = 0;
		//指定的推理显卡与设备存在的显卡对应上
		Match_device(deviceCount, select_gpu);

		//识别当前显卡算力并自动设置FP16或FP32
		for (dev = 0; dev < deviceCount; dev++)
		{
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, dev);

			ofstream ofs;
			time_t now = time(0);
			ofs.open("D:\\NewVision\\Log\\trt_Log\\Device_Log.txt", ios::app);
			ofs << ctime(&now) << "device:"<< dev << " " << deviceProp.name << endl;

			if (deviceProp.major > 6)
			{
				cudaSetPrecision = 16;
			}
			else
			{
				cudaSetPrecision = 32;
			}
			//cudaSetPrecision = 32;
		}

		const char* Dir = "D:\\NewVison\\Log\\trt_Log";
		if (_access(Dir, 0) == -1)
		{
			mkdir(Dir);
		}

		ofstream trtError;
		trtError.open(Dir);
		trtError.close();

		//解密bin文件；
		string root_path = file_path;		

		if (model_select == 0)
		{
			//Unet, 对应的encode.txt; encode.bin
			fstream in(root_path + "\\encode.txt");
			string s;
			if (in.fail())
			{
				cout << "open file error" << endl;
			}

			while (std::getline(in, s), '\n')
			{
				string str = UTF8ToGB(s.c_str()).c_str();

				if (startsWith(str, "image_H_W"))
				{
					std::vector<string> res = split(str, ":");
					std::vector<string> res1 = split(res[1], ",");
					INPUT_h = stoi(res1[0]);
					INPUT_w = stoi(res1[1]);
				}
				else if (startsWith(str, "END"))
				{
					break;
				}
			}
			in.close(); 
		}

		else if (model_select == 1)
		{

			fstream in(root_path + "\\encode_R.txt");
			string s;
			if (in.fail())
			{
				cout << "open file error" << endl;
			}

			while (getline(in, s), '\n')
			{
				string str = UTF8ToGB(s.c_str()).c_str();
				if (startsWith(str, "image_H_W"))
				{
					std::vector<string> res = split(str, ":");
					std::vector<string> ree = split(res[1], ",");
					INPUT_h = stoi(ree[0]);
					INPUT_w = stoi(ree[1]);
				}

				else if (startsWith(str, "END"))
				{
					break;
				}
			}in.close();
		}

		sEng[cam_cls][cam_thread] = convert_bin(root_path, model_select, cudaSetPrecision);
		
		//生成engine
		struct_cam[cam_cls][cam_thread].engine = sEng[cam_cls][cam_thread].eng;
		struct_cam[cam_cls][cam_thread].runtime= createInferRuntime(gLogger);
		struct_cam[cam_cls][cam_thread].context = struct_cam[cam_cls][cam_thread].engine->createExecutionContext();

		assert(struct_cam[cam_cls][cam_thread].engine->getNbBindings() == 2);

		struct_cam[cam_cls][cam_thread].input_index = struct_cam[cam_cls][cam_thread].engine->getBindingIndex(INPUT_BLOB_NAME);
		struct_cam[cam_cls][cam_thread].output_index = struct_cam[cam_cls][cam_thread].engine->getBindingIndex(OUTPUT_BLOB_NAME);

		// Create GPU buffers on device
		HANDLE_ERROR(cudaMalloc(&struct_cam[cam_cls][cam_thread].buffer[struct_cam[cam_cls][cam_thread].input_index], 1. * 3 * INPUT_h * INPUT_w * sizeof(float)), cam_cls,cam_thread);
		HANDLE_ERROR(cudaMalloc(&struct_cam[cam_cls][cam_thread].buffer[struct_cam[cam_cls][cam_thread].output_index], 1 * 2 * sizeof(float)), cam_cls, cam_thread);

		// Create stream
		HANDLE_ERROR(cudaStreamCreate(&struct_cam[cam_cls][cam_thread].stream), cam_cls, cam_thread);

		HANDLE_ERROR(cudaMallocHost((void**)&struct_cam[cam_cls][cam_thread].tdata, 3 * INPUT_h * INPUT_w * sizeof(float)),cam_cls, cam_thread);
		if (model_select == 0)
		{
			HANDLE_ERROR(cudaMallocHost((void**)&struct_cam[cam_cls][cam_thread].prob, 1 * sEng[cam_cls][cam_thread].cls * sEng[cam_cls][cam_thread].img_h * sEng[cam_cls][cam_thread].img_w * sizeof(float)), cam_cls, cam_thread);
		}
		else if (model_select == 1)
		{
			HANDLE_ERROR(cudaMallocHost((void**)&struct_cam[cam_cls][cam_thread].prob, 2 * sizeof(float)), cam_cls, cam_thread);
		}
		else
		{
			HANDLE_ERROR(cudaMallocHost((void**)&struct_cam[cam_cls][cam_thread].prob, 1 * sEng[cam_cls][cam_thread].cls * sEng[cam_cls][cam_thread].img_h * sEng[cam_cls][cam_thread].img_w * sizeof(float)), cam_cls, cam_thread);
		}


		/*struct_cam[cam_cls][cam_thread].tdata = new float[3. * INPUT_h * INPUT_w];
		
		if (model_select == 0)
		{
			struct_cam[cam_cls][cam_thread].prob = new float[1. * sEng.cls * sEng.img_h * sEng.img_w];		
		}
		else if (model_select == 1)
		{
			struct_cam[cam_cls][cam_thread].prob = new float[2];		
		}
		else
		{
			struct_cam[cam_cls][cam_thread].prob = new float[1. * sEng.cls * sEng.img_h * sEng.img_w];		
		}*/
	}

	/*******************************************************************************************************************************************
		-函数名称：Result

		-功能描述：用于对输入的图像进行推理，并输出结果；

		-输入参数：
							  img1: 传入需要推理的图片;
						camera_num: 在第几个相机上进行推理;
						thread_num: 在第几个线程上进行推理;
						select_gpu: 推理在哪个GPU上(要与Loadfile时选择的一样);
					defect_classes：训练时手动设置的缺陷数量;
						 threshold：设定的推理的阈值;
					area_threshold: 面积阈值(只在UNet起作用);
						   width_1: 原图的宽;
						  height_1: 原图的高;
						   INPUT_H: 模型要求的输入图像的高;
						   INPUT_W: 模型要求的输入图像的宽.
					  model_select:
									当设置为0时，使用UNet;
									当设置为1时，使用ResNet;
							  envs: 为0时，输出缺陷的轮廓坐标;为1时，输出模型的外接举行坐标.

		-输出参数：输出的检测的结果,详情见dll说明文档;

	*********************************************************************************************************************************************/
	
	_declspec (dllexport)char* Result(BYTE* img1, 
									int camera_num, 
									int thread_num, 
									int select_gpu, 
									int defect_classes, 
									float threshold, 
									int area_threshold, 
									int width_1, 
									int height_1, 
									int INPUT_H, 
									int INPUT_W, 
									int model_select, 
									int envs)
	{
		//若要使用UNet进行推理
		if (model_select == 0)
		{
			std::string get_axis = "";
			std::string get_area = "";

			int out_size = defect_classes * INPUT_H * INPUT_W;

			cv::Mat InputImg;

			try {
				InputImg = cv::Mat(height_1, width_1, CV_8UC3, (uchar*)img1);
			}
			catch (Exception& e)
			{
				ofstream ofs;
				ofs.open("D:\\NewVision\\Log\\trt_Log\\process_img_Error.txt", ios::app);
				ofs << "[E] MyException caught:" << e.what() << endl;

				ofs.close();
			}

			 
			cv::Mat OutputImg = cv::Mat(INPUT_H, INPUT_W, CV_8UC3);
			cv::Mat MaskImg = cv::Mat(INPUT_H, INPUT_W, CV_8UC1);
			
			//图像预处理
			Scaling_ratio get_ratio;
			get_ratio = proccess_img_UNet(InputImg, struct_cam[camera_num][thread_num].tdata, INPUT_H, INPUT_W);
			
			////对图像进行推理	
			//doInference_UNet(*struct_cam[camera_num][thread_num].context,
			//				 *struct_cam[camera_num][thread_num].engine,
			//				 struct_cam[camera_num][thread_num].stream,
			//				 struct_cam[camera_num][thread_num].buffer,
			//				 struct_cam[camera_num][thread_num].tdata,
			//				 struct_cam[camera_num][thread_num].prob,
			//				 1, INPUT_H, INPUT_W, out_size, select_gpu);
		
			//像素级的分类
			//softmax_UNet(INPUT_H, INPUT_W, defect_classes, prob, out);

			//获取每个像素中之心度最高的种类
			argmax_UNet(INPUT_H, INPUT_W, defect_classes, threshold, struct_cam[camera_num][thread_num].prob);

			//生成每类检测图然后转图片		
			uchar* ptmp_0 = NULL;

			//将检测后的缺陷类别转化为图像，找到最小外接矩形，根据设置的面积阈值，输出面积矩形的四个坐标点，将坐标以字符的形式返回出去。
			for (int p = 0; p < defect_classes; p++)
			{
				for (int i = 0; i < INPUT_H; i++)
				{
					ptmp_0 = MaskImg.ptr<uchar>(i);
					for (int j = 0; j < INPUT_W; j++)
					{
						ptmp_0[j] = (struct_cam[camera_num][thread_num].prob)[i * INPUT_W + j + INPUT_W * INPUT_H * p];
					}
				}

				//获取每张图上每个缺陷的最小外接矩形
				vector<vector<Point> > contours;
				vector<Vec4i> hierarchy;
				cv::findContours(MaskImg, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());
				float defect_h;
				float defect_w;

				if (envs == 0)
				{
					for (int v = 0; v < contours.size(); v++)
					{
						get_axis += to_string(p);
						for (int b = 0; b < contours[v].size(); b++)
						{
							get_axis += ",";
							get_axis += to_string(((double)(contours[v][b].x) - get_ratio.left) / get_ratio.ratio);
							get_axis += ",";
							get_axis += to_string(((double)(contours[v][b].y) - get_ratio.top) / get_ratio.ratio);
						}
						get_axis += ",";
						get_axis += to_string(contourArea(contours[v]));
						get_axis += ";";
					}
				}
				else if (envs == 1)
				{
					for (int i = 0; i < contours.size(); i++)
					{
						//获取最小外接矩形的坐标
						RotatedRect rect = minAreaRect(contours[i]);
						Point2f Pt[4];
						rect.points(Pt);
						for (int i = 0; i < 4; i++)
						{
							Num[i].x = Pt[i].x;
							Num[i].y = Pt[i].y;
						}
						sort(Num, Num + 4, cmp);


						double raw_0x = ((double)(Pt[0].x) - get_ratio.left) / get_ratio.ratio;
						double raw_0y = ((double)(Pt[0].y) - get_ratio.top) / get_ratio.ratio;

						double raw_1x = ((double)(Pt[1].x) - get_ratio.left) / get_ratio.ratio;
						double raw_1y = ((double)(Pt[1].y) - get_ratio.top) / get_ratio.ratio;

						double raw_2x = ((double)(Pt[2].x) - get_ratio.left) / get_ratio.ratio;
						double raw_2y = ((double)(Pt[2].y) - get_ratio.top) / get_ratio.ratio;

						double raw_3x = ((double)(Pt[3].x) - get_ratio.left) / get_ratio.ratio;
						double raw_3y = ((double)(Pt[3].y) - get_ratio.top) / get_ratio.ratio;


						//获取外接矩形的长、宽以及面积
						double tmp_a = pow(pow((raw_3x - raw_0x), 2) + pow((raw_3y - raw_0y), 2), 0.5);
						double tmp_b = pow(pow((raw_3x - raw_2x), 2) + pow((raw_3y - raw_2y), 2), 0.5);
						if (tmp_a > tmp_b)
						{
							defect_h = tmp_b;
							defect_w = tmp_a;
						}
						else
						{
							defect_h = tmp_a;
							defect_w = tmp_b;
						}
						int area = defect_h * defect_w;
						double defect_area = area;
						//判断检测到框的长和宽是否满足大于面积阈值，若大于，就记录类别、坐标、高、宽、面积，否则跳过该检测框
						if (defect_area > area_threshold)
						{
							get_axis += to_string(p);
							get_axis += ",";
							get_axis += to_string(double(raw_0x));
							get_axis += ",";
							get_axis += to_string(double(raw_0y));
							get_axis += ",";
							get_axis += to_string(double(raw_1x));
							get_axis += ",";
							get_axis += to_string(double(raw_1y));
							get_axis += ",";
							get_axis += to_string(double(raw_2x));
							get_axis += ",";
							get_axis += to_string(double(raw_2y));
							get_axis += ",";
							get_axis += to_string(double(raw_3x));
							get_axis += ",";
							get_axis += to_string(double(raw_3y));
							get_axis += ",";
							get_axis += to_string(defect_h);
							get_axis += ",";
							get_axis += to_string(defect_w);
							get_axis += ",";
							get_axis += to_string(defect_area);
							get_axis += ";";
						}
					}
				}
			}

			/*delete[]struct_cam[camera_num].RT_Eng_Cont.tdata[thread_num];
			delete[]struct_cam[camera_num].RT_Eng_Cont.prob[thread_num];
			delete[]struct_cam[camera_num].RT_Eng_Cont.out[thread_num];*/

			return strdup(get_axis.c_str());
		}
		else if (model_select == 1)
		{
			try 
			{
				uchar2Mat[camera_num][thread_num] = cv::Mat(height_1, width_1, CV_8UC3, img1);
				
				if (uchar2Mat[camera_num][thread_num].cols != width_1 || uchar2Mat[camera_num][thread_num].rows != height_1 || uchar2Mat[camera_num][thread_num].channels() !=3)
				{
					cout << "****************" << uchar2Mat[camera_num][thread_num].data << endl;
					ofstream ofs;
					time_t now = time(0);
					cv::imwrite("D:\\NewVision\\Log\\trt_Log\\error_img.bmp", uchar2Mat[camera_num][thread_num]);
					
					ofs.open("D:\\NewVision\\Log\\trt_Log\\process_img_Error.txt", ios::app);
					ofs << ctime(&now) << ":  接收到的图片不完整\n" << "Error: engine2dll.dll in line 741: uchar2Mat = cv::Mat(height_1, width_1, CV_8UC3, (uchar*)img1);" << endl;
					
					ofs.close();
					//string throw_out = "Error: engine2dll.dll in line 802: uchar2Mat = cv::Mat(height_1, width_1, CV_8UC3, (uchar*)img1);";
					//throw throw_out;
				}

				//cout << "width:" << uchar2Mat[camera_num][thread_num].cols << "----height:" << uchar2Mat[camera_num][thread_num].rows << "-----channels:" << uchar2Mat[camera_num][thread_num].channels() << endl;

			}
			catch (Exception& e)
			{
				ofstream ofs;
				time_t now = time(0);
				ofs.open("D:\\NewVision\\Log\\trt_Log\\process_img_Error.txt", ios::app);
				ofs << ctime(&now)<<"  [E] MyException caught:" << e.what() << endl;

				ofs.close();
				string throw_out = "Error: engine2dll.dll in line 741: uchar2Mat = cv::Mat(height_1, width_1, CV_8UC3, (uchar*)img1);";
			}
			
			//预处理					
			proccess_img_ResNet(uchar2Mat[camera_num][thread_num], struct_cam[camera_num][thread_num].tdata, INPUT_H, INPUT_W);
			//delete[] struct_cam[camera_num][thread_num].tdata;
			//struct_cam[camera_num][thread_num].tdata = new float[3 * INPUT_H * INPUT_W];
	
			doInference_ResNet(*struct_cam[camera_num][thread_num].context,
					*struct_cam[camera_num][thread_num].engine,
					struct_cam[camera_num][thread_num].stream,
					struct_cam[camera_num][thread_num].buffer,
					struct_cam[camera_num][thread_num].tdata,
					struct_cam[camera_num][thread_num].prob,
					INPUT_H, INPUT_W, select_gpu,
					struct_cam[camera_num][thread_num].input_index,
					struct_cam[camera_num][thread_num].output_index,
					camera_num, thread_num);

			//softmax
			float tout[2] = {0,0};
			tout[0] = exp(struct_cam[camera_num][thread_num].prob[0]) / (exp(struct_cam[camera_num][thread_num].prob[0]) + exp(struct_cam[camera_num][thread_num].prob[1]));
			tout[1] = exp(struct_cam[camera_num][thread_num].prob[1]) / (exp(struct_cam[camera_num][thread_num].prob[0]) + exp(struct_cam[camera_num][thread_num].prob[1]));
	
			if (typeid(tout[0]) == typeid(float))
			{
				string get_result="0,0,0,0,0,0,0,0,0,0,0,0;";

				if (tout[1] > tout[0])
				{
					if (tout[1] <= 1 && tout[1] >= 0)
					{
						get_result = "0," + to_string(tout[1]) + ",0,0,0,0,0,0,0,0,0,0;";
					}
					else
					{
						get_result = "0,0.501,0,0,0,0,0,0,0,0,0,0;";
					}
				}
				else {
					
					if (tout[0] >= 0 && tout[0] <= 1)
					{
						get_result = "1," + to_string(tout[0]) + ",0,0,0,0,0,0,0,0,0,0;";
					}
					else
					{
						get_result = "1,0.501,0,0,0,0,0,0,0,0,0,0;";
					}
				}
				return strdup(get_result.c_str());
			}
			else
			{
				ofstream ofs;
				time_t now = time(0);
				ofs.open("D:\\NewVision\\Log\\trt_Log\\process_img_Error.txt", ios::app);
				ofs << ctime(&now) << "  [E] Function(softmax_ResNet): softmax输出的二分类结果有问题。"<< endl;

				ofs.close();
				string throw_out = "Error: engine2dll.dll in line 830: [E] : softmax输出的二分类结果有问题。;";
				//throw throw_out;
				string get_result = "0,0.501,0,0,0,0,0,0,0,0,0,0;";
				return strdup(get_result.c_str());
			}
		}
		else
		{
			ofstream ofs;
			time_t now = time(0);
			ofs.open("D:\\NewVision\\Log\\trt_Log\\process_img_Error.txt", ios::app);
			ofs << ctime(&now) << "  [E] Function(softmax_ResNet): softmax输出的二分类结果有问题。" << endl;

			ofs.close();
			string throw_out = "Error: engine2dll.dll in line 830: [E] : softmax输出的二分类结果有问题。;";
			//throw throw_out;
			string get_result = "0,0.501,0,0,0,0,0,0,0,0,0,0;";
			return strdup(get_result.c_str());
		}
	}

	/*******************************************************
		-函数名称：ReleaseEng

		-功能描述：用于释放推理的线程；

		-输入参数：camera_cls: 释放的第几个相机；
					num_thread: 释放此相机上第几个线程；

		-输出参数：无

	*******************************************************/
	_declspec (dllexport)void ReleaseEng(int camera_cls, int num_thread)
	{
		if (Flags == true)
		{
			struct_cam[camera_cls][num_thread].context->destroy();
			struct_cam[camera_cls][num_thread].engine->destroy();
		}
	}

	_declspec (dllexport)char* getVersion()
	{
		return (char*)DLL_VERSION.c_str();
	}
}



