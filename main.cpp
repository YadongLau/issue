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

const string DLL_VERSION = "1.0.1"; 
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


float cmp(axi ax_A, axi ax_B)
{
	if (ax_A.x == ax_B.x) 
	return ax_A.y < ax_B.y;
	return ax_A.x < ax_B.x;
};


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
	_declspec (dllexport)void Loadfile(const char* file_path, 
									   int cam_cls, 
									   int cam_thread, 
									   int select_gpu, 
									   int model_select)
	
	{
		int cudaSetPrecision=32;
		int dev;
		
		Flags = true;
		char* trtModelStream{ nullptr };
		int deviceCount;
		cudaGetDeviceCount(&deviceCount);
		int INPUT_h = 0;
		int INPUT_w = 0;
		
		Match_device(deviceCount, select_gpu);

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
		string root_path = file_path;		

		if (model_select == 1)
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
		HANDLE_ERROR(cudaMallocHost((void**)&struct_cam[cam_cls][cam_thread].prob, 2 * sizeof(float)), cam_cls, cam_thread);
	
	}
	
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
			
				
		proccess_img_ResNet(uchar2Mat[camera_num][thread_num], struct_cam[camera_num][thread_num].tdata, INPUT_H, INPUT_W);
	
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



