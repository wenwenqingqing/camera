#include "opencv2/opencv.hpp"
#include "opencv2/video/tracking.hpp"

#include <sstream>
#include <fstream>
#include <dirent.h>
#include <algorithm>

#include "benchmark_dataset_reader.h"
#include "types.h"

int32_t leak_padding=2;
int32_t iteration = 10;
int32_t skip_frames = 1;

void parseArgument(char* arg);
void plotE(double_t* irra, int32_t cols, int32_t rows, std::string saveTo="");
void plotG(double_t* g_irra, std::string saveTo="");
Eigen::Vector2d rmse(double_t* g_irra, double_t* irra, std::vector<double_t> &exposure_value, std::vector<uint8_t*> &img_data_vec,  int32_t wh);

int32_t main( int32_t argc, char** argv )
{
	// parse arguments
	for(int32_t i=2; i<argc;i++)
	{
		parseArgument(argv[i]);
	}
	// load exposure times & images.
	// first parameter is dataset location.
	int32_t cols=0, rows=0, image_size=0;

	DatasetReader* reader = new DatasetReader(argv[1]);
	std::vector<double_t> exposure_value;
	std::vector<uint8_t* > img_data_vec;
	for(int32_t i=0; i<reader->GetImageNums(); i+=skip_frames)
	{
		cv::Mat img = reader->GetImageInternal(i);
		if(img.rows==0 || img.cols==0)
		{
			continue;
		}
		assert(img.type() == CV_8U);

		if((cols!=0 && cols != img.cols) || img.cols==0)
		{ 
			printf("width mismatch!\n"); 
			exit(1); 
		};
		if((rows!=0 && rows != img.rows) || img.rows==0)
		{ 
			printf("height mismatch!\n"); 
			exit(1); 
		};

		cols = img.cols;
		rows = img.rows;

		uint8_t* data = new uint8_t[img.rows*img.cols];
		std::memcpy(data, img.data, img.rows*img.cols);
		img_data_vec.push_back(data);
		exposure_value.push_back((double_t)(reader->GetExposure(i)));

		uint8_t* data2 = new uint8_t[img.rows*img.cols];
		for(int32_t it=0;it<leak_padding;it++)
		{
			std::memcpy(data2, data, img.rows*img.cols);
			for(int32_t y=1; y<rows-1; y++)
			{
				for(int32_t x=1; x<cols-1; x++)
				{
					if(data[x+y*cols]==255)
					{
						data2[x+1 + cols*(y+1)] = 255;
						data2[x+1 + cols*(y  )] = 255;
						data2[x+1 + cols*(y-1)] = 255;

						data2[x   + cols*(y+1)] = 255;
						data2[x   + cols*(y  )] = 255;
						data2[x   + cols*(y-1)] = 255;

						data2[x-1 + cols*(y+1)] = 255;
						data2[x-1 + cols*(y  )] = 255;
						data2[x-1 + cols*(y-1)] = 255;
					}
				}
			}
			std::memcpy(data, data2, img.rows*img.cols);
		}
		delete[] data2;
	}
	image_size = img_data_vec.size(); // 图片数量

	printf("loaded %d images\n", image_size);

	double_t irra[cols*rows];		// scene irradiance
	double_t irra_size[cols*rows];		// scene irradiance
	double_t g_irra[256];		// inverse response function

	for(int32_t i=0; i<image_size; i++)
	{
		for(int32_t k=0; k<cols*rows; k++)
		{
			//if(img_data_vec[i][k]==255) continue;
			irra[k] += img_data_vec[i][k];
			irra_size[k]++;
		}
	}

	for(int32_t k=0; k<cols*rows; k++)
	{
		irra[k] = irra[k] / irra_size[k]; // 平均强度
	} 

	if(-1 == system("rm -rf photoCalibResult")) 
	{
		printf("could not delete old photoCalibResult folder!\n");
	}
	if(-1 == system("mkdir photoCalibResult"))
	{
		printf("could not create photoCalibResult folder!\n");
	}

	std::ofstream log_file;
	log_file.open("photoCalibResult/log.txt", std::ios::trunc | std::ios::out);
	log_file.precision(15);

	printf("init RMSE = %f! \t", rmse(g_irra, irra, exposure_value, img_data_vec, cols*rows )[0]);
	plotE(irra,cols,rows, "photoCalibResult/irra-0");
	cv::waitKey(100);

	bool_t opt_e = true;
	bool_t opt_g = true;

	for(int32_t it=0; it<iteration; it++)
	{
		if(opt_g)
		{
			// optimize log inverse response function.
			double_t g_irra_sum[256]; //! irradiance * exposure_value of every 0~255 intensity
			double_t g_irra_num[256]; //! num of every 0~255 intensity

			for(int32_t i=0; i<image_size; i++)
			{
				for(int32_t k=0; k<cols*rows; k++)
				{
					int32_t b = img_data_vec[i][k];
					if(b == 255)
					{
						continue;
					}
					g_irra_num[b]++;                             
					g_irra_sum[b]+= irra[k] * exposure_value[i];
				}
			}

			for(int32_t i=0; i<256; i++)
			{
				g_irra[i] = g_irra_sum[i] / g_irra_num[i];
				if(!std::isfinite(g_irra[i]) && i > 1) 
				{
					g_irra[i] = g_irra[i-1] + (g_irra[i-1]-g_irra[i-2]);
				}
			}

			printf("opt_g RMSE = %f! \t", rmse(g_irra, irra, exposure_value, img_data_vec, cols*rows )[0]);
			char buf[1000]; snprintf(buf, 1000, "photoCalibResult/g_irra-%d.png", it+1);
			plotG(g_irra, buf);
		}

		if(opt_e)
		{
			// optimize scene irradiance function.
			double_t e_irra_sum[cols*rows]; //! g_irradiance * exposure_value of every pixel
			double_t e_irra_t2[cols*rows];  //! exposure_value * exposure_value of every pixel

			for(int32_t i=0;i<image_size;i++)
			{
				for(int32_t k=0;k<cols*rows;k++)
				{
					int32_t b = img_data_vec[i][k];
					if(b == 255)
					{
						continue;
					}
					e_irra_t2[k] += exposure_value[i]*exposure_value[i];
					e_irra_sum[k] += (g_irra[b]) * exposure_value[i];
				}
			}
			for(int32_t i=0; i<cols*rows;i++)
			{
				irra[i] = e_irra_sum[i] / e_irra_t2[i];
				if(irra[i] < 0)
				{
					irra[i] = 0;
				}
			}

			printf("OptE RMSE = %f!  \t", rmse(g_irra, irra, exposure_value, img_data_vec, cols*rows )[0]);

			char buf[1000]; snprintf(buf, 1000, "photoCalibResult/irra-%d", it+1);
			plotE(irra, cols, rows, buf);
		}

		// rescale such that maximum response is 255 (fairly arbitrary choice).
		double_t rescale_factor=255.0 / g_irra[255];
		for(int32_t i=0; i<cols*rows; i++)
		{
			irra[i] *= rescale_factor;
			if(i<256) 
			{
				g_irra[i] *= rescale_factor;
			}
		}
		Eigen::Vector2d err = rmse(g_irra, irra, exposure_value, img_data_vec, cols*rows );
		printf("resc RMSE = %f!  \trescale with %f!\n",  err[0], rescale_factor);

		log_file << it << " " << image_size << " " << err[1] << " " << err[0] << "\n";
		cv::waitKey(100);
	}

	log_file.flush();
	log_file.close();

	std::ofstream lg;
	lg.open("photoCalibResult/pcalib.txt", std::ios::trunc | std::ios::out);
	lg.precision(15);
	for(int32_t i=0;i<256;i++)
	{
		lg << g_irra[i] << " ";
	}
	lg << "\n";

	lg.flush();
	lg.close();

	for(int32_t i=0; i<image_size; i++)
	{
		delete[] img_data_vec[i];
	}

	return 0;
}

Eigen::Vector2d rmse(double_t* g_irra, double_t* irra, std::vector<double_t> &exposure_value, std::vector<uint8_t*> &img_data_vec,  int32_t wh)
{
	double_t e=0;		// yeah - these will be sums of a LOT of values, so we need super high precision.
	double_t num=0;

	int32_t image_size = img_data_vec.size();
	for(int32_t i=0;i<image_size;i++)
	{
		for(int32_t k=0;k<wh;k++)
		{
			if(img_data_vec[i][k] == 255) continue;
			double_t r = g_irra[img_data_vec[i][k]] - exposure_value[i]*irra[k];
			if(!std::isfinite(r)) continue;
			e += r*r*1e-10;
			num++;
		}
	}

	return Eigen::Vector2d(1e5*sqrtl((e/num)), (double_t)num);
}

void plotE(double_t* irra, int32_t cols, int32_t rows, std::string saveTo)
{

	// try to find some good color scaling for plotting.
	double_t offset = 20;
	double_t min=1e10, max=-1e10;

	double_t Emin=1e10, Emax=-1e10;

	for(int32_t i=0;i<cols*rows;i++)
	{
		double_t le = log(irra[i]+offset);
		if(le < min) min = le;
		if(le > max) max = le;

		if(irra[i] < Emin) Emin = irra[i];
		if(irra[i] > Emax) Emax = irra[i];
	}

	cv::Mat EImg = cv::Mat(rows,cols,CV_8UC3);
	cv::Mat EImg16 = cv::Mat(rows,cols,CV_16U);

	for(int32_t i=0;i<cols*rows;i++)
	{
		float_t val = 3 * (exp((log(irra[i]+offset)-min) / (max-min))-1) / 1.7183;

		int32_t icP = val;
		float_t ifP = val-icP;
		icP = icP%3;

		cv::Vec3b color;
		if(icP == 0) color= cv::Vec3b(0 ,	   	0,		     	255*ifP);
		if(icP == 1) color= cv::Vec3b(0, 		255*ifP,     	255);
		if(icP == 2) color= cv::Vec3b(255*ifP, 	255, 			255);

		EImg.at<cv::Vec3b>(i) = color;
		EImg16.at<ushort>(i) = 255* 255* (irra[i]-Emin) / (Emax-Emin);
	}

	printf("Irradiance %f - %f\n", Emin, Emax);
	cv::imshow("lnE", EImg);

	if(saveTo != "")
	{
		cv::imwrite(saveTo+".png", EImg);
		cv::imwrite(saveTo+"16.png", EImg16);
	}
}

void plotG(double_t* g_irra, std::string saveTo)
{
	cv::Mat GImg = cv::Mat(256,256,CV_32FC1);
	GImg.setTo(0);

	double_t min=1e10, max=-1e10;

	for(int32_t i=0;i<256;i++)
	{
		if(g_irra[i] < min) min = g_irra[i];
		if(g_irra[i] > max) max = g_irra[i];
	}

	for(int32_t i=0;i<256;i++)
	{
		double_t val = 256*(g_irra[i]-min) / (max-min);
		for(int32_t k=0;k<256;k++)
		{
			if(val < k)
				GImg.at<float_t>(k,i) = k-val;
		}
	}

	printf("Inv. Response %f - %f\n", min, max);
	cv::imshow("g_irra", GImg);
	if(saveTo != "") cv::imwrite(saveTo, GImg*255);
}

void parseArgument(char* arg)
{
	int32_t option;

	if(1==sscanf(arg,"leak_padding=%d",&option))
	{
		leak_padding = option;
		printf("leak_padding set to %d!\n", leak_padding);
		return;
	}
	if(1==sscanf(arg,"iterations=%d",&option))
	{
		iteration = option;
		printf("iteration set to %d!\n", iteration);
		return;
	}
	if(1==sscanf(arg,"skip=%d",&option))
	{
		skip_frames = option;
		printf("skip_frames set to %d!\n", skip_frames);
		return;
	}

	printf("could not parse argument \"%s\"!!\n", arg);
}