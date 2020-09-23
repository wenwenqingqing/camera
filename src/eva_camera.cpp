// author yangpeiwen@holomatic.com
// date 2019.11.01
// calculate noise and SNR , dynamic range

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cvdef.h>
#include <opencv2/highgui/highgui.hpp>

#include "benchmark_dataset_reader.h"

int32_t main(int32_t argc, char** argv)
{
    cv::Mat image;
    double_t exposure = 0.0, exposure_bak = 0.0;
    DatasetReader* reader = new DatasetReader(argv[1]);
    std::vector<double_t> exposure_value;
	std::vector<unsigned char*> img_data_vec;
    std::vector<double_t> noise;
    std::vector<double_t> SNR;
    std::map<double_t,std::vector<cv::Mat> > expos_img_map;
    std::vector<cv::Mat> imgVec;

    int32_t width, height;
    image = reader->GetImageInternal(0);
    width = image.cols;
    height = image.rows;
    // cv::Mat imgAve = cv::Mat(width, height, CV_8UC3);
    std::cout << "width : " << width << " ,height : " <<height << std::endl;
    float_t *imgSum = (float_t*) malloc(width*height*sizeof(float_t));
    float_t *imgAve = (float_t*) malloc(width*height*sizeof(float_t));

    float_t SIGNAL;
    float_t NOISE;
    std::vector<float_t> NOISEs;

    memset(imgSum,0,width*height*sizeof(float_t));
    memset(imgAve,0,width*height*sizeof(float_t));
    std::cout << "GetImageInternal";
	
    for(int32_t i=0;i<reader->GetImageNums();i++)
	{
        image = reader->GetImageInternal(i);
        exposure = reader->GetExposure(i);
        
        if(exposure_bak == 0 || exposure == exposure_bak)
        {
            imgVec.push_back(image);
        }
        else
        {
            expos_img_map[exposure] = imgVec; 
            for(auto iter = imgVec.begin(); iter != imgVec.end(); iter++)
            {   
                for(int32_t i = 0; i < width; i++)
                {
                    for(int32_t j = 0; j < height ; j++)
                    {
                       imgSum[j*width + i] += iter->data[j*width + i];
                       
                    }
                }
                
            }
            for(int32_t i = 0; i < width; i++)
            {
                for(int32_t j = 0; j < height ; j++)
                {
                    imgAve[j*width + i]  =  imgSum[j*width + i] / imgVec.size();
                }
            }
            
            for(auto iter = imgVec.begin(); iter != imgVec.end(); iter++)
            {   
                for(int32_t i = 0; i < width; i++)
                {
                    for(int32_t j = 0; j < height ; j++)
                    {
                       SIGNAL += iter->data[j*width + i] * iter->data[j*width + i];
                       NOISE += (iter->data[j*width + i] - imgAve[j*width + i]) * (iter->data[j*width + i] - imgAve[j*width + i]);
                    }
                } 
            }
            float_t SNRs = sqrt(SIGNAL/NOISE);
            SNR.push_back(SNRs);
            NOISEs.push_back(sqrt(NOISE /(width*height)));
            exposure_value.push_back(exposure);

            std::cout << "EXPOSURE : " << exposure << ", IMG size : " << imgVec.size();
            std::cout << " ,NOISE : "<< 20*log10(sqrt(NOISE / (width*height))) << " ,SNR : " << 20*log10(SNRs) << std::endl;

            SIGNAL = 0;
            NOISE = 0;
            memset(imgSum,0,width*height*sizeof(float_t));
            memset(imgAve,0,width*height*sizeof(float_t));
            imgVec.clear();
        }
        
        std::ofstream openfile("EvaResualts.txt", std::ios::app);
        for (int32_t i = 1; i < SNR.size(); i++)
        {
            openfile << exposure_value[i] << " " << SNR[i] << " " << NOISEs[i] << std::endl;
        }
        openfile.close();

        exposure_bak = exposure;
    }
    //cv::imshow("test", image);
    //cv::waitKey(0);
    return 0;
}