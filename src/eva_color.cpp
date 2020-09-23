#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cvdef.h>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "benchmark_dataset_reader.h"
#include <eigen3/Eigen/Eigen>
#include <stdio.h>
#include <iostream>

#define MIN(a,b)  ((a) > (b) ? (b) : (a))

cv::Rect rect_select;
bool_t rect_select_flag=false;
cv::Point origin;
cv::Mat frame;
cv::Mat frame_bak;
std::vector<cv::Rect> rectv;

struct COLOR
{
    float_t R;
    float_t G;
    float_t B;
};

int32_t calculategrid(int32_t x, int32_t y, int32_t width, int32_t height, std::vector<cv::Rect>& rects);
int32_t calculategridcolor(const cv::Mat image, std::vector<cv::Rect>& rects, std::vector<COLOR>& colors);

float_t colorList[24*3] = {128.4,84.57,72.99, 213.9,164.7,142.3, 76.76,128.1,161.3, 89.51,104.7,67.13,
                         145.5,163.4,183.5, 128.7,216.5,191.5, 166.4,142.8,50.32, 67.38,93.98,168.9,
                         164.4,102.7,111.4, 79.12,58.66,110.1, 174.8,142.8,50.32, 67.38,93.98,168.9,
                         59.19,60.10,150.3, 72.24,141.7,76.29, 162.5,48.34,46.70, 193.3,202.9,21.97,
                         173.3,93.30,165.6, 3.634,133.0,166.6, 232.2,234.5,234.3, 211.2,214.6,211.1,
                         166.4,167.5,168.2, 129.3,129.8,131.5, 89.55,89.15,90.88, 57.03,57.27,56.35};

float_t colorList0[24*3] = {128.4,84.57,72.99, 166.4,142.8,50.32, 59.19,60.10,150.3, 232.2,234.5,234.3,
                         213.9,164.7,142.3, 67.38,93.98,168.9, 72.24,141.7,76.29, 211.2,214.6,211.1,
                         76.76,128.1,161.3, 164.4,102.7,111.4, 162.5,48.34,46.70, 166.4,167.5,168.2,
                         89.51,104.7,67.13, 79.12,58.66,110.1, 193.3,202.9,21.97, 129.3,129.8,131.5,
                         145.5,163.4,183.5, 174.8,199.8,76.04, 173.3,93.30,165.6, 89.55,89.15,90.88,
                         128.7,216.5,191.5, 205.1,175.4,33.10, 3.634,133.0,166.6, 57.03,57.27,56.35};


/************************************************************************************************************************/
/****                            如果采用这个onMouse()函数的话，则可以画出鼠标拖动矩形框的4种情形                        ****/
/************************************************************************************************************************/
void onMouse(int32_t event,int32_t x,int32_t y,int32_t,void*)
{
    //Point origin;//不能在这个地方进行定义，因为这是基于消息响应的函数，执行完后origin就释放了，所以达不到效果。
    if(rect_select_flag)
    {
        rect_select.x=MIN(origin.x,x);//不一定要等鼠标弹起才计算矩形框，而应该在鼠标按下开始到弹起这段时间实时计算所选矩形框
        rect_select.y=MIN(origin.y,y);
        rect_select.width=abs(x-origin.x);//算矩形宽度和高度
        rect_select.height=abs(y-origin.y);
        rect_select&=cv::Rect(0,0,frame.cols,frame.rows);//保证所选矩形框在视频显示区域之内

    }
    if(event==CV_EVENT_LBUTTONDOWN)
    {
        rect_select_flag=true;//鼠标按下的标志赋真值
        origin=cv::Point(x,y);//保存下来单击是捕捉到的点
        rect_select=cv::Rect(x,y,0,0);//这里一定要初始化，宽和高为(0,0)是因为在opencv中Rect矩形框类内的点是包含左上角那个点的，但是不含右下角那个点
    }
    else if(event==CV_EVENT_LBUTTONUP)
    {
        rect_select_flag=false;
        frame_bak = frame;
        rectv.push_back(rect_select);
        std::cout << rect_select.x << "," << rect_select.y << "," << rect_select.width <<"," << rect_select.height << std::endl;
        std::vector<cv::Rect> rects;
        std::vector<COLOR> colors;

        calculategrid(rect_select.x, rect_select.y, rect_select.width, rect_select.height, rects);
        calculategridcolor(frame, rects, colors);
        cv::rectangle(frame_bak,rect_select,cv::Scalar(255,0,0),3,8,0);//能够实时显示在画矩形窗口时的痕迹
        // std::cout << "rects size : " << rects.size() << std::endl;
        for(int32_t i = 0; i < rects.size(); i++)
        {
            cv::rectangle(frame_bak, rects[i], cv::Scalar(0,255,0),1,8,0);
        }
        //显示视频图片到窗口
        cv::imshow("camera",frame_bak);
    }
}

int32_t main(int32_t argc, char** argv)
{
    cv::Mat image;
    double_t exposure = 0.0, exposure_bak = 0.0;
    RGBImgReader* reader = new RGBImgReader(argv[1]);
    std::vector<double_t> exposure_value;
	std::vector<unsigned char*> img_data_vec;
    
    std::vector<cv::Rect> rects;

    bool_t do_process = false;
    std::vector<double_t> SNR_R;
    std::vector<double_t> SNR_G;
    std::vector<double_t> SNR_B;

    std::map<double_t,std::vector<cv::Mat> > expos_img_map;
    std::vector<cv::Mat> imgVec;
    std::vector<cv::Mat> imgVecLast;

    int32_t width, height;
    image = reader->GetImageInternal(0);
    width = image.cols;
    height = image.rows;
    // cv::Mat imgAve = cv::Mat(width, height, CV_8UC3);
    std::cout << "width : " << width << " ,height : " <<height << std::endl;
    float_t *imgSum = (float_t*) malloc(3*width*height*sizeof(float_t));
    float_t *imgAve = (float_t*) malloc(3*width*height*sizeof(float_t));

    float_t SIGNAL_R;
    float_t SIGNAL_G;
    float_t SIGNAL_B;

    float_t NOISE_R;
    float_t NOISE_G;
    float_t NOISE_B;

    std::vector<float_t> NOISEs_R;
    std::vector<float_t> NOISEs_G;
    std::vector<float_t> NOISEs_B;

    memset(imgSum,0,3*width*height*sizeof(float_t));
    memset(imgAve,0,3*width*height*sizeof(float_t));
    std::cout << "GetImageInternal\n";
	std::cout << "reader img num : " << reader->GetImageNums() << std::endl;
    for(int32_t i=0;i<reader->GetImageNums();i++)
	{
        image = reader->GetImageInternal(i);
        exposure = reader->GetExposure(i);

        if(i == reader->GetImageNums()-1)
        {
            imgVec.push_back(image);
            imgVecLast = imgVec;
            do_process = true;
        }
        else if(exposure_bak == 0 || exposure == exposure_bak)
        {
            imgVec.push_back(image);
            do_process == false;
        }        
        else
        {
            do_process = true;
        }
        
        if(do_process)
        {
            //expos_img_map[exposure] = imgVec; 
            std::cout << "0"  << std::endl;
            for(auto iter = imgVec.begin(); iter != imgVec.end(); iter++)
            {   
                for(int32_t i = 0; i < width; i++)
                {
                    for(int32_t j = 0; j < height ; j++)
                    {
                       imgSum[j*width*3 + i*3] += iter->data[j*width*3 + i*3];
                       imgSum[j*width*3 + i*3 + 1] += iter->data[j*width*3 + i*3 + 1];
                       imgSum[j*width*3 + i*3 + 2] += iter->data[j*width*3 + i*3 + 2];
                    }
                }
                
            }
            std::cout << "1";
            for(int32_t i = 0; i < width; i++)
            {
                for(int32_t j = 0; j < height ; j++)
                {
                    imgAve[j*width*3 + i*3]  =  imgSum[j*width*3 + i*3] / imgVec.size();
                    imgAve[j*width*3 + i*3 + 1]  =  imgSum[j*width*3 + i*3 + 1] / imgVec.size();
                    imgAve[j*width*3 + i*3 + 2]  =  imgSum[j*width*3 + i*3 + 2] / imgVec.size();
                }
            }
            
            for(auto iter = imgVec.begin(); iter != imgVec.end(); iter++)
            {   
                for(int32_t i = 0; i < width; i++)
                {
                    for(int32_t j = 0; j < height ; j++)
                    {
                       SIGNAL_R += iter->data[j*width*3 + i*3] * iter->data[j*width*3 + i*3];
                       SIGNAL_G += iter->data[j*width*3 + i*3 + 1] * iter->data[j*width*3 + i*3 + 1];
                       SIGNAL_B += iter->data[j*width*3 + i*3 + 2] * iter->data[j*width*3 + i*3 + 2];

                       NOISE_R += (iter->data[j*width*3 + i*3] - imgAve[j*width*3 + i*3]) * (iter->data[j*width*3 + i*3] - imgAve[j*width*3 + i*3]);
                       NOISE_G += (iter->data[j*width*3 + i*3 + 1] - imgAve[j*width*3 + i*3 + 1]) * (iter->data[j*width*3 + i*3 + 1] - imgAve[j*width*3 + i*3 + 1]);
                       NOISE_B += (iter->data[j*width*3 + i*3 + 2] - imgAve[j*width*3 + i*3 + 2]) * (iter->data[j*width*3 + i*3 + 2] - imgAve[j*width*3 + i*3 + 2]);
                    }
                } 
            }
            float_t SNRs = sqrt(SIGNAL_R/NOISE_R);
            SNR_R.push_back(SNRs);
            NOISEs_R.push_back(sqrt(NOISE_R /(width*height)));

            float_t SNRs_G = sqrt(SIGNAL_G/NOISE_G);
            SNR_G.push_back(SNRs_G);
            NOISEs_G.push_back(sqrt(NOISE_G /(width*height)));

            float_t SNRs_B = sqrt(SIGNAL_B/NOISE_B);
            SNR_B.push_back(SNRs_B);
            NOISEs_B.push_back(sqrt(NOISE_B /(width*height)));

            exposure_value.push_back(exposure);

            std::cout << "EXPOSURE : " << exposure << ", IMG size : " << imgVec.size() << std::endl;
            std::cout << " ,R NOISE : "<< 20*log10(sqrt(NOISE_R / (width*height))) << " ,SNR : " << 20*log10(SNRs) << std::endl;
            std::cout << " ,G NOISE : "<< 20*log10(sqrt(NOISE_G / (width*height))) << " ,SNR : " << 20*log10(SNRs_G) << std::endl;
            std::cout << " ,B NOISE : "<< 20*log10(sqrt(NOISE_B / (width*height))) << " ,SNR : " << 20*log10(SNRs_B) << std::endl;
            SIGNAL_R = 0;
            SIGNAL_G = 0;
            SIGNAL_B = 0;
            
            NOISE_R = 0;
            NOISE_B = 0;
            NOISE_G = 0;
            memset(imgSum,0,3*width*height*sizeof(float_t));
            memset(imgAve,0,3*width*height*sizeof(float_t));
            
            imgVec.clear();
            imgVec.push_back(image);
            std::cout << "one process end" << std::endl;
        }
        do_process = false;
        std::cout << "No : " << i << ", exposure time : " << exposure << std::endl;
        std::ofstream openfile("EvaResualts.txt", std::ios::app);
        for (int32_t i = 1; i < SNR_R.size(); i++)
        {
            openfile << exposure_value[i] << " " << SNR_R[i] << " " << NOISEs_R[i] << " " << SNR_G[i] << " " << NOISEs_G[i] << " " << SNR_B[i] << " " << NOISEs_B[i] << std::endl;
        }
        openfile.close();
        exposure_bak = exposure;
    }

    std::vector<cv::Mat> undistortion;
    for(int32_t i = 0;i < reader->getNumUndistortImages(); i++)
    {
        cv::Mat img = reader->getUndistortImgI(i);
        undistortion.push_back(img);
    }  
    for(auto iter = undistortion.begin(); iter != undistortion.end(); iter++)
    { 
        frame = *iter;
        frame_bak = frame;
        //建立窗口
        cv::namedWindow("camera",1);//显示视频原图像的窗口
        //捕捉鼠标

        //显示视频图片到窗口
        cv::imshow("camera",frame_bak);
        cv::setMouseCallback("camera",onMouse,0);
        cv::waitKey();
    }
        
    return 0;
}

int32_t calculategrid(int32_t x, int32_t y, int32_t width, int32_t height, std::vector<cv::Rect>& rects)
{
    float_t dur_width = (float_t)width*156/1188;
    float_t dur_height = (float_t)height*134/687;
    float_t mid_width = (float_t)width*50/1188;
    float_t mid_height = (float_t)height*50/687;
    int32_t x_num = 6;
    int32_t y_num = 4;

    for(int32_t i = 0; i < x_num; i++)
    {
        for(int32_t j = 0; j < y_num; j++)
        {
            cv::Rect rect0;
            rect0.x = x + i*(dur_width + mid_width);
            rect0.y = y + j*(dur_height + mid_height);
            rect0.width = dur_width;
            rect0.height = dur_height;
            // std::cout << "rect0: " << rect0.x << ", " << rect0.y << ", " << rect0.width << ", " << rect0.height << std::endl;
            rects.push_back(rect0);
        }
    }

}

int32_t calculategridcolor(const cv::Mat image, std::vector<cv::Rect>& rects, std::vector<COLOR>& colors)
{
    uchar* data = image.data;
    float_t R_NOISE, G_NOISE, B_NOISE;
    for(int32_t i = 0; i < rects.size(); i++)
    {
        float_t RR = 0, GG = 0, BB = 0;
        cv::Rect rect0 = rects[i];
        rect0.x = rect0.x + rect0.width * 0.1;
        rect0.y = rect0.y + rect0.height * 0.1;
        rect0.width = rect0.width * 0.8;
        rect0.height = rect0.height * 0.8;
        rects[i] = rect0;
        for(int32_t i = 0; i< rect0.width; i++)
        {
            for(int32_t j= 0; j < rect0.height; j++)
            {
                int32_t index = (rect0.y + j)*image.cols + rect0.x + i;
                float_t _b = data[index*3];
                float_t _g = data[index*3+1];
                float_t _r = data[index*3+2];
                BB += _b;
                GG += _g;
                RR += _r;
            }
        }
        COLOR color;
        color.B = (float_t)BB / (float_t)(rect0.width*rect0.height);
        color.G = (float_t)GG / (float_t)(rect0.width*rect0.height);
        color.R = (float_t)RR / (float_t)(rect0.width*rect0.height);

        RR = 0; GG = 0; BB = 0;
        for(int32_t i = 0; i< rect0.width; i++)
        {
            for(int32_t j= 0; j < rect0.height; j++)
            {
                int32_t index = (rect0.y + j)*image.cols + rect0.x + i;
                float_t _b = data[index*3];
                float_t _g = data[index*3+1];
                float_t _r = data[index*3+2];
                
                BB += (_b - color.B)*(_b - color.B);
                GG += (_g - color.G)*(_g - color.G);
                RR += (_r - color.R)*(_r - color.R);
            }
        }
        R_NOISE = sqrt(RR / (float_t)(rect0.width*rect0.height));
        G_NOISE = sqrt(GG / (float_t)(rect0.width*rect0.height));
        B_NOISE = sqrt(BB / (float_t)(rect0.width*rect0.height));
        colors.push_back(color);
        //std::cout << std::fixed << std::setprecision(4) << " color RGB : " << color.R << ", " << color.G << ", " << color.B << std::endl;
        //std::cout << std::fixed << std::setprecision(4) << " color RGB noise : " << R_NOISE << ", " << G_NOISE << ", " << B_NOISE << std::endl;

        std::cout << color.R << ", " << color.G << ", " << color.B << ",";
        std::cout << R_NOISE << ", " << G_NOISE << ", " << B_NOISE << std::endl;
    }
}