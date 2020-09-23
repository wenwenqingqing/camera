#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cvdef.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/types.hpp>
#include "benchmark_dataset_reader.h"
#include <eigen3/Eigen/Eigen>
#include <stdio.h>
#include <iostream>

enum
{
    PIN = 1,
    FISHEYE
};

enum
{
    CALIBRATION = 1,
    UNDISTORTION
};

#define MIN(a,b)  ((a) > (b) ? (b) : (a))
#define CV_TERMCRIT_ITER    1
#define CV_TERMCRIT_NUMBER  CV_TERMCRIT_ITER
#define CV_TERMCRIT_EPS     2

cv::Rect rect_select;
bool_t rect_select_flag=false;
cv::Point origin;
cv::Mat frame;
cv::Mat frame_bak;
std::vector<cv::Rect> rectv;

int32_t calibration_mode = UNDISTORTION;
int32_t camera_model = FISHEYE;
cv::Size board_size = cv::Size(5,8);

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
        cv::rectangle(frame_bak,rect_select,cv::Scalar(255,0,0),3,8,0);//能够实时显示在画矩形窗口时的痕迹
        //显示视频图片到窗口
        cv::imshow("camera",frame_bak);
    }
}

int32_t detect_chess_board(cv::Mat& image, int32_t width, int32_t height, std::vector<cv::Point2f>& corners);

using namespace std;
using namespace cv;

int32_t main(int32_t argc, char** argv)
{
    std::vector<cv::Mat> image;
    cv::Mat img;
    std::vector<cv::Point2f> corners;
    RGBImgReader* reader = new RGBImgReader(argv[1]);
    std::string path = reader->getPath();

    if(calibration_mode == UNDISTORTION)
    {
        Size image_size;
        for(int32_t i = 0 ; i< reader->GetImageNums(); i++){
            img = reader->GetImageInternal(i);
            image.push_back(img);
        }
        
        image_size.width = image[0].cols;
        image_size.height = image[0].rows;

        Mat mapx = Mat(image_size, CV_32FC1);
        Mat mapy = Mat(image_size, CV_32FC1);
        Mat R = Mat::eye(3, 3, CV_32F);
        string imageFileName;
        std::stringstream StrStm;
        cv::Matx33d K;
        cv::Vec4d D;
        
        K << 990.531481, 0.000000, 937.793183,
             0.000000, 1003.992961, 560.395581,
             0.000000, 0.000000, 1.000000;

        D << -0.060434, 0.056930, -0.078662, 0.034630; 

        cv::Mat cameraMatrix=Mat(3,3,CV_32FC1,Scalar::all(0)); /* 摄像机内参数矩阵 */  
        std::vector<int32_t> point_counts;  // 每幅图像中角点的数量  
        cv::Mat distCoeffs=Mat(1,5,CV_32FC1,Scalar::all(0)); /* 摄像机的5个畸变系数：k1,k2,p1,p2,k3 */

        for (int32_t i = 0 ; i != reader->GetImageNums() ; i++)
        {
            if(camera_model == PIN)
            {
                initUndistortRectifyMap(cameraMatrix, distCoeffs, R, cameraMatrix, image_size, CV_32FC1, mapx, mapy);
            }
            else if(camera_model == FISHEYE)
            {
                fisheye::initUndistortRectifyMap(K,D,R,getOptimalNewCameraMatrix(K, D, image_size, 1, image_size, 0),image_size,CV_32FC1,mapx,mapy);
            }
                
            std::cout << "initUndistort" << std::endl;
            Mat newimage = image[i];
            remap(image[i], newimage, mapx, mapy, INTER_LINEAR);
            
            StrStm.clear();
            imageFileName.clear();
            StrStm << path << "/undistort/";
            StrStm << i+1;
            StrStm >> imageFileName;
            imageFileName += "_d.jpg";
            imshow("newimage", newimage);
            cv::imwrite(imageFileName,newimage);
            cvWaitKey(0);
        }

    }
    else
    {
        
        int32_t width, height; 

        std::vector<std::vector<cv::Point2f>> image_points_seq; /* 保存检测到的所有角点 */ 
        
        for(int32_t i = 0; i < reader->GetImageNums(); i++)
        {
            
            img = reader->GetImageInternal(i);

            image.push_back(img);
            if(detect_chess_board(image[i], board_size.height, board_size.width,corners) == 0)
                image_points_seq.push_back(corners);
                
            std::cout << "No " << i << " image" << ", corners's num " << corners.size() << std::endl;
            corners.clear();
        }
        
        int32_t image_count = image_points_seq.size();
        if(image_count == 0)
        {
            std::cout << "There is no avaliable image.";
            return -1;
        }
        Size image_size;
        image_size.width = image[0].cols;
        image_size.height = image[0].rows;
            //以下是摄像机标定  
        cout<<"Beigin calibration: "<<std::endl;  
        /*棋盘三维信息*/  
        Size square_size = Size(100,100);  /* 实际测量得到的标定板上每个棋盘格的大小 */  
        vector<vector<Point3f>> object_points; /* 保存标定板上角点的三维坐标 */  
        /*内外参数*/  
        Mat cameraMatrix=Mat(3,3,CV_32FC1,Scalar::all(0)); /* 摄像机内参数矩阵 */  
        vector<int32_t> point_counts;  // 每幅图像中角点的数量  
        Mat distCoeffs=Mat(1,5,CV_32FC1,Scalar::all(0)); /* 摄像机的5个畸变系数：k1,k2,p1,p2,k3 */  
        vector<Mat> tvecsMat;  /* 每幅图像的旋转向量 */  
        vector<Mat> rvecsMat; /* 每幅图像的平移向量 */  
        cv::Matx33d K;
        cv::Vec4d D;
        /* 初始化标定板上角点的三维坐标 */  
        int32_t i,j,t;  
        for (t=0;t<image_count;t++)   
        {  
            vector<Point3f> tempPointSet;  
            for (i=0;i<board_size.height;i++)   
            {  
                for (j=0;j<board_size.width;j++)   
                {  
                    Point3f realPoint;  
                    /* 假设标定板放在世界坐标系中z=0的平面上 */  
                    realPoint.x = i*square_size.width;  
                    realPoint.y = j*square_size.height;  
                    realPoint.z = 0;  
                    tempPointSet.push_back(realPoint);  
                }  
            }  
            object_points.push_back(tempPointSet);  
        }  
        /* 初始化每幅图像中的角点数量，假定每幅图像中都可以看到完整的标定板 */  
        for (i=0;i<image_points_seq.size();i++)  
        {  
            point_counts.push_back(board_size.width*board_size.height);  
        }     
        /* 开始标定 */ 
        if(camera_model == PIN)
        {
            calibrateCamera(object_points,image_points_seq,image_size,cameraMatrix,distCoeffs,rvecsMat,tvecsMat,0);
            std::cout << "K : " << std::endl;
            std::cout << cameraMatrix << std::endl;
            std::cout << "D :" << std::endl;
            std::cout << distCoeffs << std::endl;  
        } 
        else if(camera_model == FISHEYE)
        {
            int32_t flag = 0;
            flag |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
            flag |= cv::fisheye::CALIB_CHECK_COND;
            flag |= cv::fisheye::CALIB_FIX_SKEW;/*非常重要*/

            double_t rms = fisheye::calibrate(object_points, 
                image_points_seq, 
                image_size,
                K, D, 
                rvecsMat, tvecsMat, 
                flag, 
                cv::TermCriteria(3, 20, 1e-6));

            std::cout << "K : " << std::endl;
            std::cout << K << std::endl;
            std::cout << "D :" << std::endl;
            std::cout << D << std::endl;
        }

        cout<<"Calibration complete\n";  
        //对标定结果进行评价  
        cout<<"Begin evalution calibration results\n";  
        double_t total_err = 0.0; /* 所有图像的平均误差的总和 */  
        double_t err = 0.0; /* 每幅图像的平均误差 */  
        vector<Point2f> image_points2; /* 保存重新计算得到的投影点 */  
        cout<<"\tEvery image calibration error：\n";  
        for (i=0;i<image_count;i++)  
        {  
            vector<Point3f> tempPointSet=object_points[i];  
            /* 通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点 */ 
            if(camera_model == PIN)
            {
                projectPoints(tempPointSet,rvecsMat[i],tvecsMat[i],cameraMatrix,distCoeffs,image_points2);
            } 
            else if(camera_model ==FISHEYE)
            {
                fisheye::projectPoints(tempPointSet,image_points2,rvecsMat[i],tvecsMat[i],K,D);
            }  
            
            /* 计算新的投影点和旧的投影点之间的误差*/  
            vector<Point2f> tempImagePoint = image_points_seq[i];  
            Mat tempImagePointMat = Mat(1,tempImagePoint.size(),CV_32FC2);  
            Mat image_points2Mat = Mat(1,image_points2.size(), CV_32FC2);  
            for (int32_t j = 0 ; j < tempImagePoint.size(); j++)  
            {  
                image_points2Mat.at<Vec2f>(0,j) = Vec2f(image_points2[j].x, image_points2[j].y);  
                tempImagePointMat.at<Vec2f>(0,j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);  
            }  
            err = norm(image_points2Mat, tempImagePointMat, NORM_L2);  
            total_err += err/=  point_counts[i];     
            std::cout<<"No "<<i+1<<" image's error : "<<err<<"pixel"<<endl;        
        }     
        std::cout<<"Overall error : "<<total_err/image_count<<"pixel"<<endl;          
        std::cout<<"Evalution complete"<<endl;    
        //保存定标结果      
        std::cout<<"Beigin save calirbation results."<<endl;         
        Mat rotation_matrix = Mat(3,3,CV_32FC1, Scalar::all(0)); /* 保存每幅图像的旋转矩阵 */         

        std::string pathTemp_calib;
        pathTemp_calib = reader->getPath();
        pathTemp_calib += "/calibration.txt";
        if(camera_model == PIN)
        {
            // FILE* fp = fopen( pathTemp_calib.c_str(), "w" );
            // fprintf( fp, "The camera mode is Pin %d \n", camera_model);
            // fprintf( fp, "The average of re-projection error : %lf\n", total_err/image_count );
            // fprintf( fp, "Camera internal matrix :\n" );
            // fprintf( fp, "%lf %lf %lf\n%lf %lf %lf\n%lf %lf %lf\n", 
            // cameraMatrix(0,0), cameraMatrix(0,1), cameraMatrix(0,2),
            // cameraMatrix(1,0), cameraMatrix(1,1), cameraMatrix(1,2),
            // cameraMatrix(2,0), cameraMatrix(2,1), cameraMatrix(2,2));
            // fprintf( fp,"Distortion coefficient :\n" );
            // for ( int32_t k=0; k<distCoeffs.size(); k++)
            //     fprintf( fp, "%lf ", distCoeffs[k] );
        }
        else if(camera_model == FISHEYE)
        {
            FILE* fp = fopen( pathTemp_calib.c_str(), "w" );
            fprintf( fp, "The camera mode is Fisheye %d \n", camera_model);
            fprintf( fp, "The average of re-projection error : %lf\n", total_err/image_count );
            fprintf( fp, "Camera internal matrix :\n" );
            fprintf( fp, "%lf %lf %lf\n%lf %lf %lf\n%lf %lf %lf\n", 
            K(0,0), K(0,1), K(0,2),
            K(1,0), K(1,1), K(1,2),
            K(2,0), K(2,1), K(2,2));
            fprintf( fp,"Distortion coefficient :\n" );
            for ( int32_t k=0; k<4; k++)
                fprintf( fp, "%lf ", D[k] );
            fclose(fp);
        }

        Mat mapx = Mat(image_size, CV_32FC1);
        Mat mapy = Mat(image_size, CV_32FC1);
        Mat R = Mat::eye(3, 3, CV_32F);
        string imageFileName;
        std::stringstream StrStm;
        
        for (int32_t i = 0 ; i != reader->GetImageNums() ; i++)
        {
            if(camera_model == PIN)
            {
                initUndistortRectifyMap(cameraMatrix, distCoeffs, R, cameraMatrix, image_size, CV_32FC1, mapx, mapy);
            }
            else if(camera_model == FISHEYE)
            {
                fisheye::initUndistortRectifyMap(K,D,R,getOptimalNewCameraMatrix(K, D, image_size, 0, image_size, 0),image_size,CV_32FC1,mapx,mapy);
            }
                
            std::cout << "initUndistort" << std::endl;
            Mat newimage = image[i];
            remap(image[i], newimage, mapx, mapy, INTER_LINEAR);
            
            StrStm.clear();
            imageFileName.clear();
            StrStm << path << "/undistort/";
            StrStm << i+1;
            StrStm >> imageFileName;
            imageFileName += "_d.jpg";
            imshow("newimage", newimage);
            cv::imwrite(imageFileName,newimage);
            cvWaitKey(0);
        }
    }
    return 0;
}


int32_t detect_chess_board(cv::Mat& image, int32_t width, int32_t height, std::vector<cv::Point2f>& corners)
{
	cv::Size board_size = cv::Size(height, width);   //标定板每行，每列角点数

	cv::Mat imageGray;
	cv::cvtColor(image, imageGray, CV_RGB2GRAY);
    cv::imshow("grayimg", imageGray);
    cv::waitKey(0);
	bool_t patternfound = cv::findChessboardCorners(image, board_size, corners,  cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);//cv::CALIB_CB_FAST_CHECK +
	if (!patternfound)
	{
		std::cout << "can not find chessboard corners!" << std::endl;
		//exit(1);
        return -1;
	}
	else
	{
		//亚像素精确化
		cv::cornerSubPix(imageGray, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
	}
	//角点检测图像显示
	std::cout << "corners:";
	for (int32_t i = 0; i < corners.size(); i++)
	{
		cv::circle(image, corners[i], 5, cv::Scalar(255, 0, 255), 2);
		cv::putText(image,std::to_string(i+1) , corners[i], cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2, 2);
		std::cout << corners[i] << "|";
	}
	std::cout<<std::endl;
	cv::imshow("Extractcorner", image);
	cvWaitKey(0);
 
	return 0;
}