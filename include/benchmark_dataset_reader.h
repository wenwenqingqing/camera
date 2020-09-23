/* Copyright (c) 2016, Jakob Engel
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, 
 * this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice, 
 * this list of conditions and the following disclaimer in the documentation and/or 
 * other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its contributors 
 * may be used to endorse or promote products derived from this software without 
 * specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 * POSSIBILITY OF SUCH DAMAGE.
 */


#pragma once
#include <sstream>
#include <fstream>
#include <dirent.h>
#include <algorithm>

#include "opencv2/opencv.hpp"
#include "fov_undistorter.h"
#include "photometric_undistorter.h"
#include "types.h"
#include "zip.h"


inline int32_t getdir (std::string dir, std::vector<std::string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL)
    {
        return -1;
    }

    while ((dirp = readdir(dp)) != NULL) {
    	std::string name = std::string(dirp->d_name);

    	if(name != "." && name != "..")
    		files.push_back(name);
    }
    closedir(dp);


    std::sort(files.begin(), files.end());

    if(dir.at( dir.length() - 1 ) != '/') dir = dir+"/";
	for(int32_t i=0;i<files.size();i++)
	{
		if(files[i].at(0) != '/')
			files[i] = dir + files[i];
	}

    return files.size();
}



/*
 * provides read functionality for one of the dataset sequences.
 * * path is the folder for the sequence, with trailing slash (e.g. /home/Peter/datasets/sequenceX/
 * 		=> if it contains a filder "images", images will be read from there.
 * 		=> otherwise we assume the images will be zipped in "images.zip".
 * * MT = true: multi-threaded image loading, to allow for real-time playback (reading, decoding & rectifying images takes a while).
 */
class DatasetReader
{
public:
	DatasetReader(std::string folder)
	{
		this->path = folder;
		for(int32_t i=0;i<3;i++)
		{
			ziparchive=0;
			undistorter=0;
			databuffer=0;
		}

		getdir (path+"images/", files);
		if(files.size() > 0)
		{
			printf("Load Dataset %s: found %d files in folder /images; assuming that all images are there.\n",
					path.c_str(), (int32_t)files.size());
			isZipped=false;
		}
		else
		{
			printf("Load Dataset %s: found no in folder /images; assuming that images are zipped.\n",
					path.c_str());
			isZipped=true;

			int32_t ziperror=0;
			ziparchive = zip_open((path+"images.zip").c_str(),  ZIP_RDONLY, &ziperror);
			if(ziperror!=0)
			{
				printf("ERROR %d reading archive %s!\n", ziperror, (path+"images.zip").c_str());
				exit(1);
			}

			files.clear();
			int32_t numEntries = zip_get_num_entries(ziparchive, 0);
			for(int32_t k=0;k<numEntries;k++)
			{
				const char* name = zip_get_name(ziparchive, k,  ZIP_FL_ENC_STRICT);
				std::string nstr = std::string(name);
				if(nstr == "." || nstr == "..") continue;
				files.push_back(name);
			}

			printf("got %d entries and %d files from zipfile!\n", numEntries, (int32_t)files.size());
			std::sort(files.begin(), files.end());

		}
		loadTimestamps(path+"times.txt");


		// create undistorter.
		undistorter = new UndistorterFOV((path+"camera.txt").c_str());
		photoUndistorter = new PhotometricUndistorter(path+"pcalib.txt", path+"vignette.png",undistorter->getInputDims()[0],undistorter->getInputDims()[1]);


		// get image widths.
		widthOrg = undistorter->getInputDims()[0];
		heightOrg = undistorter->getInputDims()[1];
		width= undistorter->getOutputDims()[0];
		height= undistorter->getOutputDims()[1];

		internalTempBuffer=new float_t[widthOrg*heightOrg];

		printf("Dataset %s: Got %d files!\n", path.c_str(), (int32_t)GetImageNums());
	}
	~DatasetReader()
	{
		if(undistorter!=0) delete undistorter;
		if(photoUndistorter!=0) delete photoUndistorter;
		if(ziparchive!=0) zip_close(ziparchive);
		if(databuffer!=0) delete[] databuffer;
		delete[] internalTempBuffer;

	}

	UndistorterFOV* getUndistorter()
	{
		return undistorter;
	}

	PhotometricUndistorter* getPhotoUndistorter()
	{
		return photoUndistorter;
	}

	int32_t GetImageNums()
	{
		return files.size();
	}

	double_t getTimestamp(int32_t id)
	{
		if(id >= (int32_t)timestamps.size()) return 0;
		if(id < 0) return 0;
		return timestamps[id];
	}

	float_t GetExposure(int32_t id)
	{
		if(id >= (int32_t)exposures.size()) return 0;
		if(id < 0) return 0;
		return exposures[id];
	}

	ExposureImage* getImage(int32_t id, bool_t rectify, bool_t removeGamma, bool_t removeVignette, bool_t nanOverexposed)
	{
		assert(id >= 0 && id < (int32_t)files.size());

		cv::Mat imageRaw = GetImageInternal(id);

		if(imageRaw.rows != heightOrg || imageRaw.cols != widthOrg)
		{
			printf("ERROR: expected cv-mat to have dimensions %d x %d; found %d x %d (image %s)!\n",
					widthOrg, heightOrg, imageRaw.cols, imageRaw.rows, files[id].c_str());
			return 0;
		}

		if(imageRaw.type() != CV_8U)
		{
			printf("ERROR: expected cv-mat to have type 8U!\n");
			return 0;
		}

		ExposureImage* ret=0;


		if(removeGamma || removeVignette || nanOverexposed)
		{
			if(!rectify)
			{
				// photo undist only.
				ret = new ExposureImage(widthOrg, heightOrg, timestamps[id], exposures[id], id);
				photoUndistorter->unMapImage(imageRaw.data, ret->image, widthOrg*heightOrg, removeGamma, removeVignette, nanOverexposed );
			}
			else
			{
				// photo undist to buffer, then rect
				ret = new ExposureImage(width, height, timestamps[id], exposures[id], id);
				photoUndistorter->unMapImage(imageRaw.data, internalTempBuffer, widthOrg*heightOrg, removeGamma, removeVignette, nanOverexposed );
				undistorter->undistort<float_t>(internalTempBuffer, ret->image, widthOrg*heightOrg, width*height);
			}
		}
		else
		{
			if(rectify)
			{
				// rect only.
				ret = new ExposureImage(width, height, timestamps[id], exposures[id], id);
				undistorter->undistort<unsigned char>(imageRaw.data, ret->image, widthOrg*heightOrg, width*height);
			}
			else
			{
				// do nothing.
				ret = new ExposureImage(widthOrg, heightOrg, timestamps[id], exposures[id], id);
				for(int32_t i=0;i<widthOrg*heightOrg;i++)
					ret->image[i] = imageRaw.at<uchar>(i);
			}
		}
		return ret;
	}



	cv::Mat GetImageInternal(int32_t id)
	{
		if(!isZipped)
		{
			// CHANGE FOR ZIP FILE
			return cv::imread(files[id],CV_LOAD_IMAGE_GRAYSCALE);
		}
		else
		{
			if(databuffer==0) databuffer = new char[widthOrg*heightOrg*6+10000];
			zip_file_t* fle = zip_fopen(ziparchive, files[id].c_str(), 0);
			long readbytes = zip_fread(fle, databuffer, (long)widthOrg*heightOrg*6+10000);

			if(readbytes > (long)widthOrg*heightOrg*6)
			{
				printf("read %ld/%ld bytes for file %s. increase buffer!!\n", readbytes,(long)widthOrg*heightOrg*6+10000, files[id].c_str());
				delete[] databuffer;
				databuffer = new char[(long)widthOrg*heightOrg*60+1000000];
				fle = zip_fopen(ziparchive, files[id].c_str(), 0);
				readbytes = zip_fread(fle, databuffer, (long)widthOrg*heightOrg*60+100000);

				if(readbytes > (long)widthOrg*heightOrg*60+10000)
				{
					printf("buffer still to small (read %ld/%ld). abort.\n", readbytes,(long)widthOrg*heightOrg*60+100000);
					exit(1);
				}
			}
			return cv::imdecode(cv::Mat(readbytes,1,CV_8U, databuffer), CV_LOAD_IMAGE_GRAYSCALE);
		}
	}


private:


	inline void loadTimestamps(std::string timesFile)
	{
		std::ifstream tr;
		tr.open(timesFile.c_str());
		timestamps.clear();
		exposures.clear();
		while(!tr.eof() && tr.good())
		{
			std::string line;
			char buf[1000];
			tr.getline(buf, 1000);

			int32_t id;
			double_t stamp;
			float_t exposure = 0;

			if(3 == sscanf(buf, "%d %lf %f", &id, &stamp, &exposure))
			{
				timestamps.push_back(stamp);
				exposures.push_back(exposure);
			}

			else if(2 == sscanf(buf, "%d %lf", &id, &stamp))
			{
				timestamps.push_back(stamp);
				exposures.push_back(0);
			}
		}
		tr.close();

		if((int32_t)exposures.size()!=(int32_t)GetImageNums())
		{
			printf("DatasetReader: Mismatch between number of images and number of timestamps / exposure times. Set all to zero.");
			timestamps.clear();
			exposures.clear();
			for(int32_t i=0;i<(int32_t)GetImageNums();i++)
			{
				timestamps.push_back(0.0);
				exposures.push_back(0);
			}
		}
	}

	// data is here.
	std::vector<std::string> files;
	std::vector<double_t> timestamps;
	std::vector<float_t> exposures;

	int32_t width, height;
	int32_t widthOrg, heightOrg;

	std::string path;
	bool_t isZipped;



	// internal structures.
	UndistorterFOV* undistorter;
	PhotometricUndistorter* photoUndistorter;
	zip_t* ziparchive;
	char* databuffer;

	float_t* internalTempBuffer;
};

class RGBImgReader
{
public:
	RGBImgReader(std::string folder)
	{
		this->path = folder;
		getdir (path+"imagesrgb/", files);
		if(files.size() > 0)
		{
			printf("Load Dataset %s: found %d files in folder /images; assuming that all images are there.\n",
					path.c_str(), (int32_t)files.size());
		}
		getdir (path+"undistort/", undistortion);
		if(files.size() > 0)
		{
			printf("Load Undistortion Dataset %s: found %d files in folder /undistort; assuming that all images are there.\n",
					path.c_str(), (int32_t)files.size());
		}
		loadTimestamps(path+"times.txt");
		printf("Dataset %s: Got %d files!\n", path.c_str(), (int32_t)GetImageNums());
	}
	~RGBImgReader()
	{

	}

	int32_t GetImageNums()
	{
		return files.size();
	}

	int32_t getNumUndistortImages()
	{
		return undistortion.size();
	}

	float_t GetExposure(int32_t id)
	{
		if(id >= (int32_t)exposures.size()) return 0;
		if(id < 0) return 0;
		return exposures[id];
	}

	cv::Mat GetImageInternal(int32_t id)
	{
		return cv::imread(files[id]);
	}

	cv::Mat getUndistortImgI(int32_t id)
	{
		return cv::imread(undistortion[id]);
	}

	cv::Mat output_Gray(int32_t id)
	{
		cv::Mat RGB;
		cv::Mat GRAY;

		RGB = GetImageInternal(id);
		cv::cvtColor(RGB, GRAY, cv::COLOR_RGB2GRAY);

		return GRAY;
	}

	std::string getPath()
	{
		return path;
	}
private:
	inline void loadTimestamps(std::string timesFile)
	{
		std::ifstream tr;
		tr.open(timesFile.c_str());
		timestamps.clear();
		exposures.clear();
		while(!tr.eof() && tr.good())
		{
			std::string line;
			char buf[1000];
			tr.getline(buf, 1000);

			int32_t id;
			double_t stamp;
			float_t exposure = 0;

			if(3 == sscanf(buf, "%d %lf %f", &id, &stamp, &exposure))
			{
				timestamps.push_back(stamp);
				exposures.push_back(exposure);
			}

			else if(2 == sscanf(buf, "%d %lf", &id, &stamp))
			{
				timestamps.push_back(stamp);
				exposures.push_back(0);
			}
		}
		tr.close();

		if((int32_t)exposures.size()!=(int32_t)GetImageNums())
		{
			printf("DatasetReader: Mismatch between number of images and number of timestamps / exposure times. Set all to zero.");
			timestamps.clear();
			exposures.clear();
			for(int32_t i=0;i<(int32_t)GetImageNums();i++)
			{
				timestamps.push_back(0.0);
				exposures.push_back(0);
			}
		}
	}

	// data is here.
	std::vector<std::string> files;
	std::vector<std::string> undistortion;
	std::vector<double_t> timestamps;
	std::vector<float_t> exposures;
	std::string path;
};