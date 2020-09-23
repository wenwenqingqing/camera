/*****************************
Copyright 2011 Rafael Muñoz Salinas. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY Rafael Muñoz Salinas ''AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Rafael Muñoz Salinas OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of Rafael Muñoz Salinas.
********************************/
#ifndef _ARUCO_MarkerDetector_H
#define _ARUCO_MarkerDetector_H
#include <opencv2/core/core.hpp>
#include <cstdio>
#include <iostream>
#include "cameraparameters.h"
#include "exports.h"
#include "marker.h"
using namespace std;

namespace aruco {

/**\brief Main class for marker detection
 *
 */
class ARUCO_EXPORTS MarkerDetector {
    // Represent a candidate to be a maker
    class MarkerCandidate : public Marker {
      public:
        MarkerCandidate() {}
        MarkerCandidate(const Marker &M) : Marker(M) {}
        MarkerCandidate(const MarkerCandidate &M) : Marker(M) {
            contour = M.contour;
            idx = M.idx;
        }
        MarkerCandidate &operator=(const MarkerCandidate &M) {
            (*(Marker *)this) = (*(Marker *)&M);
            contour = M.contour;
            idx = M.idx;
            return *this;
        }

        vector< cv::Point > contour; // all the points of its contour
        int32_t idx; // index position in the global contour list
    };

  public:
    /**
     * See
     */
    MarkerDetector();

    /**
     */
    ~MarkerDetector();

    /**Detects the markers in the image passed
     *
     * If you provide information about the camera parameters and the size of the marker, then, the extrinsics of the markers are detected
     *
     * @param input input color image
     * @param detectedMarkers output vector with the markers detected
     * @param camMatrix intrinsic camera information.
     * @param distCoeff camera distorsion coefficient. If set Mat() if is assumed no camera distorion
     * @param markerSizeMeters size of the marker sides expressed in meters
     * @param setYPerperdicular If set the Y axis will be perpendicular to the surface. Otherwise, it will be the Z axis
     */
    void detect(const cv::Mat &input, std::vector< Marker > &detectedMarkers, cv::Mat camMatrix = cv::Mat(), cv::Mat distCoeff = cv::Mat(),
                float_t markerSizeMeters = -1, bool_t setYPerperdicular = false) throw(cv::Exception);
    /**Detects the markers in the image passed
     *
     * If you provide information about the camera parameters and the size of the marker, then, the extrinsics of the markers are detected
     *
     * @param input input color image
     * @param detectedMarkers output vector with the markers detected
     * @param camParams Camera parameters
     * @param markerSizeMeters size of the marker sides expressed in meters
     * @param setYPerperdicular If set the Y axis will be perpendicular to the surface. Otherwise, it will be the Z axis
     */
    void detect(const cv::Mat &input, std::vector< Marker > &detectedMarkers, CameraParameters camParams, float_t markerSizeMeters = -1,
                bool_t setYPerperdicular = false) throw(cv::Exception);

    /**This set the type of thresholding methods available
     */

    enum ThresholdMethods { FIXED_THRES, ADPT_THRES, CANNY };



    /**Sets the threshold method
     */
    void setThresholdMethod(ThresholdMethods m) { _thresMethod = m; }
    /**Returns the current threshold method
     */
    ThresholdMethods getThresholdMethod() const { return _thresMethod; }
    /**
     * Set the parameters of the threshold method
     * We are currently using the Adptive threshold ee opencv doc of adaptiveThreshold for more info
     *   @param param1: blockSize of the pixel neighborhood that is used to calculate a threshold value for the pixel
     *   @param param2: The constant subtracted from the mean or weighted mean
     */
    void setThresholdParams(double_t param1, double_t param2) {
        _thresParam1 = param1;
        _thresParam2 = param2;
    }

    /**Allows for a parallel search of several values of the param1 simultaneously (in different threads)
     * The param r1 the indicates how many values around the current value of param1 are evaluated. In other words
     * if r1>0, param1 is searched in range [param1- r1 ,param1+ r1 ]
     *
     * r2 unused yet. Added in case of future need.
     */
    void setThresholdParamRange(size_t r1 = 0, size_t r2 = 0) { _thresParam1_range = r1; }
    /**
     * This method assumes that the markers may have some of its corners joined either to another marker
     * in a chessboard like pattern) or to a rectangle. This is the case in which the subpixel refinement
     * method in opencv work best.
     *
     * Enabling this does not force you to use locked corners, normals markers will be detected also. However,
     * when using locked corners, enabling this option will increase robustness in detection at the cost of
     * higher computational time.
     * ,
     * Note for developer: Enabling this option forces a call to findCornerMaxima
     */
    void enableLockedCornersMethod(bool_t enable);

    /**
     * Set the parameters of the threshold method
     * We are currently using the Adptive threshold ee opencv doc of adaptiveThreshold for more info
     *   param1: blockSize of the pixel neighborhood that is used to calculate a threshold value for the pixel
     *   param2: The constant subtracted from the mean or weighted mean
     */
    void getThresholdParams(double_t &param1, double_t &param2) const {
        param1 = _thresParam1;
        param2 = _thresParam2;
    }


    /**Returns a reference to the internal image thresholded. It is for visualization purposes and to adjust manually
     * the parameters
     */
    const cv::Mat &getThresholdedImage() { return thres; }
    /**Methods for corner refinement
     */
    enum CornerRefinementMethod { NONE, HARRIS, SUBPIX, LINES };
    /**
     */
    void setCornerRefinementMethod(CornerRefinementMethod method) { _cornerMethod = method; }
    /**
     */
    CornerRefinementMethod getCornerRefinementMethod() const { return _cornerMethod; }
    /**Specifies the min and max sizes of the markers as a fraction of the image size. By size we mean the maximum
     * of cols and rows.
     * @param min size of the contour to consider a possible marker as valid (0,1]
     * @param max size of the contour to consider a possible marker as valid [0,1)
     *
     */
    void setMinMaxSize(float_t min = 0.03, float_t max = 0.5) throw(cv::Exception);

    /**reads the min and max sizes employed
     * @param min output size of the contour to consider a possible marker as valid (0,1]
     * @param max output size of the contour to consider a possible marker as valid [0,1)
     *
     */
    void getMinMaxSize(float_t &min, float_t &max) {
        min = _minSize;
        max = _maxSize;
    }

    /**Deprecated!!!
     *
     * Enables/Disables erosion process that is REQUIRED for chessboard like boards.
     * By default, this property is enabled
     */
    void enableErosion(bool_t enable) {}

    /**
     * Specifies a value to indicate the required speed for the internal processes. If you need maximum speed (at the cost of a lower detection rate),
     * use the value 3, If you rather a more precise and slow detection, set it to 0.
     *
     * Actually, the main differences are that in highspeed mode, we employ setCornerRefinementMethod(NONE) and internally, we use a small canonical
     * image to detect the marker. In low speed mode, we use setCornerRefinementMethod(HARRIS) and a bigger size for the canonical marker image
     */
    void setDesiredSpeed(int32_t val);
    /**
     */
    int32_t getDesiredSpeed() const { return _speed; }

    /**
     * Specifies the size for the canonical marker image. A big value makes the detection slower than a small value.
     * Minimun value is 10. Default value is 56.
     */
    void setWarpSize(int32_t val) throw(cv::Exception);
    ;
    /**
     */
    int32_t getWarpSize() const { return _markerWarpSize; }

    /**
     * Allows to specify the function that identifies a marker. Therefore, you can create your own type of markers different from these
     * employed by default in the library.
     * The marker function must have the following structure:
     *
     * int32_t myMarkerIdentifier(const cv::Mat &in,int32_t &nRotations);
     *
     * The marker function receives the image 'in' with the region that migh contain one of your markers. These are the rectangular regions with black
     *  in the image.
     *
     * As output your marker function must indicate the following information. First, the output parameter nRotations must indicate how many times the marker
     * must be rotated clockwise 90 deg  to be in its ideal position. (The way you would see it when you print it). This is employed to know
     * always which is the corner that acts as reference system. Second, the function must return -1 if the image does not contains one of your markers, and its
     *id otherwise.
     *
     */
    void setMakerDetectorFunction(int32_t (*markerdetector_func)(const cv::Mat &in, int32_t &nRotations)) { markerIdDetector_ptrfunc = markerdetector_func; }

    /**Deprecated
     *
     * Use an smaller version of the input image for marker detection.
     * If your marker is small enough, you can employ an smaller image to perform the detection without noticeable reduction in the precision.
     * Internally, we are performing a pyrdown operation
     *
     * @param level number of times the image size is divided by 2. Internally, we are performing a pyrdown.
     */
    void pyrDown(int32_t level) {}

    ///-------------------------------------------------
    /// Methods you may not need
    /// Thesde methods do the hard work. They have been set public in case you want to do customizations
    ///-------------------------------------------------

    /**
     * Thesholds the passed image with the specified method.
     */
    void thresHold(int32_t method, const cv::Mat &grey, cv::Mat &thresImg, double_t param1 = -1, double_t param2 = -1) throw(cv::Exception);
    /**
    * Detection of candidates to be markers, i.e., rectangles.
    * This function returns in candidates all the rectangles found in a thresolded image
    */
    void detectRectangles(const cv::Mat &thresImg, vector< std::vector< cv::Point2f > > &candidates);

    /**Returns a list candidates to be markers (rectangles), for which no valid id was found after calling detectRectangles
     */
    const vector< std::vector< cv::Point2f > > &getCandidates() { return _candidates; }

    /**Given the iput image with markers, creates an output image with it in the canonical position
     * @param in input image
     * @param out image with the marker
     * @param size of out
     * @param points 4 corners of the marker in the image in
     * @return true if the operation succeed
     */
    bool_t warp(cv::Mat &in, cv::Mat &out, cv::Size size, std::vector< cv::Point2f > points) throw(cv::Exception);



    /** Refine MarkerCandidate Corner using LINES method
     * @param candidate candidate to refine corners
     */
    void refineCandidateLines(MarkerCandidate &candidate, const cv::Mat &camMatrix, const cv::Mat &distCoeff);


    /**DEPRECATED!!! Use the member function in CameraParameters
     *
     * Given the intrinsic camera parameters returns the GL_PROJECTION matrix for opengl.
     * PLease NOTE that when using OpenGL, it is assumed no camera distorsion! So, if it is not true, you should have
     * undistor image
     *
     * @param CamMatrix  arameters of the camera specified.
     * @param orgImgSize size of the original image
     * @param size of the image/window where to render (can be different from the real camera image). Please not that it must be related to CamMatrix
     * @param proj_matrix output projection matrix to give to opengl
     * @param gnear,gfar: visible rendering range
     * @param invert: indicates if the output projection matrix has to yield a horizontally inverted image because image data has not been stored in the order
     *of glDrawPixels: bottom-to-top.
     */
    static void glGetProjectionMatrix(CameraParameters &CamMatrix, cv::Size orgImgSize, cv::Size size, double_t proj_matrix[16], double_t gnear, double_t gfar,
                                      bool_t invert = false) throw(cv::Exception);

  private:
    bool_t warp_cylinder(cv::Mat &in, cv::Mat &out, cv::Size size, MarkerCandidate &mc) throw(cv::Exception);
    /**
    * Detection of candidates to be markers, i.e., rectangles.
    * This function returns in candidates all the rectangles found in a thresolded image
    */
    void detectRectangles(vector< cv::Mat > &vimages, vector< MarkerCandidate > &candidates);
    // Current threshold method
    ThresholdMethods _thresMethod;
    // Threshold parameters
    double_t _thresParam1, _thresParam2, _thresParam1_range;
    // Current corner method
    CornerRefinementMethod _cornerMethod;
    // minimum and maximum size of a contour lenght
    float_t _minSize, _maxSize;

    // is corner locked
    bool_t _useLockedCorners;

    // Speed control
    int32_t _speed;
    int32_t _markerWarpSize;
    float_t _borderDistThres; // border around image limits in which corners are not allowed to be detected.
    // vectr of candidates to be markers. This is a vector with a set of rectangles that have no valid id
    vector< std::vector< cv::Point2f > > _candidates;
    // Images
    cv::Mat grey, thres;
    // pointer to the function that analizes a rectangular region so as to detect its internal marker
    int32_t (*markerIdDetector_ptrfunc)(const cv::Mat &in, int32_t &nRotations);

    /**
     */
    bool_t isInto(cv::Mat &contour, std::vector< cv::Point2f > &b);
    /**
     */
    int32_t perimeter(std::vector< cv::Point2f > &a);


    //     //GL routines
    //
    //     static void argConvGLcpara2( double_t cparam[3][4], int32_t width, int32_t height, double_t gnear, double_t gfar, double_t m[16], bool_t invert )throw(cv::Exception);
    //     static int32_t  arParamDecompMat( double_t source[3][4], double_t cpara[3][4], double_t trans[3][4] )throw(cv::Exception);
    //     static double_t norm( double_t a, double_t b, double_t c );
    //     static double_t dot(  double_t a1, double_t a2, double_t a3,
    //                         double_t b1, double_t b2, double_t b3 );
    //

    // detection of the
    void findBestCornerInRegion_harris(const cv::Mat &grey, vector< cv::Point2f > &Corners, int32_t blockSize);


    // auxiliar functions to perform LINES refinement
    void interpolate2Dline(const vector< cv::Point2f > &inPoints, cv::Point3f &outLine);
    cv::Point2f getCrossPoint(const cv::Point3f &line1, const cv::Point3f &line2);
    void distortPoints(vector< cv::Point2f > in, vector< cv::Point2f > &out, const cv::Mat &camMatrix, const cv::Mat &distCoeff);


    /**Given a vector vinout with elements and a boolean vector indicating the lements from it to remove,
     * this function remove the elements
     * @param vinout
     * @param toRemove
     */
    template < typename T > void removeElements(vector< T > &vinout, const vector< bool_t > &toRemove) {
        // remove the invalid ones by setting the valid in the positions left by the invalids
        size_t indexValid = 0;
        for (size_t i = 0; i < toRemove.size(); i++) {
            if (!toRemove[i]) {
                if (indexValid != i)
                    vinout[indexValid] = vinout[i];
                indexValid++;
            }
        }
        vinout.resize(indexValid);
    }

    // graphical debug
    void drawApproxCurve(cv::Mat &in, std::vector< cv::Point > &approxCurve, cv::Scalar color);
    void drawContour(cv::Mat &in, std::vector< cv::Point > &contour, cv::Scalar);
    void drawAllContours(cv::Mat input, std::vector< std::vector< cv::Point > > &contours);
    void draw(cv::Mat out, const std::vector< Marker > &markers);
    // method to refine corner detection in case the internal border after threshold is found
    // This was tested in the context of chessboard methods
    void findCornerMaxima(vector< cv::Point2f > &Corners, const cv::Mat &grey, int32_t wsize);



    template < typename T > void joinVectors(vector< vector< T > > &vv, vector< T > &v, bool_t clearv = false) {
        if (clearv)
            v.clear();
        for (size_t i = 0; i < vv.size(); i++)
            for (size_t j = 0; j < vv[i].size(); j++)
                v.push_back(vv[i][j]);
    }
};
};
#endif
