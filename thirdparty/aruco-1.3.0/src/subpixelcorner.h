#ifndef aruco_SUBPIXELCORNER_HPP
#define aruco_SUBPIXELCORNER_HPP

#include <opencv2/core/core.hpp> // Basic OpenCV structures (cv::Mat)

namespace aruco {

class SubPixelCorner {
  private:
    int32_t _winSize;
    int32_t _apertureSize;
    cv::TermCriteria _term;
    double_t eps;
    cv::Mat mask;
    int32_t _max_iters;

  public:
    bool_t enable;
    SubPixelCorner();

    void checkTerm();

    double_t pointDist(cv::Point2f estimate_corner, cv::Point2f curr_corner);

    /// method to refine the corners
    void RefineCorner(cv::Mat image, std::vector< cv::Point2f > &corners);

    // function to generate the mask
    void generateMask();
};
}

#endif // SUBPIXELCORNER_HPP
