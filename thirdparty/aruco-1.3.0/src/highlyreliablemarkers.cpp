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

#include "highlyreliablemarkers.h"
#include <iostream>
using namespace std;

namespace aruco {

// static variables from HighlyReliableMarkers. Need to be here to avoid linking errors
Dictionary HighlyReliableMarkers::_D;
HighlyReliableMarkers::BalancedBinaryTree HighlyReliableMarkers::_binaryTree;
int32_t HighlyReliableMarkers::_n, HighlyReliableMarkers::_ncellsBorder, HighlyReliableMarkers::_correctionDistance;
int32_t HighlyReliableMarkers::_swidth;


/**
*/
MarkerCode::MarkerCode(int32_t n) {
    // resize bits vectors and initialize to 0
    for (int32_t i = 0; i < 4; i++) {
        _bits[i].resize(n * n);
        for (int32_t j = 0; j < _bits[i].size(); j++)
            _bits[i][j] = 0;
        _ids[i] = 0; // ids are also 0
    }
    _n = n;
};


/**
 */
MarkerCode::MarkerCode(const MarkerCode &MC) {
    for (int32_t i = 0; i < 4; i++) {
        _bits[i] = MC._bits[i];
        _ids[i] = MC._ids[i];
    }
    _n = MC._n;
}



/**
 */
void MarkerCode::set(int32_t pos, bool_t val, bool_t updateIds) {
    // if not the same value
    if (get(pos) != val) {
        for (int32_t i = 0; i < 4; i++) {         // calculate bit coordinates for each rotation
            int32_t y = pos / n(), x = pos % n(); // if rotation 0, dont do anything
                                                       // else calculate bit position in that rotation
            if (i == 1) {
                int32_t aux = y;
                y = x;
                x = n() - aux - 1;
            } else if (i == 2) {
                y = n() - y - 1;
                x = n() - x - 1;
            } else if (i == 3) {
                int32_t aux = y;
                y = n() - x - 1;
                x = aux;
            }
            int32_t rotPos = y * n() + x; // calculate position in the unidimensional string
            _bits[i][rotPos] = val;            // modify value
                                               // update identifier in that rotation
            if(updateIds) {
                if (val == true)
                    _ids[i] += (int32_t)pow(float_t(2), float_t(rotPos)); // if 1, add 2^pos
                else
                    _ids[i] -= (int32_t)pow(float_t(2), float_t(rotPos)); // if 0, substract 2^pos
            }
        }
    }
}


/**
 */
int32_t MarkerCode::selfDistance(int32_t &minRot) const {
    int32_t res = _bits[0].size();    // init to n*n (max value)
    for (int32_t i = 1; i < 4; i++) { // self distance is not calculated for rotation 0
        int32_t hammdist = hammingDistance(_bits[0], _bits[i]);
        if (hammdist < res) {
            minRot = i;
            res = hammdist;
        }
    }
    return res;
}


/**
 */
int32_t MarkerCode::distance(const MarkerCode &m, int32_t &minRot) const {
    int32_t res = _bits[0].size(); // init to n*n (max value)
    for (int32_t i = 0; i < 4; i++) {
        int32_t hammdist = hammingDistance(_bits[0], m.getRotation(i));
        if (hammdist < res) {
            minRot = i;
            res = hammdist;
        }
    }
    return res;
};


/**
 */
void MarkerCode::fromString(std::string s) {
    for (int32_t i = 0; i < s.length(); i++) {
        if (s[i] == '0')
            set(i, false);
        else
            set(i, true);
    }
}

/**
 */
std::string MarkerCode::toString() const {
    std::string s;
    s.resize(size());
    for (int32_t i = 0; i < size(); i++) {
        if (get(i))
            s[i] = '1';
        else
            s[i] = '0';
    }
    return s;
}


/**
 */
cv::Mat MarkerCode::getImg(int32_t pixSize) const {
    const int32_t borderSize = 1;
    int32_t nrows = n() + 2 * borderSize;
    if (pixSize % nrows != 0)
        pixSize = pixSize + nrows - pixSize % nrows;
    int32_t cellSize = pixSize / nrows;
    cv::Mat img(pixSize, pixSize, CV_8U, cv::Scalar::all(0)); // create black image (init image to 0s)
    // double_t for to go over all the cells
    for (int32_t i = 0; i < n(); i++) {
        for (int32_t j = 0; j < n(); j++) {
            if (_bits[0][i * n() + j] != 0) { // just draw if it is 1, since the image has been init to 0
                                              // double_t for to go over all the pixels in the cell
                for (int32_t k = 0; k < cellSize; k++) {
                    for (int32_t l = 0; l < cellSize; l++) {
                        img.at< uchar >((i + borderSize) * cellSize + k, (j + borderSize) * cellSize + l) = 255;
                    }
                }
            }
        }
    }
    return img;
}


/**
 */
int32_t MarkerCode::hammingDistance(const std::vector< bool_t > &m1, const std::vector< bool_t > &m2) const {
    int32_t res = 0;
    for (int32_t i = 0; i < m1.size(); i++)
        if (m1[i] != m2[i])
            res++;
    return res;
}




/**
*/
bool_t Dictionary::fromFile(std::string filename) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    int32_t nmarkers, markersize;

    // read number of markers
    fs["nmarkers"] >> nmarkers;                     // cardinal of D
    fs["markersize"] >> markersize;                 // n
    fs["tau0"] >> tau0;

    // read each marker info
    for (int32_t i = 0; i < nmarkers; i++) {
        std::string s;
        fs["marker_" + toStr(i)] >> s;
        MarkerCode m(markersize);
        m.fromString(s);
        push_back(m);
    }
    fs.release();
    return true;
};

/**
 */
bool_t Dictionary::toFile(std::string filename) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    // save number of markers
    fs << "nmarkers" << (int32_t)size();                               // cardinal of D
    fs << "markersize" << (int32_t)((*this)[0].n());                   // n
    fs << "tau0" << (int32_t)(this->tau0); // n
    // save each marker code
    for (int32_t i = 0; i < size(); i++) {
        std::string s = ((*this)[i]).toString();
        fs << "marker_" + toStr(i) << s;
    }
    fs.release();
    return true;
};

/**
 */
int32_t Dictionary::distance(const MarkerCode &m, int32_t &minMarker, int32_t &minRot) {
    int32_t res = m.size();
    for (int32_t i = 0; i < size(); i++) {
        int32_t minRotAux;
        int32_t distance = (*this)[i].distance(m, minRotAux);
        if (distance < res) {
            minMarker = i;
            minRot = minRotAux;
            res = distance;
        }
    }
    return res;
}


/**
 */
int32_t Dictionary::minimunDistance() {
    if (size() == 0)
        return 0;
    int32_t minDist = (*this)[0].size();
    // for each marker in D
    for (int32_t i = 0; i < size(); i++) {
        // calculate self distance of the marker
        minDist = std::min(minDist, (*this)[i].selfDistance());

        // calculate distance to all the following markers
        for (int32_t j = i + 1; j < size(); j++) {
            minDist = std::min(minDist, (*this)[i].distance((*this)[j]));
        }
    }
    return minDist;
}




/**
 */
bool_t HighlyReliableMarkers::loadDictionary(Dictionary D, float_t correctionDistanceRate) {
    if (D.size() == 0)
        return false;
    _D = D;
    _n = _D[0].n();
    _ncellsBorder = (_D[0].n() + 2);
    _correctionDistance = correctionDistanceRate * ((D.tau0-1)/2);
    cerr << "aruco :: _correctionDistance = " << _correctionDistance << endl;
    _binaryTree.loadDictionary(&D);
    return true;
}

bool_t HighlyReliableMarkers::loadDictionary(std::string filename, float_t correctionDistance) {
    Dictionary D;
    D.fromFile(filename);
    return loadDictionary(D, correctionDistance);
}


/**
 */
int32_t HighlyReliableMarkers::detect(const cv::Mat &in, int32_t &nRotations) {

    assert(in.rows == in.cols);
    cv::Mat grey;
    if (in.type() == CV_8UC1)
        grey = in;
    else
        cv::cvtColor(in, grey, CV_BGR2GRAY);
    // threshold image
    cv::threshold(grey, grey, 125, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    _swidth = grey.rows / _ncellsBorder;

    // check borders, even not necesary for the highly reliable markers
    // if(!checkBorders(grey)) return -1;

    // obtain inner code
    MarkerCode candidate = getMarkerCode(grey);

    // search each marker id in the balanced binary tree
    int32_t orgPos;
    for (int32_t i = 0; i < 4; i++) {
        if (_binaryTree.findId(candidate.getId(i), orgPos)) {
            nRotations = i;
            // return candidate.getId(i);
            return orgPos;
        }
    }

    // alternative version without using the balanced binary tree (less eficient)
    //         for(uint i=0; i<_D.size(); i++) {
    //           for(uint j=0; j<4; j++) {
    //        if(_D[i].getId() == candidate.getId(j)) {
    //          nRotations = j;
    //          //return candidate.getId(j);
    //          return i;
    //        }
    //           }
    //         }

    // correct errors
    int32_t minMarker, minRot;
    if (_D.distance(candidate, minMarker, minRot) <= _correctionDistance) {
        nRotations = minRot;
        return minMarker;
        // return _D[minMarker].getId();
    }

    return -1;
}


/**
 */
bool_t HighlyReliableMarkers::checkBorders(cv::Mat grey) {
    for (int32_t y = 0; y < _ncellsBorder; y++) {
        int32_t inc = _ncellsBorder - 1;
        if (y == 0 || y == _ncellsBorder - 1)
            inc = 1; // for first and last row, check the whole border
        for (int32_t x = 0; x < _ncellsBorder; x += inc) {
            int32_t Xstart = (x) * (_swidth);
            int32_t Ystart = (y) * (_swidth);
            cv::Mat square = grey(cv::Rect(Xstart, Ystart, _swidth, _swidth));
            int32_t nZ = cv::countNonZero(square);
            if (nZ > (_swidth * _swidth) / 2) {
                return false; // can not be a marker because the border element is not black!
            }
        }
    }
    return true;
}

/**
 */
MarkerCode HighlyReliableMarkers::getMarkerCode(const cv::Mat &grey) {
    MarkerCode candidate(_n);
    for (int32_t y = 0; y < _n; y++) {
        for (int32_t x = 0; x < _n; x++) {
            int32_t Xstart = (x + 1) * (_swidth);
            int32_t Ystart = (y + 1) * (_swidth);
            cv::Mat square = grey(cv::Rect(Xstart, Ystart, _swidth, _swidth));
            int32_t nZ = countNonZero(square);
            if (nZ > (_swidth * _swidth) / 2)
                candidate.set(y * _n + x, 1);
        }
    }
    return candidate;
}



/**
 */
void HighlyReliableMarkers::BalancedBinaryTree::loadDictionary(Dictionary *D) {
    // create _orderD wich is a sorted version of D
    _orderD.clear();
    for (int32_t i = 0; i < D->size(); i++) {
        _orderD.push_back(std::pair< int32_t, int32_t >((*D)[i].getId(), i));
    }
    std::sort(_orderD.begin(), _orderD.end());

    // calculate the number of levels of the tree
    int32_t levels = 0;
    while (pow(float_t(2), float_t(levels)) <= _orderD.size())
        levels++;
    //       levels-=1; // only count full levels

    // auxiliar vector to know which elements are already in the tree
    std::vector< bool_t > visited;
    visited.resize(_orderD.size(), false);

    // calculate position of the root element
    int32_t rootIdx = _orderD.size() / 2;
    visited[rootIdx] = true; // mark it as visited
    _root = rootIdx;

    //    for(int32_t i=0; i<visited.size(); i++) std::cout << visited[i] << std::endl;

    // auxiliar vector to store the ids intervals (max and min) during the creation of the tree
    std::vector< std::pair< int32_t, int32_t > > intervals;
    // first, add the two intervals at each side of root element
    intervals.push_back(std::pair< int32_t, int32_t >(0, rootIdx));
    intervals.push_back(std::pair< int32_t, int32_t >(rootIdx, _orderD.size()));

    // init the tree
    _binaryTree.clear();
    _binaryTree.resize(_orderD.size());

    // add root information to the tree (make sure child do not coincide with self root for small sizes of D)
    if (!visited[(0 + rootIdx) / 2])
        _binaryTree[rootIdx].first = (0 + rootIdx) / 2;
    else
        _binaryTree[rootIdx].first = -1;
    if (!visited[(rootIdx + _orderD.size()) / 2])
        _binaryTree[rootIdx].second = (rootIdx + _orderD.size()) / 2;
    else
        _binaryTree[rootIdx].second = -1;

    // for each tree level
    for (int32_t i = 1; i < levels; i++) {
        int32_t nintervals = intervals.size(); // count number of intervals and process them
        for (int32_t j = 0; j < nintervals; j++) {
            // store interval information and delete it
            int32_t lowerBound, higherBound;
            lowerBound = intervals.back().first;
            higherBound = intervals.back().second;
            intervals.pop_back();

            // center of the interval
            int32_t center = (higherBound + lowerBound) / 2;

            // if center not visited, continue
            if (!visited[center])
                visited[center] = true;
            else
                continue;

            // calculate centers of the child intervals
            int32_t lowerChild = (lowerBound + center) / 2;
            int32_t higherChild = (center + higherBound) / 2;

            // if not visited (lower child)
            if (!visited[lowerChild]) {
                intervals.insert(intervals.begin(), std::pair< int32_t, int32_t >(lowerBound, center)); // add the interval to analyze later
                _binaryTree[center].first = lowerChild;                                                           // add as a child in the tree
            } else
                _binaryTree[center].first = -1; // if not, mark as no child

            // (higher child, same as lower child)
            if (!visited[higherChild]) {
                intervals.insert(intervals.begin(), std::pair< int32_t, int32_t >(center, higherBound));
                _binaryTree[center].second = higherChild;
            } else
                _binaryTree[center].second = -1;
        }
    }

    // print tree
    //     for(uint i=0; i<_binaryTree.size(); i++) std::cout << _binaryTree[i].first << " " << _binaryTree[i].second << std::endl;
    //     std::cout << std::endl;
}


int32_t count = 0;
int32_t idc = 11;

/**
 */
bool_t HighlyReliableMarkers::BalancedBinaryTree::findId(int32_t id, int32_t &orgPos) {
    int32_t pos = _root;                             // first position is root
    while (pos != -1) {                          // while having a valid position
        int32_t posId = _orderD[pos].first; // calculate id of the node
        if (posId == id) {
            orgPos = _orderD[pos].second;
            return true; // if is the desire id, return true
        } else if (posId < id)
            pos = _binaryTree[pos].second; // if desired id is higher, look in higher child
        else
            pos = _binaryTree[pos].first; // if it is lower, look in lower child
    }
    count++;
    return false; // if nothing found, return false
}
}
