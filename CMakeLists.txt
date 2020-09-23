SET(PROJECT_NAME holo_camera)

PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
set(HOLO_BASE_VERSION 0.1.0)
set(HOLO_SENSORS_VERSION 0.1.0)
set(BUILD_TYPE Release)

# set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
# message(CMAKE_CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR})

if(HOLO_BUILD_SHARED)
    set(BUILD_SHARED_LIBS ON)
else()
    set(BUILD_SHARED_LIBS OFF)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")
message(111CMAKE_MODULE_PATH=${CMAKE_MODULE_PATH})

set(EXECUTABLE_OUTPUT_PATH bin)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")
find_package(LibZip 1.1.1 REQUIRED)

set(CMAKE_CXX_FLAGS
   "${CMAKE_CXX_FLAGS_RELEASE} ${SSE_FLAGS} -std=c++0x"
)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
	${EIGEN3_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${LIBZIP_INCLUDE_DIR_ZIP}
    ${LIBZIP_INCLUDE_DIR_ZIPCONF}
)

add_executable(response_calib src/response_calib.cpp src/fov_undistorter.cpp src/photometric_undistorter.cpp)
target_link_libraries(response_calib ${OpenCV_LIBS} ${LIBZIP_LIBRARY})

add_executable(play_dataset src/playback_dataset.cpp src/fov_undistorter.cpp src/photometric_undistorter.cpp)
target_link_libraries(play_dataset ${OpenCV_LIBS} ${LIBZIP_LIBRARY})

add_executable(eva_camera src/eva_camera src/fov_undistorter.cpp src/photometric_undistorter.cpp)
target_link_libraries(eva_camera ${OpenCV_LIBS} ${LIBZIP_LIBRARY})

add_executable(eva_color src/eva_color src/fov_undistorter.cpp src/photometric_undistorter.cpp)
target_link_libraries(eva_color ${OpenCV_LIBS} ${LIBZIP_LIBRARY})

add_executable(calibration src/calibration.cpp)
target_link_libraries(calibration ${OpenCV_LIBS} ${LIBZIP_LIBRARY})

set(CMAKE_INSTALL_PREFIX "/usr/local")
message(CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX})
SET(CMAKE_MODULE_PATH ${CMAKE_INSTALL_PREFIX}/lib/cmake/)

find_package(aruco)

IF(aruco_FOUND)
	add_executable(vignette_calib src/vignette_calib.cpp src/fov_undistorter.cpp src/photometric_undistorter.cpp)
	target_link_libraries(vignette_calib ${OpenCV_LIBS} ${aruco_LIBS} ${LIBZIP_LIBRARY})
ELSE()
	message("================ aruco not found. not compiling vignetteCalib. ========================")
ENDIF()


















