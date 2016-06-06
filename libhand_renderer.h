#pragma once

#include <utility>
#include <string>
#include <chrono>
#include <map>
#include <stdlib.h>
#include <stdarg.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#define DATA_DIR "/home/user/data/libhand_onepose/"

#define POSMOD(x, m) (((x % m) + m) % m)

std::string stringFormat(const std::string fmt, ...) {
    int size = 100;
    std::string str;
    va_list ap;
    while (1) {
        str.resize(size);
        va_start(ap, fmt);
        int n = vsnprintf((char *)str.c_str(), size, fmt.c_str(), ap);
        va_end(ap);
        if (n > -1 && n < size) {
            str.resize(n);
            return str;
        }
        if (n > -1)
            size = n + 1;
        else
            size *= 2;
    }
    return str;
}

std::map<int, cv::Mat> depthCache;
std::map<int, cv::Mat> vmapCache;

cv::Mat getDepthImg(int img) {
	if (depthCache.find(img) == depthCache.end()) {
		depthCache[img] = cv::imread(stringFormat(DATA_DIR"depth/%09d.png", img), CV_LOAD_IMAGE_UNCHANGED);
	}
	return depthCache[img];
}

cv::Mat getVmapImg(int img) {
	if (vmapCache.find(img) == vmapCache.end()) {
		vmapCache[img] = cv::imread(stringFormat(DATA_DIR"vmap/%09d.png", img), CV_LOAD_IMAGE_UNCHANGED);
	}
	return vmapCache[img];
}

std::pair<int, int> samplePoses(float* img0, float* img1) {
	int r0_0 = rand() % 40,
		r0_1 = rand() % 40,
		r0_2 = rand() % 40;
	int r1_0 = r0_0 ,
		r1_1 = r0_1 ,
		r1_2 = r0_2 ;
	switch (rand() % 3) {
		case 0:
			r1_0 = POSMOD(r1_0 + (rand() % 2) * 2 - 1, 40);
			break;
		case 1:
			r1_1 = POSMOD(r1_1 + (rand() % 2) * 2 - 1, 40);
			break;
		case 2:
			r1_2 = POSMOD(r1_2 + (rand() % 2) * 2 - 1, 40);
			break;
	}
	int img0Ind = r0_0 + r0_1 * 40 + r0_2 * 40 * 40;
	int img1Ind = r1_0 + r1_1 * 40 + r1_2 * 40 * 40;

	cv::Mat img0Depth = getDepthImg(img0Ind);
	cv::Mat img0Color = getVmapImg(img0Ind);
	cv::Mat img1Depth = getDepthImg(img1Ind);
	cv::Mat img1Color = getVmapImg(img1Ind);

	while (true) {
		// try a pixel
		int x = rand() % 100;
		int y = rand() % 100;
		if (img0Depth.at<unsigned short>(y, x) == 0)
			continue;
		cv::Vec3s srcCol = img0Color.at<cv::Vec3s>(y, x);
		float bestDist = -1;
		std::pair<int, int> bestShift;
		for (int r = std::max(0, y-10); r < std::min(100, y+11); r++) {
			for (int c = std::max(0, x-10); c < std::min(100, x+11); c++) {
				if (img1Depth.at<unsigned short>(r, c) > 0) {
					cv::Vec3s destCol = img1Color.at<cv::Vec3s>(r, c);
					float dist = cv::norm(srcCol - destCol);
					if ((bestDist == -1) || (dist < bestDist)) {
						bestDist = dist;
						bestShift = std::make_pair(c - x, r - y);
					}
				}
			}
		}
		if ((bestDist >= 0) && (bestDist < 1000)) {
			for (int r = 0; r < 100; r++) {
				for (int c = 0; c < 100; c++) {
					int srcX = c - 50 + x;
					int srcY = r - 50 + y;
					if ((srcX < 0) || (srcX >= 100) || (srcY < 0) || (srcY >= 100)) {
						img0[c + 100 * r] = -1;
						img1[c + 100 * r] = -1;
					} else {
						img0[c + 100 * r] = img0Depth.at<unsigned short>(srcY, srcX) * 2.0 / ((1 << 16) - 1) - 1;
						img1[c + 100 * r] = img1Depth.at<unsigned short>(srcY, srcX) * 2.0 / ((1 << 16) - 1) - 1;
					}

				}
			}
			return bestShift;
		}
	}
}
