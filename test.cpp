#include <iostream>
#include <stdlib.h>
#include "stdio.h"
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>

#include "libhand_renderer.h"

using namespace std;
using namespace caffe;
using namespace cv;

#define RANDDOUBLE ((rand() % RAND_MAX) * 1.0 / RAND_MAX)
#define BK_GROUND -1.0f
#define FR_GROUND 1.0f

void renderRect(int imW, int imH, int rW, int rH, int cx, int cy, float rot, float* data) {
    float c = cos(rot);
    float s = sin(rot);
    for (int y = 0; y < imH; y++) {
        for (int x = 0; x < imW; x++) {
            int locx = (x - cx) * c - (y - cy) * s;
            int locy = (x - cx) * s + (y - cy) * c;
            data[x + imW * y] = (-rW / 2 < locx) && (locx < rW / 2) && (-rH / 2 < locy) && (locy < rH / 2) ? FR_GROUND : BK_GROUND;
        }
    }
}

void renderImage(int imW, int imH, int cx, int cy, float rot, float* data, const Mat& img) {
    warpAffine(img, Mat(imH, imW, CV_32F, data),
        (Mat_<float>(2, 3) << 1, 0, cx, 0, 1, cy) * // translate to desired location
        (Mat_<float>(3, 3) << cos(rot), -sin(rot), 0, sin(rot), cos(rot), 0, 0, 0, 1) * // rotate
        (Mat_<float>(3, 3) << 1, 0, -img.rows / 2, 0, 1, -img.cols / 2, 0, 0, 1), // center
        Size(imH, imW), INTER_LINEAR, BORDER_CONSTANT, Scalar(BK_GROUND));
}

void drawDat(const char* fname, int w, int h, float* dat, function<float(float)> f) {
    Mat m(h, w, CV_32F);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            m.at<float>(y, x) = f(dat[y * w + x]);
        }
    }
    imwrite(fname, m);
}

int main(int argc, char ** argv) {
    // Mat rm(100, 100, CV_32F);
    // renderRect(100, 100, 30, 30, 70, 50, 0.4, (float*)rm.data);
    // imwrite("test.png", rm);
    // return 0;
    Mat characterImg = imread("../character.png", CV_LOAD_IMAGE_GRAYSCALE);
    Mat character(50, 50, CV_32F);
    for (int j = 0; j < characterImg.cols; j++) {
        for (int i = 0; i < characterImg.rows; i++) {
            character.at<float>(i, j) = characterImg.at<unsigned char>(i, j) / -255.0 * (FR_GROUND - BK_GROUND) + FR_GROUND;
        }
    }

    srand(time(0));
    ::google::InitGoogleLogging(argv[0]);

    Caffe::set_mode(Caffe::GPU);

    std::shared_ptr<Net<float>> net(new Net<float>("../CNNNet_train_test_dual_stream.prototxt", TEST));

    net->CopyTrainedLayersFrom(argv[1]);

    const int BATCH_SIZE = net->input_blobs()[0]->shape(0);
    const int WIDTH = net->input_blobs()[0]->shape(2);
    const int HEIGHT = net->input_blobs()[0]->shape(3);
    const int NUM_LABELS = net->input_blobs()[1]->shape(1);

    int rectW = 50;
    int rectH = 50;

    // render two images:
    float img1Ptr[10000];
    float img2Ptr[10000];
    // float labPtr = labels + i * NUM_LABELS;

    // Try generate random params:
    float rot = RANDDOUBLE * M_PI * 2;
    int randx = rand() % 30 + 35;
    int randy = rand() % 30 + 35;
    int randdx = rand() % 11 - 5;
    int randdy = rand() % 11 - 5;
    float drot = RANDDOUBLE * M_PI / 9 - M_PI / 18;
    float c = cos(rot);
    float s = sin(rot);
    float cp = cos(rot + drot);
    float sp = sin(rot + drot);


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
    // int img0Ind = r0_0 + r0_1 * 40 + r0_2 * 40 * 40;
    // int img1Ind = r1_0 + r1_1 * 40 + r1_2 * 40 * 40;
    int img0Ind = stoi(argv[2]);
    int img1Ind = img0Ind + 4;
    Mat img0Depth = getDepthImg(img0Ind);
    Mat img0Color = getVmapImg(img0Ind);
    // offset by a bit
    Mat img1Depth = getDepthImg(img1Ind);
    Mat img1Color = getVmapImg(img1Ind);
    warpAffine(img1Depth, img1Depth,
        Mat((Mat_<float>(2, 3) << 1, 0, 0, 0, 1, 0)),
        Size(100, 100), INTER_LINEAR, BORDER_CONSTANT, Scalar(BK_GROUND));
    warpAffine(img1Color, img1Color,
        Mat((Mat_<float>(2, 3) << 1, 0, 2, 0, 1, 1)),
        Size(100, 100), INTER_LINEAR, BORDER_CONSTANT, Scalar(BK_GROUND));
    // renderImage(100, 100, randx, randy, rot, img1Ptr, character);
    // renderImage(100, 100, randx + randdx, randy + randdy, rot + drot, img2Ptr, character);
    // renderRect(WIDTH, HEIGHT, rectW, rectH, randx, randy, rot, img1Ptr);
    // renderRect(WIDTH, HEIGHT, rectW, rectH, randx + randdx, randy + randdy, rot + drot, img2Ptr);

    drawDat("dmap1.png", 100, 100, img1Ptr, [](float x) { return x * 127 + 128; });
    drawDat("dmap2.png", 100, 100, img2Ptr, [](float x) { return x * 127 + 128; });

    cv::Mat src(100, 100, CV_8UC3, Scalar::all(0)),
            act(100, 100, CV_8UC3, Scalar::all(0)),
            pred(100, 100, CV_8UC3, Scalar::all(0));

    int curI = 0;
    int usedColors = 0;
    int totGuesses = 0;
    int totCorrect = 0;
    while (true) {
        vector<pair<int, int>> loadedNums;
        vector<Vec3b> loadedColors;
        vector<pair<int, int>> gtGuess;
        for (int i = 0; i < BATCH_SIZE; i++) {
            int x = curI / 100;
            int y = curI % 100;
            while (img0Depth.at<unsigned short>(y, x) == 0) {
                curI++;
                if (curI == 10000)
                    break;
                x = curI / 100;
                y = curI % 100;
            }
            if (curI == 10000)
                break;
            Vec3b col(127.5 * (1 + sin(usedColors / 100.0)), 127.5 * (1 + cos(usedColors / 100.0)), 127.5 * (1 + sin(usedColors / 200.0) + sin(usedColors / 13.0)));
            usedColors++;
            src.at<Vec3b>(y, x) = col;


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

            int actx = x + bestShift.first;
            int acty = y + bestShift.second;
            int movx = actx - x;
            int movy = acty - y;

            // float locx = (x - randx) * c + (y - randy) * s;
            // float locy = -(x - randx) * s + (y - randy) * c;

            // int actx = (locx * cp - locy * sp) + (randx + randdx);
            // int acty = (locx * sp + locy * cp) + (randy + randdy);
            // int movx = actx - x;
            // int movy = acty - y;

            act.at<Vec3b>(acty, actx) = col;

            loadedColors.push_back(col);
            loadedNums.push_back(make_pair(x, y));
            gtGuess.push_back(make_pair(movx, movy));

            // net->input_blobs()[1]->mutable_cpu_data()[2 * i] = movx;
            // net->input_blobs()[1]->mutable_cpu_data()[2 * i + 1] = movy;
            float* curData = net->input_blobs()[0]->mutable_cpu_data() + 100 * 100 * 2 * i;
            for (int pry = 0; pry < 100; pry++) {
                for (int prx = 0; prx < 100; prx++) {
                    int cx = x - 50 + prx;
                    int cy = y - 50 + pry;
                    if ((cx < 0) || (cx >= 100) || (cy < 0) || (cy >= 100)) {
                        curData[prx + 100 * pry] = -1;
                        curData[prx + 100 * pry + 100 * 100] = -1;
                    } else {
                        curData[prx + 100 * pry] = img0Depth.at<unsigned short>(cy, cx) * 2.0 / ((1 << 16) - 1) - 1;
                        curData[prx + 100 * pry + 100 * 100] = img1Depth.at<unsigned short>(cy, cx) * 2.0 / ((1 << 16) - 1) - 1;
                        // curData[prx + 100 * pry] = img1Ptr[cx + 100 * cy];
                        // curData[prx + 100 * pry + 100 * 100] = img2Ptr[cx + 100 * cy];
                    }
                }
            }

            curI++;
            if (curI == 10000)
                break;
        }
        float loss;
        net->Forward(&loss);
        // cout << "loss: " << loss << endl;
        const float* guesses_x = net->blob_by_name("fc8_x")->cpu_data();
        const float* guesses_y = net->blob_by_name("fc8_y")->cpu_data();
        for (int i = 0; i < loadedNums.size(); i++) {
            int guessx = guesses_x[0];
            int guessy = guesses_y[0];
            totGuesses++;
            if ((abs(guessx - gtGuess[i].first) <= 1) && (abs(guessy - gtGuess[i].second) <= 1))
                totCorrect++;
            pred.at<Vec3b>(loadedNums[i].second + guessy, loadedNums[i].first + guessx) = loadedColors[i];
            guesses_x++;
            guesses_y++;
        }

        if (curI == 10000)
            break;
    }

    cout << "Correcness: " << totCorrect << "/" << totGuesses << endl;

    imwrite("src.png", src);
    imwrite("act.png", act);
    imwrite("pred.png", pred);

    return 0;
}
