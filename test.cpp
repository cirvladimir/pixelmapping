#include <iostream>
#include <stdlib.h>
#include "stdio.h"
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>

using namespace std;
using namespace caffe;
using namespace cv;

void renderRect(int imW, int imH, int rW, int rH, int cx, int cy, float rot, float* data) {
    float bkGround = -1;
    float col = 1;
    float c = cos(rot);
    float s = sin(rot);
    for (int y = 0; y < imH; y++) {
        for (int x = 0; x < imW; x++) {
            float locx = (x - cx) * c - (y - cy) * s;
            float locy = (x - cx) * s + (y - cy) * c;
            data[x + imW * y] = (-rW / 2 < locx) && (locx < rW / 2) && (-rH / 2 < locy) && (locy < rH / 2) ? col : bkGround;
        }
    }
}

int main(int argc, char ** argv) {
    // Mat rm(100, 100, CV_32F);
    // renderRect(100, 100, 30, 30, 70, 50, 0.4, (float*)rm.data);
    // imwrite("test.png", rm);
    // return 0;

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
    float rot = (rand() % 10000) * 1.0 / 10000.0 * M_PI / 2;
    int randx = rand() % 30 + 35;
    int randy = rand() % 30 + 35;
    int randdx = rand() % 11 - 5;
    int randdy = rand() % 11 - 5;
    float drot = 0;//(rand() % 10000) * 1.0 / 10000.0 * M_PI / 5 - M_PI / 10;
    float c = cos(rot);
    float s = sin(rot);
    float cp = cos(rot + drot);
    float sp = sin(rot + drot);
    // int locx = (50 - randx) * cos(rot) - (50 - randy) * sin(rot);
    // int locy = (50 - randx) * sin(rot) + (50 - randy) * cos(rot);
    // int movx = locx * cos(rot + drot) + locy * sin(rot + drot) + randx + randdx - 50;
    // int movy = -locx * sin(rot + drot) + locy * cos(rot + drot) + randy + randdy - 50;
    renderRect(WIDTH, HEIGHT, rectW, rectH, randx, randy, rot, img1Ptr);
    renderRect(WIDTH, HEIGHT, rectW, rectH, randx + randdx, randy + randdy, rot + drot, img2Ptr);
    // check that center is 1
    // while ((imgPtr[WIDTH / 2 + WIDTH * (HEIGHT / 2)] != 1) || ((movx > 5) || (movx < -5) || (movy > 5) || (movy < -5))) {
    //     rot = (rand() % 10000) * 1.0 / 10000.0 * M_PI / 2;
    //     randx = rand() % 30 + 35;
    //     randy = rand() % 30 + 35;
    //     randdx = rand() % 11 - 5;
    //     randdy = rand() % 11 - 5;
    //     drot = 0;//(rand() % 10000) * 1.0 / 10000.0 * M_PI / 5 - M_PI / 10;
    //     locx = (50 - randx) * cos(rot) - (50 - randy) * sin(rot);
    //     locy = (50 - randx) * sin(rot) + (50 - randy) * cos(rot);
    //     movx = locx * cos(rot + drot) + locy * sin(rot + drot) + randx + randdx - 50;
    //     movy = -locx * sin(rot + drot) + locy * cos(rot + drot) + randy + randdy - 50;
    //     renderRect(WIDTH, HEIGHT, rectW, rectH, randx, randy, rot, img1Ptr);
    //     renderRect(WIDTH, HEIGHT, rectW, rectH, randx + randdx, randy + randdy, rot + drot, img2Ptr);
    //     cout << "." << endl;
    // }



    Mat rm1(100, 100, CV_32F);
    Mat rm2(100, 100, CV_32F);
    for (int x = 0; x < 100; x++) {
       for (int y = 0; y < 100; y++) {
           rm1.at<float>(x, y) = img1Ptr[x + 100 * y] == 1 ? 255 : 0;
           rm2.at<float>(x, y) = img2Ptr[x + 100 * y] == 1 ? 255 : 0;
       }
    }
    imwrite("dmap1.png", rm1);
    imwrite("dmap2.png", rm2);

    cv::Mat src(100, 100, CV_8UC3, Scalar::all(0)),
            act(100, 100, CV_8UC3, Scalar::all(0)),
            pred(100, 100, CV_8UC3, Scalar::all(0));

    int curI = 0;
    int usedColors = 0;
    int testParity = 0;
    while (true) {
        vector<pair<int, int>> loadedNums;
        vector<Vec3b> loadedColors;
        vector<int> gtGuess;
        for (int i = 0; i < BATCH_SIZE; i++) {
            int x = curI / 100;
            int y = curI % 100;
            while (img1Ptr[x + y * 100] == -1) {
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
            src.at<Vec3b>(x, y) = col;

            float locx = (x - randx) * c - (y - randy) * s;
            float locy = (x - randx) * s + (y - randy) * c;

            int actx = (locx * cp + locy * sp) + (randx + randdx);
            int acty = (-locx * sp + locy * cp) + (randy + randdy);

            act.at<Vec3b>(actx, acty) = col;

            loadedColors.push_back(col);
            loadedNums.push_back(make_pair(x, y));
            gtGuess.push_back(((actx - x) + 5) + ((acty - y) + 5) * 11);

            // int preddx = (actx - x);
            // int preddy = (acty - y);
            // if (testParity % 8 < 4)
            //     preddx *= -1;
            // if (testParity % 4 < 2)
            //     preddy *= -1;
            // net->input_blobs()[1]->mutable_cpu_data()[i] = (testParity % 2) ? (preddx + 5) + (preddy + 5) * 11 : (preddy + 5) + (preddx + 5) * 11;
            net->input_blobs()[1]->mutable_cpu_data()[i] = ((actx - x) + 5) + ((acty - y) + 5) * 11;
            float* curData = net->input_blobs()[0]->mutable_cpu_data() + 100 * 100 * 2 * i;
            for (int pry = 0; pry < 100; pry++) {
                for (int prx = 0; prx < 100; prx++) {
                    int cx = x - 50 + prx;
                    int cy = y - 50 + pry;
                    if ((cx < 0) || (cx >= 100) || (cy < 0) || (cy >= 100)) {
                        curData[prx + 100 * pry] = -1;
                        curData[prx + 100 * pry + 100 * 100] = -1;
                    } else {
                        curData[prx + 100 * pry] = img1Ptr[cx + 100 * cy];
                        curData[prx + 100 * pry + 100 * 100] = img2Ptr[cx + 100 * cy];
                    }       
                }
            }

            curI++;
            if (curI == 10000)
                break;
        }
        // cout << (testParity % 2 ? "x then y " : "y then x ") << (testParity % 4 < 2 ? "-y " : "+y ") << (testParity % 8 < 4 ? "-x " : "+x ") << endl;
        // float* labss = (float*)malloc(BATCH_SIZE * sizeof(float) * NUM_LABELS);
        // memcpy(labss, labels, BATCH_SIZE * sizeof(float) * NUM_LABELS);
        // net->input_blobs()[1]->set_cpu_data(labss);
        testParity++;
        float loss;
        net->Forward(&loss);
        cout << "loss: " << loss << endl;
        const float* guesses = net->blob_by_name("fc8")->cpu_data();
        for (int i = 0; i < loadedNums.size(); i++) {
            int maxGuess = 0;
            for (int j = 1; j < 121; j++) {
                if (guesses[maxGuess] < guesses[j])
                    maxGuess = j;
            }
            // maxGuess = gtGuess[i];
            int guessx = maxGuess % 11 - 5;
            int guessy = maxGuess / 11 - 5;
            pred.at<Vec3b>(loadedNums[i].first + guessx, loadedNums[i].second + guessy) = loadedColors[i];
            guesses += 121;
        }

        if (curI == 10000)
            break;
    }

    imwrite("src.png", src);
    imwrite("act.png", act);
    imwrite("pred.png", pred);

    return 0;
}
