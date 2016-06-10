#include <iostream>
#include <stdlib.h>
#include "stdio.h"
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <math.h>
#include <functional>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "libhand_renderer.h"

using namespace std;
using namespace caffe;
using namespace cv;

#define RANDDOUBLE ((rand() % RAND_MAX) * 1.0 / RAND_MAX)
#define BK_GROUND -1.0f
#define FR_GROUND 1.0f

// helper function to render data to an image
// * fname: name of file
// * w: width of image
// * h: height of image
// * dat: pointer to raw image data
// * f: function which maps floats in image data to grayscale color (0 to 255)
//        example f: for the function f, pass in something like [](float x) { return (x - minX) / (maxX - miX) * 255; },
//        where minX is the minimum float in dat, and maxX is the maximum float
void drawDat(const char* fname, int w, int h, float* dat, function<float(float)> f) {
    Mat m(h, w, CV_32F);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            m.at<float>(y, x) = f(dat[y * w + x]);
        }
    }
    imwrite(fname, m);
}

int BATCH_SIZE;
int WIDTH;
int HEIGHT;
int NUM_LABELS;

float* data;
float* labels;
std::mutex mtx;
std::condition_variable cvar;
bool ready = false;

// thread to read data from the disk
void readData() {
    // locks the data and labels pointers and ready boolean
    std::unique_lock<std::mutex> lck(mtx);
    while (true) {
        // wait for computing thread to finish copying data over
        while (ready)
            cvar.wait(lck);
        // load batch into data and labels
        for (int i = 0; i < BATCH_SIZE; i++) {
            pair<int, int> mov = samplePoses(data + WIDTH * HEIGHT * 2 * i,
                data + WIDTH * HEIGHT * (2 * i + 1));
            labels[2 * i] = mov.first;
            labels[2 * i + 1] = mov.second;
        }
        // let computing thread know we're done
        ready = true;
        cvar.notify_all();
    }
}

// Program arguments:
// first argument(optional): index of gpu to use (0-n), default 0
// second argument(optional): name of .solverstate file
int main(int argc, char ** argv) {
    srand(time(0));

    // caffe initialization stuff
    ::google::InitGoogleLogging(argv[0]);
    Caffe::set_mode(Caffe::GPU);
    if (argc > 1) {
        Caffe::SetDevice(stoi(argv[1]));
    }

    SolverParameter solver_param;
    ReadSolverParamsFromTextFileOrDie("../lenet_solver.prototxt", &solver_param);

    std::shared_ptr<Solver<float> > solver;
    solver.reset(SolverRegistry<float>::CreateSolver(solver_param));

    // restore solver state if solver file provided
    if (argc > 2)
        solver->Restore(argv[2]);

    Net<float>* net = solver->net().get();

    // read known parameters from the network
    BATCH_SIZE = net->input_blobs()[0]->shape(0);
    WIDTH = net->input_blobs()[0]->shape(2);
    HEIGHT = net->input_blobs()[0]->shape(3);
    NUM_LABELS = net->input_blobs()[1]->shape(1);

    // data where the network will read in the images
    data = (float*)malloc(WIDTH * HEIGHT * 2 * BATCH_SIZE * sizeof(float));
    // labels for the data
    labels = (float*)malloc(NUM_LABELS * BATCH_SIZE * sizeof(float));

    // start the loading image thread
    std::thread loadThread(readData);

    while (true) {
        // lock data from reading thread
        std::unique_lock<std::mutex> lck(mtx);
        // wait for reading thread to finish reading data
        while (!ready)
            cvar.wait(lck);
        // copy read data into network
        memcpy(net->input_blobs()[0]->mutable_cpu_data(), data, WIDTH * HEIGHT * 2 * BATCH_SIZE * sizeof(float));
        memcpy(net->input_blobs()[1]->mutable_cpu_data(), labels, NUM_LABELS * BATCH_SIZE * sizeof(float));
        ready = false;
        lck.unlock();
        cvar.notify_all();
        // every 100 iteration check the loss
        if (solver->iter() % 100 == 0) {
            float loss;
            net->Forward(&loss);
            cout << "loss at " << solver->iter() << ": " << loss << endl;
        }
        // every 1000 iteration save the network (where it's saved is specified in lenet_solver.prototxt)
        if ((solver->iter() % 1000 == 0) && (solver->iter() > 0)) {
            solver->Snapshot();
        }
        // if (solver->iter() >= 25000) {
        //     return 0;
        // }

        // advance the solver
        solver->Step(4);
    }

    return 0;
}
