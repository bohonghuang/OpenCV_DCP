#include <iostream>
#include "opencv2/opencv.hpp"
#include "guided-filter/guidedfilter.h"

using namespace cv;

Mat getDarkChannel(const Mat &image, int blockSize) {
    if (blockSize % 2 == 0 || blockSize <= 0) throw "Invalid blockSize! ";
    int base = (blockSize - 1) / 2;
    Mat result = Mat(image.rows - base * 2, image.cols - base * 2, CV_8UC1);
    for (int y = 0; y < result.rows; y++)
        for (int x = 0; x < result.cols; x++) {
            uchar min = UCHAR_MAX;
            for (int i = y; i < y + blockSize; i++)
                for (int j = x; j < x + blockSize; j++) {
                    uchar b = image.ptr(i, j)[0];
                    uchar g = image.ptr(i, j)[1];
                    uchar r = image.ptr(i, j)[2];
                    min = std::min(min, std::min(b, std::min(g, r)));
                }
            result.ptr(y, x)[0] = min;
        }
    return result;
}

uchar getAtmosphericLight(const Mat &image, const Mat &darkChannel, float averagePercent) {
    int rows = image.rows;
    int cols = image.cols;
    int size = rows * cols;

    typedef std::tuple<int, int, uchar> Pixel;

    Pixel darkPixels[size];
    darkChannel.forEach<uchar>([cols, &darkPixels](const uchar &value, const int *position) {
        darkPixels[position[0] * cols + position[1]] = std::tuple(position[1], position[0], value);
    });
    std::sort(darkPixels, darkPixels + size, [](Pixel &a, Pixel &b) {
        return std::get<2>(a) > std::get<2>(b);
    });
    int sampleSize = std::max(int(averagePercent * float(size)), 1);
    int acc = 0;
    for (int i = 0; i < sampleSize; i++) {
        int x, y;
        uchar value;
        std::tie(x, y, value) = darkPixels[i];
        acc += std::max(image.ptr(y, x)[0], std::max(image.ptr(y, x)[1], image.ptr(y, x)[2]));
    }
    return uchar(acc / sampleSize);
}

int main() {
    Mat original = imread("pic/test.jpg", ImreadModes::IMREAD_COLOR);
    resize(original, original, Size(0, 0), 0.25, 0.25);

    Mat darkChannel = getDarkChannel(original, 15);
    resize(darkChannel, darkChannel, Size(original.cols, original.rows));

    auto atmosphericLight = float(getAtmosphericLight(original, darkChannel, 0.001f));

    float omega = 0.95f;
    Mat transmission = Mat(original.rows, original.cols, CV_32FC1);
    transmission.forEach<float>([&darkChannel, atmosphericLight, omega](float &value, const int *position) {
        value = 1.0f - omega * float(darkChannel.at<uchar>(position[0], position[1])) / atmosphericLight;
    });

    int guidedFilterR = 50;
    double guidedFilterEpsilon = 0.01;
    GuidedFilter filter = GuidedFilter(original, guidedFilterR, guidedFilterEpsilon);
    transmission = filter.filter(transmission);

    Mat result = Mat(original.rows, original.cols, CV_8UC3);
    result.forEach<Vec3b>([&original, &transmission, atmosphericLight](Vec3b &value, const int *position) {
        for (int channel = 0; channel < 3; channel++) {
            float channelValue = (float(original.at<Vec3b>(position[0], position[1])[channel]) - atmosphericLight)
                                 / transmission.at<float>(position[0], position[1]) + atmosphericLight;
            channelValue = std::clamp(channelValue, 0.0f, 255.0f);
            value[channel] = uchar(channelValue);
        }
    });

//    imshow("Original", original);
//    imshow("DarkChannel", darkChannel);
//    imshow("Transmission", transmission);
    imshow("Result", result);
    imwrite("pic/out.jpg", result);
    waitKey(0);
}
