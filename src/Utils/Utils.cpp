#include "Utils/Utils.h"
#include <vector>
#include <algorithm>
#include <opencv2/core.hpp>

int Utils::CalculateCount(const std::vector<bool> value_list, const bool value)
{
    auto count = 0;
    for (auto currentValue : value_list)
        currentValue == value ? count++ : count;
    return count;
}

cv::Mat Utils::GetGaussianKernal(const int kernel_size, const double sigma)
{
    auto kernelRadius = (kernel_size - 1) / 2;

    cv::Mat tempKernel(kernel_size, kernel_size, CV_32FC1);
    auto squareSigma = 2.0 * sigma * sigma;

    for (auto i = (-kernelRadius); i <= kernelRadius; i++)
    {
        auto row = i + kernelRadius;
        for (auto j = (-kernelRadius); j <= kernelRadius; j++)
        {
            auto col = j + kernelRadius;
            float v = exp(-(1.0 * i * i + 1.0 * j * j) / squareSigma);
            tempKernel.ptr<float>(row)[col] = v;
        }
    }

    auto elementSum = sum(tempKernel);
    cv::Mat gaussKernel;
    tempKernel.convertTo(gaussKernel, CV_32FC1, (1 / elementSum[0]));

    return gaussKernel;
}

inline float Utils::GetVectorMedian(std::vector<float>& value_list)
{
    std::sort(value_list.begin(), value_list.end());

    auto len = value_list.size();
    if (len % 2 == 1)
        return value_list[len / 2];

    return float(value_list[len / 2] + value_list[(len - 1) / 2]) / 2.0;
}

void Utils::CalculatedMedian(const cv::Mat& source_mat, cv::Mat& median_mat)
{
    auto channels = source_mat.channels();

    for (auto r = 0; r < median_mat.rows; ++r)
    {
        auto dstRowData = median_mat.ptr<float>(r);
        auto srcRowData = source_mat.ptr<float>(r);

        for (auto c = 0; c < median_mat.cols; ++c)
        {
            std::vector<float> elementVector;

            for (auto i = 0; i < channels; ++i)
                elementVector.push_back(*(srcRowData + c * channels + i));

            dstRowData[c] = GetVectorMedian(elementVector);
        }
    }
}

void Utils::Sign(const cv::Mat& src_mat, cv::Mat& dest_mat)
{
    for (auto r = 0; r < src_mat.rows; ++r)
    {
        auto perLineSrc = src_mat.ptr<float>(r);
        auto perLineDest = dest_mat.ptr<float>(r);

        for (auto c = 0; c < src_mat.cols; ++c)
        {
            if (static_cast<int>(perLineSrc[c]) > 0)
                perLineDest[c] = static_cast<float>(1);
            else if (static_cast<int>(perLineSrc[c]) < 0)
                perLineDest[c] = static_cast<float>(-1);
            else
                perLineDest[c] = static_cast<float>(0);
        }
    }
}

cv::Mat Utils::ReshapedMatColumnFirst(const cv::Mat& srcMat)
{
    cv::Mat reshapedMat(cv::Size(1, srcMat.cols * srcMat.rows), CV_32FC1);

    for (auto r = 0; r < srcMat.rows;++r)
    {
        auto nr = r;
        auto rowSrcMat = srcMat.ptr<float>(r);
        for (auto c = 0; c < srcMat.cols;++c)
        {
            reshapedMat.ptr<float>(nr)[0] = rowSrcMat[c];
            nr += srcMat.rows;
        }
    }
    return reshapedMat;
}

std::vector<cv::Mat> Utils::WarpFrames(const std::vector<cv::Mat>& srcFrames, int borderSize)
{
    std::vector<cv::Mat> warpedResult;
    warpedResult.resize(srcFrames.size());

    auto originalWidth = srcFrames[0].cols;
    auto originalHeight = srcFrames[0].rows;

    for (auto i = 0; i<srcFrames.size(); ++i)
    {
        auto subFrameSharedMemory = srcFrames[i](cv::Rect(borderSize, borderSize, originalWidth - 2 * borderSize, originalHeight - 2 * borderSize));
        subFrameSharedMemory.copyTo(warpedResult[i]);
    }
    return warpedResult;
}

double Utils::Mod(double value, double sr_factor)
{
    auto result = value - floor(value / sr_factor) * sr_factor;
    return result;
}


