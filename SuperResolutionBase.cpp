#include <opencv2/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <iostream>

#include "SuperResolutionBase.h"

#include <opencv2/imgproc.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <opencv2/core.hpp>

std::vector<double> LKOFlow::PyramidalLKOpticalFlow(cv::Mat &img1, cv::Mat &img2, cv::Rect &ROI)
{
    auto ROISize = ROI.size();

    auto levels = std::min(6, static_cast<int>(floor(log2(std::min(ROISize.height, ROISize.width)) - 2)));

    std::vector<cv::Mat> image1Pyramid;
    std::vector<cv::Mat> image2Pyramid;
    image1Pyramid.resize(levels);
    image2Pyramid.resize(levels);

    GaussianPyramid(img1, image1Pyramid, levels);
    GaussianPyramid(img2, image2Pyramid, levels);

    std::vector<double> distance = {0.0, 0.0};

    for (auto currentLevel = levels - 1; currentLevel >= 0; --currentLevel)
    {
        distance[0] *= 2;
        distance[1] *= 2;

        auto scale = pow(2, currentLevel);

        cv::Point topLeft;
        topLeft.x = std::max(static_cast<int>(ceil(ROI.x / scale)), 1);
        topLeft.y = std::max(static_cast<int>(ceil(ROI.y / scale)), 1);

        cv::Size currentSize;
        currentSize.width = floor(ROISize.width / scale);
        currentSize.height = floor(ROISize.height / scale);

        cv::Point bottomRight;
        bottomRight.x = std::min(topLeft.x + currentSize.width - 1, image1Pyramid[currentLevel].size().width - 1);
        bottomRight.y = std::min(topLeft.y + currentSize.height - 1, image1Pyramid[currentLevel].size().height - 1);

        IterativeLKOpticalFlow(image1Pyramid[currentLevel], image2Pyramid[currentLevel], topLeft, bottomRight,
                               distance);
    }

    return distance;
}

void LKOFlow::GaussianPyramid(cv::Mat &img, std::vector<cv::Mat> &pyramid, int levels)
{
    img.copyTo(pyramid[0]);

    for (auto i = 1; i < levels; ++i)
        GaussianDownSample(pyramid[i - 1], pyramid[i]);
}

void LKOFlow::IterativeLKOpticalFlow(cv::Mat &img1, cv::Mat &img2, cv::Point topLeft, cv::Point bottomRight,
                                     std::vector<double> &distance)
{
    auto oldDistance = distance;

    auto maxIterativeCount = 10;
    auto stopThrashold = 0.01;
    cv::Rect ROIRect(topLeft, bottomRight);
    auto img1Rect = img1(ROIRect);

    cv::Mat Ht, G;
    ComputeLKFlowParms(img1, Ht, G);

    auto currentIterativeIndex = 1;
    double normDistrance = 1;
    while (currentIterativeIndex < maxIterativeCount && normDistrance > stopThrashold)
    {
        auto resample_img = ResampleImg(img2, ROIRect, distance);
        cv::Mat It = img1Rect - resample_img;

        auto newIt = Utils::ReshapedMatColumnFirst(It);

        cv::Mat b = Ht * newIt;

        cv::Mat dc = G.inv() * b;
        normDistrance = norm(dc);

        distance[0] += dc.at<float>(0, 0);
        distance[1] += dc.at<float>(1, 0);

        currentIterativeIndex++;
    }
}

void LKOFlow::ComputeLKFlowParms(cv::Mat &img, cv::Mat &Ht, cv::Mat &G)
{
    cv::Mat SobelX, SobelY;
    Sobel(img, SobelX, CV_32F, 1, 0);
    Sobel(img, SobelY, CV_32F, 0, 1);

    cv::Mat kernelX = (cv::Mat_<char>(3, 3) << 1, 0, -1, 2, 0, -2, 1, 0, -1);
    cv::Mat kernelY = kernelX.t();

    cv::Mat SSobelX, SSobelY;
    //	filter2D(img, SSobelX, CV_32F, kernelX, Point(-1, -1), 0, cv::BORDER_REFLECT101);
    filter2D(img, SSobelX, CV_32F, kernelX, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
    //	filter2D(img, SSobelY, CV_32F, kernelY, Point(-1, -1), 0, cv::BORDER_REFLECT101);
    filter2D(img, SSobelY, CV_32F, kernelY, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);

    auto rectSobelX = SSobelX(cv::Rect(1, 1, SobelX.cols - 2, SobelX.rows - 2));
    auto rectSobelY = SSobelY(cv::Rect(1, 1, SobelY.cols - 2, SobelY.rows - 2));

    cv::Mat deepCopyedX, deepCopyedY;
    rectSobelX.copyTo(deepCopyedX);
    rectSobelY.copyTo(deepCopyedY);

    auto reshapedX = Utils::ReshapedMatColumnFirst(deepCopyedX);
    auto reshapedY = Utils::ReshapedMatColumnFirst(deepCopyedY);

    auto H = MergeTwoCols(reshapedX, reshapedY);
    Ht = H.t();

    G = Ht * H;
}

cv::Mat LKOFlow::MergeTwoRows(cv::Mat &up, cv::Mat &down)
{
    auto totalRows = up.rows + down.rows;

    cv::Mat mergedMat(totalRows, up.cols, up.type());

    auto submat = mergedMat.rowRange(0, up.rows);
    up.copyTo(submat);
    submat = mergedMat.rowRange(up.rows, totalRows);
    down.copyTo(submat);

    return mergedMat;
}

cv::Mat LKOFlow::MergeTwoCols(cv::Mat left, cv::Mat right)
{
    auto totalCols = left.cols + right.cols;

    cv::Mat mergedDescriptors(left.rows, totalCols, left.type());

    auto submat = mergedDescriptors.colRange(0, left.cols);
    left.copyTo(submat);
    submat = mergedDescriptors.colRange(left.cols, totalCols);
    right.copyTo(submat);

    return mergedDescriptors;
}

cv::Mat LKOFlow::ResampleImg(cv::Mat &img, cv::Rect &rect, std::vector<double> disc)
{
    cv::Mat X, Y;
    auto leftTop = rect.tl();
    auto bottomeRight = rect.br();

    Meshgrid(leftTop.x - disc[0], bottomeRight.x - 1 - disc[0], leftTop.y - disc[1], bottomeRight.y - 1 - disc[1], X,
             Y);

    cv::Mat result;
    remap(img, result, X, Y, cv::INTER_LINEAR);

    return result;
}

void LKOFlow::Meshgrid(const float lefTopX, const float rightBottomX, const float lefTopY, const float rightBottomY,
                       cv::Mat &X, cv::Mat &Y)
{
    std::vector<float> t_x, t_y;

    for (auto i = lefTopX; (i - rightBottomX) < 0.001; i++)
        t_x.push_back(i);
    for (auto j = lefTopY; (j - rightBottomY) < 0.001; j++)
        t_y.push_back(j);

    cv::repeat(cv::Mat(t_x).t(), t_y.size(), 1, X);
    cv::repeat(cv::Mat(t_y), 1, t_x.size(), Y);
}

void LKOFlow::GaussianDownSample(std::vector<cv::Mat>::const_reference srcMat, std::vector<cv::Mat>::reference destMat)
{
    cv::Mat kernel = (cv::Mat_<float>(1, 5) << 0.0625, 0.2500, 0.3750, 0.2500, 0.0625);
    cv::Mat kernelT = kernel.t();

    cv::Mat img, imgT;
    filter2D(srcMat, img, CV_32F, kernel, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
    filter2D(img, imgT, CV_32F, kernelT, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);

    cv::Size size(ceil(srcMat.cols / 2.0), ceil(srcMat.rows / 2.0));
    cv::Mat tempImg(size, CV_32FC1);

    for (auto r = 0; r < imgT.rows; r += 2)
    {
        auto rowSrcMat = imgT.ptr<float>(r);
        auto rowDstmat = tempImg.ptr<float>(ceil(r / 2.0));

        for (auto c = 0; c < imgT.cols; c += 2)
        {
            int idx = ceil(c / 2.0);
            rowDstmat[idx] = rowSrcMat[c];
        }
    }

    tempImg.copyTo(destMat);
}

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

inline float Utils::GetVectorMedian(std::vector<float> &value_list)
{
    std::sort(value_list.begin(), value_list.end());

    auto len = value_list.size();
    if (len % 2 == 1)
        return value_list[len / 2];

    return float(value_list[len / 2] + value_list[(len - 1) / 2]) / 2.0;
}

void Utils::CalculatedMedian(const cv::Mat &source_mat, cv::Mat &median_mat)
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

void Utils::Sign(const cv::Mat &src_mat, cv::Mat &dest_mat)
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

cv::Mat Utils::ReshapedMatColumnFirst(const cv::Mat &srcMat)
{
    cv::Mat reshapedMat(cv::Size(1, srcMat.cols * srcMat.rows), CV_32FC1);

    for (auto r = 0; r < srcMat.rows; ++r)
    {
        auto nr = r;
        auto rowSrcMat = srcMat.ptr<float>(r);
        for (auto c = 0; c < srcMat.cols; ++c)
        {
            reshapedMat.ptr<float>(nr)[0] = rowSrcMat[c];
            nr += srcMat.rows;
        }
    }
    return reshapedMat;
}

std::vector<cv::Mat> Utils::WarpFrames(const std::vector<cv::Mat> &srcFrames, int borderSize)
{
    std::vector<cv::Mat> warpedResult;
    warpedResult.resize(srcFrames.size());

    auto originalWidth = srcFrames[0].cols;
    auto originalHeight = srcFrames[0].rows;

    for (auto i = 0; i < srcFrames.size(); ++i)
    {
        auto subFrameSharedMemory = srcFrames[i](
                cv::Rect(borderSize, borderSize, originalWidth - 2 * borderSize, originalHeight - 2 * borderSize));
        subFrameSharedMemory.copyTo(warpedResult[i]);
    }
    return warpedResult;
}

double Utils::Mod(double value, double sr_factor)
{
    auto result = value - floor(value / sr_factor) * sr_factor;
    return result;
}


SuperResolutionBase::SuperResolutionBase(int buffer_size) : isFirstRun(false), bufferSize(buffer_size), srFactor(4),
                                                            psfSize(3), psfSigma(1.0)
{
}

bool SuperResolutionBase::SetFrameSource(const cv::Ptr<FrameSource> &frameSource)
{
    this->frameSource = frameSource;
    this->isFirstRun = true;
    return true;
}

void SuperResolutionBase::SetProps(double alpha, double beta, double lambda, double P, int maxIterationCount)
{
    props.alpha = alpha;
    props.beta = beta;
    props.lambda = lambda;
    props.P = P;
    props.maxIterationCount = maxIterationCount;
}

void SuperResolutionBase::SetSRFactor(int sr_factor)
{
    srFactor = sr_factor;
}

bool SuperResolutionBase::Reset()
{
    this->frameSource->reset();
    this->isFirstRun = true;
    return true;
}

int SuperResolutionBase::NextFrame(cv::OutputArray outputFrame)
{
    if (isFirstRun)
    {
        Init(this->frameSource);
        isFirstRun = false;
    }
    return Process(outputFrame);
}

void SuperResolutionBase::SetBufferSize(int buffer_size)
{
    bufferSize = buffer_size;
}

void SuperResolutionBase::Init(cv::Ptr<FrameSource> &frameSource)
{
//	if(this->frameBuffer)
//	{
//		delete this->frameBuffer;
//		this->frameBuffer = nullptr;
//	}
    this->frameBuffer = new FrameBuffer(bufferSize);

    cv::Mat currentFrame;
    for (auto i = 0; i < bufferSize; ++i)
    {
        frameSource->nextFrame(currentFrame);
        frameBuffer->PushGray(currentFrame);
    }

    frameSize = cv::Size(currentFrame.cols, currentFrame.rows);
    currentFrame.release();
}

void SuperResolutionBase::UpdateZAndA(cv::Mat &Z, cv::Mat &A, int x, int y, const std::vector<bool> &index,
                                      const std::vector<cv::Mat> &frames, const int len) const
{
    std::vector<cv::Mat> selectedFrames;
    for (auto i = 0; i < index.size(); ++i)
    {
        if (true == index[i])
            selectedFrames.push_back(frames[i]);
    }

    cv::Mat mergedFrame;
    cv::merge(selectedFrames, mergedFrame);

    cv::Mat medianFrame(frames[0].rows, frames[0].cols, CV_32FC1);
    Utils::CalculatedMedian(mergedFrame, medianFrame);

    for (auto r1 = y - 1, r2 = 0; r1 < Z.rows; r1 += srFactor)
    {
        auto rowOfMatZ = Z.ptr<float>(r1);
        auto rowOfMatA = A.ptr<float>(r1);
        auto rowOfMedianFrame = medianFrame.ptr<float>(r2);

        for (auto c1 = x - 1, c2 = 0; c1 < Z.cols; c1 += srFactor)
        {
            rowOfMatZ[c1] = rowOfMedianFrame[c2];
            rowOfMatA[c1] = static_cast<float>(len);
            ++c2;
        }
        ++r2;
    }
}

void SuperResolutionBase::MedianAndShift(const std::vector<cv::Mat> &interp_previous_frames,
                                         const std::vector<std::vector<double>> &current_distances,
                                         const cv::Size &new_size, cv::Mat &Z, cv::Mat &A) const
{
    Z = cv::Mat::zeros(new_size, CV_32FC1);
    A = cv::Mat::ones(new_size, CV_32FC1);

    cv::Mat markMat = cv::Mat::zeros(cv::Size(srFactor, srFactor), CV_8UC1);

    for (auto x = srFactor; x < 2 * srFactor; ++x)
    {
        for (auto y = srFactor; y < 2 * srFactor; ++y)
        {
            std::vector<bool> index;
            for (auto k = 0; k < current_distances.size(); ++k)
            {
                if (current_distances[k][0] == x && current_distances[k][1] == y)
                    index.push_back(true);
                else
                    index.push_back(false);
            }

            auto len = Utils::CalculateCount(index, true);
            if (len > 0)
            {
                markMat.at<uchar>(x - srFactor, y - srFactor) = 1;
                UpdateZAndA(Z, A, x, y, index, interp_previous_frames, len);
            }
        }
    }

    cv::Mat noneZeroMapOfMarkMat = markMat == 0;

    std::vector<int> X, Y;
    for (auto r = 0; r < noneZeroMapOfMarkMat.rows; ++r)
    {
        auto perRow = noneZeroMapOfMarkMat.ptr<uchar>(r);
        for (auto c = 0; c < noneZeroMapOfMarkMat.cols; ++c)
        {
            if (static_cast<int>(perRow[c]) == 0)
            {
                X.push_back(c);
                Y.push_back(r);
            }
        }
    }

    if (X.size() != 0)
    {
        cv::Mat meidianBlurMatOfMatZ;
        cv::medianBlur(Z, meidianBlurMatOfMatZ, 3);

        auto rowCount = Z.rows;
        auto colCount = Z.cols;

        for (auto i = 0; i < X.size(); ++i)
        {
            for (auto r = Y[i] + srFactor - 1; r < rowCount; r += srFactor)
            {
                auto rowOfMatZ = Z.ptr<float>(r);
                auto rowOfMedianBlurMatOfMatZ = meidianBlurMatOfMatZ.ptr<float>(r);

                for (auto c = X[i] + srFactor - 0; c < colCount; c += srFactor)
                {
                    auto x = rowOfMedianBlurMatOfMatZ[c];
                    rowOfMatZ[c] = x;
                }
            }
        }
    }

    cv::Mat rootedMatA;
    sqrt(A, rootedMatA);
    rootedMatA.copyTo(A);
    rootedMatA.release();
}

cv::Mat
SuperResolutionBase::FastGradientBackProject(const cv::Mat &Xn, const cv::Mat &Z, const cv::Mat &A, const cv::Mat &hpsf)
{
    cv::Mat matZAfterGaussianFilter;
    cv::filter2D(Xn, matZAfterGaussianFilter, CV_32FC1, hpsf, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);

    cv::Mat diffOfZandMedianFiltedZ;
    subtract(matZAfterGaussianFilter, Z, diffOfZandMedianFiltedZ);

    cv::Mat multiplyOfdiffZAndA = A.mul(diffOfZandMedianFiltedZ);

    cv::Mat Gsign(multiplyOfdiffZAndA.rows, multiplyOfdiffZAndA.cols, CV_32FC1);
    Utils::Sign(multiplyOfdiffZAndA, Gsign);

    cv::Mat inversedHpsf;
    cv::flip(hpsf, inversedHpsf, -1);
    cv::Mat multiplyOfGsingAndMatA = A.mul(Gsign);

    cv::Mat filterResult;
    cv::filter2D(multiplyOfGsingAndMatA, filterResult, CV_32FC1, inversedHpsf, cv::Point(-1, -1), 0,
                 cv::BORDER_REFLECT);

    return filterResult;
}

cv::Mat SuperResolutionBase::GradientRegulization(const cv::Mat &Xn, const double P, const double alpha) const
{
    cv::Mat G = cv::Mat::zeros(Xn.rows, Xn.cols, CV_32FC1);

    cv::Mat paddedXn;
    copyMakeBorder(Xn, paddedXn, P, P, P, P, cv::BORDER_REFLECT);

    for (int i = -1 * P; i <= P; ++i)
    {
        for (int j = -1 * P; j <= P; ++j)
        {
            cv::Rect shiftedXnRect(cv::Point(0 + P - i, 0 + P - j),
                                   cv::Point(paddedXn.cols - P - i, paddedXn.rows - P - j));
            auto shiftedXn = paddedXn(shiftedXnRect);

            cv::Mat diffOfXnAndShiftedXn = Xn - shiftedXn;
            cv::Mat signOfDiff(diffOfXnAndShiftedXn.rows, diffOfXnAndShiftedXn.cols, CV_32FC1);
            Utils::Sign(diffOfXnAndShiftedXn, signOfDiff);

            cv::Mat paddedSignOfDiff;
            copyMakeBorder(signOfDiff, paddedSignOfDiff, P, P, P, P, cv::BORDER_CONSTANT, 0);

            cv::Rect shiftedSignedOfDiffRect(cv::Point(0 + P + i, 0 + P + j),
                                             cv::Point(paddedSignOfDiff.cols - P + i, paddedSignOfDiff.rows - P + j));
            auto shiftedSignOfDiff = paddedSignOfDiff(shiftedSignedOfDiffRect);

            cv::Mat diffOfSignAndShiftedSign = signOfDiff - shiftedSignOfDiff;

            auto tempScale = pow(alpha, (abs(i) + abs(j)));
            diffOfSignAndShiftedSign *= tempScale;

            G += diffOfSignAndShiftedSign;
        }
    }
    return G;
}

cv::Mat SuperResolutionBase::FastRobustSR(const std::vector<cv::Mat> &interp_previous_frames,
                                          const std::vector<std::vector<double>> &current_distances, cv::Mat hpsf)
{
    cv::Mat Z, A;
    auto originalWidth = interp_previous_frames[0].cols;
    auto originalHeight = interp_previous_frames[0].rows;
    cv::Size newSize((originalWidth + 1) * srFactor - 1, (originalHeight + 1) * srFactor - 1);

    MedianAndShift(interp_previous_frames, current_distances, newSize, Z, A);

    cv::Mat HR;
    Z.copyTo(HR);

    auto iter = 1;

    while (iter < props.maxIterationCount)
    {
        auto Gback = FastGradientBackProject(HR, Z, A, hpsf);
        auto Greg = GradientRegulization(HR, props.P, props.alpha);

        Greg *= props.lambda;
        cv::Mat tempResultOfIteration = (Gback + Greg) * props.beta;
        HR -= tempResultOfIteration;

        ++iter;
    }
    return HR;
}

int SuperResolutionBase::UpdateFrameBuffer()
{
    cv::Mat currentFrame;
    frameSource->nextFrame(currentFrame);
    if (currentFrame.empty())
        return -1;

    frameBuffer->PushGray(currentFrame);
    return 0;
}

std::vector<cv::Mat> SuperResolutionBase::NearestInterp2(const std::vector<cv::Mat> &previousFrames,
                                                         const std::vector<std::vector<double>> &distances) const
{
    cv::Mat X, Y;
    LKOFlow::Meshgrid(0, frameSize.width - 1, 0, frameSize.height - 1, X, Y);

    std::vector<cv::Mat> result;
    result.resize(previousFrames.size());

    for (auto i = 0; i < distances.size(); ++i)
    {
        auto currentFrame = previousFrames[i];
        cv::remap(currentFrame, result[i], X, Y, cv::INTER_NEAREST);

        if (distances[i][0] < 0.0)
        {
            auto srcSubFrameSharedMemory = result[i](
                    cv::Rect(static_cast<int>(-distances[i][0]), 0, static_cast<int>(result[i].cols + distances[i][0]),
                             result[i].rows));
            cv::Mat tempSubFrame;
            srcSubFrameSharedMemory.copyTo(tempSubFrame);
            auto destSubFrameSharedMemory = result[i](
                    cv::Rect(0, 0, static_cast<int>(result[i].cols + distances[i][0]), result[i].rows));
            tempSubFrame.copyTo(destSubFrameSharedMemory);

            auto totalCols = result[i].cols;
            for (auto r = 0; r < result[i].rows; r++)
            {
                auto rowOfResult = result[i].ptr<float>(r);
                for (auto j = distances[i][0]; j < 0.0; ++j)
                    rowOfResult[static_cast<int>((totalCols + j))] = -1;
            }
        }

        if (distances[i][1] < 0.0)
        {
            auto srcSubFrameSharedMemory = result[i](cv::Rect(0, static_cast<int>(-distances[i][1]), result[i].cols,
                                                              static_cast<int>(result[i].rows) + distances[i][1]));
            cv::Mat tempSubFrame;
            srcSubFrameSharedMemory.copyTo(tempSubFrame);
            auto destSubFrameSharedMemory = result[i](
                    cv::Rect(0, 0, result[i].cols, static_cast<int>(result[i].rows + distances[i][1])));
            tempSubFrame.copyTo(destSubFrameSharedMemory);

            auto totalRows = result[i].rows;
            for (auto j = distances[i][1]; j < 0.0; ++j)
            {
                auto rowOfResult = result[i].ptr<float>(static_cast<int>(totalRows + j));
                for (auto c = 0; c < result[i].cols; ++c)
                    rowOfResult[c] = -1;
            }
        }

        if (distances[i][0] > 0.0)
        {
            auto srcSubFrameSharedMemory = result[i](
                    cv::Rect(0, 0, static_cast<int>(result[i].cols - distances[i][0]), result[i].rows));
            cv::Mat tempSubFrame;
            srcSubFrameSharedMemory.copyTo(tempSubFrame);
            auto destSubFrameSharedmemory = result[i](
                    cv::Rect(static_cast<int>(distances[i][0]), 0, static_cast<int>(result[i].cols - distances[i][0]),
                             result[i].rows));
            tempSubFrame.copyTo(destSubFrameSharedmemory);

            auto totalRows = result[i].rows;
            for (auto r = 0; r < totalRows; ++r)
            {
                auto rowOfResult = result[i].ptr<float>(r);
                for (auto c = 0; c < static_cast<int>(distances[i][0]); ++c)
                    rowOfResult[c] = -1;
            }
        }

        if (distances[i][1] > 0.0)
        {
            auto srcSubFrameSharedMemory = result[i](
                    cv::Rect(0, 0, result[i].cols, static_cast<int>(result[i].rows - distances[i][1])));
            cv::Mat tempSubFrame;
            srcSubFrameSharedMemory.copyTo(tempSubFrame);
            auto destSubFrameSharedmemory = result[i](cv::Rect(0, static_cast<int>(distances[i][1]), result[i].cols,
                                                               static_cast<int>(result[i].rows) - distances[i][1]));
            tempSubFrame.copyTo(destSubFrameSharedmemory);

            auto totalCols = result[i].cols;
            for (auto r = 0; r < static_cast<int>(distances[i][1]); ++r)
            {
                auto rowOfResult = result[i].ptr<float>(r);
                for (auto c = 0; c < totalCols; ++c)
                    rowOfResult[c] = -1;
            }
        }
    }
    return result;
}

int SuperResolutionBase::Process(cv::OutputArray outputFrame)
{
    auto frameList = this->frameBuffer->GetAll();
    reverse(frameList.begin(), frameList.end());
    auto registeredDistances = RegisterImages(frameList);

    std::vector<std::vector<double>> roundedDistances(registeredDistances.size(), std::vector<double>(2, 0.0));
    std::vector<std::vector<double>> restedDistances(registeredDistances.size(), std::vector<double>(2, 0.0));
    ReCalculateDistances(registeredDistances, roundedDistances, restedDistances);

    auto interpPreviousFrames = NearestInterp2(frameList, restedDistances);

    auto warpedFrames = Utils::WarpFrames(interpPreviousFrames, 2);

    auto Hpsf = Utils::GetGaussianKernal(psfSize, psfSigma);

    auto highResolutionResult = FastRobustSR(warpedFrames, roundedDistances, Hpsf);

    highResolutionResult.convertTo(outputFrame, CV_8UC1);

    return UpdateFrameBuffer();
}

std::vector<std::vector<double>> SuperResolutionBase::RegisterImages(std::vector<cv::Mat> &frames)
{
    std::vector<std::vector<double>> result;
    cv::Rect rectROI(0, 0, frames[0].cols, frames[0].rows);

    result.push_back(std::vector<double>(2, 0.0));

    for (auto i = 1; i < frames.size(); ++i)
    {
        auto currentDistance = LKOFlow::PyramidalLKOpticalFlow(frames[0], frames[i], rectROI);
        result.push_back(currentDistance);
    }

    return result;
}

void SuperResolutionBase::GetRestDistance(const std::vector<std::vector<double>> &roundedDistances,
                                          std::vector<std::vector<double>> &restedDistances, int srFactor) const
{
    for (auto i = 0; i < roundedDistances.size(); ++i)
        for (auto j = 0; j < roundedDistances[0].size(); ++j)
            restedDistances[i][j] = floor(roundedDistances[i][j] / srFactor);
}

void SuperResolutionBase::RoundAndScale(const std::vector<std::vector<double>> &registeredDistances,
                                        std::vector<std::vector<double>> &roundedDistances, int srFactor) const
{
    for (auto i = 0; i < registeredDistances.size(); ++i)
        for (auto j = 0; j < registeredDistances[0].size(); ++j)
            roundedDistances[i][j] = round(registeredDistances[i][j] * double(srFactor));
}

void SuperResolutionBase::ModAndAddFactor(std::vector<std::vector<double>> &roundedDistances, int srFactor) const
{
    for (auto i = 0; i < roundedDistances.size(); ++i)
        for (auto j = 0; j < roundedDistances[0].size(); ++j)
            roundedDistances[i][j] = Utils::Mod(roundedDistances[i][j], static_cast<double>(srFactor)) + srFactor;
}

void SuperResolutionBase::ReCalculateDistances(const std::vector<std::vector<double>> &registeredDistances,
                                               std::vector<std::vector<double>> &roundedDistances,
                                               std::vector<std::vector<double>> &restedDistances) const
{
    // NOTE: Cannot change order

    RoundAndScale(registeredDistances, roundedDistances, srFactor);

    GetRestDistance(roundedDistances, restedDistances, srFactor);

    ModAndAddFactor(roundedDistances, srFactor);
}

FrameBuffer::FrameBuffer(int bufferSize) : head(0), bufferSize(bufferSize)
{
    sourceFrames.resize(this->bufferSize);
    returnFrames.resize(this->bufferSize);
}

FrameBuffer::~FrameBuffer()
{
    currentFrame.release();
    previousFrame.release();
    sourceFrames.clear();
    returnFrames.clear();
}

void FrameBuffer::Push(cv::Mat &frame)
{
    frame.copyTo(sourceFrames[head]);
    head += 1;
    if (head >= bufferSize)
        head %= bufferSize;
}

void FrameBuffer::PushGray(cv::Mat &frame)
{
    cv::Mat grayFrame;
    if (frame.channels() == 3)
        cv::cvtColor(frame, grayFrame, CV_BGR2GRAY);
    else
        grayFrame = frame;

    cv::Mat floatGrayFrame;
    grayFrame.convertTo(floatGrayFrame, CV_32FC1);

    floatGrayFrame.copyTo(sourceFrames[head]);
    head += 1;
    if (head >= bufferSize)
        head %= bufferSize;
}

std::vector<cv::Mat> FrameBuffer::GetAll()
{
    for (auto i = head, j = 0; j < bufferSize; j++)
    {
        returnFrames[j] = sourceFrames[i];
        i += 1;
        i %= bufferSize;
    }
    return returnFrames;
}

cv::Mat &FrameBuffer::CurrentFrame()
{
    auto currentIndex = (head + bufferSize - 1) % bufferSize;
    return sourceFrames[currentIndex];
}

cv::Mat &FrameBuffer::PreviousFrame()
{
    auto previousIndex = (head + bufferSize - 2) % bufferSize;
    return sourceFrames[previousIndex];
}