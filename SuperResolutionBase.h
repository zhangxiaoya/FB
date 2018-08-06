#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <algorithm>


struct PropsStruct
{
    PropsStruct() : alpha(0.0), beta(0.0), lambda(0.0), P(0), maxIterationCount(20) {}

    double alpha;
    double beta;
    double lambda;
    double P;
    int maxIterationCount;
};

class Utils
{
public:
    static int CalculateCount(const std::vector<bool> value_list, const bool value = true);

    static cv::Mat GetGaussianKernal(const int kernel_size, const double sigma);

    static void CalculatedMedian(const cv::Mat &source_mat, cv::Mat &median_mat);

    static void Sign(const cv::Mat &src_mat, cv::Mat &dest_mat);

    static cv::Mat ReshapedMatColumnFirst(const cv::Mat &srcMat);

    static std::vector<cv::Mat> WarpFrames(const std::vector<cv::Mat> &interp_previous_frames, int borderSize);

    static double Mod(double value, double sr_factor);

private:
    static float GetVectorMedian(std::vector<float> &value_list);

};

class LKOFlow
{
public:
    static std::vector<double> PyramidalLKOpticalFlow(cv::Mat &img1, cv::Mat &img2, cv::Rect &ROI);

    static void
    Meshgrid(const float lefTopX, const float rightBottomX, const float lefTopY, const float rightBottomY, cv::Mat &X,
             cv::Mat &Y);

private:
    static void
    GaussianDownSample(std::vector<cv::Mat>::const_reference srcMat, std::vector<cv::Mat>::reference destMat);

    static void GaussianPyramid(cv::Mat &img, std::vector<cv::Mat> &pyramid, int levels);

    static void IterativeLKOpticalFlow(cv::Mat &Pyramid1, cv::Mat &Pyramid2, cv::Point topLeft, cv::Point bottomRight,
                                       std::vector<double> &disc);

    static void ComputeLKFlowParms(cv::Mat &img, cv::Mat &Ht, cv::Mat &G);

    static cv::Mat MergeTwoRows(cv::Mat &up, cv::Mat &down);

    static cv::Mat MergeTwoCols(cv::Mat left, cv::Mat right);

    static cv::Mat ResampleImg(cv::Mat &img, cv::Rect &rect, std::vector<double> disc);

};

class FrameSource
{
public:
    virtual ~FrameSource() {}

    virtual void nextFrame(cv::OutputArray frame) = 0;

    virtual void reset() = 0;
};


class FrameBuffer
{
public:
    explicit FrameBuffer(int bufferSize = 8);

    ~FrameBuffer();

    void Push(cv::Mat &frame);

    void PushGray(cv::Mat &frame);

    std::vector<cv::Mat> GetAll();

    cv::Mat &CurrentFrame();

    cv::Mat &PreviousFrame();

protected:
    int head;
    int bufferSize;
    cv::Mat currentFrame;
    cv::Mat previousFrame;
    std::vector<cv::Mat> sourceFrames;
    std::vector<cv::Mat> returnFrames;
};

class SuperResolutionBase
{
public:

    explicit SuperResolutionBase(int bufferSize = 8);

    bool SetFrameSource(const cv::Ptr<FrameSource> &frameSource);

    void SetProps(double alpha, double beta, double lambda, double P, int maxIterationCount);

    void SetSRFactor(int sr_factor);

    bool Reset();

    int NextFrame(cv::OutputArray outputFrame);

    void SetBufferSize(int buffer_size);

protected:
    void Init(cv::Ptr<FrameSource> &frameSource);

    void UpdateZAndA(cv::Mat &mat, cv::Mat &A, int x, int y, const std::vector<bool> &index,
                     const std::vector<cv::Mat> &mats, const int len) const;

    void MedianAndShift(const std::vector<cv::Mat> &interp_previous_frames,
                        const std::vector<std::vector<double>> &current_distances,
                        const cv::Size &new_size,
                        cv::Mat &mat,
                        cv::Mat &mat1) const;

    cv::Mat FastGradientBackProject(const cv::Mat &Xn, const cv::Mat &Z, const cv::Mat &A, const cv::Mat &hpsf);

    cv::Mat GradientRegulization(const cv::Mat &Xn, const double p, const double alpha) const;

    cv::Mat FastRobustSR(const std::vector<cv::Mat> &interp_previous_frames,
                         const std::vector<std::vector<double>> &current_distances, cv::Mat hpsf);

    int UpdateFrameBuffer();

    int Process(cv::OutputArray output);

    std::vector<cv::Mat> NearestInterp2(const std::vector<cv::Mat> &previousFrames,
                                        const std::vector<std::vector<double>> &currentDistances) const;

private:
    static std::vector<std::vector<double> > RegisterImages(std::vector<cv::Mat> &frames);

    void GetRestDistance(const std::vector<std::vector<double>> &roundedDistances,
                         std::vector<std::vector<double>> &restedDistances, int srFactor) const;

    void RoundAndScale(const std::vector<std::vector<double>> &registeredDistances,
                       std::vector<std::vector<double>> &roundedDistances, int srFactor) const;

    void ModAndAddFactor(std::vector<std::vector<double>> &roundedDistances, int srFactor) const;

    void ReCalculateDistances(const std::vector<std::vector<double>> &registeredDistances,
                              std::vector<std::vector<double>> &roundedDistances,
                              std::vector<std::vector<double>> &restedDistances) const;

private:
    cv::Ptr<FrameSource> frameSource;
    cv::Ptr<FrameBuffer> frameBuffer;

    bool isFirstRun;
    int bufferSize;
    cv::Size frameSize;

    int srFactor;
    int psfSize;
    double psfSigma;
    PropsStruct props;
};

class SuperResolutionFactory
{
public:
    static cv::Ptr<SuperResolutionBase> CreateSuperResolutionBase()
    {
        return new SuperResolutionBase();
    }
};


class ImageListReaderBase
{
public:
    explicit ImageListReaderBase(const std::string &file_name_format = "", int start_index = 0)
            : fileNameFormat(file_name_format),
              startIndex(start_index)
    {
    }

    virtual ~ImageListReaderBase() = default;

    virtual void ReadImageList(std::vector<cv::Mat> &image_list, int image_count) = 0;

    void SetFileNameFormat(std::string file_name_format)
    {
        fileNameFormat = file_name_format;
    }

    void SetStartIndex(int start_index)
    {
        startIndex = start_index;
    }

protected:
    std::string fileNameFormat;
    int startIndex;
};

class ImageListReader : public ImageListReaderBase
{
public:
    explicit ImageListReader(const std::string &file_name_format = "", int start_index = 0) : ImageListReaderBase(
            file_name_format, start_index)
    {
    }

    void ReadImageList(std::vector<cv::Mat> &image_list, int image_count) override
    {
        if (image_count != image_list.size())
            return;

        auto fileNameFormatCStr = fileNameFormat.c_str();

        for (auto i = startIndex; i < (image_count + startIndex); ++i)
        {
            char name[50];
            snprintf(name, sizeof(name), fileNameFormatCStr, i);

            std::string fullName(name);
            auto curImg = cv::imread(fullName, CV_LOAD_IMAGE_GRAYSCALE);

            cv::Mat floatGrayImg;
            curImg.convertTo(floatGrayImg, CV_32FC1);

            floatGrayImg.copyTo(image_list[i - startIndex]);
        }
    }
};

class EmptyFrameSource : public FrameSource
{
public:
    void nextFrame(cv::OutputArray frame) override
    {
        frame.release();
    }

    void reset() override {}
};

class CaptureFrameSource : public FrameSource
{
public:
    void nextFrame(cv::OutputArray outputFrame) override
    {
        if (outputFrame.kind() == cv::_InputArray::MAT)
        {
            videoCapture >> outputFrame.getMatRef();
        }
        else
        {
            //some thing wrong
        }
    }

protected:
    cv::VideoCapture videoCapture;
};

class VideoFrameSource : public CaptureFrameSource
{
public:
    explicit VideoFrameSource(const std::string &videoFileName)
    {
        videoCapture.release();
        videoCapture.open(videoFileName);
        CV_Assert(videoCapture.isOpened());
    }

    void reset() override { reset(); }

private:
    std::string videoFileName;
};

class ImageListFrameSource : public FrameSource
{
public:
    explicit ImageListFrameSource(int image_count, std::string file_name_format, int start_index = 0)
            :imageCount(image_count),
             fileNameFormat(file_name_format),
             startIndex(start_index)
    {
        imageListReader = new ImageListReader();
        ImageListFrameSource::reset();
    }

    ~ImageListFrameSource()
    {
        delete imageListReader;
    }

    void nextFrame(cv::OutputArray frame) override
    {
        if (currentIndex < imageCount)
        {
            imageList[currentIndex].copyTo(frame);
            ++currentIndex;
        }
        else
        {
            cv::Mat emptyMat;
            emptyMat.copyTo(frame);
        }
    }

    void reset() override
    {
        imageList.resize(imageCount);
        currentIndex = 0;

        imageListReader->SetFileNameFormat(fileNameFormat);
        imageListReader->SetStartIndex(startIndex);
        imageListReader->ReadImageList(imageList, imageCount);
    }

private:
    std::vector<cv::Mat> imageList;
    int imageCount;
    int currentIndex;
    int startIndex;
    std::string fileNameFormat;
    ImageListReader *imageListReader;
};

class FrameSourceFactory
{
public:
    static cv::Ptr<FrameSource> createEmptyFrameSource()
    {
        return new EmptyFrameSource();
    }

    static cv::Ptr<FrameSource> createFrameSourceFromVideo(const std::string &videoFileName)
    {
        return new VideoFrameSource(videoFileName);
    }

    static cv::Ptr<FrameSource>
    createFrameSourceFromImageList(const int &image_count, std::string file_name_format, int start_index = 0)
    {
        return new ImageListFrameSource(image_count, file_name_format, start_index);
    }
};
