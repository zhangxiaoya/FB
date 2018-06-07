#pragma once

#include <opencv2/core.hpp>

#include "FrameBuffer/FrameBuffer.h"
#include "FrameSource/FrameSource.h"
#include "SuperResolution/PropsStruct.h"

class SuperResolutionBase
{
public:

	explicit SuperResolutionBase(int bufferSize = 8);

	bool SetFrameSource(const cv::Ptr<FrameSource>& frameSource);

	void SetProps(double alpha, double beta, double lambda, double P, int maxIterationCount);

	void SetSRFactor(int sr_factor);

	bool Reset();

	int NextFrame(cv::OutputArray outputFrame);

	void SetBufferSize(int buffer_size);

protected:
	void Init(cv::Ptr<FrameSource>& frameSource);

	void UpdateZAndA(cv::Mat& mat, cv::Mat& A, int x, int y, const std::vector<bool>& index, const std::vector<cv::Mat>& mats, const int len) const;

	void MedianAndShift(const std::vector<cv::Mat>& interp_previous_frames,
						const std::vector<std::vector<double>>& current_distances,
						const cv::Size& new_size,
						cv::Mat& mat,
						cv::Mat& mat1) const;

	cv::Mat FastGradientBackProject(const cv::Mat& Xn, const cv::Mat& Z, const cv::Mat& A, const cv::Mat& hpsf);

	cv::Mat GradientRegulization(const cv::Mat& Xn, const double p, const double alpha) const;

	cv::Mat FastRobustSR(const std::vector<cv::Mat>& interp_previous_frames, const std::vector<std::vector<double>>& current_distances, cv::Mat hpsf);

	int UpdateFrameBuffer();

	int Process(OutputArray output);

	std::vector<cv::Mat> NearestInterp2(const std::vector<cv::Mat>& previousFrames, const std::vector<std::vector<double>>& currentDistances) const;

private:
	static std::vector<std::vector<double> > RegisterImages(std::vector<cv::Mat>& frames);

	void GetRestDistance(const std::vector<std::vector<double>>& roundedDistances, std::vector<std::vector<double>>& restedDistances, int srFactor) const;

	void RoundAndScale(const std::vector<std::vector<double>>& registeredDistances, std::vector<std::vector<double>>& roundedDistances, int srFactor) const;

	void ModAndAddFactor(std::vector<std::vector<double>>& roundedDistances, int srFactor) const;

	void ReCalculateDistances(const std::vector<std::vector<double>>& registeredDistances, std::vector<std::vector<double>>& roundedDistances, std::vector<std::vector<double>>& restedDistances) const;

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
