#include "SuperResolutionBase.h"
#include <highgui/highgui.hpp>
#include <contrib/contrib.hpp>

SuperResolutionBase::SuperResolutionBase(int bufferSize) : isFirstRun(false), bufferSize(bufferSize), srFactor(4), psfSize(3), psfSigma(1.0)
{
	this->frameBuffer = new FrameBuffer(bufferSize);
}

bool SuperResolutionBase::SetFrameSource(const cv::Ptr<FrameSource>& frameSource)
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

bool SuperResolutionBase::Reset()
{
	this->frameSource->reset();
	this->isFirstRun = true;
	return true;
}

void SuperResolutionBase::NextFrame(OutputArray outputFrame)
{
	if (isFirstRun)
	{
		Init(this->frameSource);
		isFirstRun = false;
	}
	Process(this->frameSource, outputFrame);
}

void SuperResolutionBase::Init(Ptr<FrameSource>& frameSource)
{
	Mat currentFrame;

	for (auto i = 0; i < bufferSize; ++i)
	{
		frameSource->nextFrame(currentFrame);
		frameBuffer->PushGray(currentFrame);
	}

	frameSize = Size(currentFrame.rows,currentFrame.cols);
	currentFrame.release();
}

int SuperResolutionBase::GetTrueCount(const vector<bool>& index)
{
	auto count = 0;
	for (auto curElem : index)
		curElem ? count++ : count;
	return count;
}

void SuperResolutionBase::UpdateZ(Mat& mat, int x, int y, const vector<bool>& index, const vector<Mat>& mats)
{

}

void SuperResolutionBase::MedianAndShift(const vector<Mat>& interp_previous_frames, const vector<vector<double>>& current_distances, const Size& new_size, Mat& Z, Mat& A)
{
	Z = Mat::zeros(new_size, CV_8UC1);
	A = Mat::ones(new_size, CV_8UC1);

	Mat S = Mat::zeros(Size(srFactor, srFactor), CV_8UC1);

	vector<bool> index;
	for (auto x = srFactor; x < 2 * srFactor; ++x)
	{
		for (auto y = srFactor; y < 2 * srFactor; ++y)
		{
			for (auto k = 0; k < current_distances.size(); ++k)
			{
				if (current_distances[k][0] == x && current_distances[k][1] == y)
					index.push_back(true);
				else
					index.push_back(false);
			}

			auto len = GetTrueCount(index);
			if (len > 0)
			{
				S.at<uchar>(x - srFactor, y - srFactor) = 1;

				UpdateZ(Z, x, y, index,interp_previous_frames);
//				UpdateA(A, x, y, len);
			}
		}
	}
}

void SuperResolutionBase::FastRobustSR(const vector<Mat>& interp_previous_frames, const vector<vector<double>>& current_distances)
{
	Mat Z, A;
	Size newSize((frameSize.width + 1) * srFactor - 1,(frameSize.height + 1) *srFactor -1);
	MedianAndShift(interp_previous_frames,current_distances,newSize,Z,A);
}

vector<Mat> SuperResolutionBase::NearestInterp2(const vector<Mat>& previousFrames, const vector<vector<double>>& distances) const
{
	Mat X, Y;
	LKOFlow::Meshgrid(Range(0, frameSize.width - 1), Range(0, frameSize.height - 1), X, Y);

	vector<Mat> result;
	result.resize(previousFrames.size());

	for (auto i = 0; i < distances.size(); ++i)
	{
		Mat shiftX = X + distances[i][0];
		Mat shiftY = Y + distances[i][0];

		Mat formatX, formatY;
		shiftX.convertTo(formatX, CV_32FC1);
		shiftY.convertTo(formatY, CV_32FC1);

		auto currentFrame = previousFrames[i];
		remap(currentFrame, result[i], formatX, formatY, INTER_NEAREST);
 	}
	return result;
}

void SuperResolutionBase::Process(Ptr<FrameSource>& frameSource, OutputArray outputFrame)
{
	namedWindow("Current Frame");
	namedWindow("Previous Frames");
	
	Mat currentFrame;

	while (frameBuffer->CurrentFrame().data)
	{
		imshow("Current Frame", frameBuffer->CurrentFrame());
		waitKey(100);

		auto previous_frames = frameBuffer->GetAll();
		auto currentDistances = RegisterImages(previous_frames);
		auto restDistances = CollectParms(currentDistances);
		auto interpPreviousFrames = NearestInterp2(previous_frames, restDistances);

		FastRobustSR(interpPreviousFrames, currentDistances);

		/*
		 for (auto i = 0; i < bufferSize; ++i)
		{
			imshow("Previous Frames", PreviousFrames[i]);
			waitKey(100);
		}
		 */
		
		frameSource->nextFrame(currentFrame);
		frameBuffer->PushGray(currentFrame);
	}

	currentFrame.release();
	destroyAllWindows();
}

vector<vector<double>> SuperResolutionBase::RegisterImages(vector<Mat>& frames)
{
	vector<vector<double> > result;
	Rect rectROI(0, 0, frames[0].cols, frames[0].rows);

	for(auto i =1;i<frames.size();++i)
	{
		auto currentDistance = LKOFlow::PyramidalLKOpticalFlow(frames[0], frames[i], rectROI);
		result.push_back(currentDistance);
	}

	return  result;
}

vector<vector<double>> SuperResolutionBase::GetRestDistance(const vector<vector<double>>& distances, int srFactor) const
{
	vector<vector<double> > result;
	for(auto i =0;i<distances.size();++i)
	{
		vector<double> distance;
		for (auto j = 0; j < distances[0].size(); ++j)
			distance.push_back(floor(distances[i][j] / srFactor));
		result.push_back(distance);
	}
	return  result;
}

void SuperResolutionBase::RoundAndScale(vector<vector<double>>& distances, int srFactor) const
{
	for(auto i =0;i<distances.size();++i)
		for(auto j = 0;j<distances[0].size();++j)
			distances[i][j] = round(distances[i][j] * double(srFactor));
}

void SuperResolutionBase::ModAndAddFactor(vector<vector<double>>& distances, int srFactor) const
{
	for (auto i = 0; i < distances.size(); ++i)
		for (auto j = 0; j < distances[0].size(); ++j)
			distances[i][j] = fmod(distances[i][j] , static_cast<double>(srFactor)) + srFactor;
}

vector<vector<double>> SuperResolutionBase::CollectParms(vector<vector<double> >& distances) const
{
	RoundAndScale(distances, srFactor);
	auto restDistance = GetRestDistance(distances, srFactor);
	ModAndAddFactor(distances,srFactor);

	return restDistance;
}