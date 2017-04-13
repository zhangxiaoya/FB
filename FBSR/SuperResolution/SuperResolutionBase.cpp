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