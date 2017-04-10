#include <iostream>
#include <string>
#include <opencv.hpp>

#include "FrameSource\FrameSourceFactory.h"
#include "FrameSource\FrameSource.h"
#include "SuperResolutionBase.h"
#include "SuperResolutionFactory.h"

using namespace std;
using namespace cv;

int main()
{
	const string videoFileName = "768x576.avi";

	Ptr<FrameSource> videoFrameSource = FrameSourceFactory::createFrameSourceFromVideo(videoFileName);

	Mat currentFrame;
	
	//Ptr<SuperResolutionBase> superResolution = SuperResolutionFactory::CreateSuperResolutionBTVL1();

	Ptr<SuperResolutionBase> superResolution = SuperResolutionFactory::CreateSuperResolutionBase();

	superResolution->SetFrameSource(videoFrameSource);

	superResolution->NextFrame(currentFrame);

	return 0;
}