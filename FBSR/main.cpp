#include "FrameSource/FrameSourceFactory.h"
#include "SuperResolution/SuperResolutionBase.h"
#include "SuperResolution/SuperResolutionFactory.h"

#include "SuperResolution/ReadEmilyImageList.hpp"

using namespace std;
using namespace cv;

int main()
{
		const string videoFileName = "768x576.avi";

		auto videoFrameSource = FrameSourceFactory::createFrameSourceFromVideo(videoFileName);

		Mat currentFrame;

		auto superResolution = SuperResolutionFactory::CreateSuperResolutionBase();

//		superResolution->SetFrameSource(videoFrameSource);

		superResolution->NextFrame(currentFrame);

	return 0;
}
