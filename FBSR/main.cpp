#include "FrameSource/FrameSourceFactory.h"
#include "SuperResolution/SuperResolutionBase.h"
#include "SuperResolution/SuperResolutionFactory.h"

#include "SuperResolution/ReadEmilyImageList.hpp"
#include "ReadPaperImageList.hpp"

using namespace std;
using namespace cv;

int main()
{
	/*******************************************************************************
	 *
	 * Create a Super Resolution object and set basic props
	 *
	 *******************************************************************************/
	auto superResolution = SuperResolutionFactory::CreateSuperResolutionBase();

	auto alpha = 0.7;
	auto beta = 1.0;
	auto lambda = 0.04;
	auto p = 2;
	auto maxIterationCount = 20;
	auto srFactor = 2;

	superResolution->SetProps(alpha, beta, lambda, p, maxIterationCount);
	superResolution->SetSRFactor(srFactor);

	/*******************************************************************************
	 *
	 * set Data Source
	 *
	 *******************************************************************************/

	/***********************         From Video             ***********************/
//	const string videoFileName = "Data/fog_low_gray.avi";

//	auto videoFrameSource = FrameSourceFactory::createFrameSourceFromVideo(videoFileName);

//	superResolution->SetFrameSource(videoFrameSource);

	/***********************         From Image List         ***********************/
	auto paperImageCount = 400;

	auto imageListFrameSource = FrameSourceFactory::createFrameSourceFromImageList(paperImageCount, "");

	superResolution->SetFrameSource(imageListFrameSource);

	/*******************************************************************************
	 *
	 * Processing Super Resolution
	 *
	 *******************************************************************************/

	Mat currentFrame;
	while (true)
	{
		auto currentStatus = superResolution->NextFrame(currentFrame);

		imshow("High Resolution Frame", currentFrame);

		waitKey(100);

		if (currentStatus == -1)
			break;
	}

	return 0;
}
