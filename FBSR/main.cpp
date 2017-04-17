#include "FrameSource/FrameSourceFactory.h"
#include "SuperResolution/SuperResolutionBase.h"
#include "SuperResolution/SuperResolutionFactory.h"

#include "SuperResolution/ReadEmilyImageList.hpp"

using namespace std;
using namespace cv;

int main()
{
	//	const string videoFileName = "768x576.avi";

	//	auto videoFrameSource = FrameSourceFactory::createFrameSourceFromVideo(videoFileName);

	//	Mat currentFrame;

	//	auto superResolution = SuperResolutionFactory::CreateSuperResolutionBase();

	//	superResolution->SetFrameSource(videoFrameSource);

	//	superResolution->NextFrame(currentFrame);

	auto emilyImageCount = 53;
	vector<Mat> EmilyImageList;
	EmilyImageList.resize(emilyImageCount);
	ReadEmilyImageList::ReadImageList(EmilyImageList, emilyImageCount);
	for (auto i = 0; i < emilyImageCount; ++i)
	{
		char title[20];
		snprintf(title, sizeof(title), "Emily %d", i + 1);
		string curTitle(title);
		imshow(curTitle, EmilyImageList[i]);
		waitKey(100);
		destroyAllWindows();
	}
	return 0;
}
