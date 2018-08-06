#include "SuperResolutionBase.h"

#include <iostream>
#include <highgui.h>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    /*******************************************************************************
     *
     * Create a Super Resolution object and set default props
     *
     *******************************************************************************/
    auto superResolution = SuperResolutionFactory::CreateSuperResolutionBase();

    auto alpha = 0.7;
    auto beta = 1.0;
    auto lambda = 0.04;
    auto p = 2;
    auto maxIterationCount = 20;

    auto srFactor = 2;
    auto bufferSize = 4;

    superResolution->SetProps(alpha, beta, lambda, p, maxIterationCount);
    superResolution->SetBufferSize(bufferSize);
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

    /*******************************
    *
    *  This is test case for Adyoron,
    *  uncomment it if run want get
    *  Text test case result
    *
    ******************************/
//    auto startIndex = 0;
//    auto totalImageCount = 61;
//    auto fileNameFormat = "../data/Adyoron/%06d.png";
//    auto resultNameFormat = "../result/Adyoron_4*4_result_%02d.png";
//    superResolution->SetBufferSize(20);
//    superResolution->SetSRFactor(4);
    /*******************************
    *
    *  This is test case for EIA,
    *  uncomment it if run want get
    *  Text test case result
    *
    ******************************/
//    auto startIndex = 0;
//    auto totalImageCount = 16;
//    auto fileNameFormat = "../data/eia/%06d.png";
//    auto resultNameFormat = "../result/eia_4*4_result_%02d.png";
//    superResolution->SetBufferSize(totalImageCount);
//    superResolution->SetSRFactor(4);

    /*******************************
    *
    *  This is test case for Text,
    *  uncomment it if run want get
    *  Text test case result
    *
    ******************************/
//    auto startIndex = 1;
//    auto totalImageCount = 29;
//    auto fileNameFormat = "../data/text/%06d.png";
//    auto resultNameFormat = "../result/text_4*4_result_%02d.png";
//    superResolution->SetBufferSize(29);
//    superResolution->SetSRFactor(4);

    /*******************************
     *
     *  This is test case for Emily,
     *  uncomment it if run want get
     *  Emliy test case result
     *
     ******************************/
    	auto startIndex = 1;
    	auto totalImageCount = 82;
    	auto fileNameFormat = "../data/Emily/%06d.png";
    	auto resultNameFormat = "../result/Emily_4*4_result_%02d.png";
        superResolution->SetBufferSize(53);
    	superResolution->SetSRFactor(4);


    auto imageListFrameSource = FrameSourceFactory::createFrameSourceFromImageList(totalImageCount, fileNameFormat, startIndex);

    superResolution->SetFrameSource(imageListFrameSource);

    /*******************************************************************************
     *
     * Processing Super Resolution
     *
     *******************************************************************************/
    auto index = 0;
    Mat currentFrame;
    while (true)
    {
        cout << index << "..";
        auto currentStatus = superResolution->NextFrame(currentFrame);

        cv::imshow("High Resolution Frame", currentFrame);

        cv::waitKey(1000);

        char name[50];
        sprintf(name, resultNameFormat, index);
        cv::imwrite(name, currentFrame);

        if (currentStatus == -1)
            break;

        ++index;
    }
    cv::waitKey(0);
    cv::destroyAllWindows();

    cout << endl;
    cout << "All Done!" << endl;

    return 0;
}
