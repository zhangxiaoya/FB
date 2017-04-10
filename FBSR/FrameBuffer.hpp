#include <opencv2\core\core.hpp>
#include <vector>

using namespace std;
using namespace cv;

class FrameBuffer
{
public:
	FrameBuffer(int bufferSize);
	~FrameBuffer();

protected:
	int bufferSize;
	Mat currentFrame;
	Mat previousFrame;
	vector<Mat> sourceFrames;
};

FrameBuffer::FrameBuffer(int bufferSize) : bufferSize(bufferSize)
{
	sourceFrames.resize(this->bufferSize);
}

FrameBuffer::~FrameBuffer()
{
	currentFrame.release();
	previousFrame.release();
	sourceFrames.clear();
}