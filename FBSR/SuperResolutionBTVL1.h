#pragma once

#include "FrameBuffer.h"
#include "SuperResolutionBase.h"

class SuperResolutionBTVL1 : public SuperResolutionBase
{
public:
	void Init(Ptr<FrameSource>& frameSource);
	void Process(Ptr<FrameSource>& frameSource, OutputArray outputFrame);

};