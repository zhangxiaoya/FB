#pragma once
#include "SuperResolution/SuperResolutionBase.h"

class SuperResolutionFactory
{
public:
	static Ptr<SuperResolutionBase> CreateSuperResolutionBase()
	{
		return new SuperResolutionBase();
	}
};

