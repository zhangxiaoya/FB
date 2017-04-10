#pragma once
#include "SuperResolutionBTVL1.h"

class SuperResolutionFactory
{
public:
	static Ptr<SuperResolutionBase> CreateSuperResolutionBTVL1()
	{
		return new SuperResolutionBTVL1();
	}
};

