#pragma once
#include "SuperResolutionBase.h"

class SuperResolutionFactory
{
public:
	static Ptr<SuperResolutionBase> CreateSuperResolutionBase()
	{
		return new SuperResolutionBase();
	}
};

