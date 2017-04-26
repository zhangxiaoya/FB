#pragma once
#include "SuperResolutionBTVL1.h"
#include "SuperResolutionBase.h"

class SuperResolutionFactory
{
public:
	static Ptr<SuperResolutionBase> CreateSuperResolutionBTVL1()
	{
		return new SuperResolutionBTVL1();
	}
	static Ptr<SuperResolutionBase> CreateSuperResolutionBase()
	{
		return new SuperResolutionBase();
	}
};

