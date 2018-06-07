#pragma once

struct PropsStruct
{
	PropsStruct() :alpha(0.0), beta(0.0), lambda(0.0), P(0), maxIterationCount(20) {}

	double alpha;
	double beta;
	double lambda;
	double P;
	int maxIterationCount;
};