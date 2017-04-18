#pragma once
#include <vector>

class Utils
{
public:
	static int CalculateCount(std::vector<bool> value_list, bool value = true);
	static Mat GetGaussianKernal(int kernel_size, double sigma);
};

inline int Utils::CalculateCount(std::vector<bool> value_list, bool value)
{
	auto count = 0;
	for (auto currentValue : value_list)
		currentValue == value ? count++ : count;
	return count;
}

inline Mat Utils::GetGaussianKernal(int kernel_size, double sigma)
{
	auto kernelRadius = (kernel_size - 1) / 2;

	Mat tempKernel(kernel_size, kernel_size, CV_32FC1);
	auto squareSigma = 2.0 * sigma * sigma;

	for (auto i = (-kernelRadius); i <= kernelRadius; i++)
	{
		auto row = i + kernelRadius;
		for (auto j = (-kernelRadius); j <= kernelRadius; j++)
		{
			auto col = j + kernelRadius;
			float v = exp(-(1.0 * i * i + 1.0 * j * j) / squareSigma);
			tempKernel.ptr<float>(row)[col] = v;
		}
	}

	auto elementSum = sum(tempKernel);
	Mat gaussKernel;
	tempKernel.convertTo(gaussKernel, CV_32FC1, (1 / elementSum[0]));

	return gaussKernel;
}
