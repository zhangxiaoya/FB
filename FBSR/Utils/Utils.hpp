#pragma once
#include <vector>
#include <algorithm>

class Utils
{
public:
	static int CalculateCount(const vector<bool> value_list, const bool value = true);
	static Mat GetGaussianKernal(const int kernel_size, const double sigma);
	static void CalculatedMedian(const Mat& source_mat, Mat& median_mat);

private:
	static float GetVectorMedian(vector<float>& value_list);

};

inline int Utils::CalculateCount(const vector<bool> value_list, const bool value)
{
	auto count = 0;
	for (auto currentValue : value_list)
		currentValue == value ? count++ : count;
	return count;
}

inline Mat Utils::GetGaussianKernal(const int kernel_size, const double sigma)
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

inline float Utils::GetVectorMedian(vector<float>& value_list)
{
	sort(value_list.begin(), value_list.end());

	auto len = value_list.size();
	if (len % 2 == 1)
		return value_list[len / 2];

	return float(value_list[len / 2] + value_list[(len - 1) / 2]) / 2.0;
}

inline void Utils::CalculatedMedian(const Mat& source_mat, Mat& median_mat)
{
	for (auto r = 0; r < median_mat.rows; ++r)
	{
		auto dstRowData = median_mat.ptr<float>(r);
		auto srcRowData = source_mat.ptr<float>(r);

		for (auto c = 0; c < median_mat.cols; ++c)
		{
			vector<float> elementVector;

			for (auto i = 0; i < source_mat.channels(); ++i)
				elementVector.push_back(*(srcRowData + c + i));

			dstRowData[c] = GetVectorMedian(elementVector);
		}
	}
}
