#pragma once
#include <vector>

class Utils
{
public:
	static int CalculateCount(std::vector<bool> value_list, bool value = true);
};

inline int Utils::CalculateCount(std::vector<bool> value_list, bool value)
{
	auto count = 0;
	for (auto currentValue : value_list)
		currentValue == value ? count++ : count;
	return count;
}
