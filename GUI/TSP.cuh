#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>

struct fromCUDA {
	char e;
	int cites[20];
	unsigned long CPU_time;
	unsigned long GPU_time;
};

namespace Wrapper {
	fromCUDA wrapper(int, int *, int);
}

unsigned long long factorial(int);
