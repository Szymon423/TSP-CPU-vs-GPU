﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <algorithm>
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <MMSystem.h>

#pragma comment(lib, "winmm.lib")

// Function Prototypes
long long Fact(long long n);
void check(char* arrDest, long long Max);
void display(char* arrDest, long long Max);
bool InitMMTimer(UINT wTimerRes);
void DestroyMMTimer(UINT wTimerRes, bool init);

#define NUMBER_OF_ELEMENTS 5
#define BLOCK_DIM 1024
#define OFFSET 0
// When MAX_PERM = 0, means find all permutations
#define MAX_PERM 0

__constant__ long long arr[20][20] = {
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0} ,
	{ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0} ,
	{ 0, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0} ,
	{ 0, 6, 12, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0} ,
	{ 0, 24, 48, 72, 96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0} ,
	{ 0, 120, 240, 360, 480, 600, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0} ,
	{ 0, 720, 1440, 2160, 2880, 3600, 4320, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0} ,
	{ 0, 5040, 10080, 15120, 20160, 25200, 30240, 35280, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0} ,
	{ 0, 40320, 80640, 120960, 161280, 201600, 241920, 282240, 322560, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0} ,
	{ 0, 362880, 725760, 1088640, 1451520, 1814400, 2177280, 2540160, 2903040, 3265920, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0} ,
	{ 0, 3628800, 7257600, 10886400, 14515200, 18144000, 21772800, 25401600, 29030400, 32659200, 36288000, 0, 0, 0, 0, 0, 0, 0, 0, 0} ,
	{ 0, 39916800, 79833600, 119750400, 159667200, 199584000, 239500800, 279417600, 319334400, 359251200, 399168000, 439084800, 0, 0, 0, 0, 0, 0, 0, 0} ,
	{ 0, 479001600, 958003200, 1437004800, 1916006400, 2395008000, 2874009600, 3353011200, 3832012800, 4311014400, 4790016000, 5269017600, 5748019200, 0, 0, 0, 0, 0, 0, 0} ,
	{ 0, 6227020800, 12454041600, 18681062400, 24908083200, 31135104000, 37362124800, 43589145600, 49816166400, 56043187200, 62270208000, 68497228800, 74724249600, 80951270400, 0, 0, 0, 0, 0, 0} ,
	{ 0, 87178291200, 174356582400, 261534873600, 348713164800, 435891456000, 523069747200, 610248038400, 697426329600, 784604620800, 871782912000, 958961203200, 1046139494400, 1133317785600, 1220496076800, 0, 0, 0, 0, 0} ,
	{ 0, 1307674368000, 2615348736000, 3923023104000, 5230697472000, 6538371840000, 7846046208000, 9153720576000, 10461394944000, 11769069312000, 13076743680000, 14384418048000, 15692092416000, 16999766784000, 18307441152000, 19615115520000, 0, 0, 0, 0} ,
	{ 0, 20922789888000, 41845579776000, 62768369664000, 83691159552000, 104613949440000, 125536739328000, 146459529216000, 167382319104000, 188305108992000, 209227898880000, 230150688768000, 251073478656000, 271996268544000, 292919058432000, 313841848320000, 334764638208000, 0, 0, 0} ,
	{ 0, 355687428096000, 711374856192000, 1067062284288000, 1422749712384000, 1778437140480000, 2134124568576000, 2489811996672000, 2845499424768000, 3201186852864000, 3556874280960000, 3912561709056000, 4268249137152000, 4623936565248000, 4979623993344000, 5335311421440000, 5690998849536000, 6046686277632000, 0, 0} ,
	{ 0, 6402373705728000, 12804747411456000, 19207121117184000, 25609494822912000, 32011868528640000, 38414242234368000, 44816615940096000, 51218989645824000, 57621363351552000, 64023737057280000, 70426110763008000, 76828484468736000, 83230858174464000, 89633231880192000, 96035605585920000, 102437979291648000, 108840352997376000, 115242726703104000, 0} ,
	{ 0, 121645100408832000, 243290200817664000, 364935301226496000, 486580401635328000, 608225502044160000, 729870602452992000, 851515702861824000, 973160803270656000, 1094805903679488000, 1216451004088320000, 1338096104497152000, 1459741204905984000, 1581386305314816000, 1703031405723648000, 1824676506132480000, 1946321606541312000, 2067966706950144000, 2189611807358976000, 2311256907767808000}
};

cudaError_t PermuteWithCuda(char* arrDest, long long* offset, long long* Max);

__global__ void Permute(char* arrDest, long long* offset, long long* Max)
{
	long long index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= *Max)
		return;

	long long tmpindex = index;

	index += *offset;

	char arrSrc[NUMBER_OF_ELEMENTS];
	char arrTaken[NUMBER_OF_ELEMENTS];
#pragma unroll
	for (char i = 0; i < NUMBER_OF_ELEMENTS; ++i)
	{
		arrSrc[i] = i;
		arrTaken[i] = 0;
	}

	char size = NUMBER_OF_ELEMENTS;
#pragma unroll
	for (char i = NUMBER_OF_ELEMENTS - 1; i >= 0; --i)
	{
		for (char j = i; j >= 0; --j)
		{
			if (index >= arr[i][j])
			{
				char foundcnt = 0;
				index = index - arr[i][j];
				for (char k = 0; k < NUMBER_OF_ELEMENTS; ++k)
				{
					if (arrTaken[k] == 0) // not taken
					{
						if (foundcnt == j)
						{
							arrTaken[k] = 1; // set to taken
							arrDest[(tmpindex * NUMBER_OF_ELEMENTS) + (NUMBER_OF_ELEMENTS - size)] = arrSrc[k];
							break;
						}
						foundcnt++;
					}
				}
				break;
			}
		}
		--size;
	}

}

int main()
{
	long long Max = 0;

	// obliczenie ilości dostępnych możliwości n!
	if (MAX_PERM == 0)
		Max = Fact(NUMBER_OF_ELEMENTS);
	else
		Max = MAX_PERM;

	long long offset = OFFSET;
	char* arrDest = new char[Max * NUMBER_OF_ELEMENTS];

	// Add vectors in parallel.
	cudaError_t cudaStatus = PermuteWithCuda(arrDest, &offset, &Max);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "PermuteWithCuda failed!");
		return 1;
	}
	check(arrDest, Max);
	display(arrDest, Max);

	printf("\nExecuted program successfully.\n");

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Parallel Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	delete[] arrDest;
	return 0;
}

cudaError_t PermuteWithCuda(char* arrDest, long long* offset, long long* Max)
{
	char* dev_arrDest = 0;
	long long* dev_offset = 0;
	long long* dev_Max = 0;

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers.
	cudaStatus = cudaMalloc((void**)&dev_arrDest, (*Max) * NUMBER_OF_ELEMENTS);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_offset, sizeof(long long));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_Max, sizeof(long long));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy variable from host memory to GPU buffer.
	cudaStatus = cudaMemcpy(dev_offset, offset, sizeof(long long), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Copy variable from host memory to GPU buffer.
	cudaStatus = cudaMemcpy(dev_Max, Max, sizeof(long long), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	int blocks = (*Max) / BLOCK_DIM;
	if ((*Max) % BLOCK_DIM != 0)
		++blocks;

	++blocks;

	UINT wTimerRes = 0;
	bool init = InitMMTimer(wTimerRes);
	DWORD startTime = timeGetTime();

	Permute << <blocks, BLOCK_DIM >> > (dev_arrDest, dev_offset, dev_Max);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	DWORD endTime = timeGetTime();
	printf("Timing: %dms\n", endTime - startTime);

	DestroyMMTimer(wTimerRes, init);

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(arrDest, dev_arrDest, (*Max) * NUMBER_OF_ELEMENTS, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_arrDest);
	cudaFree(dev_offset);
	cudaFree(dev_Max);

	return cudaStatus;
}

long long Fact(long long n)
{
	long long fact = 1;
	for (long long i = 2; i <= n; ++i)
	{
		fact *= i;
	}

	return fact;
}

void check(char* arrDest, long long Max)
{
	printf("\nChecking...\n");

	char check[NUMBER_OF_ELEMENTS];
	for (int i = 0; i < NUMBER_OF_ELEMENTS; ++i)
	{
		check[i] = i;
	}

	if (OFFSET != 0)
	{
		for (int i = 0; i < OFFSET; ++i)
		{
			std::next_permutation(check, check + NUMBER_OF_ELEMENTS);
		}
	}

	for (int i = 0; i < Max; ++i)
	{
		for (int j = 0; j < NUMBER_OF_ELEMENTS; ++j)
		{
			if (arrDest[i * NUMBER_OF_ELEMENTS + j] != check[j])
			{
				fprintf(stderr, "Diff check failed at %d!", i);
				return;
			}
		}

		std::next_permutation(check, check + NUMBER_OF_ELEMENTS);
	}

}

void display(char* arrDest, long long Max)
{
	for (int i = 0; i < Max; ++i)
	{
		for (int j = 0; j < NUMBER_OF_ELEMENTS; ++j)
			printf("%d", arrDest[i * NUMBER_OF_ELEMENTS + j]);
		printf("\n");
	}
}

bool InitMMTimer(UINT wTimerRes)
{
	TIMECAPS tc;

	if (timeGetDevCaps(&tc, sizeof(TIMECAPS)) != TIMERR_NOERROR)
	{
		return false;
	}

	wTimerRes = min(max(tc.wPeriodMin, 1), tc.wPeriodMax);
	timeBeginPeriod(wTimerRes);

	return true;
}

void DestroyMMTimer(UINT wTimerRes, bool init)
{
	if (init)
		timeEndPeriod(wTimerRes);
}