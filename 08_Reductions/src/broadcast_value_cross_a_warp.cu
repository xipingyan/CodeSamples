#include <stdio.h>
#include <vector>

__global__ void bcast(int arg)
{
    int laneId = threadIdx.x & 0x1f;
    int value;
    if (laneId == 0)
    {
        // Note unused variable for all threads except lane 0
        value = arg;
    }

    value = __shfl_sync(0xffffffff, value, 0); // Synchronize all threads in warp, and get "value" from lane 0
    if (value != arg)
        printf("Thread %d failed.\n", threadIdx.x);

    // Printf status
    if (laneId == 0)
    {
        printf("threadIdx.x=%d, assign value: %d, value = %d\n", threadIdx.x, arg, value);
    }
    else
    {
        printf("threadIdx.x=%d, do nothing, value = %d\n", threadIdx.x, value);
    }
}

__global__ void fast_sum(const float* __restrict dValsPtr) {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float data[32];
    data[threadIdx.x] = (id < 32 ? dValsPtr[id] : 0);

    // printf("id=%d, data[%d]=%f\n", id, threadIdx.x, data[threadIdx.x]);
    __syncwarp();

    // __syncwarp();
    // printf("threadIdx.x=%d, id=%d, data=%f\n", threadIdx.x, id, data[threadIdx.x]);
    
    float x = data[threadIdx.x];
    // Each cuda core x=1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,...,32
    x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 16);
    // First x formula=data[0]+data[0+16]=1+17=18
    // Last x formula=data[31]+data[31+16]=data[31]+data[47-32]=data[31]+data[15]=32+16=48
    // Each cuda core x=1+17,2+18,3+19,4+20,5+21,6+22,7+23,8+24,9+25...,32+16
    
    x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 8);
    x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 4);
    x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 2);
    x += __shfl_sync(0xFFFFFFFF, x, 1);
    __syncwarp();
    printf("threadIdx.x=%d, x=%f\n", threadIdx.x, x);
    if (threadIdx.x == 0)
        printf("Final result:threadIdx.x=%d, x=%f\n", threadIdx.x, x);
}
__host__ void host_fast_sum(const float* __restrict dValsPtr) {
    fast_sum<<<1, 32>>>(dValsPtr);
}

int test_bcast_a_value_cross_a_warp()
{
    // bcast<<<1, 32>>>(1234);
    // cudaDeviceSynchronize();

    float *dValsPtr;
    cudaMalloc(&dValsPtr, 32 * sizeof(float));
    std::vector<float> vals(32, 1);
    float expected_sum = 0;
    for (auto i = 0; i < 32; i++)
    {
        vals[i] = i + 1;
        expected_sum += vals[i];
    }
    printf("Expect val = %f\n", expected_sum);
    cudaMemcpy(dValsPtr, vals.data(), vals.size() * sizeof(float), cudaMemcpyHostToDevice);

    host_fast_sum(dValsPtr);

    cudaFree(dValsPtr);
    // printf("sum = %f\n", sum);
    return 0;
}