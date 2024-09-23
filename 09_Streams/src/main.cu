#include <cuda_runtime_api.h>
#include <iostream>
#include <algorithm>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include "../../shared/include/utility.h"

// A simple kernel function to keep threads busy for a while
__global__ void busy()
{
	int id = (blockIdx.x * blockDim.x + threadIdx.x);
	clock_t start_time = clock(); 
	samplesutil::WasteTime(1'000'000'000ULL);
  	clock_t stop_time = clock();
	printf("I'm awake! blockIdx.x=%d, threadIdx.x=%d, id=%d, clock=%llu ms\n", blockIdx.x, threadIdx.x, id, (stop_time-start_time));
}

constexpr unsigned int KERNEL_CALLS = 2;

// 默认使用同一个默认的stream，所以2个busy将数序执行。
void test_sequential_default_stream()
{
	std::cout << "Running sequential launches. both default stream." << std::endl;
	// Launch the same kernel several times in a row
	auto t1 = std::chrono::high_resolution_clock::now();
	for (unsigned int i = 0; i < KERNEL_CALLS; i++)
		busy<<<1, 2>>>();
	// Synchronize before continuing to get clear separation in Nsight
	cudaDeviceSynchronize();
	auto t2 = std::chrono::high_resolution_clock::now();
	std::cout << "== time=" << utils::elapse_ms(t2, t1) << " ms" << std::endl;
}

// 使用不同的stream，所以2个busy将并行执行。
void test_different_stream()
{
	std::cout << "\nRunning launches in streams. different stream." << std::endl;
	// Allocate one stream for each kernel to be launched
	auto t1 = std::chrono::high_resolution_clock::now();
	cudaStream_t streams[KERNEL_CALLS];
	for (cudaStream_t &s : streams)
	{
		// Create stream and launch kernel into it
		cudaStreamCreate(&s);
		busy<<<1, 2, 0, s>>>();
	}
	/*
	Destroy all streams. It is fine to do that immediately. Will not
	implicitly synchronize, but the GPU will continue running their
	jobs until they have all been taken care of
	*/
	for (cudaStream_t &s : streams)
		cudaStreamDestroy(s);
	cudaDeviceSynchronize();
	auto t2 = std::chrono::high_resolution_clock::now();

	std::cout << "== time=" << utils::elapse_ms(t2, t1) << " ms" << std::endl;
}

void test_threads_with_different_default_streams()
{
	/*
	If we don't specify a stream, then the kernel is launched into the default
	stream. Also, many operations like cudaDeviceSynchronize and
	cudaStreamSynchronize are submitted to the default stream. Usually, only a
	single default stream is defined per application, meaning that if you don't
	specify streams, you will not be able to benefit from kernels running
	concurrently. Hence, any elaborate CUDA application should be using streams.

	However, if the task can be cleanly separated into CPU threads, there is another
	option: using per-thread default streams. Each thread will use its own default
	stream if we pass the built-in value cudaStreamPerThread as the stream to use.
	Kernels can then run concurrently on the GPU by creating multiple CPU threads.
	Alternatively, you may set the compiler option "--default-stream per-thread".
	This way, CPU threads will use separate default streams if none are specified.
	*/
	std::cout << "\nRunning threads with different default streams" << std::endl;

	// Create mutex, condition variable and kernel counter for communication
	std::mutex mutex;
	std::condition_variable cv;
	unsigned int kernelsLaunched = 0;
	// Allocate sufficient number of threads
	std::thread threads[KERNEL_CALLS];
	// Create a separate thread for each kernel call (task)
	for (std::thread &t : threads)
	{
		t = std::thread([&mutex, &cv, &kernelsLaunched]
						{
			// Launch kernel to thread's default stream
			busy<<<1, 1, 0, cudaStreamPerThread>>>();
			/*
			 Make sure all kernels are submitted before synchronizing,
			 because cudaStreamSynchronize goes into the default 0 stream:
			 busy<1> -> sync<0>(1) -> busy<2> -> sync<0>(2)... may serialize.
			 busy<1> -> busy<2> -> sync<0>(1) -> sync<0>(2)... parallelizes.
			*/
			std::unique_lock<std::mutex> lock(mutex);
			++kernelsLaunched;
			cv.wait(lock, [&kernelsLaunched] { return kernelsLaunched == KERNEL_CALLS; });
			cv.notify_all();
			// Synchronize to wait for printf output
			cudaStreamSynchronize(cudaStreamPerThread); });
	}
	// Wait for all threads to finish launching their kernels in individual streams
	std::for_each(threads, threads + KERNEL_CALLS, [](std::thread &t)
				  { t.join(); });
}

void test_A_B()
{
	/*
	By default, custom created streams will implicitly synchronize with the
	default stream. Consider, e.g., a kernel A running in a custom stream,
	followed by a kernel B in the default stream. If we use cudaStreamCreate
	as above, then A will end before B starts. Alternatively, we may create
	custom streams with the flag cudaStreamNonBlocking. In this case, the
	custom stream will not synchronize with the default stream anymore.
	*/
	cudaStream_t customRegular, customNonblocking;
	cudaStreamCreate(&customRegular);
	cudaStreamCreateWithFlags(&customNonblocking, cudaStreamNonBlocking);

	auto testAB = [](const char *kind, cudaStream_t stream)
	{
		std::cout << "\nLaunching A (custom) -> B (default) with " << kind << " custom stream" << std::endl;
		busy<<<1, 1, 0, stream>>>();
		busy<<<1, 1>>>();
		cudaDeviceSynchronize();
	};

	testAB("regular", customRegular);		   // Will be synchronize.
	testAB("non-blocking", customNonblocking); // Will be asynchronize.

	// Clean up generated streams
	cudaStreamDestroy(customRegular);
	cudaStreamDestroy(customNonblocking);
}
int main()
{
	std::cout << "==== Sample 09 - Streams ====\n" << std::endl;
	/*
	 Expected output: "I'm awake!\n" x 4 x KERNEL_CALLS + 4

	 If you watch the output carefully or analyze the execution of 
	 this program with NVIDIA Nsight Systems, it should show that the 
	 first group of kernels run consecutively, while the second and 
	 third group run in parallel. 
	 
	 Finally, there should be two kernels running sequentially,
	 followed by two kernels running in parallel.
	*/

	test_sequential_default_stream();
	test_different_stream();
	test_threads_with_different_default_streams();
	test_A_B();

	return 0;
}

/*
Exercises:
1) Streams are a great way to bring task parallelism to the GPU. Think of a small
program that can benefit from running two different kernels at the same time and
write it, along with documentation of its inputs/outputs and usefulness.
*/