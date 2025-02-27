#include <thrust/copy.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/memory.h>
#include <thrust/scan.h>


#include <stdlib.h>
#include <cuda.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "thrust_scan.cuh"


float scan(float *output, float *input, int length, bool gpu) {
	const float arraySize = length * sizeof(float);

	if (gpu) {
        thrust::device_ptr<float> d_in = thrust::device_malloc<float>(length);
        thrust::device_ptr<float> d_out = thrust::device_malloc<float>(length);
        thrust::copy(input, input + length, d_in);

		thrust::inclusive_scan(thrust::device, d_in, d_in + length, d_out);
        thrust::copy(d_out, d_out + length, output);
        
        thrust::device_free(d_in);
        thrust::device_free(d_out);
    } else {
        void* h_in = malloc(arraySize);
        void* h_out = malloc(arraySize);
        memcpy(h_in, input, arraySize);
        memcpy(h_out, output, arraySize);
        
        float* h_in_ptr = static_cast<float*>(thrust::raw_pointer_cast(h_in));
        float* h_out_ptr = static_cast<float*>(thrust::raw_pointer_cast(h_out));

		thrust::inclusive_scan(thrust::host, h_in_ptr, h_in_ptr + length, h_out_ptr);
        thrust::copy(h_out_ptr, h_out_ptr + length, output);

        free(h_in);
        free(h_out);
	}
    return 0;
}
