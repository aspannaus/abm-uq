__global__ void prescan_arbitrary(float *g_odata, float *g_idata, int n, int powerOfTwo);
__global__ void prescan_arbitrary_unoptimized(float *g_odata, float *g_idata, int n, int powerOfTwo);

__global__ void prescan_large(float *g_odata, float *g_idata, int n, float *sums);
__global__ void prescan_large_unoptimized(float *output, float *input, int n, float *sums);

__global__ void add(float *output, int length, float *n1);
__global__ void add(float *output, int length, float *n1, float *n2);