
extern "C" {
    long sequential_scan(float *output, float *input, int length);
    float blockscan(float *output, float *input, int length, bool bcao);
    float scan(float *output, float *input, int length, bool bcao);

    void scanLargeDeviceArray(float *output, float *input, int length, bool bcao);
    void scanSmallDeviceArray(float *d_out, float *d_in, int length, bool bcao);
    void scanLargeEvenDeviceArray(float *output, float *input, int length, bool bcao);
}
