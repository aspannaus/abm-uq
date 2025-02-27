#!/usr/bin/env python3

import numpy as np
import numpy.ctypeslib as npct
import ctypes as ct
import os

# setup interface with the scan c-lib
_float_ptr = npct.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
_scan = npct.load_library('libthrustscan.so', os.path.dirname(__file__))
# Define the return type of the C function
_scan.scan.restype = ct.c_float
# Define arguments of the C function
_scan.scan.argtypes = [_float_ptr, _float_ptr, ct.c_int, ct.c_bool]

rng = np.random.default_rng()
N = 50
# wts = np.abs(rng.standard_normal(N).astype(np.float32))
wts = np.linspace(0, 1, num=N, dtype=np.float32)
out = np.zeros(N, dtype=np.float32)

time = _scan.scan(out, wts, ct.c_int(N), ct.c_bool(False))
print(f"Wall time: {time:f}\n")
print("\nOff by one index, parallel is inclusive, numpy is an exclusive sum")
print("inclusive sum is needed for systematic resampling\n")
np_out = np.cumsum(wts)
print(np.allclose(out, np_out))
