import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.cltypes as cltypes
from pyopencl.elementwise import ElementwiseKernel
from pyopencl.reduction import ReductionKernel

deviceID = 0
platformID = 0

N = 10

dev = cl.get_platforms()[platformID].get_devices()[deviceID]

ctx = cl.Context([dev])
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

a = cl.array.arange(queue, N, dtype=np.float32)
b = cl.array.arange(queue, N, dtype=np.float32)

krnl = ReductionKernel(ctx, np.float32, neutral="0",
        reduce_expr="a+b", map_expr="x[i]*y[i]",
        arguments="__global float *x, __global float *y")

prod_scalaire = krnl(a, b).get()


print(prod_scalaire)