import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.cltypes as cltypes

deviceID = 0
platformID = 0
workGroup=None

N = 100

dev = cl.get_platforms()[platformID].get_devices()[deviceID]

ctx = cl.Context([dev])
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

# # 1
data = np.zeros((N,4), dtype=np.float32)
cl_data = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=data)

# 2
# data = np.zeros(N, dtype=cltypes.float4)
# cl_data = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=data)

# # 3
# data = np.empty(N, dtype=cltypes.float4)
# cl_data = cl.array.to_device(queue,data)

# 4
# data = np.empty((N,4), dtype=np.float32)
# cl_data = cl.array.to_device(queue,data)

# # 5
# cl_data=cl.array.Array(queue,(N,1),dtype=cltypes.float4)

# # 6
#cl_data=cl.array.Array(queue,(N,4),dtype=np.float32)



prg = cl.Program(ctx, """
void print_geom()
    {
                int gdims=get_work_dim();
                printf("Number of dimensions : %d\\n",gdims);
                printf("-- global------------------\\n");
                for(int i=0;i<gdims;i++){
                    // Attention le cast est obligatoire sinon get_global_size n'est pas (toujours) évalué
                    printf("   - dim %d : %u en %u workgroups\\n",i,(uint)get_global_size(i),  (uint)get_num_groups(i));
                } 
                printf("-- local-------------------\\n");
                for(int i=0;i<gdims;i++){
                    printf("   - dim %d : %u\\n",i,(uint)get_local_size(i));
                } 


    }
__kernel void   un( __global float4* Data_In, int  N)
{
  int gid = get_global_id(0);

  Data_In[gid].xyzw = 1;
  if(gid==0) print_geom();
}
 """).build()

# 1
prg.un(queue, (N,1), workGroup, cl_data, np.int32(N))
cl.enqueue_copy(queue, data, cl_data)
print(data)

# # 2
# prg.un(queue, (N,1), workGroup, cl_data, np.int32(N))
# cl.enqueue_copy(queue, data, cl_data)
# print(data)

# # 3
# prg.un(queue, (N,1), workGroup, cl_data.data, np.int32(N))
# print(cl_data)

# # 4
# prg.un(queue, (N,1), workGroup, cl_data.data, np.int32(N))
# print(cl_data)

# # 5
# prg.un(queue, (N,1), workGroup, cl_data.data, np.int32(N))
# print(cl_data) # les opérations arithmétiques ne fonctionnent pas (float4)

# # 6
# prg.un(queue, (N,1), workGroup, cl_data.data, np.int32(N))
# print(cl_data*2) # les opérations arithmétiques FONCTIONNENT  (float32)

