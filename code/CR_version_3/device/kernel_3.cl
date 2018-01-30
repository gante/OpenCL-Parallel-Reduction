__kernel __attribute__((reqd_work_group_size(128, 1, 1))) void kernel_3( __global int *A, __global int *B){
//Vantagem: "Sequential Addressing" 
// - O acesso à shared memory é sempre feito em endereços contíguos, o que acelera o acesso à mesma
	
 //Get the indexes of the local item
 unsigned int tid = get_local_id(0);
 unsigned int wid = get_group_id(0);
 unsigned int i = get_global_id(0);
 unsigned int dim = get_local_size(0); 
 
 //Declares the shared memory
 __local int sdata[128]; // 128 -> size set for tests
 sdata[tid] = A[i];
 
 //syncs the threads (to ensure the local memory is properly loaded)
 barrier(CLK_LOCAL_MEM_FENCE);
 
 // the parallel reduction itself (workgroup = 2^7 -> 7 times)
 for (unsigned int s=dim/2; s>0; s>>=1) {
	if (tid < s) {
		sdata[tid] += sdata[tid + s];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}

 
 // write result for this block to global mem
 if (tid == 0) B[wid] = sdata[0];
};

