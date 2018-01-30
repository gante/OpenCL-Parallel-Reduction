__kernel __attribute__((reqd_work_group_size(128, 1, 1))) void kernel_5( __global int *A, __global int *B){
//Vantagem: "Unroll the last warp" 
// - As operações são feitas em conjuntos de 32 elementos. Quando se está no final do loop, a trabalhar dentro de conjuntos com no
//		máximo 32 elementos, esperar pelas outras wavefronts é desnecessário! (Aliás, o seu valor é descartável)
	
 //Get the indexes of the local item
 unsigned int tid = get_local_id(0);
 unsigned int wid = get_group_id(0);
 unsigned int dim = get_local_size(0); 
 unsigned int i = wid*dim*2 + tid;
 
 //Declares the shared memory
 __local int sdata[128]; // 128 -> size set for tests
 sdata[tid] = A[i] + A[i+dim];
 
 //syncs the threads (to ensure the local memory is properly loaded)
 barrier(CLK_LOCAL_MEM_FENCE);
 
 // the parallel reduction itself (workgroup = 2^7 -> 7 times)
 for (unsigned int s=dim/2; s>32; s>>=1) {
	if (tid < s) {
		sdata[tid] += sdata[tid + s];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}

if (tid < 32){
	if(dim > 32){
		for (unsigned int s=32; s>0; s>>=1) {
			if (tid < s) {
				sdata[tid] += sdata[tid + s];
			}
		}
	}
	else{
		for (unsigned int s=16; s>0; s>>=1) {
			if (tid < s) {
				sdata[tid] += sdata[tid + s];
			}
		}
	}
}
 
 // write result for this block to global mem
 if (tid == 0) B[wid] = sdata[0];
};
