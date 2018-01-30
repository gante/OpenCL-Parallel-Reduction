__kernel __attribute__((reqd_work_group_size(128, 1, 1))) void kernel_2( __global int *A, __global int *B){
//Vantagem: "Convergent Branching" 
// - No if dentro do ciclo for, a decisão dentro de uma wavefront(32 threads) é mais uniforme. 
// Por outras palavras: assim, em quase todas as situações, todos os elementos dentro da waveform seguem o mesmo branch, o que acelera o processamento
	
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
for(unsigned int s=1; s < dim; s *= 2) {
	int index = 2 * s * tid;
	if (index < dim) {
		sdata[index] += sdata[index + s];
	}
	
	 barrier(CLK_LOCAL_MEM_FENCE);
}

 
 // write result for this block to global mem
 if (tid == 0) B[wid] = sdata[0];
};
