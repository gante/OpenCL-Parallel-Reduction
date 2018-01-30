__kernel __attribute__((reqd_work_group_size(128, 1, 1))) void kernel_7( __global int *A, __global int *B){
//Vantagem: "Multiple Adds" 
// O Overhead de carregar a última ronda é grande. Vamos eliminar isso com mais adds no início!
//Valor ideal: 11 rondas de somas no kernel, dado que 11*2 = 22 e temos 2^22 elementos -> 4 rondas de somas "exteriores"

// Cuda - blockdim = OpenCL - localsize
// Cuda - griddim*blockdim = OpenCL - globalsize
	
 //Get the indexes of the local item
 unsigned int tid = get_local_id(0);
 unsigned int wid = get_group_id(0);
 unsigned int dim = get_local_size(0); 
 unsigned int i = wid*dim*2 + tid;
 unsigned int gridSize = get_global_size(0)*2;
 
 //Declares the shared memory
 __local int sdata[128]; // 128 -> size set for tests
 sdata[tid] = 0;
 
 // 2^4 external sums, or 2^3 * 2
for(int aux = 0; aux<8; aux++) {
	sdata[tid] += A[i] + A[i+dim];
	i += gridSize;
}
 
 //syncs the threads (to ensure the local memory is properly loaded)
 barrier(CLK_LOCAL_MEM_FENCE);
 
 // the parallel reduction itself (workgroup = 2^7 -> 7 times)
 if (tid < 64){
	sdata[tid] += sdata[tid + 64]; // s = 64
	barrier(CLK_LOCAL_MEM_FENCE);
	
	
	if (tid < 32){
	sdata[tid] += sdata[tid + 32];
	barrier(CLK_LOCAL_MEM_FENCE);
	
	//if (tid < 16){
		sdata[tid] += sdata[tid + 16];
		barrier(CLK_LOCAL_MEM_FENCE);
		
		//if (tid < 8){
			sdata[tid] += sdata[tid + 8];
			barrier(CLK_LOCAL_MEM_FENCE);
			
			//if (tid < 4){
				sdata[tid] += sdata[tid + 4];
				barrier(CLK_LOCAL_MEM_FENCE);
				
				//if (tid < 2){
					sdata[tid] += sdata[tid + 2];
					barrier(CLK_LOCAL_MEM_FENCE);
					
					if (tid < 1){
						 // write result for this block to global mem
						B[wid] = sdata[tid] + sdata[tid + 1];
					}
				//}
			//}
		//}
	//}
	}
}
 


};