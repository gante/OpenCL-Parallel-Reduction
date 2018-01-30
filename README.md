# OpenCL-Parallel-Reduction
OpenCL code for the Parallel Reduction, able to run on FPGAs (with good results on GPUs, only requires minimal code changes for GPUs)

Based on: http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf

Stage by stage optimization of a memory bound problem. 


//////////////////////////////////////////////////////

version 8 vs version 1:

Speedup for GTX860M - 12.85x @ BW=58.46GB/s

Speedup for DE5-Net (Stratix V 5SGXA7) - 30.84x @ BW = 4.35GB/s
