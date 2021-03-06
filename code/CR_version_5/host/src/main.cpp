// Copyright (C) 2013-2014 Altera Corporation, San Jose, California, USA. All rights reserved. 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this 
// software and associated documentation files (the "Software"), to deal in the Software 
// without restriction, including without limitation the rights to use, copy, modify, merge, 
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to 
// whom the Software is furnished to do so, subject to the following conditions: 
// The above copyright notice and this permission notice shall be included in all copies or 
// substantial portions of the Software. 
//  
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
// OTHER DEALINGS IN THE SOFTWARE. 
//  
// This agreement shall be governed in all respects by the laws of the State of California and 
// by the laws of the United States of America. 

///////////////////////////////////////////////////////////////////////////////////
// This host program runs a "hello world" kernel. This kernel prints out a
// message for if the work-item index matches a kernel argument.
//
// Most of this host program code is the basic elements of a OpenCL host
// program, handling the initialization and cleanup of OpenCL objects. The
// host program also makes queries through the OpenCL API to get various
// properties of the device.
///////////////////////////////////////////////////////////////////////////////////

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include "CL/opencl.h"
#include "AOCL_Utils.h"
#include <stdint.h>

using namespace aocl_utils;

#define STRING_BUFFER_LEN 1024


//2^22 numbers -> 32 bits per integer = 4 Bytes per integer (2^2) -> 2^4 MB = 16MB worth of integers 
#define VECTOR_SIZE 16777216
#define WORK_SIZE 128



// Runtime constants
// Used to define the work set over which this kernel will execute.
//static const size_t work_group_size = 8;  // 8 threads in the demo workgroup
// Defines kernel argument value, which is the workitem ID that will
// execute a printf call
//static const int thread_id_to_output = 2;

// OpenCL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_kernel kernel = NULL;
static cl_program program = NULL;

// Function prototypes
bool init();
void cleanup();
static void device_info_ulong( cl_device_id device, cl_device_info param, const char* name);
static void device_info_uint( cl_device_id device, cl_device_info param, const char* name);
static void device_info_bool( cl_device_id device, cl_device_info param, const char* name);
static void device_info_string( cl_device_id device, cl_device_info param, const char* name);
static void display_device_info( cl_device_id device );

// Entry point.
int main() {
  cl_int status;

  if(!init()) {
    return -1;
  }
  
  //-----------------------------------------------------------------------------------------------------
  // Set the memory elements
  printf("\nAllocating memory on the device.\n");
  
  // Allocate space for vectors	
	int32_t *A = (int32_t*)malloc(sizeof(int32_t)*VECTOR_SIZE);
	int32_t *B = (int32_t*)malloc(sizeof(int32_t)*(VECTOR_SIZE/(WORK_SIZE*2)));
	int32_t *C = (int32_t*)malloc(sizeof(int32_t)*(VECTOR_SIZE/(WORK_SIZE*WORK_SIZE*4)));
	int32_t *D = (int32_t*)malloc(sizeof(int32_t));

	for(int i = 0; i < VECTOR_SIZE; i++){
		A[i] = 1;   //The final result must be "VECTOR_SIZE"
	}
	
	
	// Create memory buffer on the device for the vector
	cl_mem A_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, VECTOR_SIZE * sizeof(int32_t), NULL, &status);
	checkError(status, "Failed to create memory for A");
	cl_mem B_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE, (VECTOR_SIZE/(WORK_SIZE*2)) * sizeof(int32_t), NULL, &status);
	checkError(status, "Failed to create memory for B");
	cl_mem C_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE, (VECTOR_SIZE/(WORK_SIZE*WORK_SIZE*4)) * sizeof(int32_t), NULL, &status);
	checkError(status, "Failed to create memory for C");
	cl_mem D_clmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int32_t), NULL, &status);
	checkError(status, "Failed to create memory for D");
	
	
	// Copy the Buffer A to the device
	cl_event event1, event2, event3, event4, event5;
	status = clEnqueueWriteBuffer(queue, A_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(int32_t), A, 0, NULL, &event1);
	checkError(status, "Failed to copy A");
	
	printf("\nKernel initialization is complete.\n");
	printf("Launching the kernel...\n\n");
	
	//-------------------------------------------------------------------------------------------------------
	//-------------------------------------------------------------------------------------------------------
	// 1st wave
	
	// Set the arguments of the kernel
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&A_clmem);
	checkError(status, "Failed to set kernel arg 0");
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&B_clmem);
	checkError(status, "Failed to set kernel arg 1");


	printf("\nRound 1...");

	// Configure work set over which the kernel will execute
	size_t wgSize[3] = {WORK_SIZE, 1, 1};
	size_t gSize[3] = {VECTOR_SIZE/2, 1, 1};

	// Launch the kernel
	status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, gSize, wgSize, 0, NULL, &event2);
	checkError(status, "Failed to launch kernel");
	
	printf(" Done! \n");
	
	//-------------------------------------------------------------------------------------------------------
	//-------------------------------------------------------------------------------------------------------
	// 2nd wave
	
	// Set the arguments of the kernel
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&B_clmem);
	checkError(status, "Failed to set kernel arg 0");
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&C_clmem);
	checkError(status, "Failed to set kernel arg 1");


	printf("\nRound 2...");

	// Configure work set over which the kernel will execute
	gSize[0] = gSize[0]/(wgSize[0]*2);

	// Launch the kernel
	status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, gSize, wgSize, 0, NULL, &event3);
	checkError(status, "Failed to launch kernel");
	
	printf(" Done! \n");
	
	//-------------------------------------------------------------------------------------------------------
	//-------------------------------------------------------------------------------------------------------
	// 3rd wave
	
	// Set the arguments of the kernel
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&C_clmem);
	checkError(status, "Failed to set kernel arg 0");
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&D_clmem);
	checkError(status, "Failed to set kernel arg 1");


	printf("\nRound 3...");

	// Configure work set over which the kernel will execute
	gSize[0] = gSize[0]/(wgSize[0]*2);
	if(gSize[0] < wgSize[0]) wgSize[0]=gSize[0];

	// Launch the kernel
	status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, gSize, wgSize, 0, NULL, &event4);
	checkError(status, "Failed to launch kernel");
	
	printf(" Done! \n");
	
	//-------------------------------------------------------------------------------------------------------
	
	// Read the cl memory E_clmem on device to the host variable E
	status = clEnqueueReadBuffer(queue, D_clmem, CL_TRUE, 0, sizeof(int32_t), D, 0, NULL, &event5);
	checkError(status, "Failed read the variable D");

	// Wait for command queue to complete pending events
	status = clFlush(queue);
	checkError(status, "Failed to finish");
	
	// Display the result to the screen   
	printf("\nFinal result = %d\n", D[0]);
	
	// Check times
	cl_ulong copy_start, exec_start, exec_end, copy_end;
	double total_time1, total_time2;

	clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_START, sizeof(copy_start), &copy_start, NULL);
	clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_START, sizeof(exec_start), &exec_start, NULL);
	clGetEventProfilingInfo(event4, CL_PROFILING_COMMAND_END, sizeof(exec_end), &exec_end, NULL);
	clGetEventProfilingInfo(event5, CL_PROFILING_COMMAND_END, sizeof(copy_end), &copy_end, NULL);
	total_time1 = copy_end - copy_start;
	total_time2 = exec_end - exec_start;
	printf("\n\nExecution time in milliseconds (with mem. tx.) = %0.3f ms", (total_time1 / 1000000.0) );
	printf("\nExecution time in milliseconds (without mem. tx.) = %0.3f ms", (total_time2 / 1000000.0) );
	printf("\nMemory transfer time = %0.3f ms\n", ( (total_time1-total_time2) / 1000000.0) );


	// Free the resources allocated
	cleanup();
	free(A);
	free(B);
	free(C);
	free(D);

  return 0;
}

/////// HELPER FUNCTIONS ///////

bool init() {
  cl_int status;

  if(!setCwdToExeDir()) {
    return false;
  }

  // Get the OpenCL platform.
  platform = findPlatform("Altera");
  if(platform == NULL) {
    printf("ERROR: Unable to find Altera OpenCL platform.\n");
    return false;
  }

  // User-visible output - Platform information
  {
    char char_buffer[STRING_BUFFER_LEN]; 
    printf("Querying platform for info:\n");
    printf("==========================\n");
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);
  }

  // Query the available OpenCL devices.
  scoped_array<cl_device_id> devices;
  cl_uint num_devices;

  devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));

  // We'll just use the first device.
  device = devices[0];

  // Display some device information.
  display_device_info(device);

  // Create the context.
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the command queue.
  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");

  // Create the program.   #hello_world
  std::string binary_file = getBoardBinaryFile("kernel_5", device);
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  // Create the kernel - name passed in here must match kernel name in the
  // original CL file, that was compiled into an AOCX file using the AOC tool
  const char *kernel_name = "kernel_5";  // Kernel name, as defined in the CL file #hello_world
  kernel = clCreateKernel(program, kernel_name, &status);
  checkError(status, "Failed to create kernel");

  return true;
}

// Free the resources allocated during initialization
void cleanup() {
  if(kernel) {
    clReleaseKernel(kernel);  
  }
  if(program) {
    clReleaseProgram(program);
  }
  if(queue) {
    clReleaseCommandQueue(queue);
  }
  if(context) {
    clReleaseContext(context);
  }
}

// Helper functions to display parameters returned by OpenCL queries
static void device_info_ulong( cl_device_id device, cl_device_info param, const char* name) {
   cl_ulong a;
   clGetDeviceInfo(device, param, sizeof(cl_ulong), &a, NULL);
   printf("%-40s = %lu\n", name, a);
}
static void device_info_uint( cl_device_id device, cl_device_info param, const char* name) {
   cl_uint a;
   clGetDeviceInfo(device, param, sizeof(cl_uint), &a, NULL);
   printf("%-40s = %u\n", name, a);
}
static void device_info_bool( cl_device_id device, cl_device_info param, const char* name) {
   cl_bool a;
   clGetDeviceInfo(device, param, sizeof(cl_bool), &a, NULL);
   printf("%-40s = %s\n", name, (a?"true":"false"));
}
static void device_info_string( cl_device_id device, cl_device_info param, const char* name) {
   char a[STRING_BUFFER_LEN]; 
   clGetDeviceInfo(device, param, STRING_BUFFER_LEN, &a, NULL);
   printf("%-40s = %s\n", name, a);
}

// Query and display OpenCL information on device and runtime environment
static void display_device_info( cl_device_id device ) {

   printf("Querying device for info:\n");
   printf("========================\n");
   device_info_string(device, CL_DEVICE_NAME, "CL_DEVICE_NAME");
   device_info_string(device, CL_DEVICE_VENDOR, "CL_DEVICE_VENDOR");
   device_info_uint(device, CL_DEVICE_VENDOR_ID, "CL_DEVICE_VENDOR_ID");
   device_info_string(device, CL_DEVICE_VERSION, "CL_DEVICE_VERSION");
   device_info_string(device, CL_DRIVER_VERSION, "CL_DRIVER_VERSION");
   device_info_uint(device, CL_DEVICE_ADDRESS_BITS, "CL_DEVICE_ADDRESS_BITS");
   device_info_bool(device, CL_DEVICE_AVAILABLE, "CL_DEVICE_AVAILABLE");
   device_info_bool(device, CL_DEVICE_ENDIAN_LITTLE, "CL_DEVICE_ENDIAN_LITTLE");
   device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE");
   device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE");
   device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_SIZE, "CL_DEVICE_GLOBAL_MEM_SIZE");
   device_info_bool(device, CL_DEVICE_IMAGE_SUPPORT, "CL_DEVICE_IMAGE_SUPPORT");
   device_info_ulong(device, CL_DEVICE_LOCAL_MEM_SIZE, "CL_DEVICE_LOCAL_MEM_SIZE");
   device_info_ulong(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, "CL_DEVICE_MAX_CLOCK_FREQUENCY");
   device_info_ulong(device, CL_DEVICE_MAX_COMPUTE_UNITS, "CL_DEVICE_MAX_COMPUTE_UNITS");
   device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_ARGS, "CL_DEVICE_MAX_CONSTANT_ARGS");
   device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE");
   device_info_uint(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
   device_info_uint(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
   device_info_uint(device, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE");

   {
      cl_command_queue_properties ccp;
      clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &ccp, NULL);
      printf("%-40s = %s\n", "Command queue out of order? ", ((ccp & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)?"true":"false"));
      printf("%-40s = %s\n", "Command queue profiling enabled? ", ((ccp & CL_QUEUE_PROFILING_ENABLE)?"true":"false"));
   }
}

