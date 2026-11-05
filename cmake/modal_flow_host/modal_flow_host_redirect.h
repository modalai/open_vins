/*
 * OpenVINS host OpenCL harness: force-included (-include) into libmodal-flow's
 * own translation units ONLY (never into OpenVINS sources or consumers).
 *
 * It routes the few OpenCL entry points whose behaviour must differ on a
 * desktop platform through host_runtime.cpp:
 *   - device/platform selection (libmodal-flow hard-codes CL_DEVICE_TYPE_GPU;
 *     hosts may only have a CPU or accelerator implementation),
 *   - program builds (the kernels rely on OpenCL C 2.0 generic pointers, which
 *     Adreno accepts without -cl-std; desktop compilers need it spelled out),
 *   - command-queue creation (strip CL_QUEUE_RECORDABLE_QCOM),
 *   - kernel argument tracking + NDRange capture, which back the portable
 *     cl_qcom_recordable_queues emulation.
 * The renamed functions keep the exact Khronos prototypes because <CL/cl.h>
 * itself declares them under the new names.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef OV_MODAL_FLOW_HOST_REDIRECT_H
#define OV_MODAL_FLOW_HOST_REDIRECT_H

#define clGetPlatformIDs ov_mfh_clGetPlatformIDs
#define clGetDeviceIDs ov_mfh_clGetDeviceIDs
#define clCreateCommandQueueWithProperties ov_mfh_clCreateCommandQueueWithProperties
#define clBuildProgram ov_mfh_clBuildProgram
#define clCreateKernel ov_mfh_clCreateKernel
#define clSetKernelArg ov_mfh_clSetKernelArg
#define clEnqueueNDRangeKernel ov_mfh_clEnqueueNDRangeKernel

#endif /* OV_MODAL_FLOW_HOST_REDIRECT_H */
