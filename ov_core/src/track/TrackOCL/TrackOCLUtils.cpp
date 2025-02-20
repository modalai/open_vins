#include "TrackOCLUtils.h"


OCLManager::OCLManager()
{
    
}


OCLManager::~OCLManager() 
{
    if (ocl_program) clReleaseProgram(ocl_program);
    if (context)     clReleaseContext(context);
}


int OCLManager::init(int n_cams, int width, int height, int pyr_levels)
{
    cl_int err;
    cl_uint num_platforms = 0;

    // check available platforms
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) 
    {
        printf("No OpenCL platforms found!\n");
        return 1;
    }

    // get platforms
    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    // select the first platform
    cl_platform_id platform = platforms[0];

    // get available GPU devices
    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) 
    {
        printf("No devices found!\n");
        return 1;
    }

    // select the first device
    std::vector<cl_device_id> devices(num_devices);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
    device = devices[0];

    // create context
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        printf("Could not create context!\n");
        return 1;
    }

    // put kernel code into string
    load_kernel_code(this->kernel_code);

    const char* code_ptr = kernel_code.c_str();
    size_t code_length   = kernel_code.size();
    std::string build_options = "-D WSX=1 -D WSY=1";

    // create the program
    ocl_program = clCreateProgramWithSource(context, 1, &code_ptr, &code_length, &err);
    if (err != CL_SUCCESS) 
    {
        printf("Error creating ocl_program from source: %d\n", err);
        return 1;
    }

    // build program
    err = clBuildProgram(ocl_program, 1, &device, build_options.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) 
    {
        // save build log
        size_t log_size;
        clGetProgramBuildInfo(ocl_program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

        std::vector<char> build_log(log_size);
        clGetProgramBuildInfo(ocl_program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);

        printf("Error building kernel: %s\n", build_log.data());
    }

    // put detection kernel code into string
    load_detection_kernel(this->kernel_code);

    code_ptr = kernel_code.c_str();
    code_length = kernel_code.size();

    // create the program
    detect_program = clCreateProgramWithSource(context, 1, &code_ptr, &code_length, &err);
    if (err != CL_SUCCESS) 
    {
        printf("Error creating detect_program from source: %d\n", err);
        return 1;
    }

    // build program
    err = clBuildProgram(detect_program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) 
    {
        // save build log
        size_t log_size;
        clGetProgramBuildInfo(detect_program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

        std::vector<char> build_log(log_size);
        clGetProgramBuildInfo(detect_program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);

        printf("Error building kernel: %s\n", build_log.data());
    }

    num_cams = n_cams;

    // initialize tracking for cameras
    for (int i = 0; i < num_cams; ++i) 
    {
        cl_image_format format;
        format.image_channel_order = CL_R;
        format.image_channel_data_type = CL_FLOAT;

        cam_track[i] = new OCLTracker();
        cam_track[i]->init(context, device, ocl_program, detect_program, pyr_levels+1, width, height, format);
    }


    return 0;
}


int OCLManager::load_kernel_code(std::string& dst_str)
{
    dst_str = R"CLC(
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Dachuan Zhao, dachuan@multicorewareinc.com
//    Yao Wang, bitwangyaoyao@gmail.com
//    Xiaopeng Fu, fuxiaopeng2222@163.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/



#define GRIDSIZE    3
#define LSx 8
#define LSy 8
// define local memory sizes
#define LM_W (LSx*GRIDSIZE+2)
#define LM_H (LSy*GRIDSIZE+2)
#define BUFFER  (LSx*LSy)
#define BUFFER2 BUFFER>>1

#ifdef CPU

inline void reduce3(float val1, float val2, float val3,  __local float* smem1,  __local float* smem2,  __local float* smem3, int tid)
{
    smem1[tid] = val1;
    smem2[tid] = val2;
    smem3[tid] = val3;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = BUFFER2; i > 0; i >>= 1)
    {
        if(tid < i)
        {
            smem1[tid] += smem1[tid + i];
            smem2[tid] += smem2[tid + i];
            smem3[tid] += smem3[tid + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

inline void reduce2(float val1, float val2, __local float* smem1, __local float* smem2, int tid)
{
    smem1[tid] = val1;
    smem2[tid] = val2;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = BUFFER2; i > 0; i >>= 1)
    {
        if(tid < i)
        {
            smem1[tid] += smem1[tid + i];
            smem2[tid] += smem2[tid + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

inline void reduce1(float val1, __local float* smem1, int tid)
{
    smem1[tid] = val1;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = BUFFER2; i > 0; i >>= 1)
    {
        if(tid < i)
        {
            smem1[tid] += smem1[tid + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
#else
inline void reduce3(float val1, float val2, float val3,
             __local float* smem1, __local float* smem2, __local float* smem3, int tid)
{
    smem1[tid] = val1;
    smem2[tid] = val2;
    smem3[tid] = val3;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 32)
    {
        smem1[tid] += smem1[tid + 32];
        smem2[tid] += smem2[tid + 32];
        smem3[tid] += smem3[tid + 32];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 16)
    {
        smem1[tid] += smem1[tid + 16];
        smem2[tid] += smem2[tid + 16];
        smem3[tid] += smem3[tid + 16];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 8)
    {
        smem1[tid] += smem1[tid + 8];
        smem2[tid] += smem2[tid + 8];
        smem3[tid] += smem3[tid + 8];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 4)
    {
        smem1[tid] += smem1[tid + 4];
        smem2[tid] += smem2[tid + 4];
        smem3[tid] += smem3[tid + 4];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid == 0)
    {
        smem1[0] = (smem1[0] + smem1[1]) + (smem1[2] + smem1[3]);
        smem2[0] = (smem2[0] + smem2[1]) + (smem2[2] + smem2[3]);
        smem3[0] = (smem3[0] + smem3[1]) + (smem3[2] + smem3[3]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

inline void reduce2(float val1, float val2, __local float* smem1, __local float* smem2, int tid)
{
    smem1[tid] = val1;
    smem2[tid] = val2;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 32)
    {
        smem1[tid] += smem1[tid + 32];
        smem2[tid] += smem2[tid + 32];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 16)
    {
        smem1[tid] += smem1[tid + 16];
        smem2[tid] += smem2[tid + 16];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 8)
    {
        smem1[tid] += smem1[tid + 8];
        smem2[tid] += smem2[tid + 8];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 4)
    {
        smem1[tid] += smem1[tid + 4];
        smem2[tid] += smem2[tid + 4];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid == 0)
    {
        smem1[0] = (smem1[0] + smem1[1]) + (smem1[2] + smem1[3]);
        smem2[0] = (smem2[0] + smem2[1]) + (smem2[2] + smem2[3]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

inline void reduce1(float val1, __local float* smem1, int tid)
{
    smem1[tid] = val1;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 32)
    {
        smem1[tid] += smem1[tid + 32];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 16)
    {
        smem1[tid] += smem1[tid + 16];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 8)
    {
        smem1[tid] += smem1[tid + 8];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 4)
    {
        smem1[tid] += smem1[tid + 4];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid == 0)
    {
        smem1[0] = (smem1[0] + smem1[1]) + (smem1[2] + smem1[3]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}
#endif

#define SCALE (1.0f / (1 << 20))
#define	THRESHOLD	0.01f

// Image read mode
__constant sampler_t sampler    = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

// macro to get pixel value from local memory

#define VAL(_y,_x,_yy,_xx)    (IPatchLocal[mad24(((_y) + (_yy)), LM_W, ((_x) + (_xx)))])
inline void SetPatch(local float* IPatchLocal, int TileY, int TileX,
              float* Pch, float* Dx, float* Dy,
              float* A11, float* A12, float* A22, float w)
{
    int xid=get_local_id(0);
    int yid=get_local_id(1);
    int xBase = mad24(TileX, LSx, (xid + 1));
    int yBase = mad24(TileY, LSy, (yid + 1));

    *Pch = VAL(yBase,xBase,0,0);

    *Dx = mad((VAL(yBase,xBase,-1,1) + VAL(yBase,xBase,+1,1) - VAL(yBase,xBase,-1,-1) - VAL(yBase,xBase,+1,-1)), 3.0f, (VAL(yBase,xBase,0,1) - VAL(yBase,xBase,0,-1)) * 10.0f) * w;
    *Dy = mad((VAL(yBase,xBase,1,-1) + VAL(yBase,xBase,1,+1) - VAL(yBase,xBase,-1,-1) - VAL(yBase,xBase,-1,+1)), 3.0f, (VAL(yBase,xBase,1,0) - VAL(yBase,xBase,-1,0)) * 10.0f) * w;

    *A11 = mad(*Dx, *Dx, *A11);
    *A12 = mad(*Dx, *Dy, *A12);
    *A22 = mad(*Dy, *Dy, *A22);
}
#undef VAL

inline void GetPatch(image2d_t J, float x, float y,
              float* Pch, float* Dx, float* Dy,
              float* b1, float* b2)
{
    float diff = read_imagef(J, sampler, (float2)(x,y)).x-*Pch;
    *b1 = mad(diff, *Dx, *b1);
    *b2 = mad(diff, *Dy, *b2);
}

inline void GetError(image2d_t J, const float x, const float y, const float* Pch, float* errval, float w)
{
    float diff = ((((read_imagef(J, sampler, (float2)(x,y)).x * 16384) + 256) / 512) - (((*Pch * 16384) + 256) /512)) * w;
    *errval += fabs(diff);
}


//macro to read pixel value into local memory.
#define READI(_y,_x) IPatchLocal[mad24(mad24((_y), LSy, yid), LM_W, mad24((_x), LSx, xid))] = read_imagef(I, sampler, (float2)(mad((float)(_x), (float)LSx, Point.x + xid - 0.5f), mad((float)(_y), (float)LSy, Point.y + yid - 0.5f))).x;
void ReadPatchIToLocalMem(image2d_t I, float2 Point, local float* IPatchLocal)
{
    int xid=get_local_id(0);
    int yid=get_local_id(1);
    //read (3*LSx)*(3*LSy) window. each macro call read LSx*LSy pixels block
    READI(0,0);READI(0,1);READI(0,2);
    READI(1,0);READI(1,1);READI(1,2);
    READI(2,0);READI(2,1);READI(2,2);
    if(xid<2)
    {// read last 2 columns border. each macro call reads 2*LSy pixels block
        READI(0,3);
        READI(1,3);
        READI(2,3);
    }

    if(yid<2)
    {// read last 2 row. each macro call reads LSx*2 pixels block
        READI(3,0);READI(3,1);READI(3,2);
    }

    if(yid<2 && xid<2)
    {// read right bottom 2x2 corner. one macro call reads 2*2 pixels block
        READI(3,3);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}
#undef READI

__attribute__((reqd_work_group_size(LSx, LSy, 1)))
__kernel void lkSparse(image2d_t I, image2d_t J,
                       __global const float2* prevPts, __global float2* nextPts, __global uchar* status, __global float* err,
                       const int level, const int rows, const int cols, int PATCH_X, int PATCH_Y, int c_winSize_x, int c_winSize_y, int c_iters, char calcErr)
{
    __local float smem1[BUFFER];
    __local float smem2[BUFFER];
    __local float smem3[BUFFER];

    int xid=get_local_id(0);
    int yid=get_local_id(1);
    int gid=get_group_id(0);
    int xsize=get_local_size(0);
    int ysize=get_local_size(1);
    int k;

#ifdef CPU
    float wx0 = 1.0f;
    float wy0 = 1.0f;
    int xBase = mad24(xsize, 2, xid);
    int yBase = mad24(ysize, 2, yid);
    float wx1 = (xBase < c_winSize_x) ? 1 : 0;
    float wy1 = (yBase < c_winSize_y) ? 1 : 0;
#else
#if WSX == 1
    float wx0 = 1.0f;
    int xBase = mad24(xsize, 2, xid);
    float wx1 = (xBase < c_winSize_x) ? 1 : 0;
#else
    int xBase = mad24(xsize, 1, xid);
    float wx0 = (xBase < c_winSize_x) ? 1 : 0;
    float wx1 = 0.0f;
#endif
#if WSY == 1
    float wy0 = 1.0f;
    int yBase = mad24(ysize, 2, yid);
    float wy1 = (yBase < c_winSize_y) ? 1 : 0;
#else
    int yBase = mad24(ysize, 1, yid);
    float wy0 = (yBase < c_winSize_y) ? 1 : 0;
    float wy1 = 0.0f;
#endif
#endif

    float2 c_halfWin = (float2)((c_winSize_x - 1)>>1, (c_winSize_y - 1)>>1);

    const int tid = mad24(yid, xsize, xid);

    float2 prevPt = prevPts[gid] / (float2)(1 << level);

    if (prevPt.x < 0 || prevPt.x >= cols || prevPt.y < 0 || prevPt.y >= rows)
    {
        if (tid == 0 && level == 0)
        {
            status[gid] = 0;
        }

        return;
    }
    prevPt -= c_halfWin;

    // extract the patch from the first image, compute covariation matrix of derivatives

    float A11 = 0;
    float A12 = 0;
    float A22 = 0;

    float I_patch[GRIDSIZE][GRIDSIZE];
    float dIdx_patch[GRIDSIZE][GRIDSIZE];
    float dIdy_patch[GRIDSIZE][GRIDSIZE];

    // local memory to read image with border to calc sobels
    local float IPatchLocal[LM_W*LM_H];
    ReadPatchIToLocalMem(I,prevPt,IPatchLocal);

    {
        SetPatch(IPatchLocal, 0, 0,
                 &I_patch[0][0], &dIdx_patch[0][0], &dIdy_patch[0][0],
                 &A11, &A12, &A22,1);


        SetPatch(IPatchLocal, 0, 1,
                 &I_patch[0][1], &dIdx_patch[0][1], &dIdy_patch[0][1],
                 &A11, &A12, &A22,wx0);

        SetPatch(IPatchLocal, 0, 2,
                    &I_patch[0][2], &dIdx_patch[0][2], &dIdy_patch[0][2],
                    &A11, &A12, &A22,wx1);
    }
    {
        SetPatch(IPatchLocal, 1, 0,
                 &I_patch[1][0], &dIdx_patch[1][0], &dIdy_patch[1][0],
                 &A11, &A12, &A22,wy0);


        SetPatch(IPatchLocal, 1,1,
                 &I_patch[1][1], &dIdx_patch[1][1], &dIdy_patch[1][1],
                 &A11, &A12, &A22,wx0*wy0);

        SetPatch(IPatchLocal, 1,2,
                    &I_patch[1][2], &dIdx_patch[1][2], &dIdy_patch[1][2],
                    &A11, &A12, &A22,wx1*wy0);
    }
    {
        SetPatch(IPatchLocal, 2,0,
                 &I_patch[2][0], &dIdx_patch[2][0], &dIdy_patch[2][0],
                 &A11, &A12, &A22,wy1);


        SetPatch(IPatchLocal, 2,1,
                 &I_patch[2][1], &dIdx_patch[2][1], &dIdy_patch[2][1],
                 &A11, &A12, &A22,wx0*wy1);

        SetPatch(IPatchLocal, 2,2,
                    &I_patch[2][2], &dIdx_patch[2][2], &dIdy_patch[2][2],
                    &A11, &A12, &A22,wx1*wy1);
    }


    reduce3(A11, A12, A22, smem1, smem2, smem3, tid);

    A11 = smem1[0];
    A12 = smem2[0];
    A22 = smem3[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    float D = mad(A11, A22, - A12 * A12);

    if (D < 1.192092896e-07f)
    {
        if (tid == 0 && level == 0)
            status[gid] = 0;

        return;
    }

    A11 /= D;
    A12 /= D;
    A22 /= D;

    prevPt = mad(nextPts[gid], 2.0f, - c_halfWin);

    float2 offset0 = (float2)(xid + 0.5f, yid + 0.5f);
    float2 offset1 = (float2)(xsize, ysize);
    float2 loc0 = prevPt + offset0;
    float2 loc1 = loc0 + offset1;
    float2 loc2 = loc1 + offset1;

    for (k = 0; k < c_iters; ++k)
    {
        if (prevPt.x < -c_halfWin.x || prevPt.x >= cols || prevPt.y < -c_halfWin.y || prevPt.y >= rows)
        {
            if (tid == 0 && level == 0)
                status[gid] = 0;
            break;
        }
        float b1 = 0;
        float b2 = 0;

        {
            GetPatch(J, loc0.x, loc0.y,
                     &I_patch[0][0], &dIdx_patch[0][0], &dIdy_patch[0][0],
                     &b1, &b2);


            GetPatch(J, loc1.x, loc0.y,
                     &I_patch[0][1], &dIdx_patch[0][1], &dIdy_patch[0][1],
                     &b1, &b2);

            GetPatch(J, loc2.x, loc0.y,
                        &I_patch[0][2], &dIdx_patch[0][2], &dIdy_patch[0][2],
                        &b1, &b2);
        }
        {
            GetPatch(J, loc0.x, loc1.y,
                     &I_patch[1][0], &dIdx_patch[1][0], &dIdy_patch[1][0],
                     &b1, &b2);


            GetPatch(J, loc1.x, loc1.y,
                     &I_patch[1][1], &dIdx_patch[1][1], &dIdy_patch[1][1],
                     &b1, &b2);

            GetPatch(J, loc2.x, loc1.y,
                        &I_patch[1][2], &dIdx_patch[1][2], &dIdy_patch[1][2],
                        &b1, &b2);
        }
        {
            GetPatch(J, loc0.x, loc2.y,
                     &I_patch[2][0], &dIdx_patch[2][0], &dIdy_patch[2][0],
                     &b1, &b2);


            GetPatch(J, loc1.x, loc2.y,
                     &I_patch[2][1], &dIdx_patch[2][1], &dIdy_patch[2][1],
                     &b1, &b2);

            GetPatch(J, loc2.x, loc2.y,
                        &I_patch[2][2], &dIdx_patch[2][2], &dIdy_patch[2][2],
                        &b1, &b2);
        }

        reduce2(b1, b2, smem1, smem2, tid);

        b1 = smem1[0];
        b2 = smem2[0];
        barrier(CLK_LOCAL_MEM_FENCE);

        float2 delta;
        delta.x = mad(A12, b2, - A22 * b1) * 32.0f;
        delta.y = mad(A12, b1, - A11 * b2) * 32.0f;

        prevPt += delta;
        loc0 += delta;
        loc1 += delta;
        loc2 += delta;

        if (fabs(delta.x) < THRESHOLD && fabs(delta.y) < THRESHOLD)
            break;
    }

    D = 0.0f;
    if (calcErr)
    {
        {
            GetError(J, loc0.x, loc0.y, &I_patch[0][0], &D, 1);
            GetError(J, loc1.x, loc0.y, &I_patch[0][1], &D, wx0);
        }
        {
            GetError(J, loc0.x, loc1.y, &I_patch[1][0], &D, wy0);
            GetError(J, loc1.x, loc1.y, &I_patch[1][1], &D, wx0*wy0);
        }
        if(xBase < c_winSize_x)
        {
            GetError(J, loc2.x, loc0.y, &I_patch[0][2], &D, wx1);
            GetError(J, loc2.x, loc1.y, &I_patch[1][2], &D, wx1*wy0);
        }
        if(yBase < c_winSize_y)
        {
            GetError(J, loc0.x, loc2.y, &I_patch[2][0], &D, wy1);
            GetError(J, loc1.x, loc2.y, &I_patch[2][1], &D, wx0*wy1);
            if(xBase < c_winSize_x)
                GetError(J, loc2.x, loc2.y, &I_patch[2][2], &D, wx1*wy1);
        }

        reduce1(D, smem1, tid);
    }

    if (tid == 0)
    {
        prevPt += c_halfWin;

        nextPts[gid] = prevPt;

        if (calcErr)
            err[gid] = smem1[0] / (float)(32 * c_winSize_x * c_winSize_y);
    }
}


__constant float weights[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};

__kernel void pyrDown(
    read_only  image2d_t src,      // input image
    write_only image2d_t dst,      // output image
    sampler_t sampler)             // sampler (clamped addressing, nearest filtering)
{
    // Get the coordinates of the pixel in the destination image
    int2 dst_coords = (int2)(get_global_id(0), get_global_id(1));

    // Map the destination pixel back to the source image coordinates
    int2 src_coords = (int2)(dst_coords.x * 2, dst_coords.y * 2);

    // Apply the 5x5 Gaussian kernel centered at src_coords
    float sum = 0.0f;
    for (int dy = -2; dy <= 2; dy++)
    {
        for (int dx = -2; dx <= 2; dx++)
        {
            // Read a single-channel float value from the source image
            float4 pix = read_imagef(src, sampler, src_coords + (int2)(dx, dy));

            // Since the kernel is separable, weight is product of horizontal & vertical components
            float w = weights[dx + 2] * weights[dy + 2];
            sum += pix.x * w;
        }
    }

    // Write the filtered, downsampled value to the destination image
    // Assuming a single-channel image stored in the red component
    write_imagef(dst, dst_coords, (float4)(sum, 0.0f, 0.0f, 1.0f));
}

)CLC";
        
    return 0;
}

int OCLManager::load_detection_kernel(std::string& dst_str)
{
    dst_str = dst_str = R"CLC(
// OpenCL port of the FAST corner detector.
// Copyright (C) 2014, Itseez Inc. See the license at http://opencv.org

inline int cornerScore(__global const uchar* img, int step)
{
    int k, tofs, v = img[0], a0 = 0, b0;
    int d[16];
    #define LOAD2(idx, ofs) \
        tofs = ofs; d[idx] = (short)(v - img[tofs]); d[idx+8] = (short)(v - img[-tofs])
    LOAD2(0, 3);
    LOAD2(1, -step+3);
    LOAD2(2, -step*2+2);
    LOAD2(3, -step*3+1);
    LOAD2(4, -step*3);
    LOAD2(5, -step*3-1);
    LOAD2(6, -step*2-2);
    LOAD2(7, -step-3);

    #pragma unroll
    for( k = 0; k < 16; k += 2 )
    {
        int a = min((int)d[(k+1)&15], (int)d[(k+2)&15]);
        a = min(a, (int)d[(k+3)&15]);
        a = min(a, (int)d[(k+4)&15]);
        a = min(a, (int)d[(k+5)&15]);
        a = min(a, (int)d[(k+6)&15]);
        a = min(a, (int)d[(k+7)&15]);
        a = min(a, (int)d[(k+8)&15]);
        a0 = max(a0, min(a, (int)d[k&15]));
        a0 = max(a0, min(a, (int)d[(k+9)&15]));
    }

    b0 = -a0;
    #pragma unroll
    for( k = 0; k < 16; k += 2 )
    {
        int b = max((int)d[(k+1)&15], (int)d[(k+2)&15]);
        b = max(b, (int)d[(k+3)&15]);
        b = max(b, (int)d[(k+4)&15]);
        b = max(b, (int)d[(k+5)&15]);
        b = max(b, (int)d[(k+6)&15]);
        b = max(b, (int)d[(k+7)&15]);
        b = max(b, (int)d[(k+8)&15]);

        b0 = min(b0, max(b, (int)d[k]));
        b0 = min(b0, max(b, (int)d[(k+9)&15]));
    }

    return -b0-1;
}

__kernel
void FAST_findKeypoints(
    __global const uchar * _img, int step, int img_offset,
    int img_rows, int img_cols,
    volatile __global int* kp_loc,
    int max_keypoints, int threshold )
{
    int j = get_global_id(0) + 3;
    int i = get_global_id(1) + 3;

    if (i < img_rows - 3 && j < img_cols - 3)
    {
        __global const uchar* img = _img + mad24(i, step, j + img_offset);
        int v = img[0], t0 = v - threshold, t1 = v + threshold;
        int k, tofs, v0, v1;
        int m0 = 0, m1 = 0;

        #define UPDATE_MASK(idx, ofs) \
            tofs = ofs; v0 = img[tofs]; v1 = img[-tofs]; \
            m0 |= ((v0 < t0) << idx) | ((v1 < t0) << (8 + idx)); \
            m1 |= ((v0 > t1) << idx) | ((v1 > t1) << (8 + idx))

        UPDATE_MASK(0, 3);
        if( (m0 | m1) == 0 )
            return;

        UPDATE_MASK(2, -step*2+2);
        UPDATE_MASK(4, -step*3);
        UPDATE_MASK(6, -step*2-2);

        #define EVEN_MASK (1+4+16+64)

        if( ((m0 | (m0 >> 8)) & EVEN_MASK) != EVEN_MASK &&
            ((m1 | (m1 >> 8)) & EVEN_MASK) != EVEN_MASK )
            return;

        UPDATE_MASK(1, -step+3);
        UPDATE_MASK(3, -step*3+1);
        UPDATE_MASK(5, -step*3-1);
        UPDATE_MASK(7, -step-3);
        if( ((m0 | (m0 >> 8)) & 255) != 255 &&
            ((m1 | (m1 >> 8)) & 255) != 255 )
            return;

        m0 |= m0 << 16;
        m1 |= m1 << 16;

        #define CHECK0(i) ((m0 & (511 << i)) == (511 << i))
        #define CHECK1(i) ((m1 & (511 << i)) == (511 << i))

        if( CHECK0(0) + CHECK0(1) + CHECK0(2) + CHECK0(3) +
            CHECK0(4) + CHECK0(5) + CHECK0(6) + CHECK0(7) +
            CHECK0(8) + CHECK0(9) + CHECK0(10) + CHECK0(11) +
            CHECK0(12) + CHECK0(13) + CHECK0(14) + CHECK0(15) +

            CHECK1(0) + CHECK1(1) + CHECK1(2) + CHECK1(3) +
            CHECK1(4) + CHECK1(5) + CHECK1(6) + CHECK1(7) +
            CHECK1(8) + CHECK1(9) + CHECK1(10) + CHECK1(11) +
            CHECK1(12) + CHECK1(13) + CHECK1(14) + CHECK1(15) == 0 )
            return;

        {
            int idx = atomic_inc(kp_loc);
            if( idx < max_keypoints )
            {
                kp_loc[1 + 2*idx] = j;
                kp_loc[2 + 2*idx] = i;
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// nonmaxSupression

__kernel
void FAST_nonmaxSupression(
    __global const int* kp_in, volatile __global int* kp_out,
    __global const uchar * _img, int step, int img_offset,
    int rows, int cols, int counter, int max_keypoints)
{
    const int idx = get_global_id(0);

    if (idx < counter)
    {
        int x = kp_in[1 + 2*idx];
        int y = kp_in[2 + 2*idx];
        __global const uchar* img = _img + mad24(y, step, x + img_offset);

        int s = cornerScore(img, step);

        if( (x < 4 || s > cornerScore(img-1, step)) +
            (y < 4 || s > cornerScore(img-step, step)) != 2 )
            return;
        if( (x >= cols - 4 || s > cornerScore(img+1, step)) +
            (y >= rows - 4 || s > cornerScore(img+step, step)) +
            (x < 4 || y < 4 || s > cornerScore(img-step-1, step)) +
            (x >= cols - 4 || y < 4 || s > cornerScore(img-step+1, step)) +
            (x < 4 || y >= rows - 4 || s > cornerScore(img+step-1, step)) +
            (x >= cols - 4 || y >= rows - 4 || s > cornerScore(img+step+1, step)) == 6)
        {
            int new_idx = atomic_inc(kp_out);
            if( new_idx < max_keypoints )
            {
                kp_out[1 + 3*new_idx] = x;
                kp_out[2 + 3*new_idx] = y;
                kp_out[3 + 3*new_idx] = s;
            }
        }
    }
}

)CLC";
        
    return 0;
}

int OCLTracker::init(cl_context context, cl_device_id device, cl_program program, cl_program detect_program, 
                   int pyr_levels, int base_width, int base_height, cl_image_format format)
{
    this->context = context;
    build_ocl_kernels(program, detect_program);
    create_queue(device, context);
    create_pyramids(pyr_levels, base_width, base_height, format);
    create_ocl_buf(base_width, base_height, format);
    create_detection_buffer(10000);
    create_tracking_buffers(100);
    
    return 0;
}

int OCLTracker::create_queue(cl_device_id device, cl_context context)
{
    cl_int err;

    this->context = context;
    this->queue = clCreateCommandQueue(context, device, 0, &err);
   
    if (err != CL_SUCCESS) 
    {
        printf("Failed to create command queue for OCLTracker\n");
    }

    return 0;
}


int OCLTracker::build_ocl_kernels(cl_program program, cl_program detect_program)
{
    cl_int err;

    this->downfilter_kernel = clCreateKernel(program, "pyrDown", &err);
    if (err != CL_SUCCESS) 
    {
        printf("Error creating downfilter_kernel from program!\n");
        return 1;
    }

    this->track_kernel = clCreateKernel(program, "lkSparse", &err);
    if (err != CL_SUCCESS) 
    {
        printf("Error creating track_kernel from program!\n");
        return 1;
    }

    this->extract_kernel = clCreateKernel(detect_program, "FAST_findKeypoints", &err);
    if (err != CL_SUCCESS) 
    {
        printf("Error creating extract_kernel from program!\n");
        return 1;
    }

    this->nms_kernel = clCreateKernel(detect_program, "FAST_nonmaxSupression", &err);
    if (err != CL_SUCCESS) 
    {
        printf("Error creating nms_kernel from program!\n");
        return 1;
    }

    return 0;
}


ocl_image OCLTracker::create_ocl_image(int w, int h, cl_image_format format)
{
    cl_int err;
    ocl_image image;

    image.w = w;
    image.h = h;
    image.image_format = format;

    cl_image_desc desc = {};
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = w;
    desc.image_height = h;
    desc.image_row_pitch = 0; // Let OpenCL handle row pitch

    image.img_mem = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, nullptr, &err);

    if (err != CL_SUCCESS) 
    {
        throw std::runtime_error("Failed to create OpenCL image: " + std::to_string(err));
    }

    return image;
}


void OCLTracker::create_ocl_buf(int w, int h, cl_image_format format)
{
    img_buf.w = w;
    img_buf.h = h;
    img_buf.image_format = format;

    // assuming data is single channel, 8 bit data
    size_t num_channels = 1;
    size_t data_bytes   = sizeof(uint8_t);

    cl_int err;
    img_buf.buf_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, w * h * num_channels * data_bytes, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        throw std::runtime_error("Failed to create OpenCL buffer: " + std::to_string(err));
    }

    return;
}


cv::Mat OCLTracker::save_ocl_image(ocl_image* image, std::string output_path)
{
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {image->w, image->h, 1};
    cv::Mat output(image->h, image->w, CV_32F);

    cl_int err = clEnqueueReadImage(this->queue, image->img_mem, CL_TRUE, origin, region, 0, 0, output.data, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to read OpenCL image: " << err << std::endl;
        return cv::Mat();
    }

    cv::imwrite(output_path, output);
    return output;
}


int OCLTracker::destroy_ocl_image(ocl_image* image)
{
    if (image) 
    {
        if (image->img_mem) 
        {
            clReleaseMemObject(image->img_mem);
        }

        free(image);
        return 0;
    }
    return -1;
}


int OCLTracker::create_pyramids(int levels, int base_w, int base_h, cl_image_format format)
{
    // allocate memory for prev_pyr
    prev_pyr = (ocl_pyramid*)malloc(sizeof(ocl_pyramid));
    if (!prev_pyr) 
    {
        std::cerr << "Failed to allocate memory for prev_pyr" << std::endl;
        return 1;
    }
    
    // allocate memory for next_pyr
    next_pyr = (ocl_pyramid*)malloc(sizeof(ocl_pyramid));
    if (!prev_pyr) 
    {
        std::cerr << "Failed to allocate memory for next_pyr" << std::endl;
        return 1;
    }


    // set values for 
    prev_pyr->levels = next_pyr->levels = levels;
    prev_pyr->base_w = next_pyr->base_w = base_w;
    prev_pyr->base_h = next_pyr->base_h = base_h;
    prev_pyr->pyramid_format = next_pyr->pyramid_format = format;


    prev_pyr->images = (ocl_image*)malloc(levels * sizeof(ocl_image));
    if (!prev_pyr->images) 
    {
        std::cerr << "Failed to allocate memory for prev_pyr images" << std::endl;
        free(prev_pyr);
        return 1;
    }


    next_pyr->images = (ocl_image*)malloc(levels * sizeof(ocl_image));
    if (!next_pyr->images) 
    {
        std::cerr << "Failed to allocate memory for next_pyr images" << std::endl;
        free(prev_pyr->images);
        free(prev_pyr);
        return 1;
    }


    for (int i = 0; i < levels; ++i) 
    {
        unsigned int level_w = base_w >> i;
        unsigned int level_h = base_h >> i;

        try 
        {
            prev_pyr->images[i] = create_ocl_image(level_w, level_h, format);
            next_pyr->images[i] = create_ocl_image(level_w, level_h, format);
        } 
        catch (const std::runtime_error& e)
        {
            std::cerr << e.what() << std::endl;
            destroy_pyramid(prev_pyr);
            destroy_pyramid(next_pyr);
            return 1;
        }
    }

    return 0;
}


int OCLTracker::build_next_pyramid(const void* frame)
{
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {next_pyr->base_w, next_pyr->base_h, 1};

    cl_int err = clEnqueueWriteImage(this->queue, 
                                     next_pyr->images[0].img_mem,
                                     CL_TRUE, origin, region, 0, 0, 
                                     frame, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) 
    {
        std::cerr << "Failed to write image to next_pyr level 0: " << err << std::endl;
        return -1;
    }

    cl_sampler sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP, CL_FILTER_NEAREST, &err);

    for (int i = 1; i < next_pyr->levels; ++i) 
    {
        clSetKernelArg(this->downfilter_kernel, 0, sizeof(cl_mem), &next_pyr->images[i-1].img_mem);
        clSetKernelArg(this->downfilter_kernel, 1, sizeof(cl_mem), &next_pyr->images[i].img_mem);
        clSetKernelArg(this->downfilter_kernel, 2, sizeof(cl_sampler), &sampler);

        size_t global_size[2] = {next_pyr->images[i].w, next_pyr->images[i].h};
        err = clEnqueueNDRangeKernel(this->queue, this->downfilter_kernel, 2, nullptr, global_size, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) 
        {
            std::cerr << "Failed to run downsample kernel for level " << i << ": " << err << std::endl;
            return -1;
        }
    }

    return 0;
}


int OCLTracker::destroy_pyramid(ocl_pyramid* pyramid)
{
    if (pyramid) 
    {
        if (pyramid->images) 
        {
            for (int i = 0; i < pyramid->levels; ++i) 
            {
                destroy_ocl_image(&pyramid->images[i]);
            }
            
            free(pyramid->images);
        }

        free(pyramid);
        return 0;
    }
    return -1;   
}


int OCLTracker::create_tracking_buffers(int n_points)
{
    tracking_buf.n_points = n_points;

    cl_int err;
    tracking_buf.prev_pts_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, n_points * sizeof(float) * 2, nullptr, &err);
    tracking_buf.next_pts_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, n_points * sizeof(float) * 2, nullptr, &err);
    tracking_buf.status_buf   = clCreateBuffer(context, CL_MEM_READ_WRITE, n_points * sizeof(cl_uchar),  nullptr, &err);
    tracking_buf.error_buf    = clCreateBuffer(context, CL_MEM_READ_WRITE, n_points * sizeof(cl_float),  nullptr, &err);

    if (err != CL_SUCCESS) 
    {
        printf("Failed to create buffers for tracking: %d\n", err);
        return -1;
    }

    return 0;
}


int OCLTracker::create_detection_buffer(int max_points)
{
    detection_buf.max_points = max_points;

    cl_int err;
    detection_buf.xy_pts_buf  = clCreateBuffer(context, CL_MEM_READ_WRITE, (max_points * 2 + 1) * sizeof(int), NULL, &err);
    detection_buf.xyz_pts_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, (max_points * 3 + 1) * sizeof(int), NULL, &err);

    if (err != CL_SUCCESS) 
    {
        printf("Failed to create buffers for detection: %d\n", err);
        return -1;
    }

    return 0;
}


int OCLTracker::run_tracking_step(int n_points, float* prev_pts)
{
    size_t pts_buf_size = n_points * sizeof(float) * 2;
    size_t status_buf_size = n_points * sizeof(uchar);

    uchar status[n_points];

    int pyr_levels = prev_pyr->levels - 1;

    // next points need to start as prev points scaled to smallest pyramid level
    float* next_pts = (float*)malloc(pts_buf_size);
    for (int i = 0; i < n_points; i++)
    {
        next_pts[i*2]   = (prev_pts[i*2]   / (1 << pyr_levels)) / 2.f;
        next_pts[i*2+1] = (prev_pts[i*2+1] / (1 << pyr_levels)) / 2.f;
        status[i] = 1;
    }

    // write tracking input to GPU
    clEnqueueWriteBuffer(this->queue, this->tracking_buf.prev_pts_buf, CL_TRUE, 0, pts_buf_size,    prev_pts, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(this->queue, this->tracking_buf.next_pts_buf, CL_TRUE, 0, pts_buf_size,    next_pts, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(this->queue, this->tracking_buf.status_buf,   CL_TRUE, 0, status_buf_size, status,   0, nullptr, nullptr);
    free(next_pts);
    
    int patch_x = 8;
    int patch_y = 8;
    int c_winSize_x = 21;
    int c_winSize_y = 21;
    int c_iters = 30;


    // run tracking step for number of pyramid levels
    for (int level = pyr_levels; level >= 0; level--)
    {
        size_t localThreads[3]  = {8, 8};
        size_t globalThreads[3] = {n_points * 8, 8};
        char calcErr = (0 == level) ? 1 : 0;

        cl_int err;
        err  = clSetKernelArg(this->track_kernel, 0,  sizeof(cl_mem), &prev_pyr->images[level].img_mem);
        err |= clSetKernelArg(this->track_kernel, 1,  sizeof(cl_mem), &next_pyr->images[level].img_mem);
        err |= clSetKernelArg(this->track_kernel, 2,  sizeof(cl_mem), &tracking_buf.prev_pts_buf);
        err |= clSetKernelArg(this->track_kernel, 3,  sizeof(cl_mem), &tracking_buf.next_pts_buf);
        err |= clSetKernelArg(this->track_kernel, 4,  sizeof(cl_mem), &tracking_buf.status_buf);
        err |= clSetKernelArg(this->track_kernel, 5,  sizeof(cl_mem), &tracking_buf.error_buf);
        err |= clSetKernelArg(this->track_kernel, 6,  sizeof(int),    &level);
        err |= clSetKernelArg(this->track_kernel, 7,  sizeof(int),    &prev_pyr->images[level].h);
        err |= clSetKernelArg(this->track_kernel, 8,  sizeof(int),    &prev_pyr->images[level].w);
        err |= clSetKernelArg(this->track_kernel, 9,  sizeof(int),    &patch_x);
        err |= clSetKernelArg(this->track_kernel, 10, sizeof(int),    &patch_y);
        err |= clSetKernelArg(this->track_kernel, 11, sizeof(int),    &c_winSize_x);
        err |= clSetKernelArg(this->track_kernel, 12, sizeof(int),    &c_winSize_y);
        err |= clSetKernelArg(this->track_kernel, 13, sizeof(int),    &c_iters);
        err |= clSetKernelArg(this->track_kernel, 14, sizeof(char),   &calcErr);

        if (err != CL_SUCCESS) 
        {
            printf("Error setting kernel args: %d\n", err);
        }

        err = clEnqueueNDRangeKernel(this->queue, this->track_kernel, 2, nullptr, globalThreads, localThreads, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) 
        {
            printf("Error running kernel: %d\n", err);
        }
    }

    clFinish(this->queue);

    return 0;
}


int OCLTracker::read_results(int n_points, float* next_pts_out, uchar* status_out, float* err_out)
{
    size_t pts_buf_size    = n_points * sizeof(float) * 2;
    size_t status_buf_size = n_points * sizeof(uchar);
    size_t err_buf_size    = n_points * sizeof(float);

    cl_int err;
    err  = clEnqueueReadBuffer(queue, tracking_buf.next_pts_buf, CL_TRUE, 0, pts_buf_size,    next_pts_out, 0, nullptr, nullptr);
    err |= clEnqueueReadBuffer(queue, tracking_buf.status_buf,   CL_TRUE, 0, status_buf_size, status_out,   0, nullptr, nullptr);
    err |= clEnqueueReadBuffer(queue, tracking_buf.error_buf,    CL_TRUE, 0, err_buf_size,    err_out,      0, nullptr, nullptr);

    if (err != CL_SUCCESS) 
    {
        printf("Error reading buffer for results: %d\n", err);
        return -1;
    }

    return 0;
}


void OCLTracker::swap_pyr_pointers()
{
    std::swap(prev_pyr, next_pyr);
}