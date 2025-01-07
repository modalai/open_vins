#ifndef VOXL_OCL_TRACKING_H
#define VOXL_OCL_TRACKING_H

#include <CL/cl.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#define SINGLE_CHANNEL_TYPE CL_R


struct ocl_image {
    cl_mem         image_mem;
    unsigned int   w;
    unsigned int   h;
    cl_image_format image_format;
};

struct ocl_buffer { 
    cl_mem         mem;
    unsigned int   w;
    unsigned int   h;
    cl_image_format image_format; 
};

struct ocl_pyramid {
    int levels;
    unsigned int base_w;
    unsigned int base_h;
    cl_image_format pyramid_format;
    ocl_image* images;
};

struct ocl_tracking_buffer {
    int n_points;
    cl_mem prev_pts_buf;
    cl_mem next_pts_buf;
    cl_mem status_buf;
    cl_mem error_buf;
};

namespace ov_core
{


class TrackOCL
{
    public:

        int cam_id;
        
        // cl_device_id      device;
        cl_context        context;
        cl_command_queue  queue;
        
        cl_kernel track_kernel;
        cl_kernel downfilter_kernel;

        ocl_pyramid* prev_pyr;
        ocl_pyramid* next_pyr;

        ocl_tracking_buffer tracking_buf;
        
        // Constructor
        TrackOCL() = default;

        ~TrackOCL() {
            if (queue) clReleaseCommandQueue(queue);
        }

    
        int create_queue(cl_device_id device, cl_context context);
        int build_ocl_kernels(cl_program ocl_program);

        ocl_image create_ocl_image(int w, int h, cl_image_format format);
        int destroy_ocl_image(ocl_image* image);
        cv::Mat save_ocl_image(ocl_image* image, std::string output_path);
    
        int create_pyramids(int levels, int base_w, int base_h, cl_image_format format);    
        int build_pyramid(void* frame, ocl_pyramid* pyramid);
        int destroy_pyramid(ocl_pyramid* pyramid);

        int create_tracking_buffers(int n_points);
        int run_tracking_step(ocl_pyramid* prev_pyr, ocl_pyramid* next_pyr, ocl_tracking_buffer* tracking_buf, 
                                int pyr_levels, int n_points, float* prev_pts);
        void swap_pyr_pointers(ocl_pyramid* pyr1, ocl_pyramid* pyr2);
};


class OCLManager
{
    public:
        int num_cams = 3;

        cl_device_id    device;
        cl_context      context;

        cl_program      ocl_program;

        std::string     kernel_code;

        cl_kernel       track_kernel;
        cl_kernel       downfilter_kernel;

        TrackOCL* cam_track[3];

        // Constructor
        OCLManager();

        // Destructor
        ~OCLManager();

    private:
        int init(void);
        int load_kernel_code(std::string& dst_str);

};

}


extern std::string kernel_code;


#endif // VOXL_GPU_FEATURE_TRACKER_H