#ifndef OCL_UTILS_H
#define OCL_UTILS_H

#include <CL/cl.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#define SINGLE_CHANNEL_TYPE CL_R

/**
 * @brief Simple wrapper around an OpenCL image.
 */
struct ocl_image
{
    cl_mem          img_mem = nullptr;
    unsigned int    w       = 0;
    unsigned int    h       = 0;
    cl_image_format image_format = {0};
};

/**
 * @brief Simple wrapper around a 2D OpenCL buffer
 */
struct ocl_buffer 
{
    cl_mem          buf_mem = nullptr;
    unsigned int    w       = 0;
    unsigned int    h       = 0;
    cl_image_format image_format = {0}; 
};

/**
 * @brief Represents a pyramid of images in OpenCL memory.
 */
struct ocl_pyramid 
{
    int             levels = 0;
    unsigned int    base_w = 0;
    unsigned int    base_h = 0;
    cl_image_format pyramid_format = {0};
    ocl_image*      images = nullptr;
};

/**
 * @brief Tracking-specific OpenCL buffers (e.g. feature points, status, error).
 */
struct ocl_tracking_buffer 
{
    int    n_points     = 0;
    cl_mem prev_pts_buf = nullptr;
    cl_mem next_pts_buf = nullptr;
    cl_mem status_buf   = nullptr;
    cl_mem error_buf    = nullptr;
};

struct ocl_detection_buffer
{
    int    max_points  = 0;
    cl_mem xy_pts_buf  = nullptr;
    cl_mem xyz_pts_buf = nullptr;
};

/**
 * @class OCLTracker
 * @brief Manages OpenCL-based tracking (LK optical flow).
 */
class OCLTracker
{
public:

    int cam_id;
    cl_context        context = nullptr;
    cl_command_queue  queue   = nullptr;
    
    // Kernels used for pyramid construction and point tracking
    cl_kernel track_kernel      = nullptr;
    cl_kernel downfilter_kernel = nullptr;

    // Kernels used for feature point extraction 
    cl_kernel extract_kernel = nullptr;
    cl_kernel nms_kernel     = nullptr;

    // Pointers to pyramids for the "previous" and "next" frames
    ocl_pyramid* prev_pyr = nullptr;
    ocl_pyramid* next_pyr = nullptr;

    // Buffers to store tracking data (points, status, etc.)
    ocl_tracking_buffer tracking_buf;
    
    // Buffer to store default img
    ocl_buffer img_buf;

    // Buffer to store 
    ocl_detection_buffer detection_buf;

    // Constructor
    OCLTracker() = default;

    ~OCLTracker() {
        if (queue) clReleaseCommandQueue(queue);
        if (track_kernel) clReleaseKernel(track_kernel);
        if (downfilter_kernel) clReleaseKernel(downfilter_kernel);
        if (prev_pyr) destroy_pyramid(prev_pyr);
        if (next_pyr) destroy_pyramid(next_pyr);
    }

    int init(cl_context context,
             cl_device_id device,
             cl_program program,
             cl_program detect_program,
             int pyr_levels,
             int base_width,
             int base_height,
             cl_image_format format);

    cv::Mat save_ocl_image(ocl_image* image, std::string output_path);

    void swap_pyr_pointers();
    int build_next_pyramid(const void* frame);
    int run_tracking_step(int n_points, float* prev_pts);

    int read_results(int n_points, float* next_pts_out, uchar* status_out, float* err_out);


private:
    int create_queue(cl_device_id device, cl_context context);
    int build_ocl_kernels(cl_program ocl_program, cl_program detect_program);

    void create_ocl_buf(int w, int h, cl_image_format format);
    ocl_image create_ocl_image(int w, int h, cl_image_format format);

    int create_pyramids(int levels, int base_w, int base_h, cl_image_format format);  
    int create_tracking_buffers(int n_points);
    int create_detection_buffer(int max_points);

    int destroy_ocl_image(ocl_image* image);
    int destroy_pyramid(ocl_pyramid* pyramid);
};

/**
 * @class OCLManager
 * @brief Handles creation of the OpenCL context/program/device and
 *        holds an array of OCLTracker objects for multiple cameras.
 */
class OCLManager
{
    public:
        int num_cams = 3;

        cl_device_id    device  = nullptr;
        cl_context      context = nullptr;
        cl_program      ocl_program = nullptr;
        cl_program      detect_program = nullptr;

        std::string     kernel_code;

        
        OCLTracker* cam_track[3];

        // Constructor
        OCLManager();
        ~OCLManager();

        int init(int n_cams, int width, int height, int pyr_levels);

    private:
        int load_kernel_code(std::string& dst_str);
        int load_detection_kernel(std::string& dst_str);

};


extern std::string kernel_code;


#endif // OCL_UTILS_H