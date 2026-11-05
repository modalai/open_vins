/*
 * OpenVINS host OpenCL harness: declarations of the Qualcomm recordable-queue
 * extension (cl_qcom_recordable_queues) for libmodal-flow HOST builds.
 *
 * libmodal-flow targets Adreno, whose driver ships <CL/cl_ext_qcom.h> and
 * implements recordable queues natively. Desktop OpenCL platforms (PoCL,
 * Intel compute-runtime, NVIDIA, ...) do not, so the harness provides these
 * declarations plus a portable emulation over core OpenCL 2.0
 * (cmake/modal_flow_host/host_runtime.cpp). Only the subset libmodal-flow uses
 * is declared. Device builds keep using the vendor header; this directory is on
 * the include path only when the host harness is built.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef OV_MODAL_FLOW_HOST_CL_EXT_QCOM_H
#define OV_MODAL_FLOW_HOST_CL_EXT_QCOM_H

#include <CL/cl.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Queue property requesting a recordable queue. The emulation strips it before
 * creating an ordinary in-order queue, so its value only has to be unique. */
#define CL_QUEUE_RECORDABLE_QCOM ((cl_command_queue_properties)1 << 30)

/** Opaque recording of NDRange dispatches captured from a recordable queue. */
typedef struct _cl_recording_qcom *cl_recording_qcom;

/** Replay-time kernel argument update for dispatch `dispatch_index`. */
typedef struct _cl_array_arg_qcom {
  cl_uint dispatch_index;
  cl_uint arg_index;
  size_t arg_size;
  const void *arg_value;
} cl_array_arg_qcom;

/** Replay-time global offset update for dispatch `dispatch_index` (work_dim entries used). */
typedef struct _cl_offset_qcom {
  cl_uint dispatch_index;
  size_t offsets[3];
} cl_offset_qcom;

/** Replay-time global/local work size update for dispatch `dispatch_index`
 *  (points at work_dim sizes; must stay valid for the duration of the call). */
typedef struct _cl_workgroup_qcom {
  cl_uint dispatch_index;
  const size_t *workgroup_size;
} cl_workgroup_qcom;

/** Start recording on a queue created with CL_QUEUE_RECORDABLE_QCOM. NDRange
 *  kernels enqueued on it are captured (not executed) until clEndRecordingQCOM. */
cl_recording_qcom clNewRecordingQCOM(cl_command_queue command_queue, cl_int *errcode_ret);
cl_int clEndRecordingQCOM(cl_recording_qcom recording);
cl_int clRetainRecordingQCOM(cl_recording_qcom recording);
cl_int clReleaseRecordingQCOM(cl_recording_qcom recording);

/** Replay every recorded dispatch, in order, on `command_queue`. The wait list
 *  gates the first dispatch; `event` (optional) completes with the last one. */
cl_int clEnqueueRecordingQCOM(cl_command_queue command_queue, cl_recording_qcom recording, size_t num_args,
                              const cl_array_arg_qcom *arg_array, size_t num_global_offsets,
                              const cl_offset_qcom *global_offset_array, size_t num_global_workgroups,
                              const cl_workgroup_qcom *global_workgroup_array, size_t num_local_workgroups,
                              const cl_workgroup_qcom *local_workgroup_array, cl_uint num_events_in_wait_list,
                              const cl_event *event_wait_list, cl_event *event);

#ifdef __cplusplus
}
#endif

#endif /* OV_MODAL_FLOW_HOST_CL_EXT_QCOM_H */
