/*
 * OpenVINS host OpenCL harness runtime for libmodal-flow host builds.
 *
 * 1) Device policy. libmodal-flow's OclDevice requests CL_DEVICE_TYPE_GPU.
 *    Desktop hosts may expose only a CPU (PoCL) or accelerator device, so GPU
 *    requests follow OV_OPENCL_DEVICE_TYPE:
 *      unset/auto  GPU when any platform has one, else every device of the
 *                  platform (CPU/accelerator)
 *      gpu|cpu|accelerator|all   force that device type
 *    OV_OPENCL_PLATFORM restricts platforms to those whose name or vendor
 *    contains the given text (case-insensitive).
 *
 *    Kernel language. libmodal-flow's kernels assign __global pointers to
 *    unqualified pointers, which is valid OpenCL C 2.0 (generic address space)
 *    and accepted by Adreno even without -cl-std. Desktop compilers default to
 *    OpenCL C 1.2, so programs built without -cl-std are built as CL2.0, or
 *    CL3.0 on devices that only offer 3.0 with generic address space (an
 *    explicit CL2.0 request is mapped the same way; explicit CL1.x is kept).
 *    OV_OPENCL_BUILD_OPTIONS is appended to every build.
 *
 * 2) cl_qcom_recordable_queues emulation. Adreno records NDRange dispatches
 *    once and replays them with patched arguments/sizes. Here a recording
 *    captures, per dispatch, a private kernel (same program + function name)
 *    holding the argument values that were set when the dispatch was recorded,
 *    plus its NDRange. Replay restores those values, applies the requested
 *    updates and enqueues the dispatches in order on the replay queue; the wait
 *    list gates the first dispatch, out-of-order queues are chained with events,
 *    and the returned event completes with the last dispatch. Private kernels
 *    keep replays from mutating kernels libmodal-flow still uses directly.
 *
 * 3) Non-uniform work-groups. Adreno (OpenCL 2.0) accepts a global size that is
 *    not a multiple of the local size (e.g. FAST cells of 75 px with 8x8 groups);
 *    devices without that support return CL_INVALID_WORK_GROUP_SIZE. Such
 *    launches are re-issued as up to 2^dim uniform sub-ranges -- the
 *    whole-group region plus the remainder strips, each with its own offset and
 *    a local size equal to the remainder -- so every work-item keeps its
 *    get_global_id(). This is exact for kernels indexed by global id (the
 *    libmodal-flow kernels that hit this path); a marker event joins the parts.
 *
 * Compiled WITHOUT modal_flow_host_redirect.h: calls below reach the real
 * OpenCL ICD loader.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include <CL/cl.h>
#include <CL/cl_ext_qcom.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <sys/stat.h>

#define OV_MFH_HIDDEN __attribute__((visibility("hidden")))

#ifdef OV_OPENCL_ICD_VENDORS_DEFAULT
// -DOV_OPENCL_ICD_VENDORS=<dir>: point the ICD loader at a runtime registered in
// <dir> (e.g. one installed without root) when the environment does not choose
// one. Runs as this library loads, before any OpenCL entry point initializes
// the loader (loaders read OCL_ICD_VENDORS lazily, on their first call) and
// before the process starts threads. An explicit environment always wins.
__attribute__((constructor)) static void ov_mfh_default_icd_vendors() {
  if (std::getenv("OCL_ICD_VENDORS") || std::getenv("OPENCL_VENDOR_PATH"))
    return;
  struct stat info {};
  if (stat(OV_OPENCL_ICD_VENDORS_DEFAULT, &info) == 0 && S_ISDIR(info.st_mode))
    setenv("OCL_ICD_VENDORS", OV_OPENCL_ICD_VENDORS_DEFAULT, 0);
}
#endif

namespace {

// ---------------------------------------------------------------- policy ----
std::string lower(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return (char)std::tolower(c); });
  return text;
}

std::string platform_text(cl_platform_id platform, cl_platform_info what) {
  size_t size = 0;
  if (clGetPlatformInfo(platform, what, 0, nullptr, &size) != CL_SUCCESS || size == 0)
    return {};
  std::string value(size, '\0');
  clGetPlatformInfo(platform, what, size, &value[0], nullptr);
  value.resize(std::strlen(value.c_str()));
  return value;
}

std::vector<cl_platform_id> all_platforms() {
  cl_uint count = 0;
  if (clGetPlatformIDs(0, nullptr, &count) != CL_SUCCESS || count == 0)
    return {};
  std::vector<cl_platform_id> platforms(count);
  if (clGetPlatformIDs(count, platforms.data(), nullptr) != CL_SUCCESS)
    return {};
  return platforms;
}

std::vector<cl_platform_id> selected_platforms() {
  std::vector<cl_platform_id> platforms = all_platforms();
  const char *wanted = std::getenv("OV_OPENCL_PLATFORM");
  if (wanted == nullptr || *wanted == '\0')
    return platforms;
  const std::string needle = lower(wanted);
  std::vector<cl_platform_id> kept;
  std::string seen;
  for (cl_platform_id p : platforms) {
    const std::string name = platform_text(p, CL_PLATFORM_NAME), vendor = platform_text(p, CL_PLATFORM_VENDOR);
    seen += (seen.empty() ? "" : ", ") + name;
    if (lower(name).find(needle) != std::string::npos || lower(vendor).find(needle) != std::string::npos)
      kept.push_back(p);
  }
  if (kept.empty())
    std::fprintf(stderr, "[ov_modal_flow_host] OV_OPENCL_PLATFORM='%s' matches none of: %s\n", wanted,
                 seen.empty() ? "(no platforms)" : seen.c_str());
  return kept;
}

cl_device_type forced_device_type(bool &forced) {
  forced = true;
  const char *env = std::getenv("OV_OPENCL_DEVICE_TYPE");
  const std::string value = env ? lower(env) : std::string();
  if (value == "gpu")
    return CL_DEVICE_TYPE_GPU;
  if (value == "cpu")
    return CL_DEVICE_TYPE_CPU;
  if (value == "accelerator")
    return CL_DEVICE_TYPE_ACCELERATOR;
  if (value == "all")
    return CL_DEVICE_TYPE_ALL;
  if (!value.empty() && value != "auto")
    std::fprintf(stderr, "[ov_modal_flow_host] ignoring OV_OPENCL_DEVICE_TYPE='%s' (gpu|cpu|accelerator|all|auto)\n",
                 env);
  forced = false;
  return CL_DEVICE_TYPE_GPU;
}

bool any_platform_has_gpu() {
  static const bool has_gpu = [] {
    for (cl_platform_id p : selected_platforms()) {
      cl_uint n = 0;
      if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &n) == CL_SUCCESS && n > 0)
        return true;
    }
    return false;
  }();
  return has_gpu;
}

// --------------------------------------------------------- kernel language ----
// OpenCL 3.0 device queries, spelled out so the harness builds against any
// headers >= 2.0 (values and layout from the OpenCL 3.0 specification).
constexpr cl_device_info kDeviceOpenclCAllVersions = 0x1066;  // CL_DEVICE_OPENCL_C_ALL_VERSIONS
constexpr cl_device_info kDeviceGenericAddressSpace = 0x1069; // CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT
struct NameVersion {                                          // cl_name_version
  cl_uint version;
  char name[64];
};
constexpr cl_uint version_major(cl_uint version) { return version >> 22; } // CL_VERSION_MAJOR

struct LanguageSupport {
  bool c20 = false;         // OpenCL C 2.0 (generic address space mandatory)
  bool c30_generic = false; // OpenCL C 3.0 with __opencl_c_generic_address_space
};

LanguageSupport language_support(cl_device_id device) {
  LanguageSupport support;
  size_t size = 0;
  if (clGetDeviceInfo(device, kDeviceOpenclCAllVersions, 0, nullptr, &size) == CL_SUCCESS && size > 0) {
    std::vector<NameVersion> versions(size / sizeof(NameVersion));
    clGetDeviceInfo(device, kDeviceOpenclCAllVersions, versions.size() * sizeof(NameVersion), versions.data(), nullptr);
    for (const auto &v : versions) {
      support.c20 |= version_major(v.version) == 2;
      if (version_major(v.version) == 3) {
        cl_bool generic = CL_FALSE;
        clGetDeviceInfo(device, kDeviceGenericAddressSpace, sizeof(generic), &generic, nullptr);
        support.c30_generic = generic == CL_TRUE;
      }
    }
    return support;
  }
  // OpenCL 1.x/2.x devices: "OpenCL C <major>.<minor> ..."
  char version[128] = {0};
  if (clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, sizeof(version) - 1, version, nullptr) == CL_SUCCESS) {
    int major = 0, minor = 0;
    if (std::sscanf(version, "OpenCL C %d.%d", &major, &minor) == 2)
      support.c20 = major == 2;
  }
  return support;
}

std::string build_options(cl_uint num_devices, const cl_device_id *devices, const char *options) {
  std::string opts = options ? options : "";
  LanguageSupport all{true, true};
  for (cl_uint i = 0; i < num_devices && devices; ++i) {
    const LanguageSupport s = language_support(devices[i]);
    all.c20 &= s.c20;
    all.c30_generic &= s.c30_generic;
  }
  const std::string target = all.c20 ? "-cl-std=CL2.0" : all.c30_generic ? "-cl-std=CL3.0" : "";
  if (target.empty() && num_devices > 0 && devices) {
    static std::once_flag warned;
    std::call_once(warned, [&] {
      char name[256] = {0};
      clGetDeviceInfo(devices[0], CL_DEVICE_NAME, sizeof(name) - 1, name, nullptr);
      std::fprintf(stderr,
                   "[ov_modal_flow_host] '%s' offers neither OpenCL C 2.0 nor OpenCL C 3.0 with the generic address "
                   "space; libmodal-flow's kernels need one of them (e.g. PoCL >= 7 CPU, Intel compute-runtime GPU)\n",
                   name);
    });
  }
  const size_t at = opts.find("-cl-std=");
  if (at == std::string::npos) {
    if (!target.empty())
      opts = target + (opts.empty() ? "" : " " + opts);
  } else if (opts.compare(at, 13, "-cl-std=CL2.0") == 0 && !all.c20 && !target.empty()) {
    opts.replace(at, 13, target);
  }
  if (const char *extra = std::getenv("OV_OPENCL_BUILD_OPTIONS"))
    opts += std::string(" ") + extra;
  return opts;
}

// ------------------------------------------------- non-uniform NDRanges ----
std::mutex g_split_mutex;
std::unordered_set<cl_kernel> g_split_kernels; // kernels whose non-uniform launches were rejected

bool uniform(cl_uint dim, const size_t *global, const size_t *local) {
  for (cl_uint i = 0; i < dim; ++i)
    if (local[i] == 0 || global[i] % local[i] != 0)
      return false;
  return true;
}

cl_int enqueue_split(cl_command_queue queue, cl_kernel kernel, cl_uint dim, const size_t *offset, const size_t *global,
                     const size_t *local, cl_uint num_waits, const cl_event *waits, cl_event *event) {
  std::vector<cl_event> parts;
  cl_int err = CL_SUCCESS;
  for (cl_uint mask = 0; mask < (1u << dim) && err == CL_SUCCESS; ++mask) {
    size_t o[3] = {0, 0, 0}, g[3] = {1, 1, 1}, l[3] = {1, 1, 1};
    bool empty = false;
    for (cl_uint i = 0; i < dim; ++i) {
      const size_t whole = global[i] / local[i] * local[i], rest = global[i] - whole;
      const size_t base = offset ? offset[i] : 0;
      if (mask & (1u << i)) {
        empty |= rest == 0;
        o[i] = base + whole, g[i] = rest, l[i] = rest;
      } else {
        empty |= whole == 0;
        o[i] = base, g[i] = whole, l[i] = local[i];
      }
    }
    if (empty)
      continue;
    cl_event part = nullptr;
    err = clEnqueueNDRangeKernel(queue, kernel, dim, o, g, l, num_waits, waits, &part);
    if (part)
      parts.push_back(part);
  }
  if (err == CL_SUCCESS && event)
    err = clEnqueueMarkerWithWaitList(queue, (cl_uint)parts.size(), parts.empty() ? nullptr : parts.data(), event);
  for (cl_event part : parts)
    clReleaseEvent(part);
  return err;
}

// clEnqueueNDRangeKernel with the non-uniform work-group fallback described above.
cl_int enqueue_ndrange(cl_command_queue queue, cl_kernel kernel, cl_uint dim, const size_t *offset,
                       const size_t *global, const size_t *local, cl_uint num_waits, const cl_event *waits,
                       cl_event *event) {
  const bool non_uniform =
      local != nullptr && global != nullptr && dim >= 1 && dim <= 3 && !uniform(dim, global, local);
  if (non_uniform) {
    std::lock_guard<std::mutex> lock(g_split_mutex);
    if (g_split_kernels.count(kernel))
      goto split;
  }
  {
    const cl_int err = clEnqueueNDRangeKernel(queue, kernel, dim, offset, global, local, num_waits, waits, event);
    if (err != CL_INVALID_WORK_GROUP_SIZE || !non_uniform)
      return err;
    std::lock_guard<std::mutex> lock(g_split_mutex);
    g_split_kernels.insert(kernel);
  }
split:
  return enqueue_split(queue, kernel, dim, offset, global, local, num_waits, waits, event);
}

// ------------------------------------------------------- kernel tracking ----
struct ArgValue {
  size_t size = 0;
  bool local = false; // __local argument: size only, no value
  std::vector<unsigned char> bytes;
};
using ArgMap = std::map<cl_uint, ArgValue>;

std::mutex g_state_mutex;
std::unordered_map<cl_kernel, ArgMap> g_kernel_args;
std::unordered_set<cl_command_queue> g_recordable_queues;
std::unordered_map<cl_command_queue, cl_recording_qcom> g_active_recordings;

cl_int set_arg(cl_kernel kernel, cl_uint index, const ArgValue &arg) {
  return clSetKernelArg(kernel, index, arg.size, arg.local ? nullptr : arg.bytes.data());
}

cl_kernel private_kernel(cl_kernel source, const ArgMap &args, cl_int *err) {
  cl_program program = nullptr;
  *err = clGetKernelInfo(source, CL_KERNEL_PROGRAM, sizeof(program), &program, nullptr);
  size_t length = 0;
  if (*err == CL_SUCCESS)
    *err = clGetKernelInfo(source, CL_KERNEL_FUNCTION_NAME, 0, nullptr, &length);
  if (*err != CL_SUCCESS)
    return nullptr;
  std::string name(length, '\0');
  *err = clGetKernelInfo(source, CL_KERNEL_FUNCTION_NAME, length, &name[0], nullptr);
  if (*err != CL_SUCCESS)
    return nullptr;
  cl_kernel kernel = clCreateKernel(program, name.c_str(), err);
  if (*err != CL_SUCCESS)
    return nullptr;
  for (const auto &entry : args) {
    *err = set_arg(kernel, entry.first, entry.second);
    if (*err != CL_SUCCESS) {
      clReleaseKernel(kernel);
      return nullptr;
    }
  }
  return kernel;
}

struct Dispatch {
  cl_kernel kernel = nullptr; // private copy owned by the recording
  cl_uint work_dim = 0;
  bool has_offset = false, has_local = false;
  std::array<size_t, 3> offset{}, global{}, local{};
  ArgMap args; // values at record time (restored before every replay)
};

std::string kernel_name(cl_kernel kernel) {
  size_t length = 0;
  if (clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 0, nullptr, &length) != CL_SUCCESS || length == 0)
    return "?";
  std::string name(length, '\0');
  clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, length, &name[0], nullptr);
  name.resize(std::strlen(name.c_str()));
  return name;
}

void release_events(std::vector<cl_event> &events) {
  for (cl_event e : events)
    if (e)
      clReleaseEvent(e);
  events.clear();
}

// libmodal-flow reports a failed replay only as "Replay <stage>"; say why, once per kernel/error.
void report_replay_failure(const Dispatch &dispatch, cl_int err, const std::array<size_t, 3> &global,
                           const std::array<size_t, 3> &local, bool has_local) {
  static std::mutex mutex;
  static std::set<std::pair<std::string, cl_int>> reported;
  const std::string name = kernel_name(dispatch.kernel);
  std::lock_guard<std::mutex> lock(mutex);
  if (!reported.insert({name, err}).second)
    return;
  std::string sizes;
  for (cl_uint i = 0; i < dispatch.work_dim; ++i)
    sizes += (i ? "x" : "") + std::to_string(global[i]);
  if (has_local) {
    sizes += " local ";
    for (cl_uint i = 0; i < dispatch.work_dim; ++i)
      sizes += (i ? "x" : "") + std::to_string(local[i]);
  }
  std::fprintf(stderr, "[ov_modal_flow_host] replay of kernel '%s' failed with OpenCL error %d (global %s)\n",
               name.c_str(), err, sizes.c_str());
}

} // namespace

struct _cl_recording_qcom {
  std::mutex mutex;
  std::atomic<int> references{1};
  cl_command_queue queue = nullptr;
  cl_context context = nullptr;
  bool ended = false;
  std::vector<Dispatch> dispatches;

  ~_cl_recording_qcom() {
    for (auto &d : dispatches)
      if (d.kernel)
        clReleaseKernel(d.kernel);
  }
};

extern "C" {

// --------------------------------------------------- redirected entries ----
OV_MFH_HIDDEN cl_int ov_mfh_clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms) {
  if ((num_entries == 0 && platforms != nullptr) || (platforms == nullptr && num_platforms == nullptr))
    return CL_INVALID_VALUE;
  const std::vector<cl_platform_id> kept = selected_platforms();
  if (num_platforms)
    *num_platforms = (cl_uint)kept.size();
  if (platforms)
    for (cl_uint i = 0; i < num_entries && i < kept.size(); ++i)
      platforms[i] = kept[i];
  return CL_SUCCESS;
}

OV_MFH_HIDDEN cl_int ov_mfh_clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries,
                                           cl_device_id *devices, cl_uint *num_devices) {
  bool forced = false;
  const cl_device_type policy = forced_device_type(forced);
  cl_device_type type = device_type;
  if (device_type == CL_DEVICE_TYPE_GPU) {
    if (forced)
      type = policy;
    else if (!any_platform_has_gpu())
      type = CL_DEVICE_TYPE_ALL;
  }
  const cl_int err = clGetDeviceIDs(platform, type, num_entries, devices, num_devices);
  static std::once_flag announced;
  if (err == CL_SUCCESS && devices != nullptr && num_entries > 0) {
    // One line per process naming the device TrackOCL runs on (logs, studio console).
    std::call_once(announced, [&] {
      char name[256] = {0};
      cl_device_type kind = 0;
      clGetDeviceInfo(devices[0], CL_DEVICE_NAME, sizeof(name) - 1, name, nullptr);
      clGetDeviceInfo(devices[0], CL_DEVICE_TYPE, sizeof(kind), &kind, nullptr);
      const char *what = (kind & CL_DEVICE_TYPE_GPU)   ? "GPU"
                         : (kind & CL_DEVICE_TYPE_CPU) ? "CPU"
                                                       : "accelerator";
      std::fprintf(stderr, "[ov_modal_flow_host] OpenCL device: %s (%s, %s)%s\n", name, what,
                   platform_text(platform, CL_PLATFORM_NAME).c_str(),
                   type != device_type ? "; no GPU selected (OV_OPENCL_DEVICE_TYPE=gpu|cpu|accelerator|all to override)"
                                       : "");
    });
  }
  return err;
}

OV_MFH_HIDDEN cl_int ov_mfh_clBuildProgram(cl_program program, cl_uint num_devices, const cl_device_id *device_list,
                                           const char *options, void(CL_CALLBACK *pfn_notify)(cl_program, void *),
                                           void *user_data) {
  std::vector<cl_device_id> devices(device_list, device_list + (device_list ? num_devices : 0));
  if (devices.empty()) { // all devices associated with the program
    cl_uint count = 0;
    if (clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(count), &count, nullptr) == CL_SUCCESS && count) {
      devices.resize(count);
      clGetProgramInfo(program, CL_PROGRAM_DEVICES, count * sizeof(cl_device_id), devices.data(), nullptr);
    }
  }
  const std::string opts = build_options((cl_uint)devices.size(), devices.data(), options);
  return clBuildProgram(program, num_devices, device_list, opts.c_str(), pfn_notify, user_data);
}

OV_MFH_HIDDEN cl_command_queue ov_mfh_clCreateCommandQueueWithProperties(cl_context context, cl_device_id device,
                                                                         const cl_queue_properties *properties,
                                                                         cl_int *errcode_ret) {
  std::vector<cl_queue_properties> props;
  bool recordable = false;
  if (properties) {
    for (const cl_queue_properties *p = properties; *p != 0; p += 2) {
      cl_queue_properties value = p[1];
      if (p[0] == CL_QUEUE_PROPERTIES && (value & CL_QUEUE_RECORDABLE_QCOM)) {
        recordable = true;
        value &= ~(cl_queue_properties)CL_QUEUE_RECORDABLE_QCOM;
        // Recordings replay in order on the runtime queue; the capture queue
        // itself never executes, so an in-order queue is sufficient.
        value &= ~(cl_queue_properties)CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
        if (value == 0)
          continue;
      }
      props.push_back(p[0]);
      props.push_back(value);
    }
  }
  props.push_back(0);
  cl_int err = CL_SUCCESS;
  cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props.data(), &err);
  if (errcode_ret)
    *errcode_ret = err;
  if (queue) {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    // Handles can be reused after release: always refresh the registry.
    if (recordable)
      g_recordable_queues.insert(queue);
    else
      g_recordable_queues.erase(queue);
    g_active_recordings.erase(queue);
  }
  return queue;
}

OV_MFH_HIDDEN cl_kernel ov_mfh_clCreateKernel(cl_program program, const char *kernel_name, cl_int *errcode_ret) {
  cl_kernel kernel = clCreateKernel(program, kernel_name, errcode_ret);
  if (kernel) {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    g_kernel_args.erase(kernel); // a reused handle must not inherit stale arguments
  }
  return kernel;
}

OV_MFH_HIDDEN cl_int ov_mfh_clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size,
                                           const void *arg_value) {
  const cl_int err = clSetKernelArg(kernel, arg_index, arg_size, arg_value);
  if (err == CL_SUCCESS) {
    ArgValue arg;
    arg.size = arg_size;
    arg.local = arg_value == nullptr;
    if (!arg.local)
      arg.bytes.assign((const unsigned char *)arg_value, (const unsigned char *)arg_value + arg_size);
    std::lock_guard<std::mutex> lock(g_state_mutex);
    g_kernel_args[kernel][arg_index] = std::move(arg);
  }
  return err;
}

OV_MFH_HIDDEN cl_int ov_mfh_clEnqueueNDRangeKernel(cl_command_queue queue, cl_kernel kernel, cl_uint work_dim,
                                                   const size_t *global_work_offset, const size_t *global_work_size,
                                                   const size_t *local_work_size, cl_uint num_events_in_wait_list,
                                                   const cl_event *event_wait_list, cl_event *event) {
  cl_recording_qcom recording = nullptr;
  ArgMap args;
  {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    auto active = g_active_recordings.find(queue);
    if (active != g_active_recordings.end()) {
      recording = active->second;
      auto tracked = g_kernel_args.find(kernel);
      if (tracked != g_kernel_args.end())
        args = tracked->second;
    }
  }
  if (recording == nullptr)
    return enqueue_ndrange(queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size,
                           num_events_in_wait_list, event_wait_list, event);
  // Recording: capture, do not execute.
  if (work_dim < 1 || work_dim > 3 || global_work_size == nullptr)
    return CL_INVALID_WORK_DIMENSION;
  Dispatch dispatch;
  cl_int err = CL_SUCCESS;
  dispatch.kernel = private_kernel(kernel, args, &err);
  if (err != CL_SUCCESS)
    return err;
  dispatch.work_dim = work_dim;
  dispatch.args = std::move(args);
  for (cl_uint i = 0; i < work_dim; ++i) {
    dispatch.global[i] = global_work_size[i];
    if (global_work_offset)
      dispatch.offset[i] = global_work_offset[i];
    if (local_work_size)
      dispatch.local[i] = local_work_size[i];
  }
  dispatch.has_offset = global_work_offset != nullptr;
  dispatch.has_local = local_work_size != nullptr;
  {
    std::lock_guard<std::mutex> lock(recording->mutex);
    recording->dispatches.push_back(std::move(dispatch));
  }
  if (event) {
    *event = clCreateUserEvent(recording->context, &err);
    if (err == CL_SUCCESS)
      err = clSetUserEventStatus(*event, CL_COMPLETE);
  }
  return err;
}

// ------------------------------------------- cl_qcom_recordable_queues ----
OV_MFH_HIDDEN cl_recording_qcom clNewRecordingQCOM(cl_command_queue queue, cl_int *errcode_ret) {
  auto fail = [&](cl_int code) {
    if (errcode_ret)
      *errcode_ret = code;
    return (cl_recording_qcom) nullptr;
  };
  cl_context context = nullptr;
  if (queue == nullptr ||
      clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(context), &context, nullptr) != CL_SUCCESS)
    return fail(CL_INVALID_COMMAND_QUEUE);
  std::lock_guard<std::mutex> lock(g_state_mutex);
  if (!g_recordable_queues.count(queue))
    return fail(CL_INVALID_COMMAND_QUEUE); // not created with CL_QUEUE_RECORDABLE_QCOM
  if (g_active_recordings.count(queue))
    return fail(CL_INVALID_OPERATION); // one active recording per queue
  auto *recording = new _cl_recording_qcom();
  recording->queue = queue;
  recording->context = context;
  g_active_recordings[queue] = recording;
  if (errcode_ret)
    *errcode_ret = CL_SUCCESS;
  return recording;
}

OV_MFH_HIDDEN cl_int clEndRecordingQCOM(cl_recording_qcom recording) {
  if (recording == nullptr)
    return CL_INVALID_VALUE;
  std::lock_guard<std::mutex> lock(g_state_mutex);
  auto active = g_active_recordings.find(recording->queue);
  if (recording->ended || active == g_active_recordings.end() || active->second != recording)
    return CL_INVALID_OPERATION;
  g_active_recordings.erase(active);
  recording->ended = true;
  return CL_SUCCESS;
}

OV_MFH_HIDDEN cl_int clRetainRecordingQCOM(cl_recording_qcom recording) {
  if (recording == nullptr)
    return CL_INVALID_VALUE;
  recording->references.fetch_add(1);
  return CL_SUCCESS;
}

OV_MFH_HIDDEN cl_int clReleaseRecordingQCOM(cl_recording_qcom recording) {
  if (recording == nullptr)
    return CL_INVALID_VALUE;
  if (recording->references.fetch_sub(1) != 1)
    return CL_SUCCESS;
  {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    auto active = g_active_recordings.find(recording->queue);
    if (active != g_active_recordings.end() && active->second == recording)
      g_active_recordings.erase(active);
  }
  delete recording;
  return CL_SUCCESS;
}

OV_MFH_HIDDEN cl_int clEnqueueRecordingQCOM(cl_command_queue queue, cl_recording_qcom recording, size_t num_args,
                                            const cl_array_arg_qcom *arg_array, size_t num_global_offsets,
                                            const cl_offset_qcom *global_offset_array, size_t num_global_workgroups,
                                            const cl_workgroup_qcom *global_workgroup_array,
                                            size_t num_local_workgroups, const cl_workgroup_qcom *local_workgroup_array,
                                            cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
                                            cl_event *event) {
  if (queue == nullptr)
    return CL_INVALID_COMMAND_QUEUE;
  if (recording == nullptr || !recording->ended)
    return CL_INVALID_OPERATION;
  if ((num_args && !arg_array) || (num_global_offsets && !global_offset_array) ||
      (num_global_workgroups && !global_workgroup_array) || (num_local_workgroups && !local_workgroup_array) ||
      (num_events_in_wait_list > 0) != (event_wait_list != nullptr))
    return CL_INVALID_VALUE;
  std::lock_guard<std::mutex> lock(recording->mutex);
  const size_t count = recording->dispatches.size();
  for (size_t i = 0; i < num_args; ++i)
    if (arg_array[i].dispatch_index >= count)
      return CL_INVALID_VALUE;
  for (size_t i = 0; i < num_global_offsets; ++i)
    if (global_offset_array[i].dispatch_index >= count)
      return CL_INVALID_VALUE;
  for (size_t i = 0; i < num_global_workgroups; ++i)
    if (global_workgroup_array[i].dispatch_index >= count || !global_workgroup_array[i].workgroup_size)
      return CL_INVALID_VALUE;
  for (size_t i = 0; i < num_local_workgroups; ++i)
    if (local_workgroup_array[i].dispatch_index >= count || !local_workgroup_array[i].workgroup_size)
      return CL_INVALID_VALUE;

  cl_command_queue_properties queue_props = 0;
  cl_int err = clGetCommandQueueInfo(queue, CL_QUEUE_PROPERTIES, sizeof(queue_props), &queue_props, nullptr);
  if (err != CL_SUCCESS)
    return err;
  const bool out_of_order = (queue_props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0;

  if (count == 0) // nothing recorded: still honour the wait list / event contract
    return event ? clEnqueueMarkerWithWaitList(queue, num_events_in_wait_list, event_wait_list, event) : CL_SUCCESS;

  std::vector<cl_event> chain; // events this function owns
  cl_event previous = nullptr;
  for (size_t d = 0; d < count; ++d) {
    Dispatch &dispatch = recording->dispatches[d];
    std::vector<cl_uint> patched;
    for (size_t i = 0; i < num_args; ++i) {
      const cl_array_arg_qcom &u = arg_array[i];
      if (u.dispatch_index != d)
        continue;
      err = clSetKernelArg(dispatch.kernel, u.arg_index, u.arg_size, u.arg_value);
      if (err != CL_SUCCESS)
        break;
      patched.push_back(u.arg_index);
    }
    std::array<size_t, 3> offset = dispatch.offset, global = dispatch.global, local = dispatch.local;
    bool has_offset = dispatch.has_offset, has_local = dispatch.has_local;
    for (size_t i = 0; i < num_global_offsets; ++i)
      if (global_offset_array[i].dispatch_index == d) {
        std::copy(global_offset_array[i].offsets, global_offset_array[i].offsets + dispatch.work_dim, offset.begin());
        has_offset = true;
      }
    for (size_t i = 0; i < num_global_workgroups; ++i)
      if (global_workgroup_array[i].dispatch_index == d)
        std::copy(global_workgroup_array[i].workgroup_size,
                  global_workgroup_array[i].workgroup_size + dispatch.work_dim, global.begin());
    for (size_t i = 0; i < num_local_workgroups; ++i)
      if (local_workgroup_array[i].dispatch_index == d) {
        std::copy(local_workgroup_array[i].workgroup_size, local_workgroup_array[i].workgroup_size + dispatch.work_dim,
                  local.begin());
        has_local = true;
      }
    cl_event done = nullptr;
    const bool last = d + 1 == count;
    const bool need_event = (out_of_order && !last) || (last && event != nullptr);
    if (err == CL_SUCCESS) {
      const cl_uint waits = d == 0 ? num_events_in_wait_list : (previous ? 1u : 0u);
      const cl_event *wait_list = d == 0 ? event_wait_list : (previous ? &previous : nullptr);
      err = enqueue_ndrange(queue, dispatch.kernel, dispatch.work_dim, has_offset ? offset.data() : nullptr,
                            global.data(), has_local ? local.data() : nullptr, waits, wait_list,
                            need_event ? &done : nullptr);
    }
    // Arguments are captured at enqueue time; restore the recorded values so
    // the next replay starts from the recording, not from this replay's patch.
    for (cl_uint index : patched) {
      auto recorded = dispatch.args.find(index);
      if (recorded != dispatch.args.end())
        set_arg(dispatch.kernel, index, recorded->second);
    }
    if (err != CL_SUCCESS) {
      report_replay_failure(dispatch, err, global, local, has_local);
      release_events(chain);
      return err;
    }
    if (done) {
      if (last) {
        *event = done;
      } else {
        chain.push_back(done);
        previous = done;
      }
    } else if (!out_of_order) {
      previous = nullptr; // in-order queues serialize the remaining dispatches
    }
  }
  release_events(chain); // enqueued commands keep their own references
  return CL_SUCCESS;
}

} // extern "C"
