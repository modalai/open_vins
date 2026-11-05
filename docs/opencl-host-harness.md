# OpenCL host harness for TrackOCL (libmodal-flow)

TrackOCL, the OpenCL tracker behind `use_gpu: true`, is built on libmodal-flow,
which ships for VOXL (arm64 + Adreno). The host harness builds libmodal-flow
**from its source tree for the machine you are on**, so desktop builds can run
TrackOCL on any OpenCL platform (PoCL CPU, Intel compute-runtime, NVIDIA, …),
with no arm64 emulation, cross toolchain or VOXL sysroot.

The build is agnostic about it: **if libmodal-flow is there, TrackOCL is built;
if not, the CPU trackers are**, with no flag in either case. `OV_MODAL_FLOW`
selects the policy:

| `OV_MODAL_FLOW` | Effect |
|---|---|
| `AUTO` (default) | Build TrackOCL when libmodal-flow is found, otherwise the CPU trackers; the configure log says which and why |
| `ON` | Same search, but fail when nothing usable is found |
| `OFF` | CPU trackers only, nothing is searched |

What is used, in order:

1. `OV_MODAL_FLOW_SOURCE_DIR` (CMake or environment): that source tree, built by
   the harness. An explicit tree must be valid; anything missing is an error.
2. An installed libmodal-flow (VOXL sysroot or Debian package), linked as before.
3. A source tree found automatically (not when cross-compiling):
   `OV_MODAL_FLOW_SEARCH_PATHS` (CMake list, or environment with `:`),
   `<repo>/third_party|thirdparty|external/libmodal-flow`, `<repo>/../libmodal-flow`
   (a sibling in the workspace, e.g. next to `open_vins` in a colcon `src/`), then
   `~/libmodal-flow` and `~/*/libmodal-flow` (the newest valid version wins).
   It is used when CMake ≥ 3.16 and OpenCL ≥ 2.0 headers and loader are found;
   otherwise the build continues with the CPU trackers and says what is missing.

The source tree is never modified.

## Requirements

* libmodal-flow source ≥ 1.0.10 (`pkg/control/control` is the version source).
* CMake ≥ 3.16, OpenCL headers ≥ 2.0 and the ICD loader:
  `sudo apt install opencl-headers ocl-icd-opencl-dev` (rosdep keys
  `opencl-headers`, `ocl-icd-opencl-dev`). The runtime loader `libOpenCL.so.1`
  is enough (no `-dev` symlink needed). Headers outside the system are found
  through `OV_OPENCL_ROOT` (CMake or environment) or the no-root prefix
  `~/.local/opt/ov-opencl` below; or pass
  `-DOpenCL_INCLUDE_DIR=<dir with CL/cl.h> -DOpenCL_LIBRARY=<libOpenCL.so[.1]>`.
* At runtime, an OpenCL platform whose device offers **OpenCL C 2.0, or 3.0 with
  the generic address space** (libmodal-flow's kernels use generic pointers).
  Verified: Intel compute-runtime 26.35 on a Lunar Lake (Xe2) iGPU, PoCL 7.1 CPU.
  Not sufficient: PoCL 3.0 (no generic address space). The harness says so
  explicitly instead of failing in the kernel compiler.

## Build

```bash
# automatic: e.g. with ~/OV/libmodal-flow present
cmake -S . -B build -DENABLE_ROS=OFF
#   -- Found libmodal-flow source tree /home/me/OV/libmodal-flow (-DOV_MODAL_FLOW=OFF to build without it)
#   -- OpenCL/modal-flow tracker enabled: libmodal-flow 1.0.10 host build from ...
cmake --build build -j

# a specific tree, or none (ROS 2 / ROS 1: environment variable or CMake argument)
OV_MODAL_FLOW_SOURCE_DIR=$HOME/OV/libmodal-flow colcon build --packages-up-to ov_msckf
catkin_make -DOV_MODAL_FLOW_SOURCE_DIR=$HOME/OV/libmodal-flow
OV_MODAL_FLOW=OFF colcon build --packages-up-to ov_msckf     # CPU trackers only
```

ov_core links the harness library `ov_modal_flow_host` (installed next to
`ov_core_lib`, with `$ORIGIN` RUNPATH). Its headers install under
`include/ov_modal_flow_host/` and are exported to catkin/ament consumers.

| CMake option | Effect |
|---|---|
| `OV_MODAL_FLOW` | `AUTO` (default), `ON` or `OFF`, see above. Also read from the environment. |
| `OV_MODAL_FLOW_SOURCE_DIR` | A specific libmodal-flow source tree (takes precedence over everything). Also read from the environment. |
| `OV_MODAL_FLOW_SEARCH_PATHS` | Extra places searched for a source tree: trees, or directories containing `libmodal-flow/`. Environment: `:`-separated. |
| `OV_OPENCL_ROOT` | OpenCL prefix outside the system (`include/CL/cl.h`, optional `vendors/*.icd`). Also read from the environment; `~/.local/opt/ov-opencl` is tried too. |
| `OV_MODAL_FLOW_KLT_MEANNORM` | Mirrors the tree's photometric mean-normalization switch (default `ON`). |
| `OV_OPENCL_ICD_VENDORS` | Default ICD vendors directory baked into the harness (empty by default; when the OpenCL headers come from a non-system prefix with a `vendors/` directory of `.icd` files, that directory). When the library loads and neither `OCL_ICD_VENDORS` nor `OPENCL_VENDOR_PATH` is set, it sets `OCL_ICD_VENDORS` to this directory if it exists, so a runtime installed without root works without exporting anything. An explicit environment always wins; like `OCL_ICD_VENDORS` itself, it replaces the system's `/etc/OpenCL/vendors`. |

## Run

Use any estimator config with `use_klt: true` and `use_gpu: true`. Optional environment:

| Variable | Effect |
|---|---|
| `OV_OPENCL_DEVICE_TYPE` | `auto` (default: GPU if any platform has one, else that platform's CPU/accelerator), `gpu`, `cpu`, `accelerator`, `all` |
| `OV_OPENCL_PLATFORM` | Only platforms whose name/vendor contains this text |
| `OV_OPENCL_BUILD_OPTIONS` | Appended to every kernel build |
| `OCL_ICD_VENDORS` | Standard ICD-loader variable, e.g. a directory with a user-local `pocl.icd`; overrides `OV_OPENCL_ICD_VENDORS` |

The first device selection prints one line, e.g.
`[ov_modal_flow_host] OpenCL device: Intel(R) Arc(TM) Graphics (GPU, Intel(R) OpenCL Graphics (integrated))`.
`ov_msckf_replay --capabilities` lists `"ocl"` among its trackers when the
harness is built; zstudio offers it automatically.

## Setup on a developer PC

With root, install the headers, the ICD loader and a runtime from the distribution:

```bash
sudo apt install opencl-headers ocl-icd-opencl-dev
sudo apt install intel-opencl-icd     # Intel GPUs; or pocl-opencl-icd (CPU), or a vendor driver
```

Distribution runtimes can trail new GPUs: Ubuntu 24.04's `intel-opencl-icd`
(23.43) predates Lunar Lake (Xe2), so use Intel's compute-runtime release
packages for such GPUs. Check that the device appears, and that the user can
open `/dev/dri/renderD*` (group `render`).

Without root, everything lives under `~/.local/opt` and is removed by deleting
those directories. The GPU needs only the kernel driver (`i915`/`xe`) and
membership of the `render` group.

```bash
O=~/.local/opt/ov-opencl; R=~/.local/opt/intel-compute-runtime
mkdir -p $O/bin $O/include $O/vendors $R
# static tools: patchelf (NixOS/patchelf release) and micromamba (only for the PoCL fallback)
curl -sSL https://github.com/NixOS/patchelf/releases/download/0.18.0/patchelf-0.18.0-x86_64.tar.gz | tar xz -C $O ./bin/patchelf
curl -sSL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar xj -C $O bin/micromamba
export PATH=$O/bin:$PATH   # for this shell only

# 1. Khronos headers (any recent OpenCL-Headers release)
curl -sSL https://github.com/KhronosGroup/OpenCL-Headers/archive/refs/tags/v2024.10.24.tar.gz \
  | tar xz -C /tmp && cp -r /tmp/OpenCL-Headers-2024.10.24/CL $O/include/

# 2a. Intel GPU runtime: the compute-runtime release's intel-opencl-icd + libigdgmm
#     and the matching IGC packages listed in its release notes (verify the sums)
for deb in intel-opencl-icd_*.deb libigdgmm12_*.deb intel-igc-core-2_*.deb intel-igc-opencl-2_*.deb; do
  dpkg-deb -x $deb $R
done
#     make the tree relocatable: libigdrcl loads gmmlib and dlopen()s IGC by name
patchelf --set-rpath '$ORIGIN/..:$ORIGIN/../../../local/lib' $R/usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so
for lib in $R/usr/local/lib/libigc.so.2.* $R/usr/local/lib/libigdfcl.so.2.* $R/usr/local/lib/libiga64.so.2.*; do
  patchelf --set-rpath '$ORIGIN' $lib
done
echo $R/usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so > $O/vendors/intel-neo.icd

# 2b. CPU fallback (optional): PoCL from conda-forge with a static micromamba
$O/bin/micromamba create -y -r $O/mamba -p $O/pocl -c conda-forge 'pocl>=7'
$O/bin/micromamba clean -a -y -r $O/mamba
echo $O/pocl/lib/libpocl.so.2 > $O/vendors/pocl.icd

# 3. Build: $O/include and $O/vendors are found automatically (the system
#    libOpenCL.so.1 is the ICD loader); no flags, no OCL_ICD_VENDORS needed
cmake -S . -B build-ocl -DENABLE_ROS=OFF
cmake --build build-ocl -j
ctest --test-dir build-ocl/ov_core -R test_trackocl_host
```

With both runtimes registered, `auto` picks the GPU and
`OV_OPENCL_DEVICE_TYPE=cpu` selects PoCL.

## What the harness adapts (cmake/modal_flow_host/)

* **Sources.** Builds every library source except the libmodal-pipe publishers
  (VOXL IPC, unused by TrackOCL), embeds kernels with the tree's own
  `cmake/embed_kernels.cmake` and kernel list, and generates `modal_flow/Version.hpp`.
* **Scoped redirection.** `modal_flow_host_redirect.h` is force-included into
  libmodal-flow translation units only, routing seven OpenCL calls through
  `host_runtime.cpp`:
  * **Devices.** libmodal-flow hard-codes GPU devices; the harness applies the
    device policy above.
  * **Kernel language.** Programs built without `-cl-std` get CL2.0, or CL3.0
    on 3.0-only devices with the generic address space.
  * **`cl_qcom_recordable_queues`.** Adreno-only; emulated over core OpenCL 2.0.
    Dispatches are captured with their argument values on private kernels, then
    replayed in order with argument, offset and work-size updates, the wait list
    and the returned event. Out-of-order queues are chained with events.
  * **Non-uniform work-groups.** On devices without them (e.g. FAST cells
    75 px wide with 8×8 groups), launches are split at work-group boundaries
    into uniform sub-ranges, which preserves `get_global_id`.
* **Host images.** TrackOCL uploads ordinary `cv::Mat` frames when a producer
  supplies no `modal_flow` upload views. The VOXL zero-copy path is unchanged.

## Verification

`test_trackocl_host` (registered with CTest whenever the harness is built) runs TrackOCL
and TrackKLT on the same frames. It exits 77 (skipped) when no OpenCL device
exists. Pass a directory of PNG frames to use real data:

```bash
ctest --test-dir build-ocl/ov_core -R test_trackocl_host
build-ocl/ov_core/test_trackocl_host /path/to/log/run/mpa/tracking_left --frames 300
```

Results on an Intel Core Ultra 7 258V (Lunar Lake). Intel Arc iGPU, compute-runtime
26.35 / IGC 2.41.5, no environment variables (`OV_OPENCL_ICD_VENDORS` baked in):

| | TrackOCL (GPU) | TrackOCL (PoCL CPU) | TrackKLT |
|---|---|---|---|
| Synthetic (known shift): features / ID persistence / ms per frame | 167 / 98.5 % / 3.5 | 165 / 98.5 % / 14.0 | 174 / 99.0 % / 1.0 |
| voxl-logger mono `tracking_left` VIO, 50.8 s: initialized at | 8.93 s | — | 8.93 s |
| … updates / path length / max speed | 1524 / 49.8 m / 2.6 m/s | — | 1524 / 55.4 m / 3.0 m/s |
| … mean update time | 3.2 ms | — | 3.5 ms |

Earlier results with PoCL 7.1 (CPU device):

| | TrackOCL | TrackKLT |
|---|---|---|
| Synthetic (known 2.0×1.0 px shift): features / ID persistence / step error | 166 / 98.6 % / 0.001 px | 174 / 99.0 % / 0.001 px |
| VOXL `tracking_left`, 300 frames at 1280×800: features / ID persistence | 162.5 / 98.2 % | 155.5 / 98.5 % |
| EuRoC `imu_april.bag` stereo VIO (dynamic init, 60 s): initialized at | 17.39 s | 17.34 s |
| … path length | 19.93 m | 19.94 m |
| … end-point difference | ≈3 cm | — |
| … mean update time | 29 ms | 6 ms |

## Limitations

* PoCL runs the kernels on the CPU. Expect it to be several times slower than
  TrackKLT; use a GPU ICD for speed (on the Lunar Lake iGPU a full VIO update
  costs about the same as with TrackKLT).
* TrackOCL's stereo path is still marked R&D in this fork, and its ZNCC stereo
  calibration is only filled in by the VOXL configuration path. On the host,
  stereo cameras are tracked as independent monocular streams.
* The recordable-queue emulation reproduces the semantics libmodal-flow
  relies on. It is not a general implementation of the Qualcomm extension (for
  example, only `NDRange` commands are recorded, matching the extension's scope).
