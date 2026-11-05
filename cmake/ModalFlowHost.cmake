# OpenVINS OpenCL tracker (TrackOCL) through libmodal-flow -- detection and the
# optional host harness. Included by ov_core/CMakeLists.txt.
#
# TrackOCL (ov_core/src/track/TrackOCL) runs on libmodal-flow, which ships for
# VOXL (arm64 + Adreno). The build is agnostic about it:
#
#   OV_MODAL_FLOW=AUTO (default)  build TrackOCL when libmodal-flow is available,
#                                 otherwise the CPU trackers only, with a note
#   OV_MODAL_FLOW=ON              same, but fail when it is not available
#   OV_MODAL_FLOW=OFF             CPU trackers only, nothing is searched
#
# "Available" means, in this order:
#   1. OV_MODAL_FLOW_SOURCE_DIR (CMake or environment): that libmodal-flow
#      source tree is built for this host by the harness below (must be valid).
#   2. An installed libmodal-flow (VOXL sysroot / Debian package): linked as is.
#   3. A libmodal-flow source tree found automatically (not when cross-compiling):
#        OV_MODAL_FLOW_SEARCH_PATHS entries (CMake, or environment with ':'),
#        <repo>/third_party|thirdparty|external/libmodal-flow,
#        <repo>/../libmodal-flow (a sibling in the workspace, e.g. colcon src/),
#        ~/libmodal-flow and ~/*/libmodal-flow (newest valid version wins).
#      It is used when CMake >= 3.16 and OpenCL >= 2.0 headers and the ICD
#      loader are found; otherwise the configure log says what is missing.
#
# The host harness builds libmodal-flow FROM ITS SOURCE TREE for the build host
# itself -- any architecture, any OpenCL >= 2.0 platform (PoCL, Intel
# compute-runtime, NVIDIA, ...), no emulation or cross toolchain -- so desktop
# builds can run `use_gpu: true` configurations. The tree is never modified.
# Sources that publish over libmodal-pipe (a VOXL IPC library TrackOCL does not
# use) are skipped, kernels are embedded with the tree's own
# cmake/embed_kernels.cmake, and cmake/modal_flow_host/ supplies the host pieces:
# device policy and a portable emulation of Qualcomm's cl_qcom_recordable_queues
# (see host_runtime.cpp, docs/opencl-host-harness.md).
#
# OpenCL outside the system (installed without root): OV_OPENCL_ROOT (CMake or
# environment) names a prefix with include/CL/cl.h; ~/.local/opt/ov-opencl, the
# prefix of the no-root recipe in docs/opencl-host-harness.md, is tried too. When
# the headers come from such a prefix and it has a vendors/ directory of .icd
# files, that directory becomes the default OV_OPENCL_ICD_VENDORS.

if (DEFINED ENV{OV_MODAL_FLOW} AND NOT "$ENV{OV_MODAL_FLOW}" STREQUAL "")
    set(_ov_mf_default "$ENV{OV_MODAL_FLOW}")
else ()
    set(_ov_mf_default AUTO)
endif ()
set(OV_MODAL_FLOW "${_ov_mf_default}" CACHE STRING
    "OpenCL tracker on libmodal-flow: AUTO = when found, ON = required, OFF = CPU trackers only")
set_property(CACHE OV_MODAL_FLOW PROPERTY STRINGS AUTO ON OFF)
set(OV_MODAL_FLOW_SOURCE_DIR "$ENV{OV_MODAL_FLOW_SOURCE_DIR}" CACHE PATH
    "libmodal-flow source tree to build for this host (takes precedence); empty = installed library or search")
set(OV_MODAL_FLOW_SEARCH_PATHS "" CACHE STRING
    "Extra places searched for a libmodal-flow source tree (;-list of trees or of directories containing libmodal-flow/)")
set(OV_OPENCL_ROOT "$ENV{OV_OPENCL_ROOT}" CACHE PATH
    "OpenCL headers/ICD prefix outside the system (include/CL/cl.h, optional vendors/*.icd); empty = system or ~/.local/opt/ov-opencl")
option(OV_MODAL_FLOW_KLT_MEANNORM "Host harness: photometric mean-normalization in libmodal-flow's KLT (tree default ON)" ON)
# An OpenCL runtime installed without root (e.g. under ~/.local) is registered
# in its own ICD vendors directory. Baking that directory in lets the binaries
# find it without exporting OCL_ICD_VENDORS. It is applied when the harness
# library loads, only if neither OCL_ICD_VENDORS nor OPENCL_VENDOR_PATH is set
# and the directory exists; empty keeps the system ICD configuration (or the
# OV_OPENCL_ROOT default described above).
set(OV_OPENCL_ICD_VENDORS "" CACHE PATH
    "Host harness: default OpenCL ICD vendors directory (used when OCL_ICD_VENDORS is unset); empty = system")
set(OV_MODAL_FLOW_HOST_MIN_VERSION 1.0.10) # TrackOCL consumes modal_flow/Mask.hpp (1.0.10+)
set(OV_MODAL_FLOW_HOST_DIR "${CMAKE_CURRENT_LIST_DIR}/modal_flow_host")
get_filename_component(_OV_MODAL_FLOW_REPO "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

# Version of a libmodal-flow source tree usable by the harness, or empty with a reason.
function(_ov_modal_flow_tree dir out_version out_reason)
    set(${out_version} "" PARENT_SCOPE)
    foreach (required
            library/include/modal_flow_ocl_manager.h library/include/modal_flow/Types.hpp
            library/include/modal_flow/StereoMatcher.hpp library/include/modal_flow/Mask.hpp
            library/include/modal_flow/Version.hpp.in library/CMakeLists.txt
            cmake/embed_kernels.cmake pkg/control/control)
        if (NOT EXISTS "${dir}/${required}")
            set(${out_reason} "not a libmodal-flow >= ${OV_MODAL_FLOW_HOST_MIN_VERSION} source tree (missing ${required})" PARENT_SCOPE)
            return()
        endif ()
    endforeach ()
    # Version: single source of truth is the tree's Debian control file.
    file(STRINGS "${dir}/pkg/control/control" line REGEX "^Version:")
    string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" version "${line}")
    if (NOT version)
        set(${out_reason} "pkg/control/control has no 'Version: X.Y.Z' line" PARENT_SCOPE)
    elseif (version VERSION_LESS OV_MODAL_FLOW_HOST_MIN_VERSION)
        set(${out_reason} "libmodal-flow ${version}, TrackOCL needs >= ${OV_MODAL_FLOW_HOST_MIN_VERSION}" PARENT_SCOPE)
    else ()
        set(${out_version} ${version} PARENT_SCOPE)
    endif ()
endfunction()

# First usable source tree from the search order in the header (or empty), plus
# notes about trees that were found but skipped.
function(_ov_modal_flow_discover out_tree out_notes)
    set(ordered)
    set(extra ${OV_MODAL_FLOW_SEARCH_PATHS})
    if (NOT "$ENV{OV_MODAL_FLOW_SEARCH_PATHS}" STREQUAL "")
        string(REPLACE ":" ";" env_paths "$ENV{OV_MODAL_FLOW_SEARCH_PATHS}")
        list(APPEND extra ${env_paths})
    endif ()
    foreach (path ${extra})
        list(APPEND ordered "${path}" "${path}/libmodal-flow")
    endforeach ()
    foreach (sub third_party thirdparty external)
        list(APPEND ordered "${_OV_MODAL_FLOW_REPO}/${sub}/libmodal-flow")
    endforeach ()
    list(APPEND ordered "${_OV_MODAL_FLOW_REPO}/../libmodal-flow")
    set(home)
    if (NOT "$ENV{HOME}" STREQUAL "" AND IS_DIRECTORY "$ENV{HOME}")
        file(GLOB home LIST_DIRECTORIES true "$ENV{HOME}/*/libmodal-flow")
        list(INSERT home 0 "$ENV{HOME}/libmodal-flow")
    endif ()
    set(notes)
    set(seen)
    set(best "")
    set(best_version "")
    set(group ordered)
    # In the ordered group the first usable tree wins; among home directories the
    # newest version does.
    foreach (candidate ${ordered} --home-- ${home})
        if (candidate STREQUAL "--home--")
            if (best)
                break()
            endif ()
            set(group home)
            continue()
        endif ()
        if ((best AND group STREQUAL "ordered") OR NOT IS_DIRECTORY "${candidate}")
            continue()
        endif ()
        get_filename_component(candidate "${candidate}" REALPATH)
        if (candidate IN_LIST seen)
            continue()
        endif ()
        list(APPEND seen "${candidate}")
        _ov_modal_flow_tree("${candidate}" version reason)
        get_filename_component(name "${candidate}" NAME)
        if (version)
            if (NOT best OR version VERSION_GREATER best_version)
                set(best "${candidate}")
                set(best_version ${version})
            endif ()
        elseif (name MATCHES "modal.flow" OR EXISTS "${candidate}/pkg/control/control")
            # Report trees that were skipped, not search dirs without one.
            list(APPEND notes "${candidate}: ${reason}")
        endif ()
    endforeach ()
    set(${out_tree} "${best}" PARENT_SCOPE)
    set(${out_notes} "${notes}" PARENT_SCOPE)
endfunction()

# OpenCL headers (>= 2.0) and the ICD loader, also from a prefix outside the
# system. Sets OpenCL_* (FindOpenCL) and, if empty, OV_OPENCL_ICD_VENDORS from it.
macro(_ov_modal_flow_find_opencl)
    set(_ov_mf_cl_prefixes)
    foreach (_ov_mf_prefix "${OV_OPENCL_ROOT}" "$ENV{HOME}/.local/opt/ov-opencl")
        if (NOT "${_ov_mf_prefix}" STREQUAL "" AND EXISTS "${_ov_mf_prefix}/include/CL/cl.h")
            list(APPEND _ov_mf_cl_prefixes "${_ov_mf_prefix}")
        endif ()
    endforeach ()
    if (NOT OpenCL_INCLUDE_DIR)
        set(_ov_mf_hints)
        foreach (_ov_mf_prefix ${_ov_mf_cl_prefixes})
            list(APPEND _ov_mf_hints "${_ov_mf_prefix}/include")
        endforeach ()
        find_path(OpenCL_INCLUDE_DIR NAMES CL/cl.h OpenCL/cl.h HINTS ${_ov_mf_hints})
    endif ()
    if (NOT OpenCL_LIBRARY)
        # Distributions ship the loader's unversioned libOpenCL.so only in -dev
        # packages; the runtime libOpenCL.so.1 links just as well.
        set(_ov_mf_hints)
        foreach (_ov_mf_prefix ${_ov_mf_cl_prefixes})
            list(APPEND _ov_mf_hints "${_ov_mf_prefix}/lib" "${_ov_mf_prefix}/lib64")
        endforeach ()
        find_library(OpenCL_LIBRARY NAMES OpenCL libOpenCL.so.1 HINTS ${_ov_mf_hints})
    endif ()
    find_package(OpenCL QUIET)
    if (OpenCL_FOUND AND NOT OpenCL_VERSION_STRING)
        # FindOpenCL caches its CL_VERSION_* probes; probes that failed in an
        # earlier configure (before the headers were found) would stick forever.
        foreach (_ov_mf_v 1_0 1_1 1_2 2_0 2_1 2_2 3_0)
            unset(OPENCL_VERSION_${_ov_mf_v} CACHE)
        endforeach ()
        find_package(OpenCL QUIET)
    endif ()
    if (OpenCL_FOUND AND NOT OV_OPENCL_ICD_VENDORS)
        get_filename_component(_ov_mf_prefix "${OpenCL_INCLUDE_DIR}" DIRECTORY)
        file(GLOB _ov_mf_icds "${_ov_mf_prefix}/vendors/*.icd")
        if (_ov_mf_icds AND NOT _ov_mf_prefix MATCHES "^/(usr(/local)?)?$")
            set(OV_OPENCL_ICD_VENDORS "${_ov_mf_prefix}/vendors")
        endif ()
    endif ()
endmacro()

# Build the harness from OV_MODAL_FLOW_HOST_SOURCE. On a missing prerequisite:
# FATAL_ERROR when `required`, otherwise leaves OV_MODAL_FLOW_HOST OFF and puts
# the reason in OV_MODAL_FLOW_HOST_ERROR.
macro(_ov_modal_flow_host_build required)
    set(OV_MODAL_FLOW_HOST_ERROR "")
    _ov_modal_flow_tree("${OV_MODAL_FLOW_HOST_SOURCE}" MODAL_FLOW_VERSION _ov_mf_reason)
    if (NOT MODAL_FLOW_VERSION)
        set(OV_MODAL_FLOW_HOST_ERROR "${OV_MODAL_FLOW_HOST_SOURCE}: ${_ov_mf_reason}")
    elseif (CMAKE_VERSION VERSION_LESS 3.16)
        set(OV_MODAL_FLOW_HOST_ERROR "the libmodal-flow host harness needs CMake >= 3.16 (found ${CMAKE_VERSION})")
    else ()
        _ov_modal_flow_find_opencl()
        if (NOT OpenCL_FOUND)
            string(CONCAT OV_MODAL_FLOW_HOST_ERROR
                "the libmodal-flow host harness needs OpenCL headers and the ICD loader.\n"
                "  Debian/Ubuntu: sudo apt install opencl-headers ocl-icd-opencl-dev   "
                "(rosdep keys: opencl-headers, ocl-icd-opencl-dev)\n"
                "  Without root: -DOV_OPENCL_ROOT=<prefix with include/CL/cl.h> (see docs/opencl-host-harness.md),\n"
                "  or -DOpenCL_INCLUDE_DIR=<dir containing CL/cl.h> -DOpenCL_LIBRARY=<libOpenCL.so or libOpenCL.so.1>.\n"
                "A runtime platform is also required to run TrackOCL (e.g. pocl-opencl-icd, "
                "intel-opencl-icd, or a GPU vendor driver).")
        elseif (OpenCL_VERSION_STRING AND OpenCL_VERSION_STRING VERSION_LESS 2.0)
            string(CONCAT OV_MODAL_FLOW_HOST_ERROR "libmodal-flow uses OpenCL 2.0 queue APIs, but the headers in "
                "${OpenCL_INCLUDE_DIRS} are OpenCL ${OpenCL_VERSION_STRING}")
        endif ()
    endif ()
    if (OV_MODAL_FLOW_HOST_ERROR)
        if (${required})
            message(FATAL_ERROR "OpenCL/modal-flow tracker: ${OV_MODAL_FLOW_HOST_ERROR}")
        endif ()
    else ()
        string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)" _ov_mf_match "${MODAL_FLOW_VERSION}")
        set(MODAL_FLOW_VERSION_MAJOR ${CMAKE_MATCH_1})
        set(MODAL_FLOW_VERSION_MINOR ${CMAKE_MATCH_2})
        set(MODAL_FLOW_VERSION_PATCH ${CMAKE_MATCH_3})
        add_subdirectory("${OV_MODAL_FLOW_HOST_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/ov_modal_flow_host")
        set(OV_MODAL_FLOW_HOST ON)
        set(OV_USE_MODAL_FLOW 1)
        set(MODAL_FLOW_LIBRARY ov_modal_flow_host)
        set(MODAL_FLOW_INCLUDE_DIR "")
        message(STATUS "OpenCL/modal-flow tracker enabled: libmodal-flow ${MODAL_FLOW_VERSION} host build from "
            "${OV_MODAL_FLOW_HOST_SOURCE} (OpenCL ${OpenCL_VERSION_STRING}: ${OpenCL_LIBRARIES})")
    endif ()
endmacro()

# An installed libmodal-flow (VOXL sysroot or package). Sets OV_USE_MODAL_FLOW,
# MODAL_FLOW_INCLUDE_DIR, MODAL_FLOW_LIBRARY and OpenCL_*.
macro(_ov_modal_flow_find_installed)
    find_path(MODAL_FLOW_INCLUDE_DIR NAMES modal_flow_ocl_manager.h)
    find_library(MODAL_FLOW_LIBRARY NAMES modal_flow)
    if (MODAL_FLOW_INCLUDE_DIR AND MODAL_FLOW_LIBRARY
        AND EXISTS "${MODAL_FLOW_INCLUDE_DIR}/modal_flow/Types.hpp"
        AND EXISTS "${MODAL_FLOW_INCLUDE_DIR}/modal_flow/StereoMatcher.hpp")
        if (NOT EXISTS "${MODAL_FLOW_INCLUDE_DIR}/modal_flow/Mask.hpp")
            message(FATAL_ERROR "TrackOCL mask inputs require libmodal-flow >= 1.0.10 "
                "(installed: ${MODAL_FLOW_INCLUDE_DIR}); update it or configure with -DOV_MODAL_FLOW=OFF")
        endif ()
        find_package(OpenCL QUIET)
        if (OpenCL_FOUND)
            set(OV_USE_MODAL_FLOW 1)
            message(STATUS "OpenCL/modal-flow tracker enabled: installed libmodal-flow "
                "(${MODAL_FLOW_INCLUDE_DIR}, ${MODAL_FLOW_LIBRARY})")
        endif ()
    endif ()
endmacro()

# Decide per OV_MODAL_FLOW (see the header). Sets in the caller's scope:
#   OV_USE_MODAL_FLOW             1 when TrackOCL is built (ov_core's switch)
#   OV_MODAL_FLOW_HOST            ON when the host harness is built
#   MODAL_FLOW_LIBRARY            `ov_modal_flow_host`, or the installed library
#   MODAL_FLOW_INCLUDE_DIR        installed headers; empty for the harness (its
#                                 include dirs travel as target usage requirements)
#   OV_MODAL_FLOW_HOST_INCLUDE_DIRS  harness build-tree include dirs (catkin devel exports)
#   OpenCL_* from FindOpenCL
macro(ov_modal_flow_configure)
    set(OV_USE_MODAL_FLOW 0)
    set(OV_MODAL_FLOW_HOST OFF)
    string(TOUPPER "${OV_MODAL_FLOW}" _ov_mf_mode)
    if (_ov_mf_mode MATCHES "^(OFF|NO|FALSE|0|N)$")
        message(STATUS "OpenCL/modal-flow tracker disabled (OV_MODAL_FLOW=OFF); building CPU trackers")
    elseif (NOT _ov_mf_mode MATCHES "^(AUTO|ON|YES|TRUE|1|Y)$")
        message(FATAL_ERROR "OV_MODAL_FLOW must be AUTO, ON or OFF (got '${OV_MODAL_FLOW}')")
    elseif (OV_MODAL_FLOW_SOURCE_DIR)
        # An explicit tree is a request: anything missing is an error.
        get_filename_component(OV_MODAL_FLOW_HOST_SOURCE "${OV_MODAL_FLOW_SOURCE_DIR}" REALPATH)
        _ov_modal_flow_host_build(TRUE)
    else ()
        if (_ov_mf_mode STREQUAL "AUTO")
            set(_ov_mf_required FALSE)
        else ()
            set(_ov_mf_required TRUE)
        endif ()
        _ov_modal_flow_find_installed()
        set(_ov_mf_notes)
        if (NOT OV_USE_MODAL_FLOW)
            if (CMAKE_CROSSCOMPILING)
                set(_ov_mf_notes "source trees are not searched when cross-compiling (set OV_MODAL_FLOW_SOURCE_DIR)")
            else ()
                _ov_modal_flow_discover(_ov_mf_tree _ov_mf_notes)
                if (_ov_mf_tree)
                    set(OV_MODAL_FLOW_HOST_SOURCE "${_ov_mf_tree}")
                    message(STATUS "Found libmodal-flow source tree ${_ov_mf_tree} "
                        "(-DOV_MODAL_FLOW=OFF to build without it)")
                    _ov_modal_flow_host_build(${_ov_mf_required})
                    if (OV_MODAL_FLOW_HOST_ERROR)
                        list(APPEND _ov_mf_notes "${OV_MODAL_FLOW_HOST_ERROR}")
                    endif ()
                endif ()
            endif ()
        endif ()
        if (NOT OV_USE_MODAL_FLOW)
            if (_ov_mf_notes)
                string(REPLACE ";" "\n  " _ov_mf_notes "${_ov_mf_notes}")
                set(_ov_mf_notes "\n  ${_ov_mf_notes}")
            endif ()
            if (_ov_mf_required)
                message(FATAL_ERROR "OV_MODAL_FLOW=ON but no usable libmodal-flow was found: no installed "
                    "library, no source tree in the search paths (see cmake/ModalFlowHost.cmake).${_ov_mf_notes}")
            endif ()
            message(STATUS "OpenCL/modal-flow tracker unavailable; building CPU trackers "
                "(TrackOCL builds automatically when libmodal-flow is installed or its source tree is found, "
                "or set OV_MODAL_FLOW_SOURCE_DIR)${_ov_mf_notes}")
        endif ()
    endif ()
endmacro()

# Install the harness library next to ov_core_lib and its headers under
# <include_dest>/ov_modal_flow_host (never into a shared CL/ or modal_flow/ root).
function(ov_modal_flow_host_install lib_dest bin_dest include_dest)
    install(TARGETS ov_modal_flow_host
            LIBRARY DESTINATION ${lib_dest}
            ARCHIVE DESTINATION ${lib_dest}
            RUNTIME DESTINATION ${bin_dest})
    set(dest ${include_dest}/ov_modal_flow_host)
    install(DIRECTORY "${OV_MODAL_FLOW_HOST_SOURCE}/library/include/" DESTINATION ${dest}
            FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "kernels" EXCLUDE)
    install(FILES "${OV_MODAL_FLOW_HOST_GENERATED_INCLUDE}/modal_flow/Version.hpp" DESTINATION ${dest}/modal_flow)
    install(FILES "${OV_MODAL_FLOW_HOST_DIR}/include/CL/cl_ext_qcom.h" DESTINATION ${dest}/CL)
endfunction()

# TrackOCL-on-host smoke test (synthetic frames; pass image directories as
# arguments to compare against real data). Exit 77 = skipped (no OpenCL device).
# A macro: enable_testing() must run in the calling directory's scope.
macro(ov_modal_flow_host_add_tests core_libraries)
    add_executable(test_trackocl_host "${OV_MODAL_FLOW_HOST_DIR}/test_trackocl_host.cpp")
    target_link_libraries(test_trackocl_host ${core_libraries})
    enable_testing()
    add_test(NAME test_trackocl_host COMMAND test_trackocl_host)
    set_tests_properties(test_trackocl_host PROPERTIES SKIP_RETURN_CODE 77)
endmacro()
