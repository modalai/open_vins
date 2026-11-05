# Builds libmodal-flow from ${OV_MODAL_FLOW_HOST_SOURCE} (never modified) for the
# build host as the `ov_modal_flow_host` shared library. Added by
# cmake/ModalFlowHost.cmake; see that file for the user-facing options.
find_package(Threads REQUIRED)

set(_mf ${OV_MODAL_FLOW_HOST_SOURCE})
set(_lib ${_mf}/library)
set(_gen ${CMAKE_CURRENT_BINARY_DIR}/generated)

# --- sources: everything except the libmodal-pipe publishers --------------------
file(GLOB_RECURSE _mf_sources CONFIGURE_DEPENDS ${_lib}/src/*.c ${_lib}/src/*.cpp)
set(_mf_skipped)
foreach (_src ${_mf_sources})
    file(STRINGS ${_src} _pipe_include REGEX "#[ \t]*include[ \t]*[<\"](modal_pipe|pipe_interfaces)")
    if (_pipe_include)
        list(APPEND _mf_skipped ${_src})
    endif ()
endforeach ()
if (_mf_skipped)
    list(REMOVE_ITEM _mf_sources ${_mf_skipped})
    foreach (_src ${_mf_skipped})
        get_filename_component(_name ${_src} NAME)
        message(STATUS "libmodal-flow host harness: skipping ${_name} (libmodal-pipe publisher, VOXL IPC only)")
    endforeach ()
endif ()
if (NOT _mf_sources)
    message(FATAL_ERROR "No libmodal-flow sources found under ${_lib}/src")
endif ()

# --- kernels: embed with the tree's own script and kernel list -------------------
file(READ ${_lib}/CMakeLists.txt _mf_cmake)
function(_ov_mfh_kernel var default out)
    string(REGEX MATCH "set\\(${var}[ \t\r\n]+\\\${CMAKE_CURRENT_SOURCE_DIR}/(include/kernels/[A-Za-z0-9_./-]+\\.cl)"
           _match "${_mf_cmake}")
    if (_match)
        set(${out} ${_lib}/${CMAKE_MATCH_1} PARENT_SCOPE)
    else ()
        set(${out} ${_lib}/include/kernels/${default} PARENT_SCOPE)
    endif ()
endfunction()
_ov_mfh_kernel(_mf_fast_kernel_in fast_kernel.cl _fast_kernel)
_ov_mfh_kernel(_mf_track_kernel_in track_kernel.cl _track_kernel)
_ov_mfh_kernel(_mf_stereo_kernel_in stereo_zncc_band.cl _stereo_kernel)
# ATR program: order matters (pattern tables first), so reuse the tree's list.
set(_atr_kernels)
string(REGEX MATCH "set\\(_mf_atr_kernel_ins([^)]*)\\)" _atr_block "${_mf_cmake}")
if (_atr_block)
    string(REGEX MATCHALL "/([A-Za-z0-9_]+\\.cl)" _atr_names "${CMAKE_MATCH_1}")
    foreach (_name ${_atr_names})
        list(APPEND _atr_kernels ${_lib}/include/kernels/atr${_name})
    endforeach ()
endif ()
if (NOT _atr_kernels)
    file(GLOB _atr_kernels ${_lib}/include/kernels/atr/*.cl)
    list(SORT _atr_kernels)
    list(FILTER _atr_kernels EXCLUDE REGEX "brief_pattern_256\\.cl$")
    if (EXISTS ${_lib}/include/kernels/atr/brief_pattern_256.cl)
        list(INSERT _atr_kernels 0 ${_lib}/include/kernels/atr/brief_pattern_256.cl)
    endif ()
    message(WARNING "libmodal-flow host harness: ATR kernel order not found in ${_lib}/CMakeLists.txt; using ${_atr_kernels}")
endif ()
foreach (_kernel ${_fast_kernel} ${_track_kernel} ${_stereo_kernel} ${_atr_kernels})
    if (NOT EXISTS ${_kernel})
        message(FATAL_ERROR "libmodal-flow kernel ${_kernel} not found")
    endif ()
endforeach ()
add_custom_command(
        OUTPUT ${_gen}/embedded_kernels.h
        COMMAND ${CMAKE_COMMAND}
            -DMF_KERNELS_OUT=${_gen}/embedded_kernels.h
            -DMF_FAST_KERNEL_IN=${_fast_kernel}
            -DMF_TRACK_KERNEL_IN=${_track_kernel}
            -DMF_STEREO_KERNEL_IN=${_stereo_kernel}
            "-DMF_ATR_KERNEL_INS=${_atr_kernels}"
            -P ${_mf}/cmake/embed_kernels.cmake
        DEPENDS ${_mf}/cmake/embed_kernels.cmake ${_fast_kernel} ${_track_kernel} ${_stereo_kernel} ${_atr_kernels}
        COMMENT "Embedding libmodal-flow OpenCL kernels for the host harness"
        VERBATIM)

# --- version stamp (modal_flow/Version.hpp, as the tree generates it) ------------
configure_file(${_lib}/include/modal_flow/Version.hpp.in ${_gen}/include/modal_flow/Version.hpp @ONLY)
set(OV_MODAL_FLOW_HOST_GENERATED_INCLUDE ${_gen}/include PARENT_SCOPE)
set(OV_MODAL_FLOW_HOST_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include ${_lib}/include ${_gen}/include PARENT_SCOPE)

# --- library ---------------------------------------------------------------------
add_library(ov_modal_flow_host SHARED ${_mf_sources} host_runtime.cpp ${_gen}/embedded_kernels.h)
# Route device selection, queue creation and kernel dispatch of the vendor code
# (only) through host_runtime.cpp; silence warnings in code we do not own.
set_source_files_properties(${_mf_sources} PROPERTIES
        COMPILE_OPTIONS "-include;${CMAKE_CURRENT_SOURCE_DIR}/modal_flow_host_redirect.h;-w")
target_include_directories(ov_modal_flow_host
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${_lib}/include>
            $<BUILD_INTERFACE:${_gen}/include>
        PRIVATE
            ${_lib}/src/ocl
            ${_gen}
            ${OpenCV_INCLUDE_DIRS})
target_compile_definitions(ov_modal_flow_host
        PUBLIC CL_TARGET_OPENCL_VERSION=220
        PRIVATE CL_USE_DEPRECATED_OPENCL_1_2_APIS MF_KLT_MEANNORM=$<BOOL:${OV_MODAL_FLOW_KLT_MEANNORM}>)
target_compile_features(ov_modal_flow_host PUBLIC cxx_std_17)
target_compile_options(ov_modal_flow_host PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
set_source_files_properties(host_runtime.cpp PROPERTIES COMPILE_OPTIONS "-Wall;-Wextra")
if (OV_OPENCL_ICD_VENDORS)
    set_property(SOURCE host_runtime.cpp APPEND PROPERTY COMPILE_DEFINITIONS
            "OV_OPENCL_ICD_VENDORS_DEFAULT=\"${OV_OPENCL_ICD_VENDORS}\"")
    message(STATUS "OpenCL host harness: default ICD vendors directory ${OV_OPENCL_ICD_VENDORS}")
endif ()
target_link_libraries(ov_modal_flow_host PUBLIC OpenCL::OpenCL PRIVATE Threads::Threads)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Fail at link time, not at first use, if a vendor-only symbol slips through.
    target_link_options(ov_modal_flow_host PRIVATE -Wl,--no-undefined)
endif ()
set_target_properties(ov_modal_flow_host PROPERTIES
        VERSION ${MODAL_FLOW_VERSION}
        SOVERSION ${MODAL_FLOW_VERSION_MAJOR}.${MODAL_FLOW_VERSION_MINOR}.${MODAL_FLOW_VERSION_PATCH})
