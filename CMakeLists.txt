cmake_minimum_required(VERSION 3.5)

# Project name
project(open_vins)

add_subdirectory(ov_core)
add_subdirectory(ov_init)
add_subdirectory(ov_msckf)

# Optionally build the eval module, might not needed if building for a drone/robotic system
option(BUILD_OV_EVAL "Enable or disable building of ov_eval" OFF)
if (BUILD_OV_EVAL)
	add_subdirectory(ov_eval)
endif()

# Optionally build the targetless calibrator (an on-demand tool, never part of
# the filter build). The server's --calibrate mode links ov_zcalib_lib.
option(OV_BUILD_CALIB "Enable or disable building of ov_zcalib" OFF)
if (OV_BUILD_CALIB)
	add_subdirectory(ov_zcalib)
endif()

# Make subdirectory tests discoverable from the build root. Aggregate binaries
# into one dependency graph so -j can compile independent test TUs concurrently.
if (OV_CORE_BUILD_TESTS OR OV_INIT_BUILD_TESTS OR OV_INIT_BUILD_MINI_TESTS OR OV_MSCKF_BUILD_TESTS OR OV_ZCALIB_BUILD_TESTS)
    enable_testing()
    if (NOT CMAKE_VERSION VERSION_LESS 3.7)
        add_custom_target(ov_host_tests)
        foreach(module ov_core ov_init ov_msckf ov_zcalib)
            if (TARGET ${module}_lib)
                get_property(module_targets DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${module} PROPERTY BUILDSYSTEM_TARGETS)
                set(module_test_targets)
                foreach(module_target ${module_targets})
                    if (module_target MATCHES "^(test_|zcalib_test_)")
                        list(APPEND module_test_targets ${module_target})
                    endif()
                endforeach()
                if (module_test_targets)
                    add_custom_target(${module}_tests DEPENDS ${module_test_targets})
                    add_dependencies(ov_host_tests ${module}_tests)
                endif()
            endif()
        endforeach()
    endif()
endif()
