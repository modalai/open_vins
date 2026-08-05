cmake_minimum_required(VERSION 3.3)

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

