cmake_minimum_required(VERSION 3.5)
project(ov_core)

include(${CMAKE_CURRENT_SOURCE_DIR}/../ov_core/cmake/EigenAbi.cmake)

# Include libraries (if we don't have opencv 4, then fallback to opencv 3)
# The OpenCV version needs to match the one used by cv_bridge otherwise you will get a segmentation fault!
find_package(Eigen3 REQUIRED)
# Skip find_package if OpenCV was already configured (e.g., for cross-compilation)
if (NOT OpenCV_FOUND)
    find_package(OpenCV 3 QUIET)
    if (NOT OpenCV_FOUND)
        find_package(OpenCV 4 REQUIRED)
    endif ()
endif ()
# Boost is OPTIONAL: the estimator core is boost-free (ChronoProf/FsLite/chi2 table/
# SpscRing in ov_core/src/utils). Only the ROS visualizer builds still touch it;
# ${Boost_LIBRARIES}/${Boost_INCLUDE_DIRS} expand empty when absent.
find_package(Boost QUIET COMPONENTS system filesystem thread date_time)
message(STATUS "OPENCV: " ${OpenCV_VERSION})

# If we will compile with aruco support
option(ENABLE_ARUCO_TAGS "Enable or disable aruco tag (disable if no contrib modules)" ON)
if (NOT ENABLE_ARUCO_TAGS)
    add_definitions(-DENABLE_ARUCO_TAGS=0)
    message(WARNING "DISABLING ARUCOTAG TRACKING!")
else ()
    add_definitions(-DENABLE_ARUCO_TAGS=1)
endif ()

# GPU camera payloads and TrackOCL require OpenCL AND libmodal-flow. OpenCL
# alone (common on desktop ROS machines) does not provide the VOXL types/driver.
find_package(OpenCL QUIET)
find_path(MODAL_FLOW_INCLUDE_DIR NAMES modal_flow_ocl_manager.h)
find_library(MODAL_FLOW_LIBRARY NAMES modal_flow)
set(OV_USE_MODAL_FLOW 0)
if (OpenCL_FOUND AND MODAL_FLOW_INCLUDE_DIR AND MODAL_FLOW_LIBRARY
    AND EXISTS "${MODAL_FLOW_INCLUDE_DIR}/modal_flow/Types.hpp"
    AND EXISTS "${MODAL_FLOW_INCLUDE_DIR}/modal_flow/StereoMatcher.hpp")
    set(OV_USE_MODAL_FLOW 1)
    include_directories(${OpenCL_INCLUDE_DIRS} ${MODAL_FLOW_INCLUDE_DIR})
    list(APPEND thirdparty_libraries ${OpenCL_LIBRARIES} ${MODAL_FLOW_LIBRARY})
    message(STATUS "OpenCL/modal-flow tracker enabled")
else()
    message(STATUS "OpenCL/modal-flow tracker unavailable; building CPU trackers")
endif()
add_definitions(-DHAVE_OPENCL=${OV_USE_MODAL_FLOW} -DOV_HAVE_MODAL_FLOW=${OV_USE_MODAL_FLOW})

# check if we have our python libs files (will search for python3 then python2 installs)
# sudo apt-get install python-matplotlib python-numpy python-dev
# https://cmake.org/cmake/help/v3.10/module/FindPythonLibs.html
find_package(PythonLibs QUIET)
option(DISABLE_MATPLOTLIB "Disable or enable matplotlib plot scripts in ov_eval" OFF)
set(DISABLE_MATPLOTLIB ON) # host-python headers leaked into the aarch64 build; nothing includes matplotlibcpp.h
if (PYTHONLIBS_FOUND AND NOT DISABLE_MATPLOTLIB)
    add_definitions(-DHAVE_PYTHONLIBS=1)
    message(STATUS "PYTHON VERSION: " ${PYTHONLIBS_VERSION_STRING})
    message(STATUS "PYTHON INCLUDE: " ${PYTHON_INCLUDE_DIRS})
    message(STATUS "PYTHON LIBRARIES: " ${PYTHON_LIBRARIES})
    include_directories(${PYTHON_INCLUDE_DIRS})
    list(APPEND thirdparty_libraries ${PYTHON_LIBRARIES})
endif ()

# C++17 required for structured bindings and other modern features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# Default build as release
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif ()

# Enable compile optimizations
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_CROSSCOMPILING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a -O3 -ffast-math -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops -ftree-vectorize")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3 -ffast-math -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops -ftree-vectorize")
endif()

# Enable debug flags (use if you want to debug in gdb)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -Wall -Wuninitialized -Wmaybe-uninitialized -fno-omit-frame-pointer")

# Find our ROS version!
# NOTE: Default to using the ROS1 package if both are in our enviroment
# NOTE: https://github.com/romainreignier/share_ros1_ros2_lib_demo
find_package(catkin QUIET COMPONENTS roscpp)
find_package(ament_cmake QUIET)
if (catkin_FOUND)
    message(STATUS "ROS *1* version found, building ROS1.cmake")
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROS1.cmake)
elseif (ament_cmake_FOUND)
    message(STATUS "ROS *2* version found, building ROS2.cmake")
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROS2.cmake)
else ()
    message(STATUS "No ROS versions found, building ROS1.cmake")
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROS1.cmake)
endif ()

option(OV_CORE_BUILD_TESTS "Build the independent camera projection oracle" OFF)
if (OV_CORE_BUILD_TESTS)
    # Multiprecision is a host-test oracle dependency, not a core library dependency.
    find_package(Boost REQUIRED)
    enable_testing()
    add_executable(test_camera_projection src/test_camera_projection.cpp)
    target_include_directories(test_camera_projection PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(test_camera_projection ov_core_lib ${thirdparty_libraries})
    add_test(NAME test_camera_projection COMMAND test_camera_projection)
endif ()
