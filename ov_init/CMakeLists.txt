cmake_minimum_required(VERSION 3.5)
project(ov_init)


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
find_package(Boost REQUIRED COMPONENTS system filesystem thread date_time)
# Ceres-free initialization backend (ov_init::zbft_sfm). ON by default: the dynamic initializer's MLE
# refinement + covariance recovery use the in-tree lock-free RT solver instead of Ceres, and CERES IS
# NOT REQUIRED OR LINKED. Declared here (before find_package) so the default build never pulls Ceres.
# Set -DOV_INIT_CERES_FREE=OFF to fall back to the Ceres MLE (needs voxl-ceres-solver / libceres-dev).
option(OV_INIT_CERES_FREE "Use the ceres-free ov_init::zbft_sfm solver for dynamic init" ON)
if (OV_INIT_CERES_FREE)
    add_definitions(-DUSE_CERES_FREE_INIT)
    message(STATUS "ov_init: USING CERES-FREE (ov_init::zbft_sfm) DYNAMIC INITIALIZER (Ceres not required)")
else ()
    find_package(Ceres REQUIRED)
    message(STATUS "ov_init: Ceres MLE fallback | CERES: " ${Ceres_VERSION})
endif ()
message(STATUS "OPENCV: " ${OpenCV_VERSION} " | BOOST: " ${Boost_VERSION})

# check if we have our python libs files (will search for python3 then python2 installs)
# sudo apt-get install python-matplotlib python-numpy python-dev
# https://cmake.org/cmake/help/v3.10/module/FindPythonLibs.html
find_package(PythonLibs QUIET)
option(DISABLE_MATPLOTLIB "Disable or enable matplotlib plot scripts in ov_eval" OFF)
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
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

# Enable compile optimizations
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_CROSSCOMPILING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a -O3 -ffast-math -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops -ftree-vectorize")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3 -ffast-math -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops -ftree-vectorize")
endif()

# Enable debug flags (use if you want to debug in gdb)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -Wall -Wuninitialized -Wmaybe-uninitialized")

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

