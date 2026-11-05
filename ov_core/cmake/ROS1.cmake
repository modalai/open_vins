cmake_minimum_required(VERSION 3.5)

# Find ROS build system
find_package(catkin QUIET COMPONENTS roscpp rosbag sensor_msgs cv_bridge)

# Describe ROS project
option(ENABLE_ROS "Enable or disable building with ROS (if it is found)" ON)
if (catkin_FOUND AND ENABLE_ROS)
    add_definitions(-DROS_AVAILABLE=1)
    set(OV_CORE_CATKIN_INCLUDE_DIRS src/)
    set(OV_CORE_CATKIN_LIBRARIES ov_core_lib)
    if (OV_MODAL_FLOW_HOST)
        # Devel-space consumers compile modal_flow headers through ov_core's public headers.
        list(APPEND OV_CORE_CATKIN_INCLUDE_DIRS ${OV_MODAL_FLOW_HOST_INCLUDE_DIRS})
        foreach (dir ${OpenCL_INCLUDE_DIRS})
            if (NOT dir IN_LIST CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES)
                list(APPEND OV_CORE_CATKIN_INCLUDE_DIRS ${dir})
            endif ()
        endforeach ()
        list(APPEND OV_CORE_CATKIN_LIBRARIES ov_modal_flow_host)
    endif ()
    catkin_package(
            CATKIN_DEPENDS roscpp rosbag sensor_msgs cv_bridge
            INCLUDE_DIRS ${OV_CORE_CATKIN_INCLUDE_DIRS}
            LIBRARIES ${OV_CORE_CATKIN_LIBRARIES}
            CFG_EXTRAS ov_core_abi.cmake
    )
else ()
    add_definitions(-DROS_AVAILABLE=0)
    message(WARNING "BUILDING WITHOUT ROS!")
    include(GNUInstallDirs)
    set(CATKIN_PACKAGE_LIB_DESTINATION "${CMAKE_INSTALL_LIBDIR}")
    set(CATKIN_PACKAGE_BIN_DESTINATION "${CMAKE_INSTALL_BINDIR}")
    set(CATKIN_GLOBAL_INCLUDE_DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
endif ()

# Include our header files
include_directories(
        src
        ${EIGEN3_INCLUDE_DIR}
        ${Boost_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${catkin_INCLUDE_DIRS}
)

# Set link libraries used by all binaries
list(APPEND thirdparty_libraries
        ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${catkin_LIBRARIES}
)

##################################################
# Make the core library
##################################################

# Option to disable TrackKLT (requires opencv video module)
option(DISABLE_TRACK_KLT "Disable TrackKLT to avoid opencv video module dependency" OFF)
if (DISABLE_TRACK_KLT)
    add_definitions(-DDISABLE_TRACK_KLT=1)
    message(STATUS "TrackKLT DISABLED - using TrackOCL only")
endif()

list(APPEND LIBRARY_SOURCES
        src/dummy.cpp
        src/cpi/CpiV1.cpp
        src/cpi/CpiV2.cpp
        src/sim/BsplineSE3.cpp
        src/track/TrackBase.cpp
        src/track/TrackAruco.cpp
        src/track/TrackDescriptor.cpp
        src/track/TrackSIM.cpp
        src/types/Landmark.cpp
        src/feat/Feature.cpp
        src/feat/FeatureDatabase.cpp
        src/feat/FeatureInitializer.cpp
        src/utils/print.cpp
)
# Only include TrackOCL if OpenCL is available
if (OV_USE_MODAL_FLOW)
    list(APPEND LIBRARY_SOURCES src/track/TrackOCL/TrackOCL.cpp)
endif()
# Only include TrackKLT if not disabled
if (NOT DISABLE_TRACK_KLT)
    list(APPEND LIBRARY_SOURCES src/track/TrackKLT.cpp)
endif()
file(GLOB_RECURSE LIBRARY_HEADERS "src/*.h")
add_library(ov_core_lib SHARED ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})
target_compile_definitions(ov_core_lib PUBLIC ${OV_EIGEN_ABI_DEFINITIONS}
    HAVE_OPENCL=${OV_USE_MODAL_FLOW} OV_HAVE_MODAL_FLOW=${OV_USE_MODAL_FLOW})
if (OV_USE_MODAL_FLOW)
    target_include_directories(ov_core_lib PUBLIC ${OpenCL_INCLUDE_DIRS} ${MODAL_FLOW_INCLUDE_DIR})
endif()
target_link_libraries(ov_core_lib ${thirdparty_libraries})
target_include_directories(ov_core_lib PUBLIC src/)
install(TARGETS ov_core_lib
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
if (OV_MODAL_FLOW_HOST)
    # The harness library is installed beside ov_core_lib; find it without LD_LIBRARY_PATH.
    set_property(TARGET ov_core_lib APPEND PROPERTY INSTALL_RPATH "$ORIGIN")
    ov_modal_flow_host_install(${CATKIN_PACKAGE_LIB_DESTINATION} ${CATKIN_PACKAGE_BIN_DESTINATION}
            ${CATKIN_GLOBAL_INCLUDE_DESTINATION})
endif ()
install(DIRECTORY src/
        DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

##################################################
# Make binary files!
##################################################

# if (catkin_FOUND AND ENABLE_ROS)

#     add_executable(test_tracking src/test_tracking.cpp)
#     target_link_libraries(test_tracking ov_core_lib ${thirdparty_libraries})
#     install(TARGETS test_tracking
#             ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#             LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#             RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#     )

# endif ()

# add_executable(test_webcam src/test_webcam.cpp)
# target_link_libraries(test_webcam ov_core_lib ${thirdparty_libraries})
# install(TARGETS test_webcam
#         ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#         LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#         RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )



