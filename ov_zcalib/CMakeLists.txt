# ov_zcalib: the targetless in-flight visual-inertial calibrator (upstream-OpenVINS
# layout, minimal instrumentation). Built behind OV_BUILD_CALIB; the server's
# --calibrate mode links ov_zcalib_lib, and the ov_zcalibrate CLI is the host
# replay/forensics tool.
#
# Layout follows upstream module nouns: types/ (models), cpi/ (preintegration),
# init/ (bootstrap), sim/ (synthetic worlds), utils/ (record/writeback/feeders),
# core/ (session runner), solve/ + window/ (calib-specific, no upstream analog),
# run_calibrate.cpp entrypoint, test_*.cpp at src root.
cmake_minimum_required(VERSION 3.5)
project(ov_zcalib)

find_package(Eigen3 REQUIRED)
include(GNUInstallDirs)
include(${CMAKE_CURRENT_SOURCE_DIR}/../ov_core/cmake/EigenAbi.cmake)

# This module is also a first-class catkin/ament package. The ROS-free
# superbuild already supplies the two library targets and needs no ROS discovery.
set(ZCALIB_ROS_VERSION 0)
if (CATKIN_DEVEL_PREFIX OR "$ENV{ROS_VERSION}" STREQUAL "1")
    find_package(catkin REQUIRED COMPONENTS roscpp sensor_msgs cv_bridge ov_core ov_init)
    set(ZCALIB_ROS_VERSION 1)
    catkin_package(INCLUDE_DIRS src LIBRARIES ov_zcalib_lib
        CATKIN_DEPENDS roscpp sensor_msgs cv_bridge ov_core ov_init DEPENDS Eigen3)
    include_directories(${catkin_INCLUDE_DIRS})
elseif ("$ENV{ROS_VERSION}" STREQUAL "2")
    find_package(ament_cmake REQUIRED)
    find_package(rclcpp REQUIRED)
    find_package(sensor_msgs REQUIRED)
    find_package(cv_bridge REQUIRED)
    find_package(ov_core REQUIRED)
    find_package(ov_init REQUIRED)
    set(ZCALIB_ROS_VERSION 2)
endif ()

include_directories(
        src
        ${EIGEN3_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../ov_core/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../ov_init/src
)

# Flags MUST match ov_core/ov_init exactly: Eigen objects cross the .so boundary, and a
# vectorization/alignment mismatch (e.g. AVX in ov_init, SSE2 here) makes Eigen allocate
# with one allocator and free with another -- glibc's "double free or corruption (out)".
set(CMAKE_CXX_STANDARD 17)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_CROSSCOMPILING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a -O3 -ffast-math -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops -ftree-vectorize")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3 -ffast-math -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops -ftree-vectorize")
endif()

# Port ladder (grows with task 4): core/ (runner) + run_calibrate.cpp remain; each stage
# gated on its oracle tests before the next lands. EurocFeeder is NOT ported (operator
# standing order: Stinger/VOXL data only — the cruft dies by omission).
set(OV_ZCALIB_LIB_SOURCES
        src/cpi/AciCalibPreint.cpp
        src/solve/Factor_ImuAci3.cpp
        src/solve/WindowBA.cpp
        src/solve/JointCalib.cpp
        src/solve/BiasLinkedCalib.cpp
        src/window/LinearSeed.cpp
        src/window/WindowHarvester.cpp
        src/window/WindowScorer.cpp
        src/init/HandEyeWahba.cpp
        src/init/EpipolarTimeInit.cpp
        src/utils/SessionRecord.cpp
        src/utils/ReportJson.cpp
        src/core/CalibSessionRunner.cpp
)

# Host-only ingest front-end (TrackKLT-based; the device superbuild strips KLT --
# TrackOCL is the only tracker on target, fed in-process by the server).
if (NOT DISABLE_TRACK_KLT)
    list(APPEND OV_ZCALIB_LIB_SOURCES
            src/utils/VoxlLogFeeder.cpp
    )
endif ()

add_library(ov_zcalib_lib SHARED ${OV_ZCALIB_LIB_SOURCES})
if (TARGET ov_core_lib AND TARGET ov_init_lib)
    target_link_libraries(ov_zcalib_lib ov_core_lib ov_init_lib)
endif ()
target_include_directories(ov_zcalib_lib PUBLIC src/)
target_compile_definitions(ov_zcalib_lib PUBLIC ${OV_EIGEN_ABI_DEFINITIONS})
if (ZCALIB_ROS_VERSION EQUAL 2)
    ament_target_dependencies(ov_zcalib_lib ov_core ov_init)
elseif (ZCALIB_ROS_VERSION EQUAL 1)
    target_link_libraries(ov_zcalib_lib ${catkin_LIBRARIES})
endif ()

install(TARGETS ov_zcalib_lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Test oracles. Target names carry the zcalib_ prefix; ctest names match the targets.
option(OV_ZCALIB_BUILD_TESTS "Build ov_zcalib test/oracle executables (dev/CI only)" ON)
if (OV_ZCALIB_BUILD_TESTS)
    enable_testing()
    # Header-only numerical boundary oracle; needs neither tracker nor shared libraries.
    add_executable(zcalib_test_relrot_numeric src/test_relrot_numeric.cpp)
    add_test(NAME zcalib_test_relrot_numeric COMMAND zcalib_test_relrot_numeric)
endif ()
if (OV_ZCALIB_BUILD_TESTS AND NOT DISABLE_TRACK_KLT)
    enable_testing()
    add_executable(zcalib_test_ros_no_temperature src/test_ros_no_temperature.cpp)
    target_link_libraries(zcalib_test_ros_no_temperature ov_zcalib_lib)
    add_test(NAME zcalib_test_ros_no_temperature COMMAND zcalib_test_ros_no_temperature)
    add_executable(zcalib_test_joint_numeric src/test_joint_numeric.cpp)
    target_link_libraries(zcalib_test_joint_numeric ov_zcalib_lib)
    add_test(NAME zcalib_test_joint_numeric COMMAND zcalib_test_joint_numeric)
    add_executable(zcalib_test_window_bias_separator src/test_window_bias_separator.cpp)
    target_link_libraries(zcalib_test_window_bias_separator ov_zcalib_lib)
    add_test(NAME zcalib_test_window_bias_separator COMMAND zcalib_test_window_bias_separator)
    add_executable(zcalib_test_bias_linked_calib src/test_bias_linked_calib.cpp)
    target_link_libraries(zcalib_test_bias_linked_calib ov_zcalib_lib)
    add_test(NAME zcalib_test_bias_linked_calib COMMAND zcalib_test_bias_linked_calib)
    add_executable(zcalib_test_report_json src/test_report_json.cpp)
    target_link_libraries(zcalib_test_report_json ov_zcalib_lib)
    add_test(NAME zcalib_test_report_json COMMAND zcalib_test_report_json)
    add_executable(zcalib_test_time_bootstrap src/test_time_bootstrap.cpp)
    target_link_libraries(zcalib_test_time_bootstrap ov_zcalib_lib)
    add_test(NAME zcalib_test_time_bootstrap COMMAND zcalib_test_time_bootstrap)
    add_executable(zcalib_test_camera_refinement src/test_camera_refinement.cpp)
    target_link_libraries(zcalib_test_camera_refinement ov_zcalib_lib)
    add_test(NAME zcalib_test_camera_refinement COMMAND zcalib_test_camera_refinement)

    add_executable(zcalib_test_camera_pixel_config src/test_camera_pixel_config.cpp)
    target_link_libraries(zcalib_test_camera_pixel_config ov_zcalib_lib)
    add_test(NAME zcalib_test_camera_pixel_config COMMAND zcalib_test_camera_pixel_config)
    add_executable(zcalib_test_aci3_fd src/test_aci3_fd.cpp)
    target_link_libraries(zcalib_test_aci3_fd ov_zcalib_lib)
    add_test(NAME zcalib_test_aci3_fd COMMAND zcalib_test_aci3_fd)
    add_executable(zcalib_test_preint_chain src/test_preint_chain.cpp)
    target_link_libraries(zcalib_test_preint_chain ov_zcalib_lib)
    add_test(NAME zcalib_test_preint_chain COMMAND zcalib_test_preint_chain)
    add_executable(zcalib_test_convention_ports src/test_convention_ports.cpp)
    target_link_libraries(zcalib_test_convention_ports ov_zcalib_lib)
    add_test(NAME zcalib_test_convention_ports COMMAND zcalib_test_convention_ports)
    add_executable(zcalib_test_frontend src/test_frontend.cpp)
    target_link_libraries(zcalib_test_frontend ov_zcalib_lib)
    add_test(NAME zcalib_test_frontend COMMAND zcalib_test_frontend)
    add_executable(zcalib_test_calib_e2e src/test_calib_e2e.cpp)
    target_link_libraries(zcalib_test_calib_e2e ov_zcalib_lib)
    add_test(NAME zcalib_test_calib_e2e COMMAND zcalib_test_calib_e2e)
    add_executable(zcalib_test_session_e2e src/test_session_e2e.cpp)
    target_link_libraries(zcalib_test_session_e2e ov_zcalib_lib)
    add_test(NAME zcalib_test_session_e2e COMMAND zcalib_test_session_e2e)
    add_test(NAME zcalib_test_radtan_tangent COMMAND zcalib_test_session_e2e S4-tangent)
    add_executable(zcalib_test_tg_e2e src/test_tg_e2e.cpp)
    target_link_libraries(zcalib_test_tg_e2e ov_zcalib_lib)
    add_test(NAME zcalib_test_tg_e2e COMMAND zcalib_test_tg_e2e)
    add_executable(zcalib_test_wald_mc src/test_wald_mc.cpp)
    target_link_libraries(zcalib_test_wald_mc ov_zcalib_lib)
    add_test(NAME zcalib_test_wald_mc COMMAND zcalib_test_wald_mc)
    add_executable(zcalib_test_wald_gate src/test_wald_gate.cpp)
    target_link_libraries(zcalib_test_wald_gate ov_zcalib_lib)
    add_test(NAME zcalib_test_wald_gate COMMAND zcalib_test_wald_gate)
    add_executable(zcalib_test_export_parity src/test_export_parity.cpp)
    target_link_libraries(zcalib_test_export_parity ov_zcalib_lib)
    add_test(NAME zcalib_test_export_parity COMMAND zcalib_test_export_parity)

    # Full production-session replays and Monte Carlo sweeps belong to extended
    # host validation. Keep their original records, assertions and solver settings.
    set_tests_properties(zcalib_test_calib_e2e zcalib_test_session_e2e
            zcalib_test_radtan_tangent zcalib_test_tg_e2e zcalib_test_wald_mc
            zcalib_test_export_parity PROPERTIES LABELS "extended" RUN_SERIAL TRUE)
    set_target_properties(zcalib_test_calib_e2e zcalib_test_session_e2e
            zcalib_test_tg_e2e zcalib_test_wald_mc zcalib_test_export_parity
            PROPERTIES OV_HOST_EXTENDED_TEST TRUE)
endif ()

# The CLI (upstream run_*.cpp entrypoint pattern). Builds on device too: --replay
# consumes the tracker-free session record, so DISABLE_TRACK_KLT does not
# constrain it — on-target replay is how solve-time work gets measured.
add_executable(ov_zcalibrate src/run_calibrate.cpp)
target_link_libraries(ov_zcalibrate ov_zcalib_lib)
if (ZCALIB_ROS_VERSION EQUAL 0)
    set_target_properties(ov_zcalib_lib PROPERTIES INSTALL_RPATH "$ORIGIN")
    set_target_properties(ov_zcalibrate PROPERTIES INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
    install(TARGETS ov_zcalibrate RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
else ()
    install(TARGETS ov_zcalibrate RUNTIME DESTINATION lib/${PROJECT_NAME})
    # Capture uses the existing default TrackKLT frontend. Device replay builds
    # may deliberately strip it; they still build the calibrator and the viewer.
    if (NOT DISABLE_TRACK_KLT)
        find_package(OpenCV REQUIRED)
        if (ZCALIB_ROS_VERSION EQUAL 2)
            add_executable(ov_zcalib_capture src/ros/Ros2Capture.cpp)
            ament_target_dependencies(ov_zcalib_capture rclcpp sensor_msgs cv_bridge ov_core ov_init)
        else ()
            add_executable(ov_zcalib_capture src/ros/Ros1Capture.cpp)
            target_include_directories(ov_zcalib_capture PRIVATE ${catkin_INCLUDE_DIRS})
            target_link_libraries(ov_zcalib_capture ${catkin_LIBRARIES})
        endif ()
        target_include_directories(ov_zcalib_capture PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(ov_zcalib_capture ov_zcalib_lib ${OpenCV_LIBRARIES})
        install(TARGETS ov_zcalib_capture RUNTIME DESTINATION lib/${PROJECT_NAME})
    endif ()
endif ()
# ROS-free capture from the ov_core sensor stream (utils/sensor_stream.h): the
# studio pipes voxl-logger logs and ROS 1/2 bags through the same CaptureCore
# and TrackKLT frontend as ov_zcalib_capture, without a ROS graph.
if (NOT DISABLE_TRACK_KLT)
    find_package(OpenCV REQUIRED)
    add_executable(ov_zcalib_track src/run_track.cpp)
    target_include_directories(ov_zcalib_track PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(ov_zcalib_track ov_zcalib_lib ${OpenCV_LIBRARIES})
    if (ZCALIB_ROS_VERSION EQUAL 0)
        set_target_properties(ov_zcalib_track PROPERTIES INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
        install(TARGETS ov_zcalib_track RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    else ()
        if (ZCALIB_ROS_VERSION EQUAL 2)
            ament_target_dependencies(ov_zcalib_track ov_core ov_init)
        endif ()
        install(TARGETS ov_zcalib_track RUNTIME DESTINATION lib/${PROJECT_NAME})
    endif ()
endif ()
install(DIRECTORY src/ DESTINATION include/${PROJECT_NAME}
    FILES_MATCHING PATTERN "*.h")
if (ZCALIB_ROS_VERSION EQUAL 2)
    ament_export_include_directories(include/${PROJECT_NAME})
    ament_export_libraries(ov_zcalib_lib)
    ament_export_dependencies(ov_core ov_init Eigen3)
    ament_export_definitions(${OV_EIGEN_ABI_DEFINITIONS})
    ament_package()
endif ()
